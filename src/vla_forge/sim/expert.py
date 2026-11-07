"""Scripted experts that generate demonstrations, one per skill.

They plan in end-effector space (the MJCF's `gripperframe` site at the jaw tips) and turn every waypoint into
joint targets with damped-least-squares IK: position, "jaws point straight down", and (for grasps) the jaws'
closing axis turned onto the block face normal closest to the radial direction, so the wrist never rolls more
than 45 degrees. The motions are the Stand-In pick and push experts, generalised from one fixed target to any
layout.

Every expert calls `on_tick(scene)` at 10 Hz (the policy rate), so the recorder sees exactly what the policy
will see, and the action label is the joint command in force at the end of the following 0.1 s.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from .scene import (
    BOWL_R,
    CUBE_HALF,
    GRIPPER_CLOSED,
    STEPS_PER_CONTROL,
    STEPS_PER_POLICY,
    TIMESTEP,
    Scene,
)
from .tasks import TaskSpec

DT = STEPS_PER_CONTROL * TIMESTEP
DOWN = np.array([0.0, 0.0, -1.0])

GRIP_OPEN = 0.7
GRIP_DEPTH = 0.012
CENTRE_OPEN = 0.024
CENTRE_GRASP = 0.0135
HOVER_Z = 0.07
TOUCH_GAP = 0.012  # reach: gripper frame this far above the block's top face
CENTRE_CLOSED = 0.006
JAW_REACH = 0.05
BOWL_CLEAR = 0.02
CARRY_Z = 0.075
RELEASE_Z = 0.048
TURN_STEP = 0.02  # rad per 20 ms control tick while turning a carried block
ROLL_SAFE = 2.3  # rad: the SO-101 wrist roll stops at about +-2.8
HARD_SQUEEZE = 0.12
HARD_SQ_SKILLS = ("kit", "insert", "rotate")
KIT_RELEASE_Z = 0.034  # block centre when released over a tray slot (jaw tips clear the 1.6 cm dividers)
SOCKET_RELEASE_Z = 0.040  # block bottom 5 mm over the 2 cm socket plate
TRANSIT_SPEED = 0.12
APPROACH_SPEED = 0.05
LIFT_SPEED = 0.06
GRIP_RATE = 2.5
GRIP_SQUEEZE = (
    0.08  # gentle: a saturated squeeze creeps the block out of the jaws within a second
)

PUSH_Z = 0.012
CONTACT = CUBE_HALF + 0.014
STANDOFF = CUBE_HALF + 0.035
PUSH_SPEED = 0.05
STOP_ALONG = 0.006
DONE_BOX = 0.022
LATERAL_ABORT = 0.012


class IK:
    def __init__(self, scene: Scene, w_ori=0.1, w_post=0.02, damping=1e-3):
        r = scene.robot
        self.m = scene.model
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site(r.tcp_site).id
        self.n = r.n_arm
        self.qadr = scene.qadr[: self.n]
        self.dof = scene.dofadr[: self.n]
        ids = [self.m.joint(j).id for j in r.arm_joints]
        self.lo, self.hi = self.m.jnt_range[ids, 0], self.m.jnt_range[ids, 1]
        self.home = r.home
        # a locked wrist roll cannot turn the jaws onto a block face: the heading term is dropped
        self.use_heading = self.n >= 5
        self.w_ori, self.w_post, self.damping = w_ori, w_post, damping
        self.heading: np.ndarray | None = None

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(
            3, 3
        ).copy()

    def solve(self, target, q0, iters=80, tol=5e-4):
        q = np.array(q0, dtype=float)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            pos, mat = self.fk(q)
            ep = target - pos
            eo = np.cross(mat[:, 0], DOWN)
            if self.heading is not None and self.use_heading:
                eo = eo + np.cross(mat[:, 2], self.heading)
            if np.linalg.norm(ep) < tol and np.linalg.norm(eo) < 0.02:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dof], self.w_ori * jr[:, self.dof]])
            e = np.concatenate([ep, self.w_ori * eo])
            JJt = J @ J.T + self.damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(self.n) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (self.w_post * (self.home - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        pos, _ = self.fk(q)
        return q, float(np.linalg.norm(target - pos))


def grasp_heading(block_xy, yaw: float, avoid_xy=None) -> np.ndarray:
    block_xy = np.asarray(block_xy)
    radial = block_xy / max(np.linalg.norm(block_xy), 1e-9)
    normals = [
        np.array([np.cos(yaw + k * np.pi / 2), np.sin(yaw + k * np.pi / 2), 0.0])
        for k in range(4)
    ]
    normals.sort(key=lambda n: -float(n[:2] @ radial))
    if avoid_xy is None:
        return normals[0]
    for n in normals[:3]:
        if (
            np.linalg.norm(block_xy + n[:2] * JAW_REACH - np.asarray(avoid_xy))
            > BOWL_R + BOWL_CLEAR
        ):
            return n
    return normals[0]


def site_for_block(block_xyz, heading, centre: float) -> np.ndarray:
    return np.asarray(block_xyz) + GRIP_DEPTH * DOWN - centre * heading


class Expert:
    """Drives one episode on an already reset scene."""

    def __init__(
        self,
        scene: Scene,
        spec: TaskSpec,
        on_tick: Callable[[Scene], None] | None = None,
        max_seconds: float = 20.0,
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        """noise > 0 is DART: the executed joint command carries a slowly varying perturbation (std `noise`
        rad) while `scene.cmd` keeps the clean command, so recorded labels show how to get back on track."""
        self.sc, self.spec, self.on_tick = scene, spec, on_tick
        r = self.r = (
            scene.robot
        )  # the embodiment's gripper commands, jaw-centre offsets and home pose
        self.noise_std = noise
        self.rng = rng or np.random.default_rng()
        self.noise = np.zeros(r.n_arm)
        self.ik = IK(scene)
        self.q_cmd = r.home.copy()
        self.ee_cmd = self.ik.fk(r.home)[0]
        self.g_cmd = self.g_goal = r.g_closed
        # the gripper moves through its whole stroke in the same time on every embodiment
        self.grip_rate = (
            GRIP_RATE
            if r.default
            else GRIP_RATE * abs(r.g_open - r.g_closed) / (GRIP_OPEN - GRIPPER_CLOSED)
        )
        self.step = 0
        # a firmer squeeze than the default keeps the block from sliding in the long carry of the hard skills
        self.squeeze: float | None = (
            HARD_SQUEEZE if r.default and spec.skill in HARD_SQ_SKILLS else None
        )
        self.calm = False
        self.max_ik_err = 0.0
        self.max_seconds = max_seconds
        scene.cmd = np.r_[self.q_cmd, self.g_cmd]
        if on_tick:
            on_tick(scene)

    # low level
    def tick(self, ee_goal, speed):
        delta = ee_goal - self.ee_cmd
        dist = float(np.linalg.norm(delta))
        stepmax = speed * DT
        self.ee_cmd = (
            ee_goal.copy()
            if dist <= stepmax
            else self.ee_cmd + delta * (stepmax / dist)
        )
        self.g_cmd += float(
            np.clip(self.g_goal - self.g_cmd, -self.grip_rate * DT, self.grip_rate * DT)
        )
        self.q_cmd, err = self.ik.solve(self.ee_cmd, self.q_cmd)
        self.max_ik_err = max(self.max_ik_err, err)
        d = self.sc.data
        self.sc.cmd = np.r_[self.q_cmd, self.g_cmd]
        d.ctrl[:] = self.sc.to_ctrl(np.r_[self.q_cmd + self.noise, self.g_cmd])
        for _ in range(STEPS_PER_CONTROL):
            mujoco.mj_step(self.sc.model, d)
            self.step += 1
            if self.step % STEPS_PER_POLICY == 0:
                if (
                    self.calm
                ):  # precise phase: the perturbation fades out instead of being redrawn
                    self.noise = 0.6 * self.noise
                elif self.noise_std > 0:  # OU process, ~0.5 s correlation
                    self.noise = 0.8 * self.noise + self.rng.normal(
                        0, self.noise_std * 0.6, self.r.n_arm
                    )
                if self.on_tick:
                    self.on_tick(self.sc)
        return float(np.linalg.norm(ee_goal - self.ee_cmd))

    def hold(self, seconds):
        for _ in range(round(seconds / DT)):
            self.tick(self.ee_cmd.copy(), TRANSIT_SPEED)

    def goto(self, goal, speed, settle=0.1):
        for _ in range(int(self.max_seconds / DT)):
            if (
                self.tick(np.asarray(goal, float), speed) < 1e-6
                and abs(self.g_goal - self.g_cmd) < 1e-6
            ):
                break
        self.hold(settle)

    def grip(self, goal, settle):
        self.g_goal = goal
        self.goto(self.ee_cmd.copy(), TRANSIT_SPEED, settle)

    def home(self):
        self.g_goal = self.r.g_closed
        self.ik.heading = None
        self.goto(self.ik.fk(self.r.home)[0], TRANSIT_SPEED, settle=0.5)

    # skills
    def grasp(self, avoid_xy=None):
        sc = self.sc
        c0 = sc.body_pos("target")
        self.ik.heading = grasp_heading(c0[:2], sc.body_yaw("target"), avoid_xy)
        above = site_for_block(c0, self.ik.heading, self.r.centre_open)
        self.g_goal = self.r.g_open
        self.goto([*above[:2], HOVER_Z], TRANSIT_SPEED)
        self.goto(above, APPROACH_SPEED)
        self.calm = (
            True  # the squeeze itself is precise; recovery is learned on the way down
        )
        self.goto(
            site_for_block(c0, self.ik.heading, self.r.centre_grasp),
            APPROACH_SPEED * 0.5,
            settle=0.05,
        )
        self.grip(
            self.r.g_squeeze if self.squeeze is None else self.squeeze, settle=0.4
        )
        self.calm = False
        c = sc.body_pos("target")
        self.goto(
            site_for_block([*c[:2], CARRY_Z], self.ik.heading, self.r.centre_grasp),
            LIFT_SPEED,
            settle=0.1,
        )

    def _roll_after(self, xy, turn: float) -> float:
        """Wrist roll the IK needs over xy with the jaws' axis turned by `turn` more radians."""
        h, ca, sa = self.ik.heading, np.cos(turn), np.sin(turn)
        turned = np.array([ca * h[0] - sa * h[1], sa * h[0] + ca * h[1], 0.0])
        keep, self.ik.heading = h, turned
        q, _err = self.ik.solve(
            site_for_block([*xy, CARRY_Z], turned, self.r.centre_grasp), self.q_cmd
        )
        self.ik.heading = keep
        return float(q[4]) if len(q) > 4 else 0.0

    def turn_to(self, xy, yaw_goal: float):
        """Turn the carried block about the vertical until its yaw matches `yaw_goal` modulo a quarter turn:
        the jaws' closing axis is turned by the measured yaw error, a few times over, at carry height over xy.
        The block is a cube, so a quarter turn less or more also fits; when the wrist roll would run into its
        joint limit the other way round is taken."""
        for k in range(5):
            err = (yaw_goal - self.sc.body_yaw("target")) % (np.pi / 2)
            if err > np.pi / 4:
                err -= np.pi / 2
            if k == 0:
                alt = err - np.pi / 2 if err > 0 else err + np.pi / 2
                if (
                    abs(self._roll_after(xy, err))
                    > ROLL_SAFE
                    > abs(self._roll_after(xy, alt))
                ):
                    err = alt
            if abs(err) < 0.03:
                break
            a = float(err)
            h0 = self.ik.heading.copy()
            n = max(1, int(np.ceil(abs(a) / TURN_STEP)))
            for i in range(
                1, n + 1
            ):  # turn the jaws gradually: a jump would fling the block out of them
                ca, sa = np.cos(a * i / n), np.sin(a * i / n)
                self.ik.heading = np.array(
                    [ca * h0[0] - sa * h0[1], sa * h0[0] + ca * h0[1], 0.0]
                )
                self.tick(
                    site_for_block(
                        [*xy, CARRY_Z], self.ik.heading, self.r.centre_grasp
                    ),
                    TRANSIT_SPEED,
                )
            self.hold(0.25)

    def place_at(
        self, xy, release_z, correct_slip=False, yaw_goal=None, measured=False
    ):
        """Carry over xy, correct for slip in the grasp, lower and release. DART noise fades out over the
        target (a perturbed release is a dropped block, and those demos would just be thrown away)."""
        over = site_for_block([*xy, CARRY_Z], self.ik.heading, self.r.centre_grasp)
        self.goto(over, TRANSIT_SPEED, settle=0.15)
        self.calm = True
        self.hold(0.3)
        if yaw_goal is not None:
            self.turn_to(xy, yaw_goal)
        if correct_slip:
            for _ in range(4):
                err = np.asarray(xy) - self.sc.body_pos("target")[:2]
                if np.linalg.norm(err) < 0.002:
                    break
                self.goto(
                    self.ee_cmd + np.array([*err, 0.0]), APPROACH_SPEED, settle=0.15
                )
        aim = self.ee_cmd.copy()
        aim[2] += release_z - CARRY_Z
        if measured:  # the block may hang lower in the jaws than the grasp assumed: lower to where it ends up
            aim[2] = release_z - (self.sc.body_pos("target")[2] - self.sc.ee_pos()[2])
        self.goto(aim, LIFT_SPEED, settle=0.1)
        self.grip(self.r.g_open, settle=0.4)
        self.goto([*self.ee_cmd[:2], HOVER_Z + 0.02], LIFT_SPEED, settle=0.0)
        self.calm = False

    def seat(self):
        """Closed jaws come down on the released block and press it to the bottom of the socket."""
        c = self.sc.body_pos("target")
        self.calm = True
        self.g_goal = self.r.g_closed
        tip = np.array([*c[:2], 0.0]) - self.r.centre_closed * self.ik.heading
        self.goto([*tip[:2], HOVER_Z], TRANSIT_SPEED, settle=0.05)
        self.goto([*tip[:2], 2 * CUBE_HALF + 0.004], APPROACH_SPEED, settle=0.1)
        self.goto([*tip[:2], 2 * CUBE_HALF - 0.008], APPROACH_SPEED * 0.5, settle=0.3)
        self.goto([*tip[:2], HOVER_Z], LIFT_SPEED, settle=0.0)
        self.calm = False

    def run(self):
        self.hold(0.3)
        sk = self.spec.skill
        if sk == "pick_place":
            goal = self.sc.body_pos("goal")[:2]
            self.grasp(avoid_xy=goal)
            self.place_at(goal, RELEASE_Z, correct_slip=True)
            self.home()
        elif sk == "stack":
            self.grasp()
            g = self.sc.body_pos("goal")
            self.place_at(
                g[:2],
                2 * CUBE_HALF + CUBE_HALF + 0.008 + (g[2] - CUBE_HALF),
                correct_slip=True,
            )
            self.home()
        elif sk == "kit":
            goal = self.sc.body_pos("goal")[:2]
            self.grasp()
            self.place_at(goal, KIT_RELEASE_Z, correct_slip=True, measured=True)
            self.home()
        elif sk in ("insert", "rotate"):
            sock = self.sc.body_pos("goal")[:2]
            yaw = self.sc.body_yaw("goal")
            self.grasp()
            self.place_at(
                sock,
                SOCKET_RELEASE_Z,
                correct_slip=True,
                yaw_goal=yaw,
                measured=True,
            )
            self.seat()
            self.home()
        elif sk == "lift":
            self.grasp()
            c = self.sc.body_pos("target")
            up = site_for_block([*c[:2], 0.11], self.ik.heading, self.r.centre_grasp)
            up[:2] *= 0.9  # draw it in a little toward the base while lifting
            self.goto(up, LIFT_SPEED, settle=1.2)
        elif sk == "push":
            self.push()
            self.home()
        elif sk == "reach":
            self.reach()
        else:
            raise ValueError(sk)

    def reach(self):
        """Closed jaws come down to hover just above the block's top face, and stay there."""
        c = self.sc.body_pos("target")
        self.ik.heading = grasp_heading(c[:2], self.sc.body_yaw("target"))
        self.g_goal = self.r.g_closed
        tip = (
            np.array([*c[:2], 2 * CUBE_HALF + TOUCH_GAP])
            - self.r.centre_closed * self.ik.heading
        )
        self.goto([*tip[:2], HOVER_Z], TRANSIT_SPEED)
        self.goto(tip, APPROACH_SPEED, settle=1.2)

    def push(self, max_segments=4):
        sc = self.sc
        target = sc.body_pos("goal")[:2]
        self.g_goal = self.r.g_closed
        for _ in range(max_segments):
            c0 = sc.body_pos("target")[:2]
            v = target - c0
            u = v / max(np.linalg.norm(v), 1e-9)
            if np.all(np.abs(v) < DONE_BOX):
                break
            behind = c0 - u * STANDOFF
            if self.ee_cmd[2] < HOVER_Z - 1e-3:
                self.goto(
                    np.array([*self.ee_cmd[:2], HOVER_Z]), TRANSIT_SPEED, settle=0.0
                )
            self.goto([*behind, HOVER_Z], TRANSIT_SPEED)
            self.goto([*behind, PUSH_Z], TRANSIT_SPEED * 0.6)
            end = np.array([*(target - u * CONTACT), PUSH_Z])
            for _ in range(int(self.max_seconds / DT)):
                left = self.tick(end, PUSH_SPEED)
                c = sc.body_pos("target")[:2]
                lateral = abs(float(u[0] * (c - c0)[1] - u[1] * (c - c0)[0]))
                if float(np.dot(target - c, u)) < STOP_ALONG or lateral > LATERAL_ABORT:
                    break
                if left < 1e-6:
                    self.hold(0.2)
                    break
            self.hold(0.2)
            self.goto(
                self.ee_cmd + np.array([*(-u * 0.025), 0.0]),
                TRANSIT_SPEED * 0.5,
                settle=0.0,
            )
        self.goto(np.array([*self.ee_cmd[:2], HOVER_Z]), TRANSIT_SPEED, settle=0.0)
