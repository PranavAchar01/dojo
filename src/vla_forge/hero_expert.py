"""A scripted two-arm expert for the duck table: pick one loose brick, turn it to the slot's yaw, lower it and release.

It follows sim/expert.py (damped least squares on the gripper frame, jaws closing along a chosen heading), but works
on the prefixed arm of the `Table` and on bricks of any footprint. The clutch is `Table.try_snap` after release.
"""

from __future__ import annotations

import mujoco
import numpy as np

from . import hero_bricks as db
from . import hero_sim as ds
from .sim import scene as S

DOWN = np.array([0.0, 0.0, -1.0])
G_OPEN, G_SQUEEZE, G_CLOSED = 0.7, -0.10, S.GRIPPER_CLOSED
CENTRE_OPEN, CENTRE_GRASP = 0.024, 0.0135
HOVER_Z, CARRY_Z = 0.07, 0.07
TRANSIT, APPROACH, LIFT = 0.12, 0.05, 0.06
DT = S.STEPS_PER_CONTROL * S.TIMESTEP
GRIP_RATE = 2.5
HOME_RATE = 1.5  # rad/s per joint on the way home


class ArmIK:
    def __init__(self, t: ds.Table, pre: str):
        self.m, self.pre = t.m, pre
        self.d = mujoco.MjData(t.m)
        self.site = t.m.site(pre + "gripperframe").id
        jn = [pre + j for j in S.JOINTS[:5]]
        self.qadr = [t.m.joint(j).qposadr[0] for j in jn]
        self.dof = [t.m.joint(j).dofadr[0] for j in jn]
        ids = [t.m.joint(j).id for j in jn]
        self.lo, self.hi = t.m.jnt_range[ids, 0], t.m.jnt_range[ids, 1]
        self.home = S.HOME.copy()
        self.heading: np.ndarray | None = None

    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(
            3, 3
        ).copy()

    def solve(self, target, q0, iters=80, tol=5e-4):
        q = np.array(q0, float)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            pos, mat = self.fk(q)
            ep = target - pos
            eo = np.cross(mat[:, 0], DOWN)
            if self.heading is not None:
                eo = eo + np.cross(mat[:, 2], self.heading)
            if np.linalg.norm(ep) < tol and np.linalg.norm(eo) < 0.02:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dof], 0.1 * jr[:, self.dof]])
            e = np.concatenate([ep, 0.1 * eo])
            JJt = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(5) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (0.02 * (self.home - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        pos, _ = self.fk(q)
        return q, float(np.linalg.norm(target - pos))


class Expert:
    def __init__(self, t: ds.Table, pre: str = "a_", on_tick=None):
        self.t, self.pre, self.on_tick = t, pre, on_tick
        self.ik = ArmIK(t, pre)
        self.q = S.HOME.copy()
        self.ee = self.ik.fk(self.q)[0]
        self.g = self.g_goal = G_CLOSED
        self.max_ik_err = 0.0
        self.trace = None  # optional callable(stage, brick name) for diagnosis
        base = (
            t.m.body(pre + "base").id
            if any(t.m.body(i).name == pre + "base" for i in range(t.m.nbody))
            else None
        )
        self.base_xy = (
            t.d.xpos[base][:2].copy()
            if base is not None
            else np.array(ds.ARMS[pre][0][:2])
        )

    def _tr(self, stage: str, name: str) -> None:
        if self.trace:
            self.trace(self, stage, name)

    def tick(self, goal, speed):
        delta = goal - self.ee
        dist = float(np.linalg.norm(delta))
        mx = speed * DT
        self.ee = goal.copy() if dist <= mx else self.ee + delta * (mx / dist)
        self.g += float(np.clip(self.g_goal - self.g, -GRIP_RATE * DT, GRIP_RATE * DT))
        self.q, err = self.ik.solve(self.ee, self.q)
        self.max_ik_err = max(self.max_ik_err, err)
        self.t.set_ctrl(self.pre, np.r_[self.q, self.g])
        self.t.step(S.STEPS_PER_CONTROL)
        if self.on_tick:
            self.on_tick(self.t)
        return float(np.linalg.norm(goal - self.ee))

    def hold(self, s):
        for _ in range(round(s / DT)):
            self.tick(self.ee.copy(), TRANSIT)

    def goto(self, goal, speed, settle=0.1):
        goal = np.asarray(goal, float)
        for _ in range(int(20 / DT)):
            if self.tick(goal, speed) < 1e-6 and abs(self.g_goal - self.g) < 1e-6:
                break
        self.hold(settle)

    def grip(self, goal, settle=0.4):
        self.g_goal = goal
        self.goto(self.ee.copy(), TRANSIT, settle)

    def site_for(self, p, centre):
        return np.asarray(p) + np.array([0, 0, -ds_depth(p)]) - centre * self.ik.heading

    def heading_for(self, name: str, short_axis: bool = True) -> np.ndarray:
        """Jaws close across the brick's short side, from the side facing the arm's base."""
        _, yaw = self.t.pose(name)
        part = db.brick_map()[name].part
        nx, ny = db.PART_FOOTPRINT[part]
        ax = yaw + (np.pi / 2 if nx >= ny else 0.0)  # direction of the short side
        turns = (
            4 if nx == ny else 2
        )  # a square brick can be gripped across any of its four faces
        step = np.pi / 2 if nx == ny else np.pi
        cands = [
            np.array([np.cos(ax + k * step), np.sin(ax + k * step), 0.0])
            for k in range(turns)
        ]
        p = self.t.pose(name)[0][:2]
        radial = (p - self.base_xy) / max(np.linalg.norm(p - self.base_xy), 1e-9)
        return max(cands, key=lambda n: float(n[:2] @ radial))

    def faces(self, name: str) -> list[np.ndarray]:
        """Every direction the jaws can close along the brick's short side: two for a plain brick, four for a square one."""
        _, yaw = self.t.pose(name)
        nx, ny = db.PART_FOOTPRINT[db.brick_map()[name].part]
        ax = yaw + (np.pi / 2 if nx >= ny else 0.0)
        n, step = (4, np.pi / 2) if nx == ny else (2, np.pi)
        return [
            np.array([np.cos(ax + k * step), np.sin(ax + k * step), 0.0])
            for k in range(n)
        ]

    def reach(self, name: str, heading: np.ndarray) -> float:
        """Worst IK miss (m) along the whole job with these jaws: hover, grasp, carry, turn to the seat's yaw, seat."""
        b = db.brick_map()[name]
        tp, tq = ds.target_pose(b)
        p, yaw = self.t.pose(name)
        nx, ny = db.PART_FOOTPRINT[b.part]
        sym = np.pi / 2 if nx == ny else np.pi
        turn = (2 * np.arctan2(tq[3], tq[0]) - yaw + sym / 2) % sym - sym / 2
        c, s = np.cos(turn), np.sin(turn)
        h_seat = np.array(
            [c * heading[0] - s * heading[1], s * heading[0] + c * heading[1], 0.0]
        )
        keep = self.ik.heading
        q, worst = S.HOME.copy(), 0.0
        c0 = p.copy()
        c0[2] = 0.0016
        legs = [
            (
                heading,
                self.site_for_h(c0, CENTRE_OPEN, heading)
                + np.array([0, 0, HOVER_Z - 0.0016 + 0.012]),
            ),
            (heading, self.site_for_h(c0, CENTRE_OPEN, heading)),
            (heading, self.site_for_h(c0, CENTRE_GRASP, heading)),
            (heading, self.site_for_h([*p[:2], CARRY_Z], CENTRE_GRASP, heading)),
            (h_seat, self.site_for_h([*tp[:2], CARRY_Z], CENTRE_GRASP, h_seat)),
            (
                h_seat,
                self.site_for_h(
                    [*tp[:2], tp[2] + 0.0004 + (p[2] - 0.0016)], CENTRE_GRASP, h_seat
                ),
            ),
        ]
        try:
            for h, goal in legs:
                self.ik.heading = h
                q, err = self.ik.solve(goal, q, iters=120)
                worst = max(worst, err)
        finally:
            self.ik.heading = keep
        return worst

    def site_for_h(self, p, centre, heading):
        return np.asarray(p) + np.array([0, 0, -ds_depth(p)]) - centre * heading

    def best_grasp(self, name: str) -> tuple[np.ndarray, float]:
        """The jaw direction whose full job this arm can reach most exactly (ties go to the side facing the base)."""
        p = self.t.pose(name)[0][:2]
        radial = (p - self.base_xy) / max(np.linalg.norm(p - self.base_xy), 1e-9)
        # The jaws only hold when they close along the line pointing away from the base (measured over 36 random bricks: every
        # grasp 70 degrees or more off it, or pointing back at the base, slipped); of those faces take one the arm reaches within 2.5 mm all the way, the most radial first.
        scored = [
            (
                float(h[:2] @ radial) < 0.7,
                self.reach(name, h) > 2.5e-3,
                -float(h[:2] @ radial),
                self.reach(name, h),
                i,
            )
            for i, h in enumerate(self.faces(name))
        ]
        *_, err, i = min(scored)
        return self.faces(name)[i], err

    def pick(self, name: str) -> None:
        t = self.t
        c0 = t.pose(name)[0].copy()
        c0[2] = (
            0.0016  # the pads sit low on the jaw: grip the bottom plate level of any brick
        )
        self.ik.heading = self.best_grasp(name)[0]
        self.g_goal = G_OPEN
        above = self.site_for(c0, CENTRE_OPEN)
        self.goto([*above[:2], HOVER_Z], TRANSIT)
        self._tr("hover", name)
        self.goto(above, APPROACH)
        self._tr("above", name)
        self.goto(self.site_for(c0, CENTRE_GRASP), APPROACH * 0.5, settle=0.05)
        self._tr("down", name)
        self.grip(G_SQUEEZE, 0.4)
        self._tr("squeezed", name)
        c = t.pose(name)[0]
        self.goto(self.site_for([*c[:2], CARRY_Z], CENTRE_GRASP), LIFT, 0.1)
        self._tr("picked", name)

    def turn_to(self, xy, name: str, yaw_goal: float, sym: float = np.pi) -> None:
        for _ in range(5):
            err = (yaw_goal - self.t.pose(name)[1] + sym / 2) % sym - sym / 2
            if abs(err) < 0.03:
                break
            h0 = self.ik.heading.copy()
            n = max(1, int(np.ceil(abs(err) / 0.02)))
            for i in range(1, n + 1):
                ca, sa = np.cos(err * i / n), np.sin(err * i / n)
                self.ik.heading = np.array(
                    [ca * h0[0] - sa * h0[1], sa * h0[0] + ca * h0[1], 0.0]
                )
                self.tick(self.site_for([*xy, CARRY_Z], CENTRE_GRASP), TRANSIT)
            self.hold(0.25)

    def place(self, name: str) -> bool:
        t = self.t
        b = db.brick_map()[name]
        tp, tq = ds.target_pose(b)
        xy = tp[:2]
        self.goto(self.site_for([*xy, CARRY_Z], CENTRE_GRASP), TRANSIT, 0.15)
        self.hold(0.3)
        ty = 2 * np.arctan2(tq[3], tq[0])
        nx, ny = db.PART_FOOTPRINT[b.part]
        self._tr("over", name)
        self.turn_to(xy, name, ty, sym=np.pi / 2 if nx == ny else np.pi)
        self._tr("turned", name)
        for _ in range(4):  # slip correction
            err = xy - t.pose(name)[0][:2]
            if np.linalg.norm(err) < 0.0007:
                break
            self.goto(self.ee + np.array([*err, 0.0]), APPROACH, 0.15)
        # lower until the brick's underside is 0.4 mm above its seat, as measured in the jaws
        held = t.pose(name)[0][2] - self.ee[2]
        aim = self.ee.copy()
        aim[2] = (tp[2] + 0.0004) - held
        self._tr("aligned", name)
        self.goto(aim, LIFT * 0.5, 0.1)
        self._tr("lowered", name)
        self.grip(G_OPEN, 0.4)
        self._tr("released", name)
        t.step(30)  # let the brick come to rest on the studs; one that is not seated yet drifts on its own later
        self.pre_snap = t.error(name)
        if t.try_snap(name):
            self.goto([*self.ee[:2], HOVER_Z + 0.02], LIFT, 0.0)
            return True
        self.goto([*self.ee[:2], HOVER_Z + 0.02], LIFT, 0.0)
        self._tr("lifted", name)
        t.step(150)
        self._tr("settled", name)
        self.pre_snap = t.error(name)  # measured before the clutch engages
        return t.try_snap(name)

    def home(self):
        """Back to the HOME joint angles, moving in joint space. An IK move to HOME's gripper point can end in another
        arm shape with the jaws out over the table, where the other arm's gripper runs into them at the bridge."""
        self.g_goal = G_CLOSED
        self.ik.heading = None
        q0 = self.q.copy()
        n = max(1, int(np.ceil(np.abs(S.HOME - q0).max() / (HOME_RATE * DT))))
        for i in range(1, n + 1):
            self.q = q0 + (S.HOME - q0) * (i / n)
            self.g += float(
                np.clip(self.g_goal - self.g, -GRIP_RATE * DT, GRIP_RATE * DT)
            )
            self.t.set_ctrl(self.pre, np.r_[self.q, self.g])
            self.t.step(S.STEPS_PER_CONTROL)
            if self.on_tick:
                self.on_tick(self.t)
        self.ee = self.ik.fk(self.q)[0]
        # wait (up to 2 s) until the arm really stands at HOME; no IK here, since HOME's jaws do not point straight down
        for _ in range(int(2.0 / DT)):
            if np.abs(self.t.arm_q(self.pre)[:5] - S.HOME).max() < 0.03:
                break
            self.t.step(S.STEPS_PER_CONTROL)
            if self.on_tick:
                self.on_tick(self.t)


def ds_depth(p) -> float:
    return 0.012
