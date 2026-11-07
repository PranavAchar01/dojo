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
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

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
        base = t.m.body(pre + "base").id if any(t.m.body(i).name == pre + "base" for i in range(t.m.nbody)) else None
        self.base_xy = t.d.xpos[base][:2].copy() if base is not None else np.array(ds.ARMS[pre][0][:2])

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
        cands = [np.array([np.cos(ax + k * np.pi), np.sin(ax + k * np.pi), 0.0]) for k in range(2)]
        p = self.t.pose(name)[0][:2]
        radial = (p - self.base_xy) / max(np.linalg.norm(p - self.base_xy), 1e-9)
        return max(cands, key=lambda n: float(n[:2] @ radial))

    def pick(self, name: str) -> None:
        t = self.t
        c0 = t.pose(name)[0].copy()
        c0[2] = 0.0016  # the pads sit low on the jaw: grip the bottom plate level of any brick
        self.ik.heading = self.heading_for(name)
        self.g_goal = G_OPEN
        above = self.site_for(c0, CENTRE_OPEN)
        self.goto([*above[:2], HOVER_Z], TRANSIT)
        self.goto(above, APPROACH)
        self.goto(self.site_for(c0, CENTRE_GRASP), APPROACH * 0.5, settle=0.05)
        self.grip(G_SQUEEZE, 0.4)
        c = t.pose(name)[0]
        self.goto(self.site_for([*c[:2], CARRY_Z], CENTRE_GRASP), LIFT, 0.1)

    def turn_to(self, xy, name: str, yaw_goal: float, sym: float = np.pi) -> None:
        for _ in range(5):
            err = (yaw_goal - self.t.pose(name)[1] + sym / 2) % sym - sym / 2
            if abs(err) < 0.03:
                break
            h0 = self.ik.heading.copy()
            n = max(1, int(np.ceil(abs(err) / 0.02)))
            for i in range(1, n + 1):
                ca, sa = np.cos(err * i / n), np.sin(err * i / n)
                self.ik.heading = np.array([ca * h0[0] - sa * h0[1], sa * h0[0] + ca * h0[1], 0.0])
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
        self.turn_to(xy, name, ty, sym=np.pi)
        for _ in range(4):  # slip correction
            err = xy - t.pose(name)[0][:2]
            if np.linalg.norm(err) < 0.0007:
                break
            self.goto(self.ee + np.array([*err, 0.0]), APPROACH, 0.15)
        # lower until the brick's underside is 0.4 mm above its seat, as measured in the jaws
        held = t.pose(name)[0][2] - self.ee[2]
        aim = self.ee.copy()
        aim[2] = (tp[2] + 0.0004) - held
        self.goto(aim, LIFT * 0.5, 0.1)
        self.grip(G_OPEN, 0.4)
        self.goto([*self.ee[:2], HOVER_Z + 0.02], LIFT, 0.0)
        t.step(150)
        self.pre_snap = t.error(name)  # measured before the clutch engages
        return t.try_snap(name)

    def home(self):
        self.g_goal = G_CLOSED
        self.ik.heading = None
        self.goto(self.ik.fk(S.HOME)[0], TRANSIT, 0.3)


def ds_depth(p) -> float:
    return 0.012
