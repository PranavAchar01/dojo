# Robot parts: where every model comes from

The Robot tab only offers parts that exist as published open-source MuJoCo models. Nothing in this list was
modelled for this project: the MJCF files and meshes are used as published, and the builder only composes them
(attaches an effector to an arm's last link, removes the arm's own jaw when another effector replaces it, locks a
wrist joint, adds a tool-centre site). Every catalog card in the page shows the licence and links to the source.

MuJoCo Menagerie files are copied from commit `98a70c569c4c1b6da467a20012f766ee1f8dc521` of
https://github.com/google-deepmind/mujoco_menagerie into `vendor/menagerie/<model>/`, each folder with its own
`LICENSE` and `README.md` from upstream.

## Arms

| Id | Part | Maker / model author | Licence | Source | File |
|---|---|---|---|---|---|
| `so101` | SO-101 follower arm | TheRobotStudio (SO-ARM100), used by Hugging Face LeRobot | Apache-2.0 | https://github.com/TheRobotStudio/SO-ARM100 | `vendor/so101/so101_new_calib.xml` |
| `so100` | SO-100 (SO-ARM100) arm | TheRobotStudio; MJCF by MuJoCo Menagerie | Apache-2.0 | https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/trs_so_arm100 | `vendor/menagerie/trs_so_arm100/so_arm100.xml` |
| `koch` | Koch v1.1 low-cost arm | Alexander Koch, updated by Hugging Face LeRobot; MJCF by MuJoCo Menagerie | Apache-2.0 | https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/low_cost_robot_arm | `vendor/menagerie/low_cost_robot_arm/low_cost_robot_arm.xml` |
| `wx250s` | WidowX 250 6DOF | Trossen Robotics (Interbotix); MJCF by MuJoCo Menagerie | BSD-3-Clause | https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/trossen_wx250s | `vendor/menagerie/trossen_wx250s/wx250s.xml` |
| `vx300s` | ViperX 300 6DOF | Trossen Robotics (Interbotix); MJCF by MuJoCo Menagerie | BSD-3-Clause | https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/trossen_vx300s | `vendor/menagerie/trossen_vx300s/vx300s.xml` |

## Wrists

| Id | Option | What it does |
|---|---|---|
| `native` | the arm's own wrist | every wrist joint driven |
| `locked` | roll locked | the wrist roll joint is fixed at zero and its actuator removed (one action dimension fewer). Same sourced links, no new part |

## End effectors

| Id | Part | Maker / model author | Licence | Source | File |
|---|---|---|---|---|---|
| `native` | the arm's own jaw | as the arm | as the arm | as the arm | as the arm |
| `robotiq_2f85` | Robotiq 2F-85 | Robotiq; ROS-Industrial description; MJCF by MuJoCo Menagerie | BSD-2-Clause | https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/robotiq_2f85 | `vendor/menagerie/robotiq_2f85/2f85.xml` |
| `franka_hand` | Franka Hand | Franka Robotics; MJCF by MuJoCo Menagerie | Apache-2.0 | https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda | `vendor/menagerie/franka_hand/hand.xml` |

## Factory line equipment

The Line tab's cells add a conveyor and totes to the table. Both come from published sources; nothing was modelled
from scratch.

| Part | Source | Licence | How it is used |
|---|---|---|---|
| Tote | robosuite `bins_arena.xml`, body `bin1` (floor box + four wall boxes), https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/models/assets/arenas/bins_arena.xml | MIT (`vendor/robosuite/LICENSE`) | `vendor/robosuite/bins_arena.xml` is read at run time; `sim/line.py` takes bin1's collision boxes and scales them so the opening is 9 cm square and the rim about 3.4 cm high (an SO-101 reaches over it), floor on the table. Colour per station. The visual legs, textures and second bin are not used. |
| Conveyor | MuJoCo issue #547, "Conveyor belt", answer by maintainer Yuval Tassa: a box on a slide joint with large damping driven by a constant force settles at force / damping and carries what rests on it by friction, https://github.com/google-deepmind/mujoco/issues/547 | recipe (MuJoCo itself is Apache-2.0) | `sim/line.py add_belt`: same construction (slide joint, damping 1000 N s/m), force applied through `qfrc_applied` so the policy's action vector keeps its size; a contact exclude keeps the belt off the table. 0.12 m/s while running; it stops when a part reaches the station's photo-eye. |
| Arms | `so101` and `vx300s` above | as above | Stations use the same arm models as the rest of the app. |

## Fixtures for the harder skills

| Item | Source | Licence | Note |
|---|---|---|---|
| Kitting tray (2 x 2 slots) | built procedurally from boxes in `src/vla_forge/sim/props.py` | none needed | No downloaded mesh. Slots are 5.5 cm inside, dividers 1.6 cm tall. |
| Insertion socket (and the 45 degree "diamond" socket) | built procedurally from boxes in `props.py` | none needed | Square hole with 3.5 mm clearance around the 3 cm cube and a 45 degree lead-in made of thin tilted boxes. No downloaded mesh. |

No outside mesh was used, so nothing here is attributed to an outside source.

## Looked at and left out

- **Suction cup.** No open-source MuJoCo model of a suction gripper was found in these sources, and MuJoCo has no
  suction primitive, so one would have to be modelled here. The parts rule forbids that.
- **UMI gripper.** The Menagerie UMI model is the hand-held data-collection rig (a 1.56 kg body on a free base),
  not a finger set that mounts on a wrist.
