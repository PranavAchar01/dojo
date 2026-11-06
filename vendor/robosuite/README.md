# robosuite bins arena

`bins_arena.xml` and `LICENSE` are copied unmodified from robosuite (ARISE Initiative, MIT License),
https://github.com/ARISE-Initiative/robosuite, commit 5ce6643f3092639d08f7b0f90ed1c6a84f50552c,
path `robosuite/models/assets/arenas/bins_arena.xml`.

VLA Forge reads the collision geoms of body `bin1` (a floor box and four wall boxes) from this file and
scales them to a tote that fits the SO-101's reach (`src/vla_forge/sim/line.py`, `bin_geoms`).
