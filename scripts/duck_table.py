"""Render the MuJoCo table (two SO-101 arms, six loose bricks). Usage: duck_table.py OUT.png"""

import sys

from PIL import Image

from vla_forge import duck_sim as ds

t = ds.Table.new()
t.reset()
Image.fromarray(t.render("overview", 1600, 1000)).save(sys.argv[1])
