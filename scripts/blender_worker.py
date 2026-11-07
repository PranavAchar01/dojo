"""Blender side of the optimizer montage. Run headless by src/vla_forge/blender_montage.py:

    blender -b --factory-startup -P scripts/blender_worker.py -- JOBS.json

JOBS.json: {"res": [w, h], "samples": n, "jobs": [{"geo": npz, "frames": npy, "idx": [frame indices], "out": [png paths]}]}

The physics is MuJoCo's. Each job is one recorded rollout: the geometry of the compiled scene (exported once per layout
structure as a .npz of geoms and meshes) is built in Blender and every requested frame sets each geom from the body
poses MuJoCo recorded, then renders it. Nothing here moves a body by itself.
"""

from __future__ import annotations

import json
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

LOOKAT = Vector((0.15, 0.0, 0.07))
AZIMUTH, ELEVATION = math.radians(148.0), math.radians(-34.0)
FOVY = math.radians(45.0)
BOX, SPHERE, CAPSULE, CYLINDER, MESH, PLANE, ELLIPSOID = 6, 2, 3, 5, 7, 0, 4


def lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def material(cache: dict, rgba, rough: float, metal: float = 0.0):
    key = (tuple(round(float(v), 3) for v in rgba[:3]), rough, metal)
    if key not in cache:
        m = bpy.data.materials.new(f"m{len(cache)}")
        m.use_nodes = True
        bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = (
            *(lin(float(v)) for v in rgba[:3]),
            1.0,
        )
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metal
        cache[key] = m
    return cache[key]


def concrete(name: str = "concrete"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 90.0
    tex.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.19, 0.195, 0.2, 1)
    ramp.color_ramp.elements[1].color = (0.27, 0.275, 0.285, 1)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.15
    coord = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.88
    return m


def primitive(kind: int, size: np.ndarray):
    """A mesh for a MuJoCo primitive, in the geom frame."""
    import bmesh

    bm = bmesh.new()
    if kind == BOX:
        bmesh.ops.create_cube(bm, size=2.0)
        bmesh.ops.scale(bm, vec=Vector(size.tolist()), verts=bm.verts)
    elif kind == SPHERE:
        bmesh.ops.create_uvsphere(
            bm, u_segments=24, v_segments=14, radius=float(size[0])
        )
    elif kind == ELLIPSOID:
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=1.0)
        bmesh.ops.scale(bm, vec=Vector(size.tolist()), verts=bm.verts)
    else:  # cylinder, capsule (drawn as a cylinder with its hemispheres folded into the length)
        r, h = float(size[0]), float(size[1])
        bmesh.ops.create_cone(
            bm, segments=24, radius1=r, radius2=r, depth=2 * h, cap_ends=True
        )
    me = bpy.data.meshes.new("prim")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = kind in (SPHERE, ELLIPSOID)
    return me


class Scene:
    def __init__(self, geo_path: str, res: tuple[int, int], samples: int):
        z = np.load(geo_path)
        self.z = z
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene
        self.sc = sc
        self.mats: dict = {}
        self.objs: list = []
        coll = sc.collection
        concrete_mat = concrete()
        n = len(z["gbody"])
        for j in range(n):
            kind = int(z["gtype"][j])
            if kind == MESH:
                a, c = int(z["vadr"][j]), int(z["vnum"][j])
                fa, fc = int(z["fadr"][j]), int(z["fnum"][j])
                verts = z["verts"][a : a + c]
                faces = z["faces"][fa : fa + fc]
                me = bpy.data.meshes.new(f"g{j}")
                me.vertices.add(len(verts))
                me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
                me.loops.add(len(faces) * 3)
                me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
                me.polygons.add(len(faces))
                me.polygons.foreach_set(
                    "loop_start", np.arange(0, len(faces) * 3, 3, dtype=np.int32)
                )
                me.polygons.foreach_set(
                    "loop_total", np.full(len(faces), 3, dtype=np.int32)
                )
                me.update(calc_edges=True)
                for p in me.polygons:
                    p.use_smooth = True
            elif kind == PLANE:
                continue
            else:
                me = primitive(kind, z["gsize"][j])
            ob = bpy.data.objects.new(f"g{j}", me)
            coll.objects.link(ob)
            rgba = z["grgba"][j]
            role = str(z["grole"][j])
            if role == "table":
                ob.data.materials.append(concrete_mat)
            else:
                ob.data.materials.append(
                    material(self.mats, rgba, 0.55 if role == "arm" else 0.45)
                )
            self.objs.append((j, ob))
        self.floor(coll, concrete_mat)
        self.light(coll)
        self.camera(coll, res)
        self.engine(res, samples)

    def floor(self, coll, mat) -> None:
        me = bpy.data.meshes.new("floor")
        me.from_pydata(
            [(-3, -3, -0.0005), (3, -3, -0.0005), (3, 3, -0.0005), (-3, 3, -0.0005)],
            [],
            [(0, 1, 2, 3)],
        )
        ob = bpy.data.objects.new("floor", me)
        ob.data.materials.append(mat)
        coll.objects.link(ob)

    def light(self, coll) -> None:
        sun = bpy.data.lights.new("sun", "SUN")
        sun.energy = 1.6
        sun.angle = math.radians(14)
        so = bpy.data.objects.new("sun", sun)
        so.rotation_euler = (math.radians(48), math.radians(8), math.radians(-35))
        coll.objects.link(so)
        fill = bpy.data.lights.new("fill", "AREA")
        fill.energy = 25
        fill.size = 1.2
        fo = bpy.data.objects.new("fill", fill)
        fo.location = (0.7, -0.5, 0.7)
        fo.rotation_euler = (math.radians(55), 0, math.radians(55))
        coll.objects.link(fo)
        w = bpy.data.worlds.new("w")
        w.use_nodes = True
        bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
        bg.inputs["Color"].default_value = (0.62, 0.64, 0.68, 1)
        bg.inputs["Strength"].default_value = 0.35
        self.sc.world = w

    def camera(self, coll, res) -> None:
        cam = bpy.data.cameras.new("cam")
        cam.sensor_fit = "VERTICAL"
        cam.angle_y = FOVY
        co = bpy.data.objects.new("cam", cam)
        coll.objects.link(co)
        self.cam = co
        self.sc.camera = co
        self.dist = 0.46

    def place_camera(self, distance: float) -> None:
        f = Vector(
            (
                math.cos(ELEVATION) * math.cos(AZIMUTH),
                math.cos(ELEVATION) * math.sin(AZIMUTH),
                math.sin(ELEVATION),
            )
        )
        pos = LOOKAT - f * distance
        self.cam.location = pos
        self.cam.rotation_euler = f.to_track_quat("-Z", "Y").to_euler()

    def engine(self, res, samples: int) -> None:
        r = self.sc.render
        for name in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            try:
                r.engine = name
                break
            except TypeError:
                continue
        r.resolution_x, r.resolution_y = res
        r.resolution_percentage = 100
        r.image_settings.file_format = "PNG"
        self.sc.view_settings.view_transform = "Standard"
        try:
            self.sc.eevee.taa_render_samples = samples
        except AttributeError:
            pass

    def pose(self, frame: np.ndarray) -> None:
        f = np.asarray(frame, float).reshape(-1, 7)
        z = self.z
        for j, ob in self.objs:
            b = int(z["gbody"][j])
            bp = f[b - 1] if b > 0 else np.array([0, 0, 0, 1, 0, 0, 0.0])
            mb = Matrix.LocRotScale(Vector(bp[:3]), Quaternion(bp[3:]), None)
            mg = Matrix.LocRotScale(
                Vector(z["gpos"][j]), Quaternion(z["gquat"][j]), None
            )
            ob.matrix_world = mb @ mg

    def render(
        self,
        frame,
        path: str,
        res: tuple[int, int] | None = None,
        distance: float = 0.46,
    ) -> None:
        if res:
            self.sc.render.resolution_x, self.sc.render.resolution_y = res
        self.place_camera(distance)
        self.pose(frame)
        self.sc.render.filepath = path
        bpy.ops.render.render(write_still=True)


def main() -> None:
    with open(sys.argv[sys.argv.index("--") + 1]) as fh:
        spec = json.load(fh)
    res = tuple(spec["res"])
    last = None
    scene = None
    for job in spec["jobs"]:
        if job["geo"] != last or scene is None:
            scene = Scene(job["geo"], res, spec["samples"])
            last = job["geo"]
        frames = np.load(job["frames"])
        for i, path in zip(job["idx"], job["out"], strict=True):
            scene.render(
                frames[i], path, tuple(job.get("res", res)), job.get("distance", 0.46)
            )


main()
