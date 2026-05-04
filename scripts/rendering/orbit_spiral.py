import bpy
import math
import mathutils
import sys
import argparse
import os
import json


# -----------------------------
# VERIFY BLENDER VERSION
# -----------------------------
version = bpy.app.version

if version[0] < 4:
    raise Exception(f"Blender 4+ required, found {version}")

print(f"[INFO] Blender version OK: {version}")


# -----------------------------
# ARGPARSE
# -----------------------------
def get_args():
    argv = sys.argv
    if "--" not in argv:
        return None

    argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser()

    parser.add_argument("--ply", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--trajectory", default="spiral",
                        choices=["spiral", "orbit"])

    parser.add_argument("--turns", type=float, default=2.0)

    parser.add_argument("--height", type=float, default=0.3)

    parser.add_argument("--frames", type=int, default=240)

    return parser.parse_args(argv)


args = get_args()
if args is None:
    raise Exception("Missing args")


# -----------------------------
# INPUT CHECK
# -----------------------------
if not os.path.isfile(args.ply):
    raise FileNotFoundError(args.ply)

PLY_PATH = os.path.abspath(args.ply)


# -----------------------------
# OUTPUT CHECK
# -----------------------------
OUTPUT_DIR = os.path.abspath(args.output_dir)

if not os.path.isdir(OUTPUT_DIR):
    raise NotADirectoryError(OUTPUT_DIR)

if not os.access(OUTPUT_DIR, os.W_OK):
    raise PermissionError(OUTPUT_DIR)

print(f"[INFO] Output OK: {OUTPUT_DIR}")


TRAJECTORY = args.trajectory
ROTATIONS = args.turns
HEIGHT_AMPLITUDE = args.height
NB_FRAMES = args.frames


# -----------------------------
# CLEAN SCENE
# -----------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)


# -----------------------------
# IMPORT PLY
# -----------------------------
bpy.ops.wm.ply_import(filepath=PLY_PATH)

obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj


# -----------------------------
# BBOX CENTER + SIZE
# -----------------------------
coords = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]

min_x = min(v.x for v in coords)
max_x = max(v.x for v in coords)
min_y = min(v.y for v in coords)
max_y = max(v.y for v in coords)
min_z = max(v.z for v in coords)
max_z = max(v.z for v in coords)

center = mathutils.Vector((
    (min_x + max_x) / 2,
    (min_y + max_y) / 2,
    (min_z + max_z) / 2,
))

radius = max(max_x - min_x, max_y - min_y, max_z - min_z) * 1.2


# -----------------------------
# CAMERA
# -----------------------------
cam_data = bpy.data.cameras.new("OrbitCam")
cam = bpy.data.objects.new("OrbitCam", cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam


# -----------------------------
# CAMERA MATRIX (NERFSTUDIO)
# -----------------------------
def look_at_c2w(pos, target):
    pos = mathutils.Vector(pos)
    target = mathutils.Vector(target)

    forward = (target - pos).normalized()
    up_world = mathutils.Vector((0, 0, 1))

    right = forward.cross(up_world).normalized()
    up = right.cross(forward).normalized()

    # MATRICE 4x4 PROPRE (NO TRANSPOSE)
    c2w = mathutils.Matrix((
        (right.x, up.x, forward.x, pos.x),
        (right.y, up.y, forward.y, pos.y),
        (right.z, up.z, forward.z, pos.z),
        (0.0,     0.0,   0.0,      1.0)
    ))

    return c2w


# -----------------------------
# TRAJECTORY + EXPORT
# -----------------------------
frames_out = []

for i in range(NB_FRAMES):

    t = i / NB_FRAMES
    angle = t * ROTATIONS * 2 * math.pi

    if TRAJECTORY == "orbit":
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        z = center.z

    else:  # spiral
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        z = center.z + (HEIGHT_AMPLITUDE * radius * (t - 0.5))

    pos = (x, y, z)

    cam.location = pos

    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    cam.keyframe_insert(data_path="location", frame=i)
    cam.keyframe_insert(data_path="rotation_euler", frame=i)

    # -----------------------------
    # NERFSTUDIO FRAME EXPORT
    # -----------------------------
    c2w = look_at_c2w(pos, center)

    frames_out.append({
        "file_path": f"frame_{i:04d}.png",
        "transform_matrix": [list(row) for row in c2w]
    })


# -----------------------------
# FRAME RANGE
# -----------------------------
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = NB_FRAMES


# -----------------------------
# WRITE transforms.json
# -----------------------------
output_json = {
    "fl_x": 1200,
    "fl_y": 1200,
    "cx": 640,
    "cy": 360,
    "w": 1280,
    "h": 720,
    "frames": frames_out
}

output_path = os.path.join(OUTPUT_DIR, "transforms.json")

with open(output_path, "w") as f:
    json.dump(output_json, f, indent=2)

print("[INFO] Nerfstudio transforms written to:", output_path)
