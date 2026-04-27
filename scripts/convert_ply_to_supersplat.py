import numpy as np
from plyfile import PlyData, PlyElement
import sys

# ======================
# ARGS
# ======================
inp = sys.argv[1]
out = sys.argv[2]

print(f"📥 Loading: {inp}")

ply = PlyData.read(inp)
v = np.asarray(ply['vertex'].data)

# ======================
# POSITIONS
# ======================
x = v['x']
y = v['y']
z = v['z']

x_new = x
y_new = z
z_new = -y

# ======================
# ROTATION (QUATERNIONS)
# Nerfstudio: rot_0, rot_1, rot_2, rot_3
# ======================
qx = v['rot_0']
qy = v['rot_1']
qz = v['rot_2']
qw = v['rot_3']

qx_new = qx
qy_new = -qz
qz_new = qy
qw_new = qw

# ======================
# REBUILD
# ======================
new_v = np.empty(len(v), dtype=v.dtype)

for name in v.dtype.names:
    if name == 'x':
        new_v['x'] = x_new
    elif name == 'y':
        new_v['y'] = y_new
    elif name == 'z':
        new_v['z'] = z_new

    elif name == 'rot_0':
        new_v['rot_0'] = qx_new
    elif name == 'rot_1':
        new_v['rot_1'] = qy_new
    elif name == 'rot_2':
        new_v['rot_2'] = qz_new
    elif name == 'rot_3':
        new_v['rot_3'] = qw_new

    else:
        new_v[name] = v[name]

# ======================
# WRITE
# ======================
el = PlyElement.describe(new_v, 'vertex')
PlyData([el], text=False).write(out)

print(f"✅ Saved corrected SuperSplat PLY: {out}")
