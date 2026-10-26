"""Convert nebula3d frames (.npy) to render-ready VDBs (one file per frame).
Runs in Blender's python (bundled openvdb):

blender -b --factory-startup --python nebula_to_vdb.py -- <run> [first last step] [--scale 10]

Grids (all float, saved as half):
  dust : neutral gas + 8% of the ionized gas (dust survives partially in HII regions)
  ha   : (rho*xi)^2        recombination emission measure  -> H-alpha / H-beta / Paschen
  oiii : ha, deep inside the HII region (high-ionization zone near hot stars)
  sii  : ha, just inside the ionization front (low-ionization zone)
  g0   : FUV radiation field (dust-attenuated, 1/r^2) -> PAH/PDR glow and baked dust scattering
  shock: compression proxy max(0,-div v)*rho -> shock-excited lines (jets, Herbig-Haro knots)
"""
import sys, os, json, time
import numpy as np
import openvdb as vdb

argv = sys.argv[sys.argv.index("--") + 1:]
scale = 10.0
if "--scale" in argv:
    i = argv.index("--scale"); scale = float(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
run = argv[0]
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim3d", run)
meta = json.load(open(os.path.join(root, "meta.json")))
n_frames = len(meta["times"])
first, last, step = (int(argv[1]), int(argv[2]), int(argv[3])) if len(argv) >= 4 else (0, n_frames - 1, 1)
out = os.path.join(root, "vdb")
os.makedirs(out, exist_ok=True)
vs = meta["dx"] * scale
xf = vdb.createLinearTransform(voxelSize=vs)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def grid(name, arr, tol):
    g = vdb.FloatGrid()
    g.copyFromArray(np.ascontiguousarray(arr, dtype=np.float32), tolerance=tol)
    g.name = name
    g.transform = xf
    g.saveFloatAsHalf = True
    return g


for f in range(first, last + 1, step):
    t0 = time.time()
    p = os.path.join(root, "frames", f"f{f:04d}")
    rho = np.load(p + "_rho.npy").astype(np.float32)
    xi = np.load(p + "_xi.npy").astype(np.float32)
    if xi.max() > 1.5:
        xi /= 255.0
    df = np.load(p + "_df.npy").astype(np.float32)
    g0 = np.load(p + "_g0.npy").astype(np.float32)
    ni = rho * xi
    ha = ni * ni
    grids = [
        grid("dust", rho * (1 - xi) + 0.08 * ni, 1e-4),
        grid("ha", ha, 1e-6),
        grid("oiii", ha * sstep(-0.03, -0.12, df), 1e-6),
        grid("sii", ha * sstep(-0.05, -0.005, df), 1e-6),
        grid("g0", g0, 1e-3),
    ]
    if os.path.exists(p + "_sh.npy"):           # shock-excited emission (jets, HH knots, turbulence)
        grids.append(grid("shock", np.load(p + "_sh.npy").astype(np.float32), 1e-3))
    dst = os.path.join(out, f"neb_{f:04d}.vdb")
    vdb.write(dst, grids=grids, metadata={"run": run, "frame": f, "time": float(meta["times"][f]),
                                           "scale": scale})
    print(f"VDB {dst} {os.path.getsize(dst)/1e6:.0f} MB {time.time()-t0:.1f}s", flush=True)
