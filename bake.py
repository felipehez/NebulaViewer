"""Offline high-resolution bake of a NebulaViewer session (the 'B' key writes bakes/bake_<stamp>.json).

Runs the same Sim class as the viewer, at bake resolution, with:
  * the parameters the preview was reset with, the same seed and the same 'matched' noise
    -> same large-scale structure as the preview (finer detail on top);
  * the preview's star list replayed (incl. triggered births at their birth times; triggering off);
  * the live parameter changes (q_main / wind / jet ...) replayed at the sim times they were made;
  * full-quality radiative transfer (1024x2048 ionizing rays, FUV 256x256x512) like nebula3d.py.
Writes nebula3d-format frames (f####_rho/xi/df/g0/sh.npy + meta.json), so nebula_preview.py,
nebula_to_vdb.py and build_nebula_scene.py work unchanged on the result.

python bake.py bakes/bake_20261001_223535.json --res 384
python bake.py bakes/bake_....json --res 256 --frames 48 --png         # + quick PNG/GIF previews
python bake.py bakes/bake_....json --out sim3d/my_run   # run name for the Blender scripts in blender/
"""
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warp as wp  # noqa: E402
from sim import Sim, grid_shape  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bake_json")
    ap.add_argument("--res", type=int, default=None, help="grid cells along z (default: json high_res)")
    ap.add_argument("--t_end", type=float, default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--out", default=None, help="run directory (default: sim3d/<run>, where blender/ looks)")
    ap.add_argument("--run", default=None)
    ap.add_argument("--rays", type=int, nargs=2, default=[1024, 2048])
    ap.add_argument("--fuv_grid", type=int, nargs=3, default=[256, 256, 512])
    ap.add_argument("--no_replay_stars", action="store_true", help="re-run triggered star formation instead")
    ap.add_argument("--no_events", action="store_true", help="ignore live parameter changes, use params_now")
    ap.add_argument("--png", action="store_true", help="also render each frame with the NebulaViewer renderer")
    ap.add_argument("--png_size", type=int, nargs=2, default=[960, 960])
    a = ap.parse_args()

    doc = json.load(open(a.bake_json))
    scen = doc["scenario"]
    hr = doc["high_res"]
    res = a.res or hr["res"]
    t_end = a.t_end or hr["t_end"]
    frames = a.frames or hr["frames"]
    run = a.run or f"{os.path.splitext(os.path.basename(a.bake_json))[0].replace('bake_', scen + '_')}_{res}"
    root = a.out or os.path.join(HERE, "sim3d", run)
    os.makedirs(os.path.join(root, "frames"), exist_ok=True)
    P = dict(doc["params_now"] if a.no_events else doc["params_reset"])
    P["t_end"] = t_end
    P["frames"] = frames
    events = [] if a.no_events else sorted(doc.get("events", []), key=lambda e: e[0])
    ic = doc["preview"]["ic_mode"]
    if ic != "matched":
        print("note: the preview used ic_mode='nebula3d'; the realization will differ at another resolution")

    t0 = time.time()
    sim = Sim(scen, res=res, params=P, ic_mode=ic, rays=tuple(a.rays), fuv_grid=tuple(a.fuv_grid), fast=False)
    replay = not a.no_replay_stars
    if replay:
        sim.P["trigger"] = 0.0          # births come from the preview's list
    nx, ny, nz = sim.shape
    dx = sim.dx
    print(f"bake {scen} {nx}x{ny}x{nz} t_end={t_end} frames={frames} -> {root}", flush=True)
    while sim.phase == "pre":
        sim.step(1)
    if replay:
        sim.replace_stars(doc["stars"], doc["star_kinds"], doc["jets"], doc["jet_kinds"])
    wp.synchronize()
    print(f"pre-turbulence: {sim.ip} steps, Mach ~{sim.mach or 0:.1f} ({time.time()-t0:.0f}s)", flush=True)

    renderer = cam = look = None
    if a.png:
        from PIL import Image
        from render import Renderer, OrbitCamera, make_look
        renderer = Renderer(*a.png_size)
        c = doc["camera"]
        cam = OrbitCamera(target=c["target"], dist=c["dist"], yaw=c["yaw"], pitch=c["pitch"], fov=c.get("fov_deg", 39.6))
        look = make_look(scen, doc.get("look"))

    times = []
    dt_frame = t_end / max(frames - 1, 1)
    next_frame, frame, ev_i = 0.0, 0, 0
    tl = time.time()
    pngs = []
    while True:
        while ev_i < len(events) and events[ev_i][0] <= sim.t + 1e-12:
            _, name, val = events[ev_i]
            sim.P[name] = val
            sim._wkey = None
            print(f"   event t={sim.t:.4f}: {name} = {val}", flush=True)
            ev_i += 1
        if sim.t >= next_frame - 1e-9:
            sim.update_render_fields(force=True)
            p = os.path.join(root, "frames", f"f{frame:04d}")
            np.save(p + "_g0.npy", np.minimum(sim.g0.numpy(), 6e4).astype(np.float16))
            np.save(p + "_rho.npy", sim.rho.numpy().astype(np.float16))
            np.save(p + "_xi.npy", np.round(np.clip(sim.xi.numpy(), 0, 1) * 255).astype(np.uint8))
            np.save(p + "_df.npy", np.clip(sim.dfront.numpy(), -4, 4).astype(np.float16))
            np.save(p + "_sh.npy", np.minimum(sim.shock.numpy(), 6e4).astype(np.float16))
            times.append(sim.t)
            if renderer is not None:
                renderer.render(sim, cam, look, step_vox=0.6)
                pp = os.path.join(root, f"preview_{frame:04d}.png")
                Image.fromarray(np.ascontiguousarray(renderer.image()[..., :3])).save(pp)
                pngs.append(pp)
            print(f"frame {frame}/{frames} t={sim.t:.4f} step {sim.it} ({time.time()-tl:.0f}s)", flush=True)
            frame += 1
            next_frame += dt_frame
            if frame >= frames:
                break
        sim.dt_cap = next_frame - sim.t + 1e-7
        sim.step(1)
    stars, jets = sim.stars_absolute()
    meta = dict(sim.P, run=run, dx=dx, shape=[nx, ny, nz], stars=stars, jets=jets, times=times, steps=sim.it,
                wall_s=time.time() - t0, nu=a.rays[0], nphi=a.rays[1], fuv_grid=list(a.fuv_grid),
                bake_source=os.path.abspath(a.bake_json), ic_mode=ic, events=events,
                generator="NebulaViewer/bake.py")
    json.dump(meta, open(os.path.join(root, "meta.json"), "w"), indent=1)
    if pngs:
        from PIL import Image
        ims = [Image.open(p).convert("RGB") for p in pngs]
        w, h = ims[0].size
        small = [im.resize((w // 2, h // 2), Image.LANCZOS).quantize(colors=240) for im in ims]
        small[0].save(os.path.join(root, "preview.gif"), save_all=True, append_images=small[1:], duration=80, loop=0)
    print(f"done: {sim.it} steps, {frame} frames, {time.time()-t0:.0f}s -> {root}")
    print("next (Blender): the run folder must be sim3d/<run> next to blender/ (the default --out); then "
          f"blender -b --factory-startup --python blender/nebula_to_vdb.py -- {os.path.basename(root)}  and  "
          f"blender -b --factory-startup --python blender/build_nebula_scene.py -- {os.path.basename(root)}")


if __name__ == "__main__":
    main()
