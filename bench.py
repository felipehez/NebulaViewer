"""Timing table for NebulaViewer (sim ms/step, render ms/frame) on this GPU.

python bench.py                    # bipolar, res 128 192 256
python bench.py --scenario bubbles --res 128 192
"""
import argparse, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import warp as wp  # noqa: E402
from sim import Sim  # noqa: E402
from render import Renderer, OrbitCamera, make_look  # noqa: E402


def timed(fn, n):
    fn()
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    wp.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="bipolar")
    ap.add_argument("--res", type=int, nargs="*", default=[128, 192, 256])
    ap.add_argument("--sizes", nargs="*", default=["512x512", "1280x720", "1920x1080"])
    ap.add_argument("--steps", type=int, default=4, help="steps/frame for the fps estimate")
    a = ap.parse_args()
    rows = []
    for res in a.res:
        s = Sim(a.scenario, res=res, verbose=False)
        t0 = time.perf_counter()
        n_pre = 0
        while s.phase == "pre":
            s.step(1)
            n_pre += 1
        wp.synchronize()
        pre_ms = (time.perf_counter() - t0) / max(n_pre, 1) * 1e3
        s.step(300)                                   # let the HII region / lobes develop
        main_ms = timed(lambda: s.step(1), 200)
        nu, nphi = s.rfront.shape
        ion_ms = timed(lambda: s.ionize(), 20)
        rf_ms = timed(lambda: s.update_render_fields(force=True), 10)
        look = make_look(a.scenario)
        r_ms = {}
        for sz in a.sizes:
            W, H = map(int, sz.split("x"))
            r = Renderer(W, H)
            cam = OrbitCamera()
            cam.blender_view(a.scenario, s.L, aspect=W / H)
            r.render(s, cam, look)
            r_ms[sz] = timed(lambda: r.render(s, cam, look), 20)                 # ray march + post
            if sz == "1280x720":
                lk = dict(look, filter="cubic")
                r_ms["cubic 1280x720"] = timed(lambda: r.render(s, cam, lk), 10)

                def repack():
                    r.packed_key = None
                    r.render(s, cam, look)
                r_ms["repack+1280x720"] = timed(repack, 10)
            del r
        rows.append((res, s.shape, n_pre, pre_ms, main_ms, (nu, nphi), ion_ms, rf_ms, r_ms, s.t))
        print(f"res {res} {s.shape}: pre {pre_ms:.2f} ms/step ({n_pre} steps), main {main_ms:.2f} ms/step, "
              f"ionize {ion_ms:.2f} ms ({nu}x{nphi} rays), G0+shock {rf_ms:.2f} ms, render "
              + ", ".join(f"{k} {v:.2f}" for k, v in r_ms.items()), flush=True)
        del s
    print("\n| grid | sim ms/step | ionize ms (rays) | G0+shock ms | " + " | ".join(f"render {k}" for k in a.sizes)
          + " | est. fps @1280x720, %d st/f |" % a.steps)
    print("|---" * (5 + len(a.sizes)) + "|")
    for res, shp, n_pre, pre_ms, main_ms, rays, ion_ms, rf_ms, r_ms, t in rows:
        frame = a.steps * main_ms + r_ms.get("repack+1280x720", r_ms.get("1280x720", 0)) + rf_ms / 2
        print(f"| {shp[0]}x{shp[1]}x{shp[2]} | {main_ms:.2f} | {ion_ms:.2f} ({rays[0]}x{rays[1]}) | {rf_ms:.2f} | "
              + " | ".join(f"{r_ms[k]:.2f}" for k in a.sizes) + f" | {1000.0 / frame:.0f} |")


if __name__ == "__main__":
    main()
