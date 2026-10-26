"""NebulaViewer: interactive nebula3d simulation + GPU ray-marched render in a window ("EmberGen for nebulae").

python viewer.py                              # window, bipolar 128^3
python viewer.py --scenario bubbles --res 192
python viewer.py --headless --frames 600 --save_every 100 --out media/bipolar_128 --gif [--mp4]

Window: options panel on the right (Tab hides it), timeline at the bottom (scrub / play the cache),
video recording to MP4 (F9 or the REC button). Keys: README.md or F1.
"""
import argparse, datetime, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import warp as wp  # noqa: E402
import sim as simmod  # noqa: E402
from sim import Sim, make_params, grid_shape, NEB_DEFAULTS, PRE_TURB  # noqa: E402
from render import Renderer, OrbitCamera, make_look, PALETTE_NAMES  # noqa: E402
from timeline import Timeline  # noqa: E402
from recorder import Recorder, VIDEO_SIZES, crop_tans  # noqa: E402

SCENARIOS = ["bipolar", "bubbles", "cliff"]
TONEMAPS = ["agx", "agx_base", "filmic"]
# past 384: the sim + renderer need 48 B/cell, and cropped domains (P['crop']) simulate only a window of the box
RES_OPTIONS = [96, 128, 160, 192, 224, 256, 288, 320, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736,
               768, 800]
HELP = """\
mouse L-drag orbit | R-drag / Shift+L-drag pan | wheel zoom        Tab panel   H HUD   F1 help   Esc quit
Space play/pause   N step   R reset (applies the panel's * values)   1/2/3 bipolar/bubbles/cliff
Left/Right previous/next cached frame   Home/End first frame / live      F9 record video
Q q_main   W wind   J jet (Shift = lower)   [ ] or PgUp/PgDn steps/frame   U pre-turbulence   Z grid res
D dust   E emission   A ambient G0   G G0 sat   X detail   +/- exposure   P palette   T tone map   I filter
K spikes   L bg stars   V render scale   M march step   O auto-orbit   C Blender shot (Shift: whole box)
S/F12 screenshot (Shift+S: 2x HQ)   B bake (json + commands)"""


def kv_list(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def fmt_flag(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return " ".join(fmt_flag(x) for x in v)
    return str(v)


class App:
    """everything shared by the window and the headless mode"""

    def __init__(self, a):
        self.a = a
        self.scenario = a.scenario
        self.res = a.res
        self.res_next = a.res
        self.steps = a.steps
        self.running = not a.paused
        self.pre_override = {"on": True, "off": False}.get(a.pre)
        self.sim_over = kv_list(a.set)
        self.look_over = kv_list(a.look)
        self.render_scale = a.render_scale
        self.step_vox = a.step_vox
        self.auto_orbit = a.orbit
        self.last_msg = ""
        self.msg_t = 0.0
        self.fps = 0.0
        self.t_sim = self.t_render = 0.0
        self.frame_i = 0
        self.dirty = True
        self.end_pause = a.end_pause and not a.headless
        self.end_reached = False
        self.pending = {}          # reset-only parameters staged in the panel (applied by R)
        self.timeline = Timeline(a.cache_gb)
        self.cursor = None         # None = live sim; int = showing timeline.snaps[cursor]
        self.play_fps = 24.0
        self.play_acc = 0.0
        self.loop = False
        self.recorder = Recorder()
        self.rec = dict(size="1920x1080", fps=30, crf=16, png=False, hq=False, guide=True, orbit_s=12.0)
        self.rec_renderer = None
        self.last_video = ""
        self.frame_changed = False
        self.on_rebuild = None     # window: rebuild the panel when the scenario changes
        self.sim = None
        self.cam = OrbitCamera(fov=a.fov)
        self.new_scenario(self.scenario, first=True)
        if a.yaw is not None:
            self.cam.yaw = a.yaw
        if a.pitch is not None:
            self.cam.pitch = a.pitch
        if a.dist is not None:
            self.cam.dist = a.dist
        self.renderer = Renderer(*self.render_size(a.width, a.height))

    # ------------------------------------------------------------------ state
    def sim_params(self, scenario):
        p = make_params(scenario, None)
        if self.pre_override is True and p["pre_t"] <= 0:
            p.update(PRE_TURB)
        elif self.pre_override is False:
            p["pre_t"] = 0.0
        p.update(self.sim_over)
        p["seed"] = self.a.seed if self.a.seed is not None else p["seed"]
        return p

    def new_scenario(self, scenario, first=False):
        if self.recorder.active:
            self.stop_recording()
        self.scenario = scenario
        lk = make_look(scenario, self.look_over)
        if not first:
            lk["palette"] = self.look["palette"]
            lk["tonemap"] = self.look["tonemap"]
        elif self.a.palette:
            lk["palette"] = self.a.palette
        if first and self.a.exposure is not None:
            lk["exposure"] = self.a.exposure
        self.look = lk
        self.default_look = dict(lk)
        self.pending = {}
        params = self.sim_params(scenario)
        self.res_next = self.fit_res(params, self.res_next)
        self.res = self.res_next
        self.default_params = dict(params)
        if self.sim is None:
            self.sim = Sim(scenario, res=self.res, params=params, ic_mode=self.a.ic, verbose=True)
        else:
            self.sim.reset(scenario, self.res, params)
        self.timeline.clear(self.sim)
        self.cursor = None
        self.end_reached = False
        self.frame_camera()
        self.dirty = True
        if self.on_rebuild:
            self.on_rebuild()

    def reset(self):
        """re-run the ICs (+ pre-turbulence) with the current parameters (live changes kept) + staged ones"""
        p = dict(self.sim.P)
        staged = dict(self.pending)
        p.update(staged)
        self.res_next = self.fit_res(p, self.res_next)
        self.pending = {}
        if self.pre_override is True and p["pre_t"] <= 0:
            p.update({k: v for k, v in PRE_TURB.items() if k not in staged})
        elif self.pre_override is False:
            p["pre_t"] = 0.0
        elif self.pre_override is None and "pre_t" not in staged:
            p["pre_t"] = self.sim_params(self.scenario)["pre_t"]
        old_L = self.sim.L
        self.res = self.res_next
        self.sim.reset(self.scenario, self.res, p)
        self.timeline.clear(self.sim)
        self.cursor = None
        self.end_reached = False
        if tuple(old_L) != tuple(self.sim.L):
            self.frame_camera()
        self.dirty = True

    def grid_of(self, P, res):
        """grid a reset with these parameters builds at res (the crop window if P['crop'])"""
        box = max(float(P.get("box", 1.0)), 1.0)
        if P.get("crop"):
            import crop
            return crop.crop_window(self.scenario, res, box, P["crop"])[2]
        return grid_shape(self.scenario, res, box)

    def grid_info(self):
        P = dict(self.sim.P, **self.pending)
        sh = self.grid_of(P, self.res_next)
        return f"next reset: {sh[0]}x{sh[1]}x{sh[2]} = {sh[0]*sh[1]*sh[2]/1e6:.0f}M cells, ~{simmod.grid_gb(sh):.1f} GB"

    def fit_res(self, P, res):
        """largest resolution <= res (res itself or one of RES_OPTIONS) whose grid fits on the GPU (leaves room
        for the timeline)"""
        box = max(float(P.get("box", 1.0)), 1.0)
        opts = sorted(set([r for r in RES_OPTIONS if r <= res] + [res])) or [RES_OPTIONS[0]]
        for r in reversed(opts):
            sh = self.grid_of(P, r)
            if simmod.grid_gb(sh) <= self.a.max_grid_gb:
                if r != res:
                    self.message(f"res {res} with box {box:g} does not fit on the GPU (limit {self.a.max_grid_gb:g} GB): "
                                 f"using {r} ({sh[0]}x{sh[1]}x{sh[2]})")
                return r
        return RES_OPTIONS[0]

    def frame_camera(self, whole=False):
        """default shot = the Blender camera of build_nebula_scene.py (whole=True: fit the whole box). A cropped
        domain keeps the camera of the full box (shifted into window coordinates), so shot offsets keep meaning."""
        if whole:
            self.cam.frame_box(self.sim.L, aspect=self.a.width / self.a.height)
            self.cam.yaw, self.cam.pitch = 0.0, 0.0
        elif getattr(self.sim, "L_full", None) is not None:
            self.cam.blender_view(self.scenario, self.sim.L_full, aspect=self.a.width / self.a.height)
            self.cam.target = self.cam.target - np.asarray(self.sim.crop_origin)
        else:
            self.cam.blender_view(self.scenario, self.sim.L, aspect=self.a.width / self.a.height)

    def render_size(self, W, H):
        return max(64, int(W * self.render_scale)), max(64, int(H * self.render_scale))

    def message(self, s):
        self.last_msg = s
        self.msg_t = time.perf_counter()
        print(s, flush=True)

    # ------------------------------------------------------------------ timeline
    def view_tau(self):
        if self.cursor is not None and self.timeline.snaps:
            return self.timeline.snaps[self.cursor]["tau"]
        return self.sim.tau()

    def _keep_live_state(self):
        """before leaving the live state, store it so the end of the timeline is exactly where we were"""
        tl = self.timeline
        if self.cursor is None and (not tl.snaps or self.sim.tau() > tl.snaps[-1]["tau"] + 1e-9):
            tl.capture(self.sim)

    def goto(self, i):
        tl = self.timeline
        if not tl.snaps:
            return
        self._keep_live_state()
        i = max(0, min(i, len(tl.snaps) - 1))
        if tl.restore(self.sim, i):
            self.cursor = i
            self.dirty = True
            self.frame_changed = True

    def scrub(self, tau):
        self.running = False
        i = self.timeline.nearest(tau)
        if i is not None and i != self.cursor:
            self.goto(i)

    def go_live(self):
        """leave the cache: continue simulating from the last snapshot"""
        tl = self.timeline
        if self.cursor is None:
            return
        if self.cursor != len(tl.snaps) - 1:
            tl.restore(self.sim, len(tl.snaps) - 1)
        self.cursor = None
        self.dirty = True

    def branch(self):
        """a simulation change while looking at the past: drop the cached future, simulate from here"""
        if self.cursor is None:
            return
        tau = self.timeline.snaps[self.cursor]["tau"]
        n_drop = len(self.timeline.snaps) - 1 - self.cursor
        self.timeline.truncate_after(tau)
        self.cursor = None
        if n_drop:
            self.message(f"new branch from t={tau:.4f} ({n_drop} future frames dropped)")

    def transport(self, name):
        tl = self.timeline
        if name == "play":
            self.running = not self.running
            self.play_acc = 0.0
        elif name == "|<":
            self.running = False
            self.goto(0)
        elif name == "<":
            self.running = False
            self.goto((self.cursor if self.cursor is not None else len(tl.snaps)) - 1)
        elif name == ">":
            self.running = False
            if self.cursor is not None and self.cursor < len(tl.snaps) - 1:
                self.goto(self.cursor + 1)
            else:
                self.go_live()
                self.advance(self.steps)
        elif name == ">|":
            self.go_live()
        elif name == "rec":
            self.toggle_recording()

    # ------------------------------------------------------------------ per frame
    def advance(self, k=None):
        """simulate k steps (live); while looking at the cache this starts a new branch"""
        if self.cursor is not None:
            self.branch()
        t0 = time.perf_counter()
        self.sim.step(self.steps if k is None else k)
        self.timeline.maybe_capture(self.sim)
        wp.synchronize()
        self.t_sim = time.perf_counter() - t0
        self.dirty = True
        self.frame_changed = True
        s = self.sim
        if self.end_pause and not self.end_reached and s.phase == "main" and s.t >= s.P["t_end"]:
            self.end_reached = True
            self.running = False
            self.message(f"reached t_end={s.P['t_end']:g}: paused (Space continues, R restarts)")

    def tick(self, dt):
        """one displayed frame: play the cache or simulate"""
        if not self.running:
            self.t_sim = 0.0
            return
        tl = self.timeline
        if self.cursor is not None:
            self.play_acc += dt * self.play_fps
            n = int(self.play_acc)
            if n <= 0:
                return
            self.play_acc -= n
            if self.recorder.active:
                n = 1                       # recording: every cached frame once
            nxt = self.cursor + n
            if nxt >= len(tl.snaps) - 1:
                if self.loop:
                    self.goto(0)
                else:
                    self.goto(len(tl.snaps) - 1)
                    self.cursor = None      # end of the cache: continue simulating live
            else:
                self.goto(nxt)
            self.t_sim = 0.0
        else:
            self.advance()

    def render(self, out=None, flip=False):
        t0 = time.perf_counter()
        rgba = self.renderer.render(self.sim, self.cam, self.look, step_vox=self.step_vox, flip=flip)
        if out is not None:
            wp.copy(out, rgba)
        wp.synchronize()
        self.t_render = time.perf_counter() - t0
        self.dirty = False
        return rgba

    # ------------------------------------------------------------------ live parameters
    def set_sim(self, name, v):
        if self.cursor is not None:
            self.branch()
        v = self.sim.set_param(name, v)
        self.dirty = True
        return v

    def set_look(self, name, v):
        self.look[name] = v
        self.dirty = True

    def bump_sim(self, name, factor=None, delta=None, lo=0.0, hi=1e9):
        v = self.sim.P[name]
        v = v * factor if factor is not None else v + delta
        v = self.set_sim(name, min(max(v, lo), hi))
        self.message(f"{name} = {v:.4g}")

    def bump_look(self, name, factor=None, delta=None, lo=0.0, hi=1e9):
        v = self.look[name]
        v = v * factor if factor is not None else v + delta
        self.look[name] = min(max(v, lo), hi)
        self.dirty = True
        self.message(f"{name} = {self.look[name]:.4g}")

    # ------------------------------------------------------------------ video
    def rec_size(self):
        for lbl, w, h in VIDEO_SIZES:
            if lbl == self.rec["size"]:
                return (w, h) if w else (self.a.width - self.a.width % 2, self.a.height - self.a.height % 2)
        return 1920, 1080

    def rec_tans(self, w, h):
        return crop_tans(self.cam.tans(self.a.width, self.a.height), self.a.width, self.a.height, w, h)

    def render_rec_frame(self):
        w, h = self.rec_size()
        if self.rec_renderer is None or (self.rec_renderer.W, self.rec_renderer.H) != (w, h):
            self.rec_renderer = Renderer(w, h)
        lk = dict(self.look, filter="cubic") if self.rec["hq"] else self.look
        step = 0.5 if self.rec["hq"] else min(self.step_vox, 0.75)
        self.rec_renderer.render(self.sim, self.cam, lk, step_vox=step, tans=self.rec_tans(w, h)[0])
        return self.rec_renderer.image()

    def start_recording(self, tag=""):
        w, h = self.rec_size()
        path = self.recorder.start(f"{self.scenario}{tag}", (w, h), fps=int(self.rec["fps"]),
                                   crf=int(self.rec["crf"]), png=self.rec["png"])
        self.frame_changed = True
        self.message(f"recording {w}x{h} @ {self.rec['fps']} fps -> {path}")
        return path

    def stop_recording(self):
        n = self.recorder.n
        p = self.recorder.stop()
        if p:
            self.last_video = p
            self.message(f"video saved ({n} frames): {p}")
        return p

    def toggle_recording(self):
        if self.recorder.active:
            self.stop_recording()
        else:
            self.start_recording()

    def record_if_needed(self):
        """window: add a video frame whenever the image changed (sim, playback or camera)"""
        if self.recorder.active and self.frame_changed:
            self.recorder.add(self.render_rec_frame())
        self.frame_changed = False

    def export_timeline(self):
        """the whole cache -> MP4 (one video frame per cached snapshot)"""
        tl = self.timeline
        self._keep_live_state()
        if len(tl.snaps) < 2:
            self.message("the timeline is empty: simulate first")
            return None
        was, was_running = self.cursor, self.running
        self.running = False
        self.start_recording("_timeline")
        for i in range(len(tl.snaps)):
            tl.restore(self.sim, i)
            self.cursor = i
            self.recorder.add(self.render_rec_frame())
            if i % 20 == 0:
                print(f"   export {i + 1}/{len(tl.snaps)}", flush=True)
        p = self.stop_recording()
        self.goto(was if was is not None else len(tl.snaps) - 1)
        if was is None:
            self.cursor = None
        self.running = was_running
        return p

    def export_orbit(self):
        """360 deg orbit around the current state -> MP4"""
        n = max(2, int(self.rec["orbit_s"] * self.rec["fps"]))
        was_running = self.running
        self.running = False
        yaw0 = self.cam.yaw
        self.start_recording("_orbita")
        for i in range(n):
            self.cam.yaw = yaw0 + 360.0 * i / n
            self.recorder.add(self.render_rec_frame())
        self.cam.yaw = yaw0
        p = self.stop_recording()
        self.running = was_running
        self.dirty = True
        return p

    def save_png(self, path=None, img=None, hq=False):
        """save the current frame; hq=True renders a separate 2x-size still (tricubic, fine steps)"""
        from PIL import Image
        if path is None:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(HERE, "media", f"shot_{self.scenario}_{stamp}{'_hq' if hq else ''}.png")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if hq and img is None:
            W, H = self.a.width * 2, self.a.height * 2
            r = Renderer(W, H)
            r.render(self.sim, self.cam, dict(self.look, filter="cubic"), step_vox=0.4)
            img = r.image()
            del r
        img = self.renderer.image() if img is None else img
        Image.fromarray(np.ascontiguousarray(img[..., :3])).save(path)
        self.message(f"saved {path}")
        return path

    # ------------------------------------------------------------------ bake
    def bake(self, res=None):
        """write bakes/bake_<stamp>.json: nebula3d.py command (same params, high res) + bake.py command
        (same realization via the NebulaViewer Sim) + look/camera for Blender."""
        res = res or self.a.bake_res
        s = self.sim
        P = dict(s.P)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run = f"{self.scenario}_{stamp}"
        nx, ny, nz = grid_shape(self.scenario, res)       # nebula3d.py has no box: unit box
        t_end = s.t if (s.phase == "main" and s.t > 0.02) else P["t_end"]
        frames = max(24, int(round(t_end * 240)))
        skip = {"run", "nx", "ny", "nz", "t_end", "frames", "nu", "nphi", "fuv_grid", "scenario"}
        flags = ["--run", run, "--scenario", self.scenario, "--nx", str(nx), "--ny", str(ny), "--nz", str(nz),
                 "--t_end", fmt_flag(t_end), "--frames", str(frames), "--nu", "1024", "--nphi", "2048",
                 "--fuv_grid", "256", "256", "512"]
        for k in NEB_DEFAULTS:
            if k in skip or k not in P:
                continue
            if P[k] != NEB_DEFAULTS[k]:
                flags += [f"--{k}", fmt_flag(P[k])]
        neb = os.path.join(simmod.NEBULA_SCRIPTS, "nebula3d.py")
        cmd = "python " + neb + " " + " ".join(flags)
        path = os.path.join(HERE, "bakes", f"bake_{stamp}.json")
        cam = self.cam
        pos, fwd, right, up = cam.basis()
        lk = self.look
        doc = {
            "created": stamp,
            "scenario": self.scenario,
            "nebula3d_cmd": cmd,
            "bake_cmd": f"python {os.path.join(HERE, 'bake.py')} {path} --res {res}",
            "notes": ("nebula3d_cmd: same parameters, but a different random realization (nebula3d draws its "
                      "noise per resolution), live parameter changes are not replayed, and the viewer-only "
                      "options (box, q_ramp, solver hll) do not exist there. bake_cmd: NebulaViewer's Sim at high "
                      "resolution with the same matched noise, same stars and the event log replayed -> same "
                      "large-scale structure as the preview; writes nebula3d-format frames."),
            "preview": {"res": self.res, "shape": list(s.shape), "ic_mode": s.ic_mode, "t": s.t, "phase": s.phase,
                        "steps": s.it, "pre_steps": s.ip, "mach": s.mach},
            "params_reset": getattr(s, "P_reset", P),
            "params_now": P,
            "events": s.events,
            "stars": [[float(v) for v in st] for st in s.stars],
            "star_kinds": s.kinds,
            "jets": [list(map(float, j)) for j in s.jets],
            "jet_kinds": s.jkinds,
            "high_res": {"res": res, "shape": [nx, ny, nz], "t_end": t_end, "frames": frames},
            "look": {k: v for k, v in lk.items()},
            "blender_nebula_volume": {
                "Dust Opacity": lk["dust_opacity"], "Emission Scale": lk["emission"], "H-alpha": lk["ha"],
                "OIII": lk["oiii"], "SII": lk["sii"], "PAH": lk["pah"], "Dust Glow": lk["glow"],
                "Shock": lk["shock"], "G0 Saturation": lk["gs"], "Ambient G0": lk["ambient"],
                "Edge Fade": lk["edge_fade"], "palette": lk["palette"], "view_exposure": lk["exposure"]},
            "camera": {"yaw": cam.yaw, "pitch": cam.pitch, "dist": cam.dist, "fov_deg": cam.fov,
                       "fov_note": "on the larger image side (Blender sensor fit Auto); 39.6 = 50 mm lens, 36 mm sensor",
                       "lens_mm": 18.0 / math.tan(math.radians(cam.fov) * 0.5),
                       "target": cam.target.tolist(), "location_domain": pos.tolist(),
                       "location_blender_BU": cam.blender_location()[0],
                       "rotation_euler_deg_blender": cam.blender_location()[1],
                       "image_size": [self.renderer.W, self.renderer.H]},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(doc, f, indent=1, default=float)
        print("\n=== BAKE ===============================================================")
        print(cmd)
        print("# same realization as the preview (NebulaViewer Sim, matched noise):")
        print(doc["bake_cmd"])
        print(f"-> {path}\n", flush=True)
        self.message(f"bake saved: bakes/bake_{stamp}.json")
        return path, doc

    # ------------------------------------------------------------------ keyboard (window independent)
    def on_key(self, name, shift=False):
        """handle a key by pyglet symbol name ('Q', '_1', 'BRACKETRIGHT', ...). Shift = lower / reverse.
        Returns an action the window layer must do ('quit', 'render_scale', 'hud', 'help', 'panel') or None."""
        s = self.sim
        up = -1 if shift else 1
        lk = self.look
        if name == "ESCAPE":
            return "quit"
        if name == "SPACE":
            self.transport("play")
            self.message("play" if self.running else "pause")
        elif name in ("N", "PERIOD"):
            self.go_live()
            self.advance(1)
        elif name == "LEFT":
            self.transport("<")
        elif name == "RIGHT":
            self.transport(">")
        elif name == "HOME":
            self.transport("|<")
        elif name == "END":
            self.transport(">|")
        elif name == "F9":
            self.toggle_recording()
        elif name == "TAB":
            return "panel"
        elif name == "R":
            self.reset()
            self.message(f"reset {self.scenario}")
        elif name in ("_1", "_2", "_3", "NUM_1", "NUM_2", "NUM_3"):
            self.new_scenario(SCENARIOS[int(name[-1]) - 1])
            self.message(f"scenario {self.scenario}")
        elif name == "Z":
            i = min(range(len(RES_OPTIONS)), key=lambda j: abs(RES_OPTIONS[j] - self.res))
            self.res_next = RES_OPTIONS[max(0, min(len(RES_OPTIONS) - 1, i + up))]
            self.reset()
            self.message(f"grid resolution {self.res} (reset)")
        elif name == "U":
            cur = s.P["pre_t"] > 0 if self.pre_override is None else self.pre_override
            self.pre_override = not cur
            self.message(f"pre-turbulence {'on' if self.pre_override else 'off'} at next reset (R)")
        elif name in ("BRACKETRIGHT", "BRACERIGHT", "PAGEUP", "BRACKETLEFT", "BRACELEFT", "PAGEDOWN"):
            more = name in ("BRACKETRIGHT", "BRACERIGHT", "PAGEUP")
            self.steps = min(self.steps + 1, 64) if more else max(self.steps - 1, 1)
            self.message(f"steps/frame {self.steps}")
        elif name == "Q":
            self.bump_sim("q_main", factor=1.25 ** up, lo=1e-6, hi=1.0)
        elif name == "W":
            self.bump_sim("wind", delta=0.1 * up, lo=0.0, hi=5.0)
        elif name == "J":
            self.bump_sim("jet", delta=0.5 * up, lo=0.0, hi=20.0)
        elif name == "D":
            self.bump_look("dust_opacity", factor=1.25 ** up, lo=0.01, hi=100.0)
        elif name == "E":
            self.bump_look("emission", factor=1.25 ** up, lo=0.01, hi=1e4)
        elif name == "A":
            self.bump_look("ambient", delta=0.05 * up, lo=0.0, hi=50.0)
        elif name == "G":
            self.bump_look("gs", factor=2.0 ** up, lo=0.1, hi=1e5)
        elif name == "X":
            self.bump_look("detail", delta=0.5 * up, lo=0.0, hi=4.0)
        elif name in ("PLUS", "EQUAL", "NUM_ADD"):
            self.bump_look("exposure", delta=0.25, lo=-10, hi=10)
        elif name in ("MINUS", "NUM_SUBTRACT", "UNDERSCORE"):
            self.bump_look("exposure", delta=-0.25, lo=-10, hi=10)
        elif name == "P":
            i = (PALETTE_NAMES.index(lk["palette"]) + up) % len(PALETTE_NAMES)
            lk["palette"] = PALETTE_NAMES[i]
            self.message(f"palette {lk['palette']}")
        elif name == "T":
            lk["tonemap"] = TONEMAPS[(TONEMAPS.index(lk["tonemap"]) + up) % len(TONEMAPS)]
            self.message(f"tone map {lk['tonemap']}")
        elif name == "K":
            lk["spikes"] = not lk["spikes"]
        elif name == "L":
            lk["bg_stars"] = 0.0 if lk["bg_stars"] > 0 else 1.0
        elif name == "I":
            opts = ["linear", "smooth", "cubic"]
            lk["filter"] = opts[(opts.index(lk["filter"]) + up) % 3]
            self.message(f"volume filter {lk['filter']}")
        elif name == "M":
            opts = [0.5, 0.75, 1.0, 1.5]
            self.step_vox = opts[(opts.index(self.step_vox) + up) % 4] if self.step_vox in opts else 0.75
            self.message(f"march step {self.step_vox} voxel")
        elif name == "V":
            opts = [1.0, 0.75, 0.5]
            self.render_scale = opts[(opts.index(self.render_scale) + 1) % 3] if self.render_scale in opts else 1.0
            self.message(f"render scale {self.render_scale}")
            self.dirty = True
            return "render_scale"
        elif name == "O":
            self.auto_orbit = 0.0 if self.auto_orbit else 0.3
        elif name == "C":
            self.frame_camera(whole=shift)
        elif name in ("S", "F12"):
            if shift:
                self.save_png(hq=True)
            else:
                self.save_png()
        elif name == "B":
            self.bake()
        elif name == "H":
            return "hud"
        elif name == "F1":
            return "help"
        else:
            return None
        self.dirty = True
        return None

    # ------------------------------------------------------------------ HUD
    def hud(self):
        s = self.sim
        nx, ny, nz = s.shape
        lines = [
            f"{self.fps:5.1f} fps   sim {self.t_sim*1e3:5.1f} ms ({self.steps} st/f, "
            f"{(self.t_sim/max(self.steps,1))*1e3:.2f} ms/st)   render {self.t_render*1e3:5.1f} ms   "
            f"{self.renderer.W}x{self.renderer.H}",
            f"{self.scenario} {nx}x{ny}x{nz}  box {s.L[2]:g}  {s.time_label()}  step {s.it}  dt {s.dt:.1e}  "
            f"{'PLAY' if self.running else 'PAUSE'}{' (cache)' if self.cursor is not None else ''}  "
            f"stars {len(s.active_stars())}/{len(s.stars)}" + (f"  Mach~{s.mach:.1f}" if s.mach else ""),
        ]
        if self.last_msg and time.perf_counter() - self.msg_t < 5.0:
            lines.append(">> " + self.last_msg)
        return "\n".join(lines)


# ----------------------------------------------------------------------------------- panel contents
def build_panel(app):
    """the widget list of the options panel for the current scenario"""
    import ui
    scen = app.scenario
    P = lambda: app.sim.P
    dP = app.default_params
    W = []

    def live(label, name, lo, hi, log=False, fmt="{:.3g}", help="", integer=False):
        W.append(ui.Slider(label, lambda: P()[name], lambda v: app.set_sim(name, v), lo, hi, log=log, fmt=fmt,
                           default=dP.get(name), integer=integer, help=help))

    def staged(label, name, lo, hi, log=False, fmt="{:.3g}", help="", integer=False):
        ref = lambda: app.sim.P_reset.get(name, P().get(name))
        W.append(ui.Slider(label, lambda: app.pending.get(name, ref()),
                           lambda v: app.pending.__setitem__(name, v), lo, hi, log=log, fmt=fmt,
                           default=dP.get(name), integer=integer, pending=lambda: name in app.pending
                           and app.pending[name] != ref(), help=help + "  (applied with 'Apply and reset')"))

    def look(label, name, lo, hi, log=False, fmt="{:.3g}", help=""):
        W.append(ui.Slider(label, lambda: app.look[name], lambda v: app.set_look(name, v), lo, hi, log=log,
                           fmt=fmt, default=app.default_look.get(name), help=help))

    def set_steps(v):
        app.steps = int(v)

    # --- simulation (live)
    W.append(ui.Section("Simulation (live)"))
    W.append(ui.Info(lambda: f"{scen}  {app.sim.shape[0]}x{app.sim.shape[1]}x{app.sim.shape[2]}  "
                             f"{app.sim.time_label()}  stars {len(app.sim.active_stars())}/{len(app.sim.stars)}"))
    live("Star Q (ionizing photons)", "q_main", 1e-5, 1e-1, log=True,
         help="Ionizing photons per unit time of the main stars: Strömgren radius ~ Q^(1/3).")
    live("Star turn-on time", "q_ramp", 0.0, 0.1, fmt="{:.3f}",
         help="Time for a new star to reach its full UV output (contraction to the main sequence, "
              "~10^4-10^5 years). 0 = instant. Its HII region starts trapped and grows with it.")
    live("Stellar wind / radiation pressure", "wind", 0.0, 3.0,
         help="Radial push on the ionized gas (1/r^2): empties the bubbles and compresses their walls.")
    if scen == "bipolar":
        live("Bipolar jets", "jet", 0.0, 15.0, help="Momentum injected in two narrow cones along the axis.")
        live("Jet pulsing", "jet_pulse", 0.0, 1.0,
             help="Jet velocity variability: makes knots (Herbig-Haro objects).")
        live("Pulse period", "jet_period", 0.005, 0.1, log=True)
    else:
        live("Protostellar jets", "jet_proto", 0.0, 10.0, help="Jets of the new / embedded stars.")
    live("Dust FUV opacity", "kappa_fuv", 1.0, 60.0, log=True,
         help="How much dust blocks the non-ionizing UV (PAH glow / photodissociation).")
    live("Sustained turbulence", "drive_main", 0.0, 1.0,
         help="Keeps stirring the gas during the main phase (fraction of the initial forcing).")
    W.append(ui.Slider("Sim steps per frame", lambda: app.steps, set_steps, 1, 32, integer=True, fmt="{:d}",
                       default=4, help="More steps = the sim advances faster per frame (lower fps)."))
    W.append(ui.Row(ui.Button("Play / Pause", lambda: app.transport("play"), help="Space"),
                    ui.Button("Step", lambda: (app.go_live(), app.advance(1)), help="N"),
                    ui.Button("Reset", app.reset, color=(120, 70, 50, 255), help="R: restart with the * values")))

    # --- initial conditions (reset)
    W.append(ui.Section("Initial conditions (reset)", collapsed=False))
    W.append(ui.Choice("Scenario", SCENARIOS, lambda: app.scenario, lambda v: app.new_scenario(v),
                       help="1/2/3. Changing the scenario restarts the simulation."))
    W.append(ui.Choice("Resolution (cells per unit)", RES_OPTIONS, lambda: app.res_next,
                       lambda v: setattr(app, "res_next", v), pending=lambda: app.res_next != app.res,
                       help="More cells = more detail (filaments, thin fronts) but slower: the cost grows "
                            "with res^4. 128 interactive, 192-256 to watch, 320+ to record."))
    W.append(ui.Info(lambda: app.grid_info()))
    staged("Domain size", "box", 1.0, 2.5, fmt="{:.2f}",
           help="Box size (the nebula keeps its size, and so does the detail): more room for the gas to "
                "expand without the box showing. Costs cells: x1.3 = ~1.7x (bubbles/cliff), "
                "x1.6 = ~4x (bipolar).")
    staged("Organic cloud edge", "cloud_edge", 0.0, 2.0, fmt="{:.2f}",
           help="The gas thins out with an irregular edge before the box walls, so the ionized region "
                "never shows the rectangle. 0 = off (as in nebula3d).")
    if scen == "bipolar":
        staged("Cloud radius", "cloud_r", 0.3, 1.0, fmt="{:.2f}",
               help="Where the cloud starts to thin out (with the organic edge).")
    W.append(ui.Toggle("Pre-turbulence (turbulent initial cloud)",
                       lambda: (P()["pre_t"] > 0) if app.pre_override is None else app.pre_override,
                       lambda v: setattr(app, "pre_override", bool(v)),
                       help="Supersonic turbulence phase before the stars are born: filaments and cores."))
    staged("Pre-turbulence strength", "pre_amp", 0.0, 15.0, help="Turbulence forcing (7 ~ Mach 7).")
    staged("Forcing fine scale (k max)", "pre_kmax", 2.0, 10.0,
           help="Highest wavenumber stirred: low = large eddies, high = fine structure (smoke).")
    staged("Pre-turbulence duration", "pre_t", 0.05, 0.5, fmt="{:.2f}")
    staged("Initial density contrast", "turb_sigma", 0.2, 1.8, help="Sigma of the initial log-normal field.")
    staged("Spectrum slope", "turb_slope", 2.5, 4.5,
           help="Power spectrum of the initial noise: high = smoother, low = clumpier.")
    staged("Seed", "seed", 0, 999, integer=True, fmt="{:d}", help="Another random realization.")
    W.append(ui.Choice("Hydro solver", ["hll", "rusanov"], lambda: app.pending.get("solver", P()["solver"]),
                       lambda v: app.pending.__setitem__("solver", v),
                       pending=lambda: "solver" in app.pending and app.pending["solver"] != P()["solver"],
                       help="hll: HLL flux + limiter, less numerical diffusion (sharper walls and fingers). "
                            "rusanov: the one in nebula3d.py, softer."))
    W.append(ui.Choice("Limiter (hll only)", ["minmod", "vanleer", "mc"],
                       lambda: app.pending.get("limiter", P()["limiter"]),
                       lambda v: app.pending.__setitem__("limiter", v),
                       pending=lambda: "limiter" in app.pending and app.pending["limiter"] != P()["limiter"],
                       help="minmod: robust. vanleer / mc: even sharper, but they make 2-3 cell striations "
                            "(numerical noise, not physical) in shocked shells."))
    if scen == "bipolar":
        staged("Axis tilt", "tilt", 0.0, 90.0, fmt="{:.0f}°")
        staged("Torus radius", "torus_R", 0.02, 0.15)
        staged("Torus thickness", "torus_a", 0.005, 0.05)
        staged("Torus density", "torus_rho", 5.0, 300.0, log=True)
        staged("Envelope radius", "env_r0", 0.03, 0.3)
        staged("Equatorial contrast", "eq_contrast", 0.0, 20.0)
    elif scen == "bubbles":
        staged("Number of bubbles", "n_bubbles", 1, 12, integer=True, fmt="{:d}")
        staged("Minimum separation", "min_sep", 0.1, 0.8)
        staged("Star formation threshold", "trigger", 0.0, 10.0,
               help="Local compression that triggers a star birth in the walls (0 = off).")
        staged("Max. births", "max_births", 0, 24, integer=True, fmt="{:d}")
        staged("New stars Q", "q_trig", 1e-6, 1e-3, log=True)
    else:
        staged("Globules", "globules", 0, 200, integer=True, fmt="{:d}")
        staged("Embedded stars", "n_embedded", 0, 12, integer=True, fmt="{:d}")
        staged("Cloud height", "cloud_top", 0.3, 0.8)
        staged("Edge roughness", "edge_amp", 0.0, 0.2)
    W.append(ui.Button("Apply and reset", app.reset, color=(150, 90, 40, 255),
                       help="Restarts the sim with the values marked * (R)."))

    # --- shader
    W.append(ui.Section("Shader (Nebula Volume)"))
    W.append(ui.Choice("Palette", PALETTE_NAMES, lambda: app.look["palette"], lambda v: app.set_look("palette", v),
                       help="jwst: NIRCam-like · sho: Hubble SII/Ha/OIII · natural: true color (red Ha)."))
    W.append(ui.Choice("Tone map", TONEMAPS, lambda: app.look["tonemap"], lambda v: app.set_look("tonemap", v)))
    look("Exposure", "exposure", -6.0, 4.0, fmt="{:+.2f}")
    look("Dust opacity", "dust_opacity", 0.05, 50.0, log=True)
    look("Emission (scale)", "emission", 0.1, 100.0, log=True)
    look("H-alpha", "ha", 0.0, 5.0, help="Hydrogen recombination: all the ionized gas (rho*xi)^2.")
    look("OIII", "oiii", 0.0, 5.0, help="Doubly ionized oxygen: inner zone, near the hot stars.")
    look("SII", "sii", 0.0, 5.0, help="Ionized sulfur: the layer just inside the ionization front.")
    look("PAH (PDR)", "pah", 0.0, 0.5, help="PAH molecules excited by FUV on the cloud skins (JWST orange).")
    look("Dust glow", "glow", 0.0, 0.02, fmt="{:.4f}")
    look("Shocks", "shock", 0.0, 0.05, fmt="{:.4f}", help="Shock emission: jets, HH knots, turbulence.")
    look("G0 saturation", "gs", 0.5, 1e4, log=True)
    look("Ambient G0", "ambient", 0.0, 2.0)
    look("Detail (sub-voxel noise)", "detail", 0.0, 4.0, help="Procedural noise that modulates dust and emission.")
    look("Detail scale", "detail_scale", 0.5, 12.0, log=True)
    look("Edge fade", "edge_fade", 0.0, 0.3)
    look("Star brightness", "star_gain", 0.0, 3.0)
    look("Star size", "star_size", 0.3, 3.0)
    look("Background stars", "bg_stars", 0.0, 3.0)
    W.append(ui.Toggle("Diffraction spikes (JWST)", lambda: app.look["spikes"],
                       lambda v: app.set_look("spikes", bool(v))))

    # --- render
    W.append(ui.Section("Render", collapsed=True))
    W.append(ui.Choice("Volume filter", ["linear", "smooth", "cubic"], lambda: app.look["filter"],
                       lambda v: app.set_look("filter", v),
                       help="linear: sharpest · smooth: [1 4 1] prefilter (like Cycles cubic) · "
                            "cubic: tricubic B-spline (soft, 8x more reads)."))
    W.append(ui.Slider("March step (voxels)", lambda: app.step_vox,
                       lambda v: (setattr(app, "step_vox", v), setattr(app, "dirty", True)), 0.3, 2.0,
                       default=0.75, help="Smaller = more accurate and slower."))
    W.append(ui.Slider("Render scale", lambda: app.render_scale, lambda v: app.set_render_scale(v), 0.25, 1.0,
                       default=1.0, fmt="{:.2f}", help="Internal viewport resolution (V)."))
    W.append(ui.Slider("Auto-orbit (°/frame)", lambda: app.auto_orbit,
                       lambda v: setattr(app, "auto_orbit", v), 0.0, 2.0, default=0.0, fmt="{:.2f}"))
    W.append(ui.Slider("Playback speed (fps)", lambda: app.play_fps,
                       lambda v: setattr(app, "play_fps", v), 2.0, 60.0, default=24.0, fmt="{:.0f}",
                       help="Cached frames per second when playing the timeline."))
    W.append(ui.Toggle("Loop playback", lambda: app.loop, lambda v: setattr(app, "loop", bool(v))))
    W.append(ui.Slider("Timeline memory (GB)", lambda: app.timeline.budget / 1e9,
                       lambda v: app.timeline.set_budget(v), 1.0, 16.0, default=app.a.cache_gb, fmt="{:.1f}",
                       help="GPU cache. When it fills up, every other frame is dropped (the whole history "
                            "stays, with fewer frames)."))
    W.append(ui.Row(ui.Button("Default camera", lambda: app.frame_camera(), help="C"),
                    ui.Button("Frame the box", lambda: app.frame_camera(whole=True), help="Shift+C"),
                    ui.Button("HQ screenshot", lambda: app.save_png(hq=True), help="Shift+S: 2x tricubic PNG")))
    W.append(ui.Button("Bake (JSON + high-res command)", app.bake, help="B"))

    # --- video
    W.append(ui.Section("Video"))
    W.append(ui.Choice("Size", [s[0] for s in VIDEO_SIZES], lambda: app.rec["size"],
                       lambda v: app.rec.__setitem__("size", v),
                       help="The video is the frame marked in the viewport, re-rendered at this size."))
    W.append(ui.Slider("Video FPS", lambda: app.rec["fps"], lambda v: app.rec.__setitem__("fps", int(v)),
                       12, 60, integer=True, fmt="{:d}", default=30))
    W.append(ui.Slider("Quality (CRF, lower = better)", lambda: app.rec["crf"],
                       lambda v: app.rec.__setitem__("crf", int(v)), 10, 28, integer=True, fmt="{:d}", default=16))
    W.append(ui.Toggle("High quality (tricubic, fine step)", lambda: app.rec["hq"],
                       lambda v: app.rec.__setitem__("hq", bool(v)), help="Slower per frame."))
    W.append(ui.Toggle("Also save PNGs", lambda: app.rec["png"], lambda v: app.rec.__setitem__("png", bool(v))))
    W.append(ui.Toggle("Show frame guide", lambda: app.rec["guide"], lambda v: app.rec.__setitem__("guide", bool(v))))
    W.append(ui.Button("● Record", app.toggle_recording,
                       label_fn=lambda: "■ Stop recording" if app.recorder.active else "● Record (F9)",
                       color_fn=lambda: (200, 50, 50, 255) if app.recorder.active else (110, 40, 40, 255),
                       help="Records every new frame (sim, playback or camera) straight to H.264 MP4 in videos/."))
    W.append(ui.Button("Export timeline to video", app.export_timeline,
                       help="Renders every cached frame, first to last, to an MP4."))
    W.append(ui.Slider("Orbit duration (s)", lambda: app.rec["orbit_s"],
                       lambda v: app.rec.__setitem__("orbit_s", v), 3.0, 30.0, default=12.0, fmt="{:.0f}"))
    W.append(ui.Button("360° orbit video", app.export_orbit, help="A full turn around the current state."))
    W.append(ui.Info(lambda: ("last: " + os.path.basename(app.last_video)) if app.last_video else "videos go to NebulaViewer/videos"))
    return W


# ----------------------------------------------------------------------------------- headless
def run_headless(a):
    from PIL import Image
    app = App(a)
    out = a.out or os.path.join(HERE, "media", f"{a.scenario}_{a.res}")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    gif_frames, saved = [], []
    t_sim, t_ren, n_ren = 0.0, 0.0, 0
    wall0 = time.perf_counter()
    sim = app.sim
    sim.profile = a.profile
    if a.keytest:
        # every control, through the same App.on_key the window uses (no GL needed)
        names = ["SPACE", "SPACE", "N", "PERIOD", "P", "P", "P", "T", "T", "T", "K", "K", "L", "L", "I", "I", "I",
                 "M", "M", "M", "M", "Q", "W", "J", "D", "E", "A", "G", "X", "PLUS", "MINUS", "NUM_ADD",
                 "NUM_SUBTRACT", "BRACKETRIGHT", "BRACELEFT", "PAGEUP", "PAGEDOWN", "O", "O", "C", "V", "V", "V",
                 "U", "U", "B", "S", "F12", "H", "F1", "TAB", "LEFT", "LEFT", "RIGHT", "HOME", "END", "_2", "R",
                 "_3", "Z", "NUM_1", "R", "ESCAPE", "F5"]
        shifted = {"Q", "W", "J", "D", "E", "A", "G", "X", "C", "Z", "S"}
        for nm in names:
            for sh in ([False, True] if nm in shifted else [False]):
                act = app.on_key(nm, sh)
                if act == "render_scale":
                    app.renderer.resize(*app.render_size(a.width, a.height))
                app.advance(2)
                app.render()
                print(f"   key {'Shift+' if sh else ''}{nm}: ok ({act})", flush=True)
        print("KEYTEST OK", flush=True)
        return app
    if a.mp4:
        app.rec["size"] = "window"
        app.rec["fps"] = a.mp4_fps
        app.start_recording("_headless")
    for f in range(a.frames):
        app.advance()
        stop = a.until_t is not None and sim.phase == "main" and sim.t >= a.until_t
        t_sim += app.t_sim
        if a.orbit:
            app.cam.yaw += a.orbit
        need = (f % a.save_every == 0) or f == a.frames - 1 or stop
        gif_need = a.gif and (f % a.gif_every == 0)
        if need or gif_need or a.render_all:
            app.render()
            t_ren += app.t_render
            n_ren += 1
        if a.mp4:
            app.record_if_needed()
        if need:
            p = f"{out}_f{f:04d}.png"
            app.save_png(p)
            saved.append(p)
            print(f"   frame {f} {sim.time_label()} sim {app.t_sim*1e3:.1f} ms render {app.t_render*1e3:.1f} ms",
                  flush=True)
        if gif_need:
            im = Image.fromarray(np.ascontiguousarray(app.renderer.image()[..., :3]))
            w, h = im.size
            gif_frames.append(im.resize((int(w * a.gif_scale), int(h * a.gif_scale)), Image.LANCZOS))
        if stop:
            break
    if a.mp4:
        app.stop_recording()
    wall = time.perf_counter() - wall0
    steps = sim.prof.get("n_steps", 1)
    print(f"\nheadless done: {sim.time_label()} after {steps} steps, wall {wall:.1f}s")
    print(f"  sim    : {t_sim/steps*1e3:.2f} ms/step ({steps} steps)")
    print(f"  cache  : {app.timeline.status()}")
    if n_ren:
        print(f"  render : {t_ren/n_ren*1e3:.2f} ms/frame at {app.renderer.W}x{app.renderer.H} ({n_ren} frames)")
    if a.profile:
        print("  profile:", {k: round(v, 3) for k, v in sim.prof.items()})
    if gif_frames:
        gp = out + ".gif"
        q = [g.quantize(colors=240, method=Image.Quantize.MEDIANCUT) for g in gif_frames]
        q[0].save(gp, save_all=True, append_images=q[1:], duration=a.gif_ms, loop=0)
        print("  gif   :", gp, len(gif_frames), "frames")
    if a.turntable:
        # orbit 360 deg around the final state (sim paused) -> <out>_turntable.gif
        frames_tt = []
        yaw0 = app.cam.yaw
        for i in range(a.turntable):
            app.cam.yaw = yaw0 + 360.0 * i / a.turntable
            app.render()
            im = Image.fromarray(np.ascontiguousarray(app.renderer.image()[..., :3]))
            w, h = im.size
            frames_tt.append(im.resize((int(w * a.gif_scale), int(h * a.gif_scale)), Image.LANCZOS))
            if i % max(a.turntable // 4, 1) == 0:
                app.save_png(f"{out}_turn{i:03d}.png")
        app.cam.yaw = yaw0
        q = [g.quantize(colors=240, method=Image.Quantize.MEDIANCUT) for g in frames_tt]
        q[0].save(out + "_turntable.gif", save_all=True, append_images=q[1:], duration=a.gif_ms, loop=0)
        print("  turntable:", out + "_turntable.gif", len(q), "frames")
    if a.timeline_test:
        tl = app.timeline
        print(f"  timeline test: {len(tl.snaps)} snapshots, scrub to 1/3, play to the end, branch")
        app.scrub(tl.snaps[len(tl.snaps) // 3]["tau"])
        app.render()
        app.save_png(f"{out}_scrub.png")
        app.running = True
        for _ in range(len(tl.snaps)):
            app.tick(1.0 / app.play_fps)
        print(f"   after playback: cursor {app.cursor}, {sim.time_label()}")
        app.scrub(tl.snaps[len(tl.snaps) // 2]["tau"])
        app.set_sim("wind", sim.P["wind"] * 1.5)
        app.advance(4)
        print(f"   branched: cursor {app.cursor}, {len(tl.snaps)} snapshots, {sim.time_label()}")
        p = app.export_timeline()
        print("   timeline video:", p)
    if a.bake_test:
        app.bake()
    return app


# ----------------------------------------------------------------------------------- window
def run_window(a):
    import pyglet
    from pyglet import gl
    from pyglet.window import key, mouse
    import ui

    app = App(a)
    window = pyglet.window.Window(a.width + (ui.PANEL_W if a.panel else 0), a.height + ui.TimelineBar.h,
                                  caption="NebulaViewer", resizable=True, vsync=a.vsync)
    panel = ui.Panel()
    panel.visible = bool(a.panel)
    bar = ui.TimelineBar(app)
    batch = pyglet.graphics.Batch()
    hud_bg = pyglet.shapes.Rectangle(0, 0, 10, 10, color=(0, 0, 0, 150), batch=batch)
    hud = pyglet.text.Label("", x=8, y=a.height - 8, anchor_y="top", multiline=True, width=1400,
                            font_name="Consolas", font_size=10, color=(235, 235, 235, 255), batch=batch)
    help_lbl = pyglet.text.Label(HELP, x=8, y=8, anchor_y="bottom", multiline=True, width=1400,
                                 font_name="Consolas", font_size=10, color=(200, 220, 255, 255))
    guide_batch = pyglet.graphics.Batch()
    guide = [pyglet.shapes.Rectangle(0, 0, 0, 0, color=(0, 0, 0, 140), batch=guide_batch) for _ in range(4)]
    st = {"hud": True, "help": False, "tex": None, "pbo": None, "glbuf": None, "interop": a.interop,
          "frames": 0, "t_fps": time.perf_counter(), "start": time.perf_counter(), "fps_log": [],
          "render_ms": [], "sim_ms": [], "owner": None, "t_last": time.perf_counter()}

    def viewport():
        """(x, y, w, h) of the 3D view inside the window"""
        pw = panel.width if panel.visible else 0
        return 0, ui.TimelineBar.h, max(64, window.width - pw), max(64, window.height - ui.TimelineBar.h)

    def rebuild_panel():
        panel.set_widgets(build_panel(app))
        panel.layout(window.width, window.height)

    app.on_rebuild = rebuild_panel

    def setup_gl_image():
        W, H = app.renderer.W, app.renderer.H
        st["tex"] = pyglet.image.Texture.create(W, H, internalformat=gl.GL_RGBA8,
                                                min_filter=gl.GL_LINEAR, mag_filter=gl.GL_LINEAR)
        if st["pbo"] is not None:
            st["glbuf"] = None
            gl.glDeleteBuffers(1, st["pbo"])
            st["pbo"] = None
        if st["interop"]:
            try:
                pbo = gl.GLuint()
                gl.glGenBuffers(1, pbo)
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
                gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, W * H * 4, None, gl.GL_STREAM_DRAW)
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
                st["pbo"] = pbo
                st["glbuf"] = wp.RegisteredGLBuffer(int(pbo.value), "cuda:0", flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
                                                    fallback_to_copy=False)
            except Exception as e:  # pragma: no cover - driver dependent
                print("CUDA-GL interop unavailable, falling back to host upload:", e)
                st["interop"] = False
                st["glbuf"] = None

    def upload():
        W, H = app.renderer.W, app.renderer.H
        tex = st["tex"]
        gl.glBindTexture(tex.target, tex.id)
        if st["interop"] and st["glbuf"] is not None:
            arr = st["glbuf"].map(dtype=wp.uint32, shape=(W * H,))
            app.render(out=arr, flip=True)
            st["glbuf"].unmap()
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, st["pbo"])
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, W, H, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, 0)
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            app.render(flip=True)
            img = np.ascontiguousarray(app.renderer.rgba.numpy())
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, W, H, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                               img.ctypes.data_as(gl.POINTER(gl.GLubyte)))

    def resize_render():
        vx, vy, vw, vh = viewport()
        app.a.width, app.a.height = vw, vh
        rw, rh = app.render_size(vw, vh)
        if (rw, rh) != (app.renderer.W, app.renderer.H):
            app.renderer.resize(rw, rh)
            setup_gl_image()
        app.dirty = True

    def set_render_scale(v):
        app.render_scale = float(v)
        resize_render()

    app.set_render_scale = set_render_scale
    setup_gl_image()
    rebuild_panel()
    bar.layout(viewport()[2])

    def relayout():
        panel.layout(window.width, window.height)
        bar.layout(viewport()[2])
        resize_render()

    @window.event
    def on_resize(w, h):
        if w > 0 and h > 0:
            relayout()

    def in_view(x, y):
        vx, vy, vw, vh = viewport()
        return vx <= x < vx + vw and vy <= y < vy + vh

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        st["owner"] = None
        if bar.on_press(x, y, button, modifiers):
            st["owner"] = "bar"
        elif panel.on_press(x, y, button, modifiers):
            st["owner"] = "panel"
        elif in_view(x, y):
            st["owner"] = "view"

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        if st["owner"] == "bar":
            bar.on_release(x, y, button, modifiers)
        elif st["owner"] == "panel":
            panel.on_release(x, y, button, modifiers)
        st["owner"] = None

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        panel.on_motion(x, y)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if st["owner"] == "bar":
            bar.on_drag(x, y, dx, dy, buttons, modifiers)
            return
        if st["owner"] == "panel":
            panel.on_drag(x, y, dx, dy, buttons, modifiers)
            app.dirty = True
            return
        if st["owner"] != "view":
            return
        c = app.cam
        if (buttons & mouse.LEFT) and not (modifiers & key.MOD_SHIFT):
            c.yaw -= dx * 0.3
            c.pitch = min(max(c.pitch - dy * 0.3, -89.0), 89.0)
        elif buttons & (mouse.RIGHT | mouse.MIDDLE) or (buttons & mouse.LEFT and modifiers & key.MOD_SHIFT):
            pos, fwd, right, up = c.basis()
            vh = viewport()[3]
            s = c.dist * c.tans(app.a.width, app.a.height)[1] * 2.0 / max(vh, 1)
            c.target = c.target - right * dx * s - up * dy * s
        app.dirty = True
        app.frame_changed = True

    @window.event
    def on_mouse_scroll(x, y, sx, sy):
        if panel.on_scroll(x, y, sy):
            return
        if in_view(x, y):
            app.cam.dist = min(max(app.cam.dist * (0.9 ** sy), 0.05), 20.0)
            app.dirty = True
            app.frame_changed = True

    @window.event
    def on_key_press(symbol, mod):
        action = app.on_key(key.symbol_string(symbol), bool(mod & key.MOD_SHIFT))
        if action == "quit":
            if app.recorder.active:
                app.stop_recording()
            window.close()
        elif action == "render_scale":
            resize_render()
        elif action == "hud":
            st["hud"] = not st["hud"]
        elif action == "help":
            st["help"] = not st["help"]
        elif action == "panel":
            panel.visible = not panel.visible
            relayout()
        return pyglet.event.EVENT_HANDLED

    def draw_guide():
        """darken what is outside the recorded frame"""
        vx, vy, vw, vh = viewport()
        w, h = app.rec_size()
        _, (gx, gy, gw, gh) = app.rec_tans(w, h)
        rects = [(vx, vy, vw, gy), (vx, vy + gy + gh, vw, vh - gy - gh), (vx, vy + gy, gx, gh),
                 (vx + gx + gw, vy + gy, vw - gx - gw, gh)]
        for r, (x, y, ww, hh) in zip(guide, rects):
            r.position = (x, y)
            r.width, r.height = max(ww, 0), max(hh, 0)
            r.color = (60, 0, 0, 150) if app.recorder.active else (0, 0, 0, 140)
        guide_batch.draw()

    @window.event
    def on_draw():
        now = time.perf_counter()
        dt = now - st["t_last"]
        st["t_last"] = now
        app.tick(dt)
        if app.auto_orbit:
            app.cam.yaw += app.auto_orbit
            app.dirty = True
            app.frame_changed = True
        if app.dirty:
            upload()
            app.frame_changed = True            # any visible change is a new video frame
            st["render_ms"].append(app.t_render * 1e3)
            if app.running:
                st["sim_ms"].append(app.t_sim * 1e3)
        app.record_if_needed()
        window.clear()
        vx, vy, vw, vh = viewport()
        st["tex"].blit(vx, vy, width=vw, height=vh)
        if app.rec["guide"] and (app.recorder.active or app.rec["size"] != "window"):
            draw_guide()
        if st["hud"]:
            if time.perf_counter() - st.get("hud_t", 0.0) > 0.2:      # text layout is slow-ish: 5 Hz
                hud.text = app.hud()
                st["hud_t"] = time.perf_counter()
            hud.y = vy + vh - 8
            hud_bg.x, hud_bg.y = 0, vy + vh - hud.content_height - 14
            hud_bg.width, hud_bg.height = min(hud.content_width + 16, vw), hud.content_height + 14
            batch.draw()
        if st["help"]:
            help_lbl.y = vy + 8
            help_lbl.draw()
        panel.draw()
        bar.draw()
        st["frames"] += 1
        if now - st["t_fps"] > 0.5:
            app.fps = st["frames"] / (now - st["t_fps"])
            st["fps_log"].append(app.fps)
            st["frames"], st["t_fps"] = 0, now

    def autoclose(dt):
        if a.autoclose_shot:
            try:
                pyglet.image.get_buffer_manager().get_color_buffer().save(a.autoclose_shot)
                print("window grab saved:", a.autoclose_shot)
            except Exception as e:
                print("window grab failed:", e)
        if app.recorder.active:
            app.stop_recording()
        window.close()
        pyglet.app.exit()

    if a.selftest:
        # scripted interaction (no human needed): keys, panel clicks/drags, timeline scrub, recording
        px = lambda: window.width - panel.width + 60

        def panel_click(label, button=mouse.LEFT, drag_to=None):
            target = [w_ for w_ in panel.widgets if getattr(w_, "label", None) == label]
            if target and not target[0].items[0].visible:      # scroll it into view
                panel.scroll = 0
                panel.layout()
                for _ in range(80):
                    if target[0].items[0].visible:
                        break
                    window.dispatch_event("on_mouse_scroll", px(), 300, 0, -0.5)
            for wdg in panel.widgets:
                if getattr(wdg, "label", None) == label and wdg.items and wdg.items[0].visible:
                    x, y = wdg.x + wdg.w * 0.5, wdg.y + wdg.h * 0.5
                    window.dispatch_event("on_mouse_press", x, y, button, 0)
                    if drag_to is not None:
                        window.dispatch_event("on_mouse_drag", wdg.x + wdg.w * drag_to, y, 5, 0, button, 0)
                    window.dispatch_event("on_mouse_release", x, y, button, 0)
                    return True
            raise RuntimeError(f"panel widget not found / hidden: {label}")

        acts = [(1.0, "key", key.P, 0), (1.3, "key", key.T, 0), (1.5, "panel", "Stellar wind / radiation pressure", 0.7),
                (1.8, "panel", "Exposure", 0.6), (2.0, "panel", "Palette", None), (2.2, "size", 1400, 800),
                (2.6, "drag", 40, -15), (2.8, "scroll", 2), (3.0, "key", key.F9, 0), (5.0, "key", key.F9, 0),
                (5.2, "scrub", 0.3), (5.5, "scrub", 0.6), (5.8, "key", key.LEFT, 0), (6.0, "key", key.SPACE, 0),
                (7.0, "panel", "Seed", 0.3), (7.2, "panel", "Apply and reset", None),
                (8.5, "key", key.TAB, 0), (8.8, "key", key.TAB, 0), (9.0, "key", key._2, 0),
                (10.5, "pscroll", -6), (10.8, "key", key.B, 0), (11.0, "key", key.S, 0), (11.2, "key", key._1, 0)]

        def act(dt, item):
            kind = item[1]
            try:
                if kind == "key":
                    window.dispatch_event("on_key_press", item[2], item[3])
                elif kind == "size":
                    window.set_size(item[2], item[3])
                elif kind == "drag":
                    vx, vy, vw, vh = viewport()
                    window.dispatch_event("on_mouse_press", vw // 2, vy + vh // 2, mouse.LEFT, 0)
                    window.dispatch_event("on_mouse_drag", vw // 2, vy + vh // 2, item[2], item[3], mouse.LEFT, 0)
                    window.dispatch_event("on_mouse_release", vw // 2, vy + vh // 2, mouse.LEFT, 0)
                elif kind == "scroll":
                    window.dispatch_event("on_mouse_scroll", 100, 300, 0, item[2])
                elif kind == "pscroll":
                    window.dispatch_event("on_mouse_scroll", px(), 300, 0, item[2])
                elif kind == "panel":
                    panel_click(item[2], drag_to=item[3])
                elif kind == "scrub":
                    x = bar.tx0 + (bar.tx1 - bar.tx0) * item[2]
                    window.dispatch_event("on_mouse_press", x, 30, mouse.LEFT, 0)
                    window.dispatch_event("on_mouse_drag", x + 5, 30, 5, 0, mouse.LEFT, 0)
                    window.dispatch_event("on_mouse_release", x + 5, 30, mouse.LEFT, 0)
                print(f"selftest t={item[0]:.1f} {kind} {item[2:]} ok  ({app.sim.time_label()}, cursor {app.cursor})",
                      flush=True)
            except Exception as e:  # noqa
                st["selftest_err"] = repr(e)
                print("SELFTEST ERROR", repr(e), flush=True)
                import traceback
                traceback.print_exc()
        for it in acts:
            pyglet.clock.schedule_once(act, it[0], it)
    if a.autoclose:
        pyglet.clock.schedule_once(autoclose, a.autoclose)
    print("NebulaViewer window open. F1 = help, Tab = panel. interop:", st["interop"], flush=True)
    pyglet.app.run(interval=1.0 / a.max_fps)
    if app.recorder.active:
        app.stop_recording()
    if a.selftest:
        print("SELFTEST", "FAILED: " + st["selftest_err"] if st.get("selftest_err") else "OK", flush=True)
    fl = st["fps_log"][2:] or st["fps_log"]
    if fl:
        print(f"window session: mean fps {np.mean(fl):.1f} (min {np.min(fl):.1f}, max {np.max(fl):.1f}); "
              f"render {np.median(st['render_ms'] or [0]):.1f} ms, sim {np.median(st['sim_ms'] or [0]):.1f} ms "
              f"per frame (median); interop {st['interop']}; {app.sim.time_label()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="bipolar", choices=SCENARIOS)
    ap.add_argument("--res", type=int, default=128, help="grid cells along z (x/y follow the scenario aspect)")
    ap.add_argument("--width", type=int, default=1280, help="3D view width (the panel is added to the window)")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--render_scale", type=float, default=1.0, help="internal render resolution / window")
    ap.add_argument("--step_vox", type=float, default=0.75, help="ray-march step in voxels")
    ap.add_argument("--steps", type=int, default=4, help="sim steps per displayed frame")
    ap.add_argument("--paused", action="store_true")
    ap.add_argument("--ic", default="matched", choices=["matched", "nebula3d"],
                    help="matched: resolution-independent noise (preview ~ bake); nebula3d: exact nebula3d.py ICs")
    ap.add_argument("--pre", default="auto", choices=["auto", "on", "off"], help="pre-turbulence phase")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", help="sim params, e.g. --set wind=0.6 jet=4 box=2 solver=rusanov")
    ap.add_argument("--look", nargs="*", help="render params, e.g. --look dust_opacity=2 emission=8")
    ap.add_argument("--palette", default=None, choices=PALETTE_NAMES)
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--fov", type=float, default=39.6,
                    help="field of view of the larger image side (deg); 39.6 = Blender 50 mm lens")
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=None)
    ap.add_argument("--dist", type=float, default=None)
    ap.add_argument("--orbit", type=float, default=0.0, help="auto-orbit, degrees per frame")
    ap.add_argument("--bake_res", type=int, default=384)
    ap.add_argument("--cache_gb", type=float, default=8.0, help="GPU memory for the timeline cache")
    ap.add_argument("--max_grid_gb", type=float, default=14.0, help="refuse grids needing more GPU memory")
    # window
    ap.add_argument("--panel", type=int, default=1, help="show the options panel")
    ap.add_argument("--vsync", action="store_true")
    ap.add_argument("--max_fps", type=float, default=240.0)
    ap.add_argument("--interop", type=int, default=1, help="CUDA-GL interop (1) or host upload (0)")
    ap.add_argument("--autoclose", type=float, default=0.0, help="close the window after N seconds")
    ap.add_argument("--autoclose_shot", default=None, help="save a window grab (PNG) before auto-closing")
    ap.add_argument("--selftest", action="store_true", help="scripted UI test (use with --autoclose)")
    ap.add_argument("--end_pause", action="store_true", help="pause when the sim reaches t_end (old behaviour)")
    # headless
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--until_t", type=float, default=None, help="stop when the sim reaches this time")
    ap.add_argument("--out", default=None, help="output prefix (PNG/GIF), default media/<scenario>_<res>")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif_every", type=int, default=5)
    ap.add_argument("--gif_scale", type=float, default=0.5)
    ap.add_argument("--gif_ms", type=int, default=60)
    ap.add_argument("--mp4", action="store_true", help="headless: record every frame to videos/*.mp4")
    ap.add_argument("--mp4_fps", type=int, default=30)
    ap.add_argument("--render_all", action="store_true", help="render every frame (timing)")
    ap.add_argument("--turntable", type=int, default=0, help="headless: N-frame 360 deg orbit GIF at the end")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--bake_test", action="store_true", help="headless: write a bake json at the end")
    ap.add_argument("--keytest", action="store_true", help="headless: exercise every keyboard control")
    ap.add_argument("--timeline_test", action="store_true", help="headless: scrub / play / branch / export")
    a = ap.parse_args()
    if a.headless:
        run_headless(a)
    else:
        run_window(a)


if __name__ == "__main__":
    main()
