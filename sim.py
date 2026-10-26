"""NebulaViewer simulation: the nebula3d.py toy radiation-hydro, wrapped as a resident-on-GPU class.

Reuses the Warp kernels and initial conditions of nebula3d.py (next to this file; the env var NEBULA_SCRIPTS
can point to another copy). It is imported, never modified. What is new here:
  * the time loop of nebula3d.main() refactored into Sim.reset() / Sim.step(k), with the
    pre-turbulence phase advanced step by step too (you can watch the turbulence form);
  * live parameters (q_main, wind, jet, ...) read every step, logged as events for the bake;
  * amortised radiative transfer for interactive use (fewer ionizing rays, coarse FUV grid,
    G0 / shock only recomputed when a frame is rendered);
  * 'matched' initial conditions: the random fields (turbulence, sheet warps, stirring) are
    generated resolution-independently, so a 128^3 preview and a 384^3 bake (bake.py) share
    the same large-scale structure. ic_mode='nebula3d' reproduces nebula3d.py bit for bit.
  * low-memory storage: 40 bytes per cell instead of 68-72, same results (hydro bit for bit).
    RK2 with one stage copy (stage 2 writes in place), one 16 B/cell scratch block (`stage`: the RK stage,
    xi_tmp, fp32 accumulators, the renderer's packing grids), dfront / g0 in float16, no shock grid.
  * cropped domains (P['crop'], crop.py): simulate only the window of the full box a shot sees.

Domain units as in nebula3d: dx = 1/nz, box = (nx*dx, ny*dx, 1), arrays [i,j,k] = x,y,z (z up).
"""
import argparse, ast, contextlib, math, os, sys, time
import numpy as np
import warp as wp

NEBULA_SCRIPTS = os.environ.get("NEBULA_SCRIPTS", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NEBULA_SCRIPTS)
_dwb = sys.dont_write_bytecode
sys.dont_write_bytecode = True       # never write a __pycache__ next to nebula3d.py (it may be a read-only copy)
import nebula3d as N  # noqa: E402  (runs wp.init())
sys.dont_write_bytecode = _dwb

wp.config.quiet = True


# ----------------------------------------------------------------------------------- parameters
def nebula3d_defaults():
    """All argparse defaults of nebula3d.main(), read from its source (stays in sync with the file)."""
    tree = ast.parse(open(N.__file__, encoding="utf-8").read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            name = node.args[0].value.lstrip("-")
            kw = {k.arg: k.value for k in node.keywords}
            out[name] = ast.literal_eval(kw["default"]) if "default" in kw else None
    return out


NEB_DEFAULTS = nebula3d_defaults()

# Felipe's reference parameter sets (README of Nebula + lookdev of 2026-10-01). 'aspect' = box shape / nz.
PRESETS = {
    "bipolar": dict(scenario="bipolar", q_main=8e-4, turb_sigma=0.9, pre_t=0.2, pre_amp=7.0, pre_kmax=4.0,
                    wind=0.4, jet=3.0, jet_pulse=0.7, jet_period=0.02, tilt=25.0, torus_R=0.056,
                    torus_a=0.02, env_r0=0.095, kappa_fuv=10.0, t_end=0.4, frames=96),
    "bubbles": dict(scenario="bubbles", q_main=3e-4, turb_sigma=1.1, wind=0.5, trigger=4.0, trigger_min=2.0,
                    q_trig=5e-5, n_bubbles=6, min_sep=0.5, trigger_sep=0.18, max_stars=28, max_births=8,
                    birth_gap=0.012, kappa_fuv=10.0, t_end=0.25, frames=60),
    "cliff": dict(scenario="cliff", q_main=5e-3, globules=80, n_embedded=6, star_y=0.3, edge_amp=0.08,
                  kappa_fuv=10.0, t_end=0.2, frames=60),
}
ASPECT = {"bipolar": (1.0, 1.0, 1.0), "bubbles": (1.5, 0.75, 1.0), "cliff": (1.5, 0.875, 1.0)}
# NebulaViewer-only parameters (not in nebula3d.py):
#   box    : domain size in nebula units (bipolar: all axes; bubbles / cliff: x and z, the view plane).
#            The nebula keeps its size and the box grows around it; 'res' stays = cells per nebula unit,
#            so a bigger box costs cells, not detail.
#   cloud_edge: organic edge of the cloud. The gas fades to the floor density before the box faces
#            (noisy boundary), so an ionized region never shows the rectangular box (0 = off).
#   q_ramp : star turn-on time. A massive star is born bloated and cool and reaches its full ionizing
#            output while it contracts to the main sequence (~1e4-1e5 yr; 1 time unit ~ 1 Myr for a
#            ~10 pc box at c_i ~ 10 km/s). Q(age) = Q * smoothstep(age / q_ramp)^3 -> R_Stromgren ~ age.
#   solver : "hll" (HLL flux, upwinded: sharper shells/fingers) | "rusanov" (nebula3d.py, more diffusive)
#   limiter: hll slope limiter "minmod" (default) | "vanleer" | "mc" (sharper, but grid-scale striations)
#   clumps : bipolar only, number of dense clumps in the envelope for the lobes to sweep (0 = off); clump_amp =
#            peak density / smooth envelope, clump_size = gaussian radius, between clump_rin and clump_rmax
#            from the star (nebula units), 15-65 deg from the axis (see add_bipolar_clumps)
EXTRA_DEFAULTS = dict(box=1.0, q_ramp=0.0, solver="rusanov", limiter="minmod", cloud_edge=0.0, cloud_r=0.6,
                      clumps=0, clump_amp=40.0, clump_size=0.022, clump_rin=0.12, clump_rmax=0.4)
VIEWER_DEFAULTS = dict(q_ramp=0.02, solver="hll", cloud_edge=1.0)
BOX_DEFAULTS = {"bipolar": 1.6, "bubbles": 1.3, "cliff": 1.3}   # Felipe 2026-10-01: the box showed in videos
# optional pre-turbulence for scenarios that don't use it by default (Felipe: Mach ~7 is the sweet spot)
PRE_TURB = dict(pre_t=0.2, pre_amp=7.0, pre_kmax=4.0)
# parameters that can be changed while the sim runs (all others need a reset)
LIVE = ("q_main", "wind", "jet", "jet_proto", "jet_pulse", "jet_period", "kappa_fuv", "drive_main", "pre_amp",
        "q_ramp")


def grid_shape(scenario, res, box=1.0):
    """cells for 'res' cells per nebula unit; box scales all axes (bipolar) or x and z (bubbles, cliff)"""
    ax, ay, az = ASPECT[scenario]
    r8 = lambda v: max(8, int(round(v / 8.0)) * 8)
    by = box if scenario == "bipolar" else 1.0
    return r8(ax * res * box), r8(ay * res * by), r8(az * res * box)


BYTES_PER_CELL = 40          # Sim fields; the renderer adds an 8 B/cell RGBA16F texture


def grid_gb(shape):
    """GPU memory of a Sim + Renderer at this grid (40 B/cell of fields + the 8 B/cell texture)"""
    return shape[0] * shape[1] * shape[2] * (BYTES_PER_CELL + 8) / 1e9


def make_params(scenario, overrides=None):
    p = dict(NEB_DEFAULTS)
    p.update(EXTRA_DEFAULTS)
    p.update(VIEWER_DEFAULTS)
    p.update(PRESETS[scenario])
    p["box"] = BOX_DEFAULTS[scenario]
    if overrides:
        p.update(overrides)
    return p


def organic_noise(shape, dx, seed, kmin=1.2, kmax=3.5, n=10):
    """smooth unit-variance 3D field (sum of random plane waves): resolution independent"""
    r = np.random.default_rng([int(seed), 4242])
    x = ((np.arange(shape[0]) + 0.5) * dx).astype(np.float32)
    y = ((np.arange(shape[1]) + 0.5) * dx).astype(np.float32)
    z = ((np.arange(shape[2]) + 0.5) * dx).astype(np.float32)
    f = np.zeros(shape, np.float32)
    for _ in range(n):
        d = r.standard_normal(3)
        d /= np.linalg.norm(d)
        k = 2 * np.pi * r.uniform(kmin, kmax) * d
        ph = r.uniform(0, 2 * np.pi)
        f += np.sin(np.float32(k[0]) * x[:, None, None] + np.float32(k[1]) * y[None, :, None]
                    + np.float32(k[2]) * z[None, None, :] + np.float32(ph))
    return f / np.float32(np.sqrt(n / 2.0))


def _sst(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def cloud_taper(scenario, shape, dx, L, P):
    """1 inside the cloud, -> 0 before the box faces, with a noisy (organic) boundary"""
    amt = float(P.get("cloud_edge", 0.0))
    if amt <= 0:
        return None
    n = organic_noise(shape, dx, P["seed"], kmin=1.0, kmax=2.6) * np.float32(0.13 * amt)
    ax = [((np.arange(m) + 0.5) * dx - 0.5 * l).astype(np.float32) / np.float32(0.5 * l) for m, l in zip(shape, L)]
    if scenario == "bipolar":
        u = np.sqrt(ax[0][:, None, None] ** 2 * (0.5 * L[0]) ** 2 + ax[1][None, :, None] ** 2 * (0.5 * L[1]) ** 2
                    + ax[2][None, None, :] ** 2 * (0.5 * L[2]) ** 2) / np.float32(0.5 * min(L))
        u0 = min(float(P.get("cloud_r", 0.6)) / (0.5 * min(L)), 0.85)
        return (1.0 - _sst(u0, 1.0, u + n)).astype(np.float32)
    # bubbles / cliff: fade toward the x faces and the z faces (cliff: top only); content half sizes 0.75 / 0.5
    T = np.ones(shape, np.float32)
    hx, hz = 0.75 / (0.5 * L[0]), 0.5 / (0.5 * L[2])
    ux = np.abs(ax[0])[:, None, None]
    T *= 1.0 - _sst(min(hx - 0.1, 0.88), 1.0, ux + n)
    uz = ax[2][None, None, :]
    if scenario == "cliff":
        uz = np.maximum(uz, 0.0)
    else:
        uz = np.abs(uz)
    T *= 1.0 - _sst(min(hz - 0.1, 0.88), 1.0, uz + n)
    return T.astype(np.float32)


def ic_bubbles_box(nx, ny, nz, dx, args, rng):
    """nebula3d.initial_conditions_bubbles on any box (the sheet spans the whole x-z plane)"""
    y = (np.arange(ny) + 0.5) * dx
    Ly = ny * dx
    turb = N.grf3((nx, ny, nz), args.turb_slope, rng)
    warp = 0.05 * N.grf2((nx, nz), 3.2, rng)
    mod = 1.0 + 0.45 * N.grf2((nx, nz), 3.6, rng)
    y0 = 0.5 * Ly + warp[:, None, :]
    half = args.sheet_w * Ly
    prof = 1.0 / (1.0 + np.exp((np.abs(y[None, :, None] - y0) - half) / 0.025))
    sig = args.turb_sigma
    rho = np.exp(sig * turb - 0.5 * sig * sig) * (0.02 + 1.2 * prof * np.clip(mod, 0.3, None)[:, None, :])
    rho = np.maximum(rho, args.floor).astype(np.float32)
    return rho, prof.astype(np.float32), None


def ic_cliff_box(nx, ny, nz, dx, args, rng, off):
    """nebula3d.initial_conditions (cliff) on a wider/taller box: the wall continues sideways and down,
    off = (x, z) offset of the original 1.5 x 1 content region"""
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    # broadcast axes instead of a meshgrid (same values; 2026-10-25: the meshgrid was 24 of the 64 B/cell host peak)
    X, Y, Z = x[:, None, None], y[None, :, None], z[None, None, :]
    turb = N.grf3((nx, ny, nz), args.turb_slope, rng)
    top = off[1] + args.cloud_top + args.edge_amp * N.grf2((nx, ny), 2.8, rng)[:, :, None]
    top = top + 0.06 * np.sin(np.pi * Y / y.max()) - 0.03          # (nx, ny, 1): the wall top per column
    body = 1.0 / (1.0 + np.exp((Z - top) / 0.012))
    sig = args.turb_sigma
    rho = np.exp(sig * turb - 0.5 * sig * sig) * (0.02 + 1.5 * body)
    ng = int(round(args.globules * x.max() / 1.5))          # same globule density along the wider wall
    for _ in range(ng):
        gx, gy = rng.uniform(0.05, 0.95) * x.max(), rng.uniform(0.1, 0.9) * y.max()
        ix, iy = min(int(gx / dx), nx - 1), min(int(gy / dx), ny - 1)
        gz = float(top[ix, iy, 0]) + rng.uniform(-0.08, 0.03)
        gr = rng.uniform(0.008, 0.02)
        amp = rng.uniform(3.0, 8.0)
        h = int(4 * gr / dx) + 1
        sl = tuple(slice(max(c - h, 0), min(c + h + 1, n)) for c, n in
                   ((ix, nx), (iy, ny), (int(gz / dx), nz)))
        rr = (x[sl[0], None, None] - gx) ** 2 + (y[None, sl[1], None] - gy) ** 2 + (z[None, None, sl[2]] - gz) ** 2
        rho[sl] += (np.exp(-rr / (2 * gr * gr)) * amp).astype(np.float32)
    rho = np.maximum(rho, args.floor).astype(np.float32)
    return rho, body.astype(np.float32), top


def add_bipolar_clumps(rho, x, y, z, c, n, env, eq, P):
    """Felipe 2026-10-24: the ignition sweeps a couple of dense clumps and he wants more of them.
    P['clumps'] dense blobs in the envelope, 15-65 deg from the axis (inside / along the walls of the future
    lobes, both hemispheres; the equator is shadowed by the torus) at clump_rin..clump_rmax from the star,
    so both lobes open first and then sweep them into cometary knots. Peak density = clump_amp x the smooth
    envelope there, textured by the envelope's own log-normal noise (sqrt) so they are not perfect spheres.
    Positions come from their own seeded rng in nebula units: same clumps at any res, and the other random
    draws are untouched."""
    nc = int(P.get("clumps", 0))
    if nc <= 0:
        return
    r_ = np.random.default_rng([int(P["seed"]), 5151])
    dx = x[1] - x[0]
    th0, th1 = np.radians(15.0), np.radians(65.0)
    e1 = np.array([n[2], 0.0, -n[0]])                    # n = (sin tilt, 0, cos tilt): e1, e2 complete the frame
    e2 = np.array([0.0, 1.0, 0.0])
    rin, rmax = float(P["clump_rin"]), float(P["clump_rmax"])
    size, amp = float(P["clump_size"]), float(P["clump_amp"])
    for _ in range(nc):
        u = r_.uniform(np.cos(th1), np.cos(th0)) * r_.choice([-1.0, 1.0])
        ph = r_.uniform(0, 2 * np.pi)
        d = r_.uniform(rin, rmax)
        p = np.asarray(c) + d * (u * n + np.sqrt(1 - u * u) * (np.cos(ph) * e1 + np.sin(ph) * e2))
        a = size * r_.uniform(0.6, 1.5)
        A = amp * r_.uniform(0.5, 1.5)
        h = int(3.5 * a / dx) + 1
        ii = [int(p[0] / dx), int(p[1] / dx), int(p[2] / dx)]
        sl = tuple(slice(max(q - h, 0), min(q + h + 1, m)) for q, m in zip(ii, rho.shape))
        if any(s.start >= s.stop for s in sl):
            continue
        X, Y, Z = x[sl[0]][:, None, None], y[sl[1]][None, :, None], z[sl[2]][None, None, :]
        g = np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2) / (2 * a * a))
        rr = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) + 1e-6
        env_s = np.clip((rr / P["env_r0"]) ** -1.5, 0.05, 6.0)
        rho[sl] += 0.3 * eq[sl] * np.sqrt(env[sl] * env_s) * A * g


def ic_bipolar_box(nx, ny, nz, dx, args, rng):
    """nebula3d.initial_conditions_bipolar on a box of any size (nebula at the box centre)."""
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    c = np.array([nx * dx * 0.5, ny * dx * 0.5, nz * dx * 0.5])
    tl = np.radians(args.tilt)
    n = np.array([np.sin(tl), 0.0, np.cos(tl)])
    # broadcast axes instead of a meshgrid (same values, ~half the host memory at reset)
    RX, RY, RZ = (x - c[0])[:, None, None], (y - c[1])[None, :, None], (z - c[2])[None, None, :]
    r = np.sqrt(RX ** 2 + RY ** 2 + RZ ** 2) + 1e-6
    h = RX * n[0] + RY * n[1] + RZ * n[2]
    del RX, RY, RZ
    rc = np.sqrt(np.maximum(r * r - h * h, 0))
    turb = N.grf3((nx, ny, nz), args.turb_slope, rng)
    sig = args.turb_sigma
    env = np.clip((r / args.env_r0) ** -1.5, 0.05, 6.0) * np.exp(sig * turb - 0.5 * sig * sig)
    eq = 1.0 + args.eq_contrast * np.exp(-(h / (0.35 * r + 0.01)) ** 2)
    torus = args.torus_rho * np.exp(-((rc - args.torus_R) ** 2 + h ** 2) / (2 * args.torus_a ** 2))
    rho = np.maximum(0.3 * env * eq, args.floor)
    add_bipolar_clumps(rho, x, y, z, c, n, env, eq, vars(args))
    rho = rho.astype(np.float32)
    return rho, torus.astype(np.float32), None, (tuple(float(v) for v in c), tuple(float(v) for v in n))


# ----------------------------------------------------------------------------------- matched random fields
class MatchedNoise:
    """Resolution-independent Gaussian random fields. Every integer Fourier mode |m| <= K gets the
    same random coefficient whatever the grid size (a seeded 'master' spectrum), so the same seed
    gives the same large-scale field at 128^3 and 384^3; only modes above K differ (they carry a
    few % of the variance for the k^-3.8 spectra used here). Drop-in for nebula3d.grf3/grf2/solenoidal."""

    def __init__(self, seed, K=32):
        self.seed = int(seed)
        self.K = K
        self.calls = 0

    def _coeffs(self, shape, real_last):
        """complex coefficients on the (r)fft grid of `shape`, matched for |m| <= K."""
        self.calls += 1
        K = self.K
        mshape = tuple([2 * K + 1] * (len(shape) - 1) + [(K + 1) if real_last else 2 * K + 1])
        rm = np.random.default_rng([self.seed, 7919, self.calls])
        master = rm.standard_normal(mshape) + 1j * rm.standard_normal(mshape)
        tshape = tuple(list(shape[:-1]) + [shape[-1] // 2 + 1 if real_last else shape[-1]])
        rh = np.random.default_rng([self.seed, 104729, self.calls, *shape])
        out = rh.standard_normal(tshape) + 1j * rh.standard_normal(tshape)
        idx_t, idx_m = [], []
        for a, n in enumerate(shape):
            last = a == len(shape) - 1 and real_last
            if last:
                m = np.arange(0, min(K, n // 2) + 1)
                it, im = m, m
            else:
                m = np.arange(-min(K, (n - 1) // 2), min(K, (n - 1) // 2) + 1)
                it, im = m % n, m % (2 * K + 1)
            idx_t.append(it)
            idx_m.append(im)
        out[np.ix_(*idx_t)] = master[np.ix_(*idx_m)]
        return out

    def grf3(self, shape, slope, rng=None):
        kx = np.fft.fftfreq(shape[0])[:, None, None]
        ky = np.fft.fftfreq(shape[1])[None, :, None]
        kz = np.fft.rfftfreq(shape[2])[None, None, :]
        # physical wavenumber (cycles per domain unit); dx is the same on every axis
        k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2) * shape[2]
        k[0, 0, 0] = 1.0
        amp = k ** (-slope / 2.0)
        amp[0, 0, 0] = 0.0
        f = np.fft.irfftn(amp * self._coeffs(shape, True), s=shape, axes=(0, 1, 2)).astype(np.float32)
        return (f - f.mean()) / f.std()

    def grf2(self, shape, slope, rng=None, nz=None):
        kx = np.fft.fftfreq(shape[0])[:, None]
        ky = np.fft.rfftfreq(shape[1])[None, :]
        k = np.sqrt(kx ** 2 + ky ** 2) * (nz or 1)
        k[0, 0] = 1.0
        amp = k ** (-slope / 2.0)
        amp[0, 0] = 0.0
        f = np.fft.irfftn(amp * self._coeffs(shape, True), s=shape, axes=(0, 1)).astype(np.float32)
        return (f - f.mean()) / f.std()

    def solenoidal(self, shape, rng=None, kmin=1.0, kmax=6.0):
        ks = np.meshgrid(*[np.fft.fftfreq(n) * n for n in shape], indexing="ij")
        K = np.sqrt(sum(k * k for k in ks))
        amp = np.where((K >= kmin) & (K <= kmax), 1.0 / np.maximum(K, 1e-6), 0.0)
        F = [amp * self._coeffs(shape, False) for _ in range(3)]
        kd = sum(k * f for k, f in zip(ks, F)) / np.maximum(K * K, 1e-12)
        F = [f - k * kd for k, f in zip(ks, F)]
        v = [np.fft.ifftn(f).real for f in F]
        rms = np.sqrt(sum((c * c).mean() for c in v))
        return [np.ascontiguousarray(c / rms, dtype=np.float32) for c in v]


@contextlib.contextmanager
def patched_noise(noise, nz):
    """route nebula3d's IC functions through the matched noise generator"""
    old3, old2 = N.grf3, N.grf2
    N.grf3 = noise.grf3
    N.grf2 = lambda shape, slope, rng: noise.grf2(shape, slope, rng, nz=nz)
    try:
        yield
    finally:
        N.grf3, N.grf2 = old3, old2


# ----------------------------------------------------------------------------------- extra kernels
@wp.kernel
def add_torus(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
              mz: wp.array3d(dtype=float), torus: wp.array3d(dtype=float)):
    """the bipolar disk/torus forms after the turbulent phase, at rest (as in nebula3d.main)"""
    i, j, k = wp.tid()
    r = rho[i, j, k]
    tq = torus[i, j, k]
    f = 1.0 - tq / (tq + r)
    rho[i, j, k] = r + tq
    mx[i, j, k] = mx[i, j, k] * f
    my[i, j, k] = my[i, j, k] * f
    mz[i, j, k] = mz[i, j, k] * f


@wp.func
def mc_lim(a: float, b: float):
    """monotonized-central slope limiter (sharper than minmod, still TVD)"""
    if a * b <= 0.0:
        return 0.0
    s = wp.where(a > 0.0, 1.0, -1.0)
    return s * wp.min(wp.min(2.0 * wp.abs(a), 2.0 * wp.abs(b)), 0.5 * wp.abs(a + b))


@wp.func
def vl_lim(a: float, b: float):
    """van Leer (harmonic) slope limiter: between minmod and MC"""
    if a * b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


@wp.func
def slope(a: float, b: float, lim: int):
    if lim == 2:
        return mc_lim(a, b)
    if lim == 1:
        return vl_lim(a, b)
    return N.minmod(a, b)


@wp.func
def lim4(a: wp.vec4, b: wp.vec4, lim: int):
    return wp.vec4(slope(a[0], b[0], lim), slope(a[1], b[1], lim), slope(a[2], b[2], lim), slope(a[3], b[3], lim))


@wp.func
def flux_hll(qa: wp.vec4, qb: wp.vec4, qc: wp.vec4, qd: wp.vec4, c2b: float, c2c: float,
             e: wp.vec4, floor: float, lim: int):
    """HLL flux (isothermal, Davis wave speeds) on the face between cells b and c, limited primitive
    reconstruction (lim 0 minmod, 1 van Leer, 2 MC). nebula3d.face_flux = Rusanov + minmod."""
    L = qb + 0.5 * lim4(qb - qa, qc - qb, lim)
    R = qc - 0.5 * lim4(qc - qb, qd - qc, lim)
    rL = wp.max(L[0], floor)
    rR = wp.max(R[0], floor)
    unL = wp.dot(wp.vec4(0.0, L[1], L[2], L[3]), e)
    unR = wp.dot(wp.vec4(0.0, R[1], R[2], R[3]), e)
    UL = wp.vec4(rL, rL * L[1], rL * L[2], rL * L[3])
    UR = wp.vec4(rR, rR * R[1], rR * R[2], rR * R[3])
    FL = UL * unL + e * (rL * c2b)
    FR = UR * unR + e * (rR * c2c)
    cb = wp.sqrt(c2b)
    cc = wp.sqrt(c2c)
    SL = wp.min(unL - cb, unR - cc)
    SR = wp.max(unL + cb, unR + cc)
    if SL >= 0.0:
        return FL
    if SR <= 0.0:
        return FR
    return (SR * FL - SL * FR + (SL * SR) * (UR - UL)) / (SR - SL)


@wp.kernel
def hydro_stage_hll(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
                    mz: wp.array3d(dtype=float), xi: wp.array3d(dtype=float),
                    r0: wp.array3d(dtype=float), mx0: wp.array3d(dtype=float), my0: wp.array3d(dtype=float),
                    mz0: wp.array3d(dtype=float),
                    o_r: wp.array3d(dtype=float), o_mx: wp.array3d(dtype=float), o_my: wp.array3d(dtype=float),
                    o_mz: wp.array3d(dtype=float),
                    stars: wp.array(dtype=wp.vec4), nstars: int, wind: float,
                    jets: wp.array(dtype=wp.vec4), jet_cos: float, jet_r: float, pulse: float,
                    dt: float, dx: float, a: float, b: float, floor: float, cn2: float, ci2: float, vmax: float,
                    lim: int):
    """nebula3d.hydro_stage with the HLL face flux (sources identical)"""
    i, j, k = wp.tid()
    div = wp.vec4(0.0)
    for axis in range(3):
        oi = 0
        oj = 0
        ok = 0
        if axis == 0:
            oi = 1
        if axis == 1:
            oj = 1
        if axis == 2:
            ok = 1
        e = wp.vec4(0.0, float(oi), float(oj), float(ok))
        qm2 = N.prim(rho, mx, my, mz, i - 2 * oi, j - 2 * oj, k - 2 * ok, floor)
        qm1 = N.prim(rho, mx, my, mz, i - oi, j - oj, k - ok, floor)
        q0 = N.prim(rho, mx, my, mz, i, j, k, floor)
        qp1 = N.prim(rho, mx, my, mz, i + oi, j + oj, k + ok, floor)
        qp2 = N.prim(rho, mx, my, mz, i + 2 * oi, j + 2 * oj, k + 2 * ok, floor)
        cm1 = N.csq(xi, i - oi, j - oj, k - ok, cn2, ci2)
        c0 = N.csq(xi, i, j, k, cn2, ci2)
        cp1 = N.csq(xi, i + oi, j + oj, k + ok, cn2, ci2)
        fR = flux_hll(qm1, q0, qp1, qp2, c0, cp1, e, floor, lim)
        fL = flux_hll(qm2, qm1, q0, qp1, cm1, c0, e, floor, lim)
        div = div + (fR - fL)

    U = wp.vec4(rho[i, j, k], mx[i, j, k], my[i, j, k], mz[i, j, k])
    dU = -div / dx
    p = (wp.vec3(float(i), float(j), float(k)) + wp.vec3(0.5)) * dx
    x_ion = xi[i, j, k]
    if wind > 0.0 and x_ion > 0.01:
        acc = wp.vec3(0.0)
        for s in range(nstars):
            st = stars[s]
            if st[3] > 0.0:
                rel = p - wp.vec3(st[0], st[1], st[2])
                r2 = wp.dot(rel, rel) + 0.0004
                acc = acc + rel * (st[3] / (r2 * wp.sqrt(r2)))
        acc = acc * (wind * x_ion)
        dU = dU + wp.vec4(0.0, U[0] * acc[0], U[0] * acc[1], U[0] * acc[2])
    accj = wp.vec3(0.0)
    for s in range(nstars):
        jt = jets[s]
        st = stars[s]
        if jt[3] > 0.0 and st[3] > 0.0:
            rel = p - wp.vec3(st[0], st[1], st[2])
            r = wp.length(rel)
            if r < jet_r and r > 1e-6:
                ax = wp.vec3(jt[0], jt[1], jt[2])
                c = wp.dot(rel, ax) / r
                if wp.abs(c) > jet_cos:
                    accj = accj + ax * (wp.where(c > 0.0, 1.0, -1.0) * jt[3] * pulse / (r * r + 0.0004))
    dU = dU + wp.vec4(0.0, U[0] * accj[0], U[0] * accj[1], U[0] * accj[2])
    Un = U + dt * dU
    U0 = wp.vec4(r0[i, j, k], mx0[i, j, k], my0[i, j, k], mz0[i, j, k])
    out = a * U0 + b * Un
    r = wp.max(out[0], floor)
    vel = wp.vec3(out[1], out[2], out[3]) / r
    sp = wp.length(vel)
    if sp > vmax:
        vel = vel * (vmax / sp)
    o_r[i, j, k] = r
    o_mx[i, j, k] = r * vel[0]
    o_my[i, j, k] = r * vel[1]
    o_mz[i, j, k] = r * vel[2]


@wp.kernel
def mach_sums(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
              mz: wp.array3d(dtype=float), out: wp.array(dtype=float)):
    i, j, k = wp.tid()
    r = rho[i, j, k]
    m2 = mx[i, j, k] * mx[i, j, k] + my[i, j, k] * my[i, j, k] + mz[i, j, k] * mz[i, j, k]
    wp.atomic_add(out, 0, m2 / r)
    wp.atomic_add(out, 1, r)


@wp.kernel
def to_half(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=wp.float16)):
    i, j, k = wp.tid()
    dst[i, j, k] = wp.float16(src[i, j, k])


@wp.kernel
def every2(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float)):
    """dst = src[::2, ::2, ::2]"""
    i, j, k = wp.tid()
    dst[i, j, k] = src[2 * i, 2 * j, 2 * k]


@wp.kernel
def every2_half(src: wp.array3d(dtype=wp.float16), dst: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    dst[i, j, k] = float(src[2 * i, 2 * j, 2 * k])


class _Every2:
    """stands in for a full grid in nebula3d.find_triggered, which only reads a.shape and a[::2, ::2, ::2]"""

    def __init__(self, sub, shape):
        self.sub, self.shape = sub, shape

    def __getitem__(self, key):
        if key != (slice(None, None, 2),) * 3:
            raise IndexError("only a[::2, ::2, ::2] is available")
        return self.sub


# ----------------------------------------------------------------------------------- Sim
class Sim:
    """GPU-resident nebula3d simulation.

    sim = Sim("bipolar", res=128); sim.step(4); sim.update_render_fields()
    Fields for rendering (wp.array3d, [i,j,k]): rho, xi (float32), dfront, g0 (float16, or float32 with
    P['fp32_fields']); sim.shock computes the shock proxy on demand. sim.stage = 4 float planes of scratch,
    free between steps (render.Renderer packs into it). Stars: sim.stars = [(x, y, z, Q, t_on)],
    sim.active_stars() for the renderer.
    """

    def __init__(self, scenario="bipolar", res=128, params=None, ic_mode="matched", rays=None,
                 fuv_grid=None, ion_every=None, device="cuda:0", verbose=True, fast=True):
        self.device = device
        # fast=True (viewer): ray step 1 voxel, ionization every 3 steps, FUV every 8-16 steps.
        # fast=False (bake.py): nebula3d.py accuracy (half-voxel rays, P['ion_every'])
        self.fast = fast
        self.verbose = verbose
        self.ic_mode = ic_mode
        self.rays_override = rays
        self.fuv_override = fuv_grid
        self.ion_every_override = ion_every
        self.prof = {}
        self.reset(scenario, res, params)

    # ------------------------------------------------------------------ helpers
    def log(self, *a):
        if self.verbose:
            print("[sim]", *a, flush=True)

    def _z(self):
        return wp.zeros(self.shape, dtype=float, device=self.device)

    def _L(self, kernel, inputs, dim=None):
        wp.launch(kernel, dim=dim or self.shape, inputs=inputs, device=self.device)

    def _tick(self, key, t0):
        self.prof[key] = self.prof.get(key, 0.0) + (time.perf_counter() - t0)

    def args(self):
        return argparse.Namespace(**self.P)

    # ------------------------------------------------------------------ setup
    def _drop_fields(self):
        """release the big grids before a reset builds the new ones (lower peak on resets)"""
        for k in ("rho", "mx", "my", "mz", "xi", "xi_tmp", "dfront", "g0", "r1", "mx1", "my1", "mz1", "stage",
                  "df_half", "torus", "torus0", "volume"):
            self.__dict__.pop(k, None)

    def reset(self, scenario=None, res=None, params=None):
        t_setup = time.perf_counter()
        self._drop_fields()
        if scenario is not None:
            self.scenario = scenario
        if res is not None:
            self.res = int(res)
        if params is not None:
            self.P = dict(params)
        elif scenario is not None or not hasattr(self, "P"):
            self.P = make_params(self.scenario)
        self.P["scenario"] = self.scenario
        P = self.P
        args = self.args()
        for k_, v_ in EXTRA_DEFAULTS.items():
            P.setdefault(k_, v_)
        box = max(float(P["box"]), 1.0)
        P["box"] = box
        crop = P.get("crop")                      # a window of the full box (crop.py)
        if crop:
            import crop as C
            if self.ic_mode != "matched":
                raise ValueError("crop needs ic_mode='matched'")
            cfull, coff, (nx, ny, nz) = C.crop_window(self.scenario, self.res, box, crop)
        else:
            nx, ny, nz = grid_shape(self.scenario, self.res, box)
        self.shape = (nx, ny, nz)
        P["nx"], P["ny"], P["nz"] = nx, ny, nz
        dx = 1.0 / self.res                       # res = cells per nebula unit, whatever the box
        self.dx = dx
        self.L = (nx * dx, ny * dx, nz * dx)
        Lx, Ly = self.L[0], self.L[1]
        # offset of the original nebula3d content region (1.5 x 1 in x-z) inside a bigger box
        ox = 0.5 * (Lx - ASPECT[self.scenario][0]) if self.scenario != "bipolar" else 0.0
        oz = 0.5 * (self.L[2] - 1.0) if self.scenario != "bipolar" else 0.0
        self.content_off = (ox, oz)
        unit = abs(box - 1.0) < 1e-9 and nz == self.res and not crop
        rng = np.random.default_rng(P["seed"])
        self.rng = rng
        n_emb = P.get("n_embedded")
        if crop:
            Lf = tuple(n * dx for n in cfull)
            o = tuple(i * dx for i in coff)
            self.crop_full, self.crop_off, self.crop_origin, self.L_full = cfull, coff, o, Lf
            self._crop_faces = [(coff[a] > 0, coff[a] + self.shape[a] < cfull[a]) for a in range(3)]
            sfull = tuple(int(math.ceil(L_ / (1.0 / 24.0))) + 2 for L_ in Lf)   # stirring grid of the full box
            plane = (0, 1) if self.scenario == "cliff" else (0, 2)
            self.noise = C.WindowNoise(P["seed"], cfull, coff, self.shape, sfull, plane)
            if self.scenario == "cliff":
                # the wall top from the full domain's content offset; embedded protostars: as many as the full
                # domain puts in the window's share of the 1.5 wide wall
                args.cloud_top = P["cloud_top"] + (0.5 * (Lf[2] - 1.0) - o[2]) - oz
                cx0 = 0.5 * (Lf[0] - ASPECT["cliff"][0])
                share = max(0.0, min(o[0] + Lx, cx0 + 1.5) - max(o[0], cx0)) / 1.5
                n_emb = int(round(n_emb * share))
        else:
            self.crop_origin, self.L_full = (0.0, 0.0, 0.0), None
            self.noise = MatchedNoise(P["seed"]) if self.ic_mode == "matched" else None
        self.ext_stars = []
        ctx = patched_noise(self.noise, nz) if self.noise else contextlib.nullcontext()
        torus_np = None
        # star kinds: 'main' (Q = qf * q_main, follows the live q_main), 'abs' (absolute Q)
        # jet kinds:  'jet' (strength = live jet), 'proto' (= live jet_proto), 'none'
        with ctx:
            if self.scenario == "bubbles":
                if unit:
                    rho_np, body, top = N.initial_conditions_bubbles(nx, ny, nz, args, rng)
                else:
                    rho_np, body, top = ic_bubbles_box(nx, ny, nz, dx, args, rng)
                stars = [(s[0] + ox, s[1], s[2] + oz, s[3], s[4])
                         for s in N.place_bubble_stars(args, rng, Lx - 2 * ox, Ly)]
                kinds = ["main"] * len(stars)
                jets, jkinds = [], []
            elif self.scenario == "bipolar":
                if unit:
                    rho_np, torus_np, top, (c, n) = N.initial_conditions_bipolar(nx, ny, nz, args, rng)
                else:
                    rho_np, torus_np, top, (c, n) = ic_bipolar_box(nx, ny, nz, dx, args, rng)
                if P["pre_t"] <= 0:
                    rho_np = rho_np + torus_np
                    torus_np = None
                stars = [(c[0], c[1], c[2], P["q_main"], 0.0)]
                kinds = ["main"]
                jets, jkinds = [(n[0], n[1], n[2], P["jet"])], ["jet"]
            else:
                if unit:
                    rho_np, body, top = N.initial_conditions(nx, ny, nz, args, rng)
                else:
                    rho_np, body, top = ic_cliff_box(nx, ny, nz, dx, args, rng, (ox, oz))
                zs, ys = P["star_z"] + oz, P["star_y"]
                Lc = Lx - 2 * ox                          # width of the original content region
                stars = [(ox + 0.50 * Lc, ys * Ly, zs, P["q_main"], 0.0),
                         (ox + 0.30 * Lc, (ys + 0.08) * Ly, zs + 0.06, 0.5 * P["q_main"], 0.0),
                         (ox + 0.74 * Lc, (ys - 0.07) * Ly, zs + 0.03, 0.4 * P["q_main"], 0.02)]
                kinds = ["main"] * 3
                xc = (np.arange(nx) + 0.5) * dx
                inner = ((xc > ox) & (xc < Lx - ox)).astype(np.float32)[:, None, None]
                for p_, ton in N.pick_embedded_stars(rho_np, body * inner, top, n_emb, rng, dx,
                                                     (0.15 * P["t_end"], 0.6 * P["t_end"])):
                    stars.append((float(p_[0]), float(p_[1]), float(p_[2]), P["q_embedded"], float(ton)))
                    kinds.append("abs")
                jets, jkinds = [], []
        if crop:                                   # the full domain's edge + an organic fade at the cut faces
            T = C.window_taper(self.scenario, (nx, ny, nz), dx, self.L_full, self.crop_origin, self._crop_faces, P)
        else:
            T = cloud_taper(self.scenario, (nx, ny, nz), dx, self.L, P)
        if T is not None:                          # organic cloud edge: fade to the floor before the faces
            fl = np.float32(P["floor"])
            rho_np = fl + (rho_np - fl) * T
            if torus_np is not None:
                torus_np = torus_np * T
            del T
        for i in range(len(jets), len(stars)):     # main stars: no jets; embedded protostars: jets
            is_proto = stars[i][3] < 0.3 * P["q_main"]
            a_ = self._rand_axis()
            jets.append((float(a_[0]), float(a_[1]), float(a_[2]), P["jet_proto"] if is_proto else 0.0))
            jkinds.append("proto" if is_proto else "none")
        self.q_main0 = P["q_main"]
        # per-star Q factor: Q = qf * q_main for 'main' stars, absolute otherwise
        self.stars = [(s[0], s[1], s[2], s[3] / P["q_main"] if k == "main" else s[3], s[4])
                      for s, k in zip(stars, kinds)]
        self.kinds = kinds
        self.jets = [j[:3] for j in jets]
        self.jkinds = jkinds
        if crop:
            self._crop_stars()

        dev = self.device
        z = self._z
        sh = self.shape
        self.rho = wp.array(rho_np.astype(np.float32), dtype=float, device=dev)
        del rho_np
        self.mx, self.my, self.mz, self.xi = z(), z(), z(), z()
        # one scratch block: the RK2 stage between the two hydro kernels, free between steps (xi_tmp, fp32
        # accumulators of dfront / G0, the torus upload, render.Renderer's packing grids)
        self.stage = wp.zeros((4,) + sh, dtype=float, device=dev)
        self.r1, self.mx1, self.my1, self.mz1 = (self.stage[c] for c in range(4))
        self.xi_tmp = self.stage[0]
        # dfront / g0 (render, timeline, star-formation trigger) in float16; P['fp32_fields'] keeps them float32
        hdt = float if P.get("fp32_fields") else wp.float16
        self.dfront = wp.empty(sh, dtype=hdt, device=dev)
        self.dfront.fill_(1e3)
        self.g0 = wp.zeros(sh, dtype=hdt, device=dev)
        # the trigger reads dfront on the 2x-subsampled grid: an exact fp32 copy of just those cells
        self.df_half = wp.full(tuple((n + 1) // 2 for n in sh), 1e3, dtype=float, device=dev)             if P["trigger"] > 0 else None
        # the bipolar torus waits on the host (added at the end of the turbulent phase; kept for timeline restores)
        self.torus = torus_np.astype(np.float32) if torus_np is not None else None
        self.torus0 = self.torus
        self.smax = wp.zeros(1, dtype=float, device=dev)
        self.cn2 = (1.0 / P["ci_over_cn"]) ** 2
        self.ci2 = 1.0
        self.diag = math.sqrt(Lx ** 2 + Ly ** 2 + self.L[2] ** 2) + 0.6
        self.dr = (1.0 if self.fast else 0.5) * dx
        self.nsteps = int(self.diag / self.dr) + 1
        self.cap = max(len(self.stars), P["max_stars"])
        self.star_arr = wp.zeros(self.cap, dtype=wp.vec4, device=dev)
        self.jet_arr = wp.zeros(self.cap, dtype=wp.vec4, device=dev)
        self._wkey = None         # fresh star/jet arrays -> force the upload of the weights
        self.generation = getattr(self, "generation", 0) + 1
        self.bmax = wp.vec3(Lx, Ly, self.L[2])
        # interactive radiative transfer budget (nebula3d defaults: 768x1536 rays, FUV 192x256x512)
        # (measured on a 3090: 512x1024 rays = 0.7 ms at 128^3, 1.1 ms at 192^3; FUV 128x128x256 = 1/3 ms)
        nu = self.rays_override[0] if self.rays_override else (512 if nz <= 256 else 1024)
        nphi = self.rays_override[1] if self.rays_override else 2 * nu
        self.rfront = wp.zeros((nu, nphi), dtype=float, device=dev)
        fg = self.fuv_override or ((128, 128, 256) if nz <= 192 else ((160, 160, 320) if nz <= 256 else (256, 256, 512)))
        self.att = wp.zeros(tuple(fg), dtype=float, device=dev)
        self.ion_every = self.ion_every_override or (max(P["ion_every"], 3) if self.fast else P["ion_every"])
        self.fuv_every = (8 if nz <= 160 else 16) if self.fast else 8
        # stirring fields (low-res, solenoidal), cross-faded every pre_corr
        if self.ic_mode == "matched":
            # fixed physical resolution (24 cells per domain unit) -> same forcing at any grid size
            self.dxs = 1.0 / 24.0
            self.sshape = tuple(int(math.ceil(L_ / self.dxs)) + 2 for L_ in self.L)
        else:
            sfac = 6
            self.sshape = (nx // sfac + 2, ny // sfac + 2, nz // sfac + 2)
            self.dxs = sfac * dx
        self.drv = None
        # clocks / bookkeeping
        self.t = 0.0
        self.it = 0
        self.tp = 0.0
        self.ip = 0
        self.dt = 0.0
        self.version = 0          # bumps on every step (renderer repacks when it changes)
        self.rf_version = -1      # version of g0/shock
        self.fuv_version = -10 ** 9
        self.n_born = 0
        self.last_birth = -1e9
        self.events = []          # (t, name, value): live parameter changes, replayed by bake.py
        self.births = []
        self.mach = None
        self.sm0 = None
        self.P_reset = dict(P)    # parameters the ICs were built with (bake.py starts from these)
        self.phase = "pre" if P["pre_t"] > 0 else "main"
        if self.phase == "pre":
            self.drv = {"A": self._mkF(), "B": self._mkF(), "clock": 0.0}
        else:
            self._start_main()
        wp.synchronize()
        self.setup_s = time.perf_counter() - t_setup
        self.log(f"{self.scenario} grid {nx}x{ny}x{nz} ({nx*ny*nz/1e6:.1f}M cells, box {box:g}"
                 f"{', crop ' + str(list(crop)) if crop else ''}) rays {nu}x{nphi} fuv {tuple(fg)} ic={self.ic_mode} "
                 f"solver {P['solver']} setup {self.setup_s:.2f}s, {len(self.stars)} stars")

    def _rand_axis(self):
        v = self.rng.standard_normal(3)
        return v / np.linalg.norm(v)

    def _mkF(self):
        gen = self.noise.solenoidal if self.noise else (lambda s, r, a, b: N.solenoidal(s, r, a, b))
        return [wp.array(c_, dtype=float, device=self.device)
                for c_ in gen(self.sshape, self.rng, 1.0, self.P["pre_kmax"])]

    def _start_main(self):
        """end of the turbulent phase: add the torus, record compression reference, first ionization"""
        if self.torus is not None:                # uploaded into the scratch block (it waits on the host)
            t = self.stage[0]
            t.assign(self.torus)
            self._L(add_torus, [self.rho, self.mx, self.my, self.mz, t])
            self.torus = None
        if self.P["trigger"] > 0:
            from scipy import ndimage
            self.sm0 = ndimage.gaussian_filter(self._every2(self.rho), 1.0)
        self.phase = "main"
        self._set_star_weights()
        self.ionize()

    # ------------------------------------------------------------------ storage helpers
    def render_scratch(self):
        """two RGBA16F grids (vec4h views of the stage block), free between sim steps (render.Renderer.pack)"""
        n = self.shape[0] * self.shape[1] * self.shape[2]
        return (wp.array(ptr=self.stage.ptr, dtype=wp.vec4h, shape=self.shape, device=self.device),
                wp.array(ptr=self.stage.ptr + 8 * n, dtype=wp.vec4h, shape=self.shape, device=self.device))

    @property
    def shock(self):
        """compression heating proxy max(0, -div v) rho, computed now into a new fp32 grid (bake.py); the
        renderer computes it inline while packing"""
        out = wp.empty(self.shape, dtype=float, device=self.device)
        self._L(N.shock_field, [self.rho, self.mx, self.my, self.mz, self.dx, out])
        return out

    def _every2(self, a):
        """a[::2, ::2, ::2] as a numpy array, subsampled on the GPU (no full-grid download)"""
        out = wp.empty(tuple((n + 1) // 2 for n in self.shape), dtype=float, device=self.device)
        self._L(every2 if a.dtype == wp.float32 else every2_half, [a, out], dim=out.shape)
        return out.numpy()

    def _refresh_df_half(self):
        self._L(every2 if self.dfront.dtype == wp.float32 else every2_half, [self.dfront, self.df_half],
                dim=self.df_half.shape)

    def _crop_stars(self):
        """cropped window: stars at their full-domain places (reset placed them for a box the size of the window)"""
        Lf, o, L = self.L_full, self.crop_origin, self.L
        if self.scenario != "bipolar":
            sx = (0.5 * (Lf[0] - ASPECT[self.scenario][0]) - o[0]) - 0.5 * (L[0] - ASPECT[self.scenario][0])
            sz = (0.5 * (Lf[2] - 1.0) - o[2]) - 0.5 * (L[2] - 1.0)
            st = []
            for s, k in zip(self.stars, self.kinds):
                if k == "main":              # placed from the content region (embedded ones come from the window gas)
                    y = s[1] if self.scenario == "bubbles" else s[1] * Lf[1] / L[1] - o[1]
                    s = (s[0] + sx, y, s[2] + sz) + tuple(s[3:])
                st.append(s)
            self.stars = st
            self.content_off = (self.content_off[0] + sx, self.content_off[1] + sz)
        m = 0.02

        def inside(s, k):                    # outside only if the window cut it off (stars placed beyond the
            if self.scenario == "cliff" and k == "main":     # full box on purpose stay); the cliff's main stars
                return True                  # light the whole wall from above the gas: kept, wherever they are
            for a in range(3):
                full = -o[a] - m <= s[a] <= Lf[a] - o[a] + m
                if full and not (-m <= s[a] <= L[a] + m):
                    return False
            return True
        keep = [i for i, (s, k) in enumerate(zip(self.stars, self.kinds)) if inside(s, k)]
        # bubbles: stars outside the window still light it in the FUV (PAH glow); they do not ionize or push
        self.ext_stars = [(self.stars[i], self.kinds[i]) for i in range(len(self.stars)) if i not in keep]
        if self.ext_stars:
            self.log(f"crop: {len(self.ext_stars)} stars outside the window (FUV only)")
        self.stars = [self.stars[i] for i in keep]
        self.kinds = [self.kinds[i] for i in keep]
        self.jets = [self.jets[i] for i in keep]
        self.jkinds = [self.jkinds[i] for i in keep]

    # ------------------------------------------------------------------ physics pieces
    def q_of(self, i):
        """full ionizing output of star i (no turn-on ramp)"""
        s = self.stars[i]
        return s[3] * self.P["q_main"] if self.kinds[i] == "main" else s[3]

    def ramp(self, i):
        """turn-on factor of star i: 0 before t_on, smoothstep(age/q_ramp)^3 -> 1 (see EXTRA_DEFAULTS)"""
        age = self.t - self.stars[i][4]
        if self.phase != "main" or age < 0.0:
            return 0.0
        tr = float(self.P.get("q_ramp", 0.0))
        if tr <= 0.0 or age >= tr:
            return 1.0
        x = age / tr
        s = x * x * (3.0 - 2.0 * x)
        return max(s * s * s, 1e-4)

    def q_eff(self, i):
        return self.q_of(i) * self.ramp(i)

    def jet_of(self, i):
        k = self.jkinds[i]
        return self.P["jet"] if k == "jet" else (self.P["jet_proto"] if k == "proto" else 0.0)

    def active_stars(self):
        """[(x, y, z, Q/q_main, age)] of switched-on stars (for the renderer)"""
        out = []
        if self.phase != "main":
            return out
        for i, s in enumerate(self.stars):
            if self.t >= s[4]:
                out.append((s[0], s[1], s[2], self.q_eff(i) / self.P["q_main"], self.t - s[4]))
        return out

    def _set_star_weights(self):
        qm = self.P["q_main"]
        w = [(s[0], s[1], s[2], (self.q_eff(i) / qm) if self.t >= s[4] else 0.0) for i, s in enumerate(self.stars)]
        w += [(0.0, 0.0, 0.0, 0.0)] * (self.cap - len(w))
        key = tuple(round(v[3], 9) for v in w) + tuple(self.jet_of(i) for i in range(len(self.stars)))
        if key == getattr(self, "_wkey", None):
            return
        self._wkey = key
        self.star_arr.assign(np.array(w, dtype=np.float32))
        jw = [(a[0], a[1], a[2], self.jet_of(i)) for i, a in enumerate(self.jets)]
        jw += [(0.0, 0.0, 1.0, 0.0)] * (self.cap - len(jw))
        self.jet_arr.assign(np.array(jw, dtype=np.float32))

    def ionize(self):
        t0 = time.perf_counter()
        df = self.stage[1]                        # fp32 dfront in the scratch, rounded into self.dfront at the end
        self.xi.zero_()
        df.fill_(1e3)
        nu, nphi = self.rfront.shape
        for i, (sx, sy, sz, _, ton) in enumerate(self.stars):
            if self.t < ton:
                continue
            sv = wp.vec3(sx, sy, sz)
            self._L(N.cast_rays, [self.rho, sv, self.q_eff(i), self.dr, self.nsteps, self.dx, self.diag,
                                  self.bmax, self.rfront], dim=(nu, nphi))
            self._L(N.apply_front, [self.rfront, sv, self.dx, 1.2 * self.dx, self.xi, df])
        self._L(N.smooth_xi, [self.xi, self.xi_tmp])
        wp.copy(self.xi, self.xi_tmp)
        self._store(df, self.dfront)
        if self.df_half is not None:
            self._L(every2, [df, self.df_half], dim=self.df_half.shape)
        if self.profile:
            wp.synchronize()
            self._tick("ionize", t0)

    def fuv_field(self):
        """G0 = sum_star L exp(-tau_FUV) / r^2 on a coarse spherical grid (render only), accumulated in fp32
        in the scratch block and stored in self.g0"""
        if self.phase != "main":
            self.g0.zero_()
            return
        acc = self.stage[2]
        acc.zero_()
        fnr, fnu, fnp = self.att.shape
        binr = self.diag / fnr
        for i, (sx, sy, sz, _, ton) in enumerate(self.stars):
            if self.t < ton:
                continue
            sv = wp.vec3(sx, sy, sz)
            self._L(N.cast_fuv, [self.rho, self.xi, sv, self.P["kappa_fuv"], self.dr, self.nsteps, self.dx,
                                 binr, self.bmax, self.att], dim=(fnu, fnp))
            self._L(N.apply_fuv, [self.att, sv, self.q_eff(i) / self.P["q_main"], self.dx, binr, acc])
        for (sx, sy, sz, qf, ton), kind in self.ext_stars:     # cropped domain: stars outside the window (FUV only)
            age = self.t - ton
            if age < 0.0:
                continue
            tr = float(self.P.get("q_ramp", 0.0))
            x = min(age / tr, 1.0) if tr > 0 else 1.0
            ramp = max((x * x * (3.0 - 2.0 * x)) ** 3, 1e-4)
            q = (qf * self.P["q_main"] if kind == "main" else qf) * ramp
            sv = wp.vec3(sx, sy, sz)
            self._L(N.cast_fuv, [self.rho, self.xi, sv, self.P["kappa_fuv"], self.dr, self.nsteps, self.dx,
                                 binr, self.bmax, self.att], dim=(fnu, fnp))
            self._L(N.apply_fuv, [self.att, sv, q / self.P["q_main"], self.dx, binr, acc])
        self._store(acc, self.g0)

    def _store(self, src, dst):
        """fp32 scratch -> dfront / g0 (float16, or float32 with P['fp32_fields'])"""
        if dst.dtype == wp.float16:
            self._L(to_half, [src, dst])
        else:
            wp.copy(dst, src)

    fuv_every = 8   # steps between FUV (G0) updates (set per resolution in reset); G0 only feeds the render

    def update_render_fields(self, force=False):
        """recompute G0 if the sim advanced since the last call (the shock proxy is computed while packing)"""
        if self.rf_version == self.version and not force:
            return False
        t0 = time.perf_counter()
        n_on = sum(1 for s in self.stars if self.t >= s[4])
        tr = float(self.P.get("q_ramp", 0.0))
        ramping = tr > 0 and any(0.0 <= self.t - s[4] < tr for s in self.stars)
        if ramping and self.version - getattr(self, "fuv_version", -10 ** 9) >= max(2, self.fuv_every // 4):
            force = True
        if (force or self.version - getattr(self, "fuv_version", -10 ** 9) >= self.fuv_every
                or n_on != getattr(self, "fuv_nstars", -1) or self.phase != getattr(self, "fuv_phase", None)):
            self.fuv_field()
            self.fuv_version, self.fuv_nstars, self.fuv_phase = self.version, n_on, self.phase
        self.rf_version = self.version
        if self.profile:
            wp.synchronize()
            self._tick("render_fields", t0)
        return True

    def _compute_dt(self):
        self.smax.zero_()
        self._L(N.max_speed, [self.rho, self.mx, self.my, self.mz, self.xi, self.cn2, self.ci2, self.smax])
        return self.P["cfl"] * self.dx / float(self.smax.numpy()[0])

    def _hydro(self, dt, nstars, wind, pulse):
        P = self.P
        hll = P.get("solver", "rusanov") == "hll"
        hs = hydro_stage_hll if hll else N.hydro_stage
        common = [self.star_arr, nstars, wind, self.jet_arr, math.cos(math.radians(P["jet_angle"])),
                  P["jet_r"], pulse, dt, self.dx]
        tail = [P["floor"], self.cn2, self.ci2, 6.0]
        if hll:
            tail.append({"minmod": 0, "vanleer": 1, "mc": 2}[P.get("limiter", "minmod")])
        U = [self.rho, self.mx, self.my, self.mz]
        U1 = [self.r1, self.mx1, self.my1, self.mz1]
        self._L(hs, U + [self.xi] + U + U1 + common + [0.0, 1.0] + tail)
        # stage 2 writes U^{n+1} = 0.5 U^n + 0.5 (U1 + dt L(U1)) over U^n: each cell reads its own U^n once,
        # before writing it, so one stage copy is enough (same arithmetic as two copies, bit for bit)
        self._L(hs, U1 + [self.xi] + U + U + common + [0.5, 0.5] + tail)

    def _stir(self, dt, amp):
        d = self.drv
        blend = min(d["clock"] / self.P["pre_corr"], 1.0)
        if self.L_full is not None:               # cropped window: the full domain's stirring at the window offset
            import crop as C
            self._L(C.drive_window, [self.rho, self.mx, self.my, self.mz] + d["A"] + d["B"]
                    + [blend, amp, dt, self.dx, self.dxs, wp.vec3(*self.crop_origin)])
        else:
            self._L(N.drive, [self.rho, self.mx, self.my, self.mz] + d["A"] + d["B"]
                    + [blend, amp, dt, self.dx, self.dxs])
        d["clock"] += dt
        if d["clock"] >= self.P["pre_corr"]:
            d["A"], d["B"], d["clock"] = d["B"], self._mkF(), 0.0

    def _pre_step(self):
        dt = self._compute_dt()
        self._hydro(dt, 0, 0.0, 0.0)
        self._stir(dt, self.P["pre_amp"])
        self.tp += dt
        self.ip += 1
        self.dt = dt
        if self.tp >= self.P["pre_t"]:
            s = wp.zeros(2, dtype=float, device=self.device)
            self._L(mach_sums, [self.rho, self.mx, self.my, self.mz, s])
            a, b = s.numpy()
            self.mach = math.sqrt(a / b) * self.P["ci_over_cn"]
            self.log(f"pre-turbulence done: {self.ip} steps t={self.tp:.3f} mass-weighted Mach ~{self.mach:.1f}")
            if self.P["drive_main"] <= 0:
                self.drv = None
            self._start_main()

    def _check_triggers(self):
        P = self.P
        if not (P["trigger"] > 0 and self.it > 0 and self.it % P["trigger_every"] == 0 and len(self.stars) < self.cap
                and self.n_born < P["max_births"] and self.t - self.last_birth >= P["birth_gap"]):
            return
        t0 = time.perf_counter()
        st_abs = [(s[0], s[1], s[2], self.q_of(i), s[4]) for i, s in enumerate(self.stars)]
        sh = self.shape
        if self.df_half is None:                  # trigger switched on after the reset
            self.df_half = wp.empty(tuple((n + 1) // 2 for n in sh), dtype=float, device=self.device)
            self._refresh_df_half()
        # find_triggered reads only a[::2, ::2, ::2]: subsample on the GPU instead of downloading the grids
        xi_s = self._every2(self.xi)
        if self.L_full is not None:               # cropped window: no births in the organic fade at the cut faces
            w = float(P.get("crop_edge", 0.1)) + 0.06     # (its low initial density inflates the compression ratio)
            for a, (lo_cut, hi_cut) in enumerate(self._crop_faces):
                c = (np.arange(xi_s.shape[a]) * 2 + 0.5) * self.dx
                sl = [None, None, None]
                sl[a] = slice(None)
                near = np.zeros(xi_s.shape[a], bool)
                if lo_cut:
                    near |= c < w
                if hi_cut:
                    near |= c > self.L[a] - w
                xi_s = np.where(near[tuple(sl)], np.float32(1.0), xi_s)     # 'ionized' -> not a candidate
        born = N.find_triggered(_Every2(self._every2(self.rho), sh), _Every2(xi_s, sh),
                                _Every2(self.df_half.numpy(), sh), self.sm0, st_abs, self.args(), self.dx, self.t)
        for b in born[: self.cap - len(self.stars)]:
            self.stars.append(b)
            self.kinds.append("abs")
            self.jets.append(tuple(float(v) for v in self._rand_axis()))
            self.jkinds.append("proto")
            self.births.append(b)
            self.n_born += 1
            self.last_birth = self.t
            self.log(f"* star born t={self.t:.4f} at {tuple(round(v, 3) for v in b[:3])}")
        self._tick("triggers", t0)

    def _main_step(self):
        P = self.P
        self._check_triggers()
        if self.it % self.ion_every == 0:
            self.ionize()
        self._set_star_weights()
        dt = self._compute_dt()
        if self.dt_cap is not None:          # bake: land exactly on frame times (as nebula3d.main)
            dt = min(dt, self.dt_cap)
        pulse = max(0.0, 1.0 + P["jet_pulse"] * math.sin(2 * math.pi * self.t / P["jet_period"]))
        self._hydro(dt, len(self.stars), P["wind"], pulse)
        if P["drive_main"] > 0 and self.drv is not None:
            self._stir(dt, P["pre_amp"] * P["drive_main"])
        self.t += dt
        self.it += 1
        self.dt = dt

    # ------------------------------------------------------------------ public API
    profile = False
    dt_cap = None

    def replace_stars(self, stars, kinds, jets, jkinds):
        """use an explicit star list (bake.py replays the preview's stars incl. triggered births)"""
        self.stars = [tuple(float(v) for v in s) for s in stars]
        self.kinds = list(kinds)
        self.jets = [tuple(float(v) for v in j[:3]) for j in jets]
        self.jkinds = list(jkinds)
        if len(self.stars) > self.cap:
            self.cap = len(self.stars)
            self.star_arr = wp.zeros(self.cap, dtype=wp.vec4, device=self.device)
            self.jet_arr = wp.zeros(self.cap, dtype=wp.vec4, device=self.device)
        self._wkey = None
        if self.phase == "main":
            self._set_star_weights()
            self.ionize()

    def stars_absolute(self):
        """nebula3d meta.json format: (x, y, z, Q, t_on) with absolute Q; jets (ax, ay, az, strength)"""
        st = [(s[0], s[1], s[2], self.q_of(i), s[4]) for i, s in enumerate(self.stars)]
        jt = [(j[0], j[1], j[2], self.jet_of(i)) for i, j in enumerate(self.jets)]
        return st, jt

    def step(self, k=1):
        """advance k hydro steps (pre-turbulence or main phase). Returns wall seconds."""
        t0 = time.perf_counter()
        for _ in range(k):
            if self.phase == "pre":
                self._pre_step()
            else:
                self._main_step()
            self.version += 1
        self._tick("step", t0)
        self.prof["n_steps"] = self.prof.get("n_steps", 0) + k
        return time.perf_counter() - t0

    def set_param(self, name, value):
        """change a parameter; live ones take effect on the next step and are logged for the bake"""
        old = self.P.get(name)
        if isinstance(old, int) and not isinstance(old, bool):
            value = int(round(value))
        self.P[name] = value
        if name == "drive_main" and value > 0 and self.drv is None and self.phase == "main":
            self.drv = {"A": self._mkF(), "B": self._mkF(), "clock": 0.0}
        if name in LIVE:
            self.events.append((float(self.t if self.phase == "main" else 0.0), name, value))
            if name in ("q_main", "jet", "jet_proto"):
                self._wkey = None
        return value

    # ------------------------------------------------------------------ timeline (see timeline.py)
    def tau(self):
        """timeline time: < 0 during the pre-turbulence phase, = t afterwards"""
        return self.t if self.phase == "main" else self.tp - self.P["pre_t"]

    def get_state_meta(self):
        """everything except the big fields needed to resume from this moment"""
        import copy
        d = self.drv
        return dict(
            t=self.t, it=self.it, tp=self.tp, ip=self.ip, dt=self.dt, phase=self.phase, mach=self.mach,
            stars=list(self.stars), kinds=list(self.kinds), jets=list(self.jets), jkinds=list(self.jkinds),
            births=list(self.births), n_born=self.n_born, last_birth=self.last_birth, P=dict(self.P),
            n_events=len(self.events), sm0=self.sm0,
            drv=None if d is None else {"A": list(d["A"]), "B": list(d["B"]), "clock": d["clock"]},
            rng=copy.deepcopy(self.rng.bit_generator.state), noise_calls=self.noise.calls if self.noise else 0)

    def set_state_meta(self, m):
        """after the fields were restored (timeline.Timeline.restore)"""
        import copy
        for k in ("t", "it", "tp", "ip", "dt", "phase", "mach", "n_born", "last_birth", "sm0"):
            setattr(self, k, m[k])
        self.stars, self.kinds = list(m["stars"]), list(m["kinds"])
        self.jets, self.jkinds = list(m["jets"]), list(m["jkinds"])
        self.births = list(m["births"])
        self.P = dict(m["P"])
        self.events = self.events[:m["n_events"]]
        d = m["drv"]
        self.drv = None if d is None else {"A": list(d["A"]), "B": list(d["B"]), "clock": d["clock"]}
        self.rng.bit_generator.state = copy.deepcopy(m["rng"])
        if self.noise:
            self.noise.calls = m["noise_calls"]
        self.torus = self.torus0 if self.phase == "pre" else None
        self._wkey = None
        if self.phase == "main":
            self._set_star_weights()
        self.version += 1
        # xi / dfront / g0 come from the snapshot: no ray casting needed to show it
        if self.df_half is not None:              # the trigger reads the restored (fp16) dfront
            self._refresh_df_half()
        self.rf_version = self.version
        self.fuv_version = self.version
        self.fuv_nstars = sum(1 for s in self.stars if self.t >= s[4])
        self.fuv_phase = self.phase

    def time_label(self):
        if self.phase == "pre":
            return f"pre-turb {self.tp:.3f}/{self.P['pre_t']:.2f}"
        return f"t={self.t:.4f}"

    def stats(self):
        """quick diagnostics (forces a sync): ionized fraction, density range"""
        xi = self.xi.numpy()
        r = self.rho.numpy()
        return dict(ion=float(xi.mean()), rho_max=float(r.max()), rho_min=float(r.min()))
