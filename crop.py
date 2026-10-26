"""Cropped domains: simulate only the part of the nebula a shot sees (used by sim.Sim when P["crop"] is set).

crop = [x0, x1, y0, y1, z0, z1]: the window as fractions of the full box of the scenario (sim.grid_shape at the
same res and box). The window is the same nebula, cut out:
  * matched noise of the FULL domain evaluated on the window: every mode |m| <= K (the large scales that the
    'matched' ICs share between a 128 preview and a 384 bake) is exact; the smaller scales are a new realization
    with the same spectrum and variance, exactly like a run at another resolution;
  * the cloud's organic edge of the full domain, times an organic fade (width `crop_edge`) at every window face
    that cuts through the full domain, so the cut walls hide like the box faces do;
  * stars at their full-domain positions (those outside the window are dropped), the turbulent stirring of the
    full domain (sampled with the window offset).
Limits: bipolar windows are centred on the star (its ICs are built around the box centre), the depth
axis (y) of bubbles / cliff is cropped symmetrically, the globules / embedded protostars of the cliff are drawn
inside the window (same density, other places). Coordinates inside the sim are window coordinates:
sim.crop_origin is the window corner in full-domain units (camera target = full-domain target - crop_origin).
"""
import math

import numpy as np
import warp as wp

import sim as S

N = S.N


def crop_window(scenario, res, box, crop):
    """(full grid shape, window offset in cells, window shape) for crop fractions [x0, x1, y0, y1, z0, z1]"""
    full = S.grid_shape(scenario, res, box)
    off, shp = [], []
    for a in range(3):
        lo, hi = float(crop[2 * a]), float(crop[2 * a + 1])
        # centred axes (see the module docstring); a crop given symmetric about 0.5 stays exactly centred, so the
        # content offsets the ICs get (sim.Sim.reset: ox, oz) are the full domain's ones (custom ICs need no change)
        sym = scenario == "bipolar" or a == 1 or abs(lo + hi - 1.0) < 1e-6
        if sym:
            h = max(0.5 - lo, hi - 0.5)
            lo, hi = 0.5 - h, 0.5 + h
            n = min(full[a], max(16, int(math.ceil(2 * h * full[a] / 16.0)) * 16))
            i0 = (full[a] - n) // 2
        else:
            i0 = max(0, int(math.floor(lo * full[a] / 8.0)) * 8)
            n = min(full[a], max(16, int(math.ceil(hi * full[a] / 8.0)) * 8)) - i0
        off.append(i0)
        shp.append(n)
    return full, tuple(off), tuple(shp)


# ----------------------------------------------------------------------------------- matched noise on a window
class WindowNoise(S.MatchedNoise):
    """S.MatchedNoise of the full grid `full`, evaluated on the window (offset `off`, shape `win`)."""

    def __init__(self, seed, full, off, win, sfull, plane=(0, 2), K=32):
        super().__init__(seed, K)
        self.full, self.off, self.win = tuple(full), tuple(off), tuple(win)
        self.sfull = tuple(sfull)          # stirring grid of the full domain
        self.plane = plane                 # axes of the 2D fields (bubbles: x-z sheet warps, cliff: x-y wall top)

    def _axes(self, shape):
        if len(shape) == 3:
            return (0, 1, 2)
        for ax in (self.plane, (0, 2), (0, 1), (1, 2)):
            if tuple(self.win[a] for a in ax) == tuple(shape):
                return ax
        raise ValueError(f"window noise: no window axes for shape {shape}")

    def _field(self, shape, slope):
        """unit-variance field with spectrum k^-slope: full-domain modes |m| <= K exact, the rest resynthesized"""
        self.calls += 1
        K, d = self.K, len(shape)
        axes = self._axes(shape)
        F = [self.full[a] for a in axes]
        O = [self.off[a] for a in axes]
        W = list(shape)
        mshape = tuple([2 * K + 1] * (d - 1) + [K + 1])
        rm = np.random.default_rng([self.seed, 7919, self.calls])
        master = rm.standard_normal(mshape) + 1j * rm.standard_normal(mshape)
        ms, ims = [], []
        for a, n in enumerate(F):                      # the matched modes, as MatchedNoise._coeffs places them
            if a == d - 1:
                m = np.arange(0, min(K, n // 2) + 1)
                im = m
            else:
                m = np.arange(-min(K, (n - 1) // 2), min(K, (n - 1) // 2) + 1)
                im = m % (2 * K + 1)
            ms.append(m)
            ims.append(im)

        def amp_of(kk):                                # kk: per-axis wavenumbers (cycles per cell), broadcastable
            k = np.sqrt(sum(x * x for x in kk))
            a_ = np.where(k > 0, np.maximum(k, 1e-30) ** (-slope / 2.0), 0.0)
            return a_

        grids = np.ix_(*[m / n for m, n in zip(ms, F)])
        Z = amp_of(grids) * master[np.ix_(*ims)]
        wl = np.where(ms[-1] == 0, 1.0, 2.0)
        Nf = float(np.prod(F))
        # exact variance of the matched part over the full domain (band-limited: a 2K+2 grid samples it exactly)
        C = [2 * K + 2] * d
        Zc = np.zeros(tuple(C[:-1]) + (C[-1] // 2 + 1,), complex)
        Zc[np.ix_(*[m % c if a < d - 1 else m for a, (m, c) in enumerate(zip(ms, C))])] = Z
        fc = np.fft.irfftn(Zc, s=C) * (float(np.prod(C)) / Nf)
        var_low = float((fc * fc).mean())
        # expected variance of the other modes of the full grid (independent unit complex normals)
        var_all = 0.0
        mz = np.arange(F[-1] // 2 + 1)
        cz = np.where((mz == 0) | ((F[-1] % 2 == 0) & (mz == F[-1] // 2)), 1.0, 4.0)
        kz = mz / F[-1]
        if d == 3:
            ky = np.fft.fftfreq(F[1])[:, None]
            for mx in np.fft.fftfreq(F[0]):
                a2 = amp_of((mx, ky, kz[None, :])) ** 2
                var_all += float((a2 * cz[None, :]).sum())
        else:
            kx = np.fft.fftfreq(F[0])[:, None]
            var_all = float((amp_of((kx, kz[None, :])) ** 2 * cz[None, :]).sum())
        ca = np.where(ms[-1] == 0, 1.0, 4.0)
        low_sum = float((amp_of(grids) ** 2 * ca.reshape([1] * (d - 1) + [-1])).sum())
        var_high = max(var_all - low_sum, 0.0) / Nf ** 2
        # matched part at the window points: separable partial DFT
        E = [np.exp(2j * np.pi * np.outer(m, o + np.arange(w)) / n) for m, o, w, n in zip(ms, O, W, F)]
        Zw = Z * wl.reshape([1] * (d - 1) + [-1])
        out = np.empty(tuple(W), np.float32)
        if d == 3:
            G = np.einsum("abc,cz->abz", Zw, E[2], optimize=True)
            G = np.einsum("abz,by->ayz", G, E[1], optimize=True).reshape(len(ms[0]), -1)
            Gr, Gi = G.real.copy(), G.imag.copy()
            del G
            Er, Ei = E[0].real.T.copy(), E[0].imag.T.copy()
            step = max(1, int(2e8 // max(Gr.shape[1], 1) // 8))
            for i in range(0, W[0], step):
                out[i:i + step] = ((Er[i:i + step] @ Gr - Ei[i:i + step] @ Gi) / Nf).reshape(-1, W[1], W[2])
        else:
            G = Zw @ E[1]
            out[:] = (E[0].real.T @ G.real - E[0].imag.T @ G.imag) / Nf
        # the other scales: a new realization on the window lattice, scaled to the full-domain variance
        tshape = tuple(W[:-1]) + (W[-1] // 2 + 1,)
        rh = np.random.default_rng([self.seed, 104729, self.calls, *W, 31337])
        R = rh.standard_normal(tshape) + 1j * rh.standard_normal(tshape)
        kw = [np.fft.fftfreq(w) for w in W[:-1]] + [np.fft.rfftfreq(W[-1])]
        kw = np.ix_(*kw)
        low = np.ones(tshape, bool)
        for k_, n in zip(kw, F):
            low = low & (np.abs(k_) <= K / n + 1e-12)
        R *= amp_of(kw) * ~low
        fh = np.fft.irfftn(R, s=W)
        del R
        vh = float((fh * fh).mean())
        if vh > 0 and var_high > 0:
            out += (fh * math.sqrt(var_high / vh)).astype(np.float32)
        del fh
        out /= np.float32(math.sqrt(var_low + var_high))
        return out

    def grf3(self, shape, slope, rng=None):
        if tuple(shape) != self.win:
            return super().grf3(shape, slope, rng)
        return self._field(shape, slope)

    def grf2(self, shape, slope, rng=None, nz=None):
        try:
            self._axes(shape)
        except ValueError:
            return super().grf2(shape, slope, rng, nz)
        return self._field(shape, slope)

    def solenoidal(self, shape, rng=None, kmin=1.0, kmax=6.0):
        """stirring of the full domain (drive_window samples it with the window offset)"""
        return super().solenoidal(self.sfull, rng, kmin, kmax)


# ----------------------------------------------------------------------------------- organic edges
def organic_noise_at(shape, dx, seed, origin, kmin=1.2, kmax=3.5, n=10):
    """sim.organic_noise evaluated on a window starting at `origin` (same plane waves)"""
    r = np.random.default_rng([int(seed), 4242])
    x, y, z = (((np.arange(m) + 0.5) * dx + o).astype(np.float32) for m, o in zip(shape, origin))
    f = np.zeros(shape, np.float32)
    for _ in range(n):
        d = r.standard_normal(3)
        d /= np.linalg.norm(d)
        k = 2 * np.pi * r.uniform(kmin, kmax) * d
        ph = r.uniform(0, 2 * np.pi)
        f += np.sin(np.float32(k[0]) * x[:, None, None] + np.float32(k[1]) * y[None, :, None]
                    + np.float32(k[2]) * z[None, None, :] + np.float32(ph))
    return f / np.float32(np.sqrt(n / 2.0))


def window_taper(scenario, shape, dx, Lfull, origin, faces, P):
    """sim.cloud_taper of the full domain on the window x an organic fade at the cut faces"""
    amt = float(P.get("cloud_edge", 0.0))
    n = organic_noise_at(shape, dx, P["seed"], origin, kmin=1.0, kmax=2.6)
    T = np.ones(shape, np.float32)
    if amt > 0:
        nn = n * np.float32(0.13 * amt)
        L = Lfull
        ax = [(((np.arange(m) + 0.5) * dx + o) - 0.5 * l).astype(np.float32) / np.float32(0.5 * l)
              for m, l, o in zip(shape, L, origin)]
        if scenario == "bipolar":
            u = np.sqrt(ax[0][:, None, None] ** 2 * (0.5 * L[0]) ** 2 + ax[1][None, :, None] ** 2 * (0.5 * L[1]) ** 2
                        + ax[2][None, None, :] ** 2 * (0.5 * L[2]) ** 2) / np.float32(0.5 * min(L))
            u0 = min(float(P.get("cloud_r", 0.6)) / (0.5 * min(L)), 0.85)
            T *= 1.0 - S._sst(u0, 1.0, u + nn)
        else:
            hx, hz = 0.75 / (0.5 * L[0]), 0.5 / (0.5 * L[2])
            T *= 1.0 - S._sst(min(hx - 0.1, 0.88), 1.0, np.abs(ax[0])[:, None, None] + nn)
            uz = ax[2][None, None, :]
            uz = np.maximum(uz, 0.0) if scenario == "cliff" else np.abs(uz)
            T *= 1.0 - S._sst(min(hz - 0.1, 0.88), 1.0, uz + nn)
    w = float(P.get("crop_edge", 0.1))
    for a, (lo_cut, hi_cut) in enumerate(faces):     # organic fade at the faces that cut the full domain
        c = (np.arange(shape[a]) + 0.5) * dx
        sl = [None, None, None]
        sl[a] = slice(None)
        for cut, dist in ((lo_cut, c), (hi_cut, shape[a] * dx - c)):
            if cut:
                g = (dist.astype(np.float32)[tuple(sl)] + np.float32(0.35 * w) * n) / np.float32(w)
                T *= S._sst(0.15, 1.0, g).astype(np.float32)
    return T.astype(np.float32)


@wp.kernel
def drive_window(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
                 mz: wp.array3d(dtype=float), fx: wp.array3d(dtype=float), fy: wp.array3d(dtype=float),
                 fz: wp.array3d(dtype=float), gx: wp.array3d(dtype=float), gy: wp.array3d(dtype=float),
                 gz: wp.array3d(dtype=float), blend: float, amp: float, dt: float, dx: float, dxs: float,
                 origin: wp.vec3):
    """nebula3d.drive with the full-domain stirring fields sampled at (window position + origin)"""
    i, j, k = wp.tid()
    p = (wp.vec3(float(i), float(j), float(k)) + wp.vec3(0.5)) * dx + origin
    fa = wp.vec3(N.sample_n(fx, p, dxs), N.sample_n(fy, p, dxs), N.sample_n(fz, p, dxs))
    fb = wp.vec3(N.sample_n(gx, p, dxs), N.sample_n(gy, p, dxs), N.sample_n(gz, p, dxs))
    f = (1.0 - blend) * fa + blend * fb
    r = rho[i, j, k] * amp * dt
    mx[i, j, k] = mx[i, j, k] + r * f[0]
    my[i, j, k] = my[i, j, k] + r * f[1]
    mz[i, j, k] = mz[i, j, k] + r * f[2]


# ----------------------------------------------------------------------------------- shot -> crop
def symmetric(crop):
    """the smallest crop centred on the box centre (0.5) on every axis that contains `crop`"""
    out = []
    for a in range(3):
        h = max(0.5 - crop[2 * a], crop[2 * a + 1] - 0.5)
        out += [round(0.5 - h, 4), round(0.5 + h, 4)]
    return out


def vertical_tans(cam, W=16, H=9):
    """(tan_x, tan_y) of the 9:16 strip that edit_v2 --vertical cuts out of a W x H (16:9) frame of `cam`"""
    tx, ty = cam.tans(W, H)
    return ty * 9.0 / 16.0, ty


def crop_for_shot(scenario, res, box, cams, tans, margin=0.12, nsamp=9, sphere=None, xshift=0.0):
    """crop fractions that contain what the cameras (render.OrbitCamera, full-domain coordinates) see with
    (tan_x, tan_y), plus `margin` (nebula units) for the organic fade; cams = every camera of the shot.
    sphere = (centre, radius): only count what is inside it (bipolar: the cloud, not the empty box corners).
    xshift: horizontal offset of the frame centre in tan units (edit_v2's 9:16 cut at xc: (2 xc - 1) tan_x16)"""
    full = S.grid_shape(scenario, res, box)
    L = np.array(full, float) / res
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for cam in cams:
        pos, fwd, right, up = cam.basis()
        tx, ty = tans
        for sx in np.linspace(-1, 1, nsamp):
            for sy in np.linspace(-1, 1, nsamp):
                d = fwd + (xshift + sx * tx) * right + sy * ty * up
                d /= np.linalg.norm(d)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ta = np.where(np.abs(d) > 1e-9, (0.0 - pos) / d, -np.inf)
                    tb = np.where(np.abs(d) > 1e-9, (L - pos) / d, np.inf)
                t0 = max(np.max(np.minimum(ta, tb)), 0.0)
                t1 = np.min(np.maximum(ta, tb))
                if t1 <= t0:
                    continue
                for t in np.linspace(t0, t1, 48):
                    p = pos + d * t
                    if sphere is not None and np.linalg.norm(p - np.asarray(sphere[0])) > sphere[1]:
                        continue
                    lo, hi = np.minimum(lo, p), np.maximum(hi, p)
    lo = np.clip(lo - margin, 0.0, L)
    hi = np.clip(hi + margin, 0.0, L)
    return [float(v) for a in range(3) for v in (lo[a] / L[a], hi[a] / L[a])]
