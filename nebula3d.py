"""Nebula 3D: ionization fronts carving a turbulent molecular cloud, on the GPU (NVIDIA Warp).

Same toy physics as nebula_toy2d.py, in 3D:
  * isothermal Euler, c^2 = c_n^2 + x_ion (c_i^2 - c_n^2)   (hot ionized gas ~100x pressure)
  * Stromgren ionization along rays from each star:  ionized while  int n^2 r^2 dr < Q
    -> per star we cast nu*nphi rays and store only the FRONT RADIUS per direction (the
       ionized fraction along a ray is monotonic), then every cell looks up its ray.
  * per-cell dfront = r - r_front (signed, min over stars) gives the stratification used for
    rendering: OIII deep inside, H-alpha everywhere ionized, SII just inside the front and the
    PDR glow (FUV-heated PAH skin, orange in JWST images) just outside it.
Numerics: MUSCL (minmod) + Rusanov flux, Heun RK2, outflow boundaries, one fused kernel per stage.

python nebula3d.py --run cliff_a --nx 288 --ny 160 --nz 192 --t_end 0.25 --frames 60
"""
import argparse, json, math, os, time
import numpy as np
import warp as wp

wp.config.log_level = wp.LOG_WARNING if hasattr(wp, "LOG_WARNING") else None
wp.init()

# ----------------------------------------------------------------------------------- helpers
@wp.func
def ldc(a: wp.array3d(dtype=float), i: int, j: int, k: int):
    return a[wp.clamp(i, 0, a.shape[0] - 1), wp.clamp(j, 0, a.shape[1] - 1), wp.clamp(k, 0, a.shape[2] - 1)]


@wp.func
def prim(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
         mz: wp.array3d(dtype=float), i: int, j: int, k: int, floor: float):
    r = wp.max(ldc(rho, i, j, k), floor)
    return wp.vec4(r, ldc(mx, i, j, k) / r, ldc(my, i, j, k) / r, ldc(mz, i, j, k) / r)


@wp.func
def csq(xi: wp.array3d(dtype=float), i: int, j: int, k: int, cn2: float, ci2: float):
    return cn2 + ldc(xi, i, j, k) * (ci2 - cn2)


@wp.func
def minmod(a: float, b: float):
    if a * b <= 0.0:
        return 0.0
    if wp.abs(a) < wp.abs(b):
        return a
    return b


@wp.func
def minmod4(a: wp.vec4, b: wp.vec4):
    return wp.vec4(minmod(a[0], b[0]), minmod(a[1], b[1]), minmod(a[2], b[2]), minmod(a[3], b[3]))


@wp.func
def face_flux(qa: wp.vec4, qb: wp.vec4, qc: wp.vec4, qd: wp.vec4, c2b: float, c2c: float,
              e: wp.vec4, floor: float):
    """Rusanov flux on the face between cells b and c (a, d are the outer neighbours).
    q = (rho, u, v, w); e = (0, axis one-hot)."""
    L = qb + 0.5 * minmod4(qb - qa, qc - qb)
    R = qc - 0.5 * minmod4(qc - qb, qd - qc)
    rL = wp.max(L[0], floor)
    rR = wp.max(R[0], floor)
    vL = wp.vec4(0.0, L[1], L[2], L[3])
    vR = wp.vec4(0.0, R[1], R[2], R[3])
    unL = wp.dot(vL, e)
    unR = wp.dot(vR, e)
    UL = wp.vec4(rL, rL * L[1], rL * L[2], rL * L[3])
    UR = wp.vec4(rR, rR * R[1], rR * R[2], rR * R[3])
    FL = UL * unL + e * (rL * c2b)
    FR = UR * unR + e * (rR * c2c)
    amax = wp.max(wp.abs(unL) + wp.sqrt(c2b), wp.abs(unR) + wp.sqrt(c2c))
    return 0.5 * (FL + FR) - 0.5 * amax * (UR - UL)


# ----------------------------------------------------------------------------------- hydro
@wp.kernel
def hydro_stage(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
                mz: wp.array3d(dtype=float), xi: wp.array3d(dtype=float),
                r0: wp.array3d(dtype=float), mx0: wp.array3d(dtype=float), my0: wp.array3d(dtype=float),
                mz0: wp.array3d(dtype=float),
                o_r: wp.array3d(dtype=float), o_mx: wp.array3d(dtype=float), o_my: wp.array3d(dtype=float),
                o_mz: wp.array3d(dtype=float),
                stars: wp.array(dtype=wp.vec4), nstars: int, wind: float,
                jets: wp.array(dtype=wp.vec4), jet_cos: float, jet_r: float, pulse: float,
                dt: float, dx: float, a: float, b: float, floor: float, cn2: float, ci2: float, vmax: float):
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
        qm2 = prim(rho, mx, my, mz, i - 2 * oi, j - 2 * oj, k - 2 * ok, floor)
        qm1 = prim(rho, mx, my, mz, i - oi, j - oj, k - ok, floor)
        q0 = prim(rho, mx, my, mz, i, j, k, floor)
        qp1 = prim(rho, mx, my, mz, i + oi, j + oj, k + ok, floor)
        qp2 = prim(rho, mx, my, mz, i + 2 * oi, j + 2 * oj, k + 2 * ok, floor)
        cm1 = csq(xi, i - oi, j - oj, k - ok, cn2, ci2)
        c0 = csq(xi, i, j, k, cn2, ci2)
        cp1 = csq(xi, i + oi, j + oj, k + ok, cn2, ci2)
        fR = face_flux(qm1, q0, qp1, qp2, c0, cp1, e, floor)
        fL = face_flux(qm2, qm1, q0, qp1, cm1, c0, e, floor)
        div = div + (fR - fL)

    U = wp.vec4(rho[i, j, k], mx[i, j, k], my[i, j, k], mz[i, j, k])
    dU = -div / dx
    # optional stellar wind / radiation pressure on ionized gas: a = wind * w_s * rhat / r^2
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
    # bipolar jets: momentum injection in two narrow cones along each star's axis (any gas)
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
def max_speed(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
              mz: wp.array3d(dtype=float), xi: wp.array3d(dtype=float), cn2: float, ci2: float,
              out: wp.array(dtype=float)):
    i, j, k = wp.tid()
    r = rho[i, j, k]
    v = wp.vec3(mx[i, j, k], my[i, j, k], mz[i, j, k]) / r
    c = wp.sqrt(cn2 + xi[i, j, k] * (ci2 - cn2))
    wp.atomic_max(out, 0, wp.abs(v[0]) + wp.abs(v[1]) + wp.abs(v[2]) + c)


@wp.kernel
def drive(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
          mz: wp.array3d(dtype=float), fx: wp.array3d(dtype=float), fy: wp.array3d(dtype=float),
          fz: wp.array3d(dtype=float), gx: wp.array3d(dtype=float), gy: wp.array3d(dtype=float),
          gz: wp.array3d(dtype=float), blend: float, amp: float, dt: float, dx: float, dxs: float):
    """turbulent stirring: a = amp * lerp(F, G, blend), F/G low-res solenoidal fields"""
    i, j, k = wp.tid()
    p = (wp.vec3(float(i), float(j), float(k)) + wp.vec3(0.5)) * dx
    fa = wp.vec3(sample_n(fx, p, dxs), sample_n(fy, p, dxs), sample_n(fz, p, dxs))
    fb = wp.vec3(sample_n(gx, p, dxs), sample_n(gy, p, dxs), sample_n(gz, p, dxs))
    f = (1.0 - blend) * fa + blend * fb
    r = rho[i, j, k] * amp * dt
    mx[i, j, k] = mx[i, j, k] + r * f[0]
    my[i, j, k] = my[i, j, k] + r * f[1]
    mz[i, j, k] = mz[i, j, k] + r * f[2]


@wp.kernel
def shock_field(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
                mz: wp.array3d(dtype=float), dx: float, out: wp.array3d(dtype=float)):
    """compression heating proxy  max(0, -div v) * rho  -> shock-excited line emission"""
    i, j, k = wp.tid()
    du = ldc(mx, i + 1, j, k) / ldc(rho, i + 1, j, k) - ldc(mx, i - 1, j, k) / ldc(rho, i - 1, j, k)
    dv = ldc(my, i, j + 1, k) / ldc(rho, i, j + 1, k) - ldc(my, i, j - 1, k) / ldc(rho, i, j - 1, k)
    dw = ldc(mz, i, j, k + 1) / ldc(rho, i, j, k + 1) - ldc(mz, i, j, k - 1) / ldc(rho, i, j, k - 1)
    div = (du + dv + dw) / (2.0 * dx)
    out[i, j, k] = wp.max(-div, 0.0) * rho[i, j, k]


# ----------------------------------------------------------------------------------- ionization
@wp.func
def sample_n(rho: wp.array3d(dtype=float), p: wp.vec3, dx: float):
    """trilinear density at world position p, 0 outside the domain"""
    g = p / dx - wp.vec3(0.5)
    nx = rho.shape[0]
    ny = rho.shape[1]
    nz = rho.shape[2]
    if g[0] < -0.5 or g[1] < -0.5 or g[2] < -0.5:
        return 0.0
    if g[0] > float(nx) - 0.5 or g[1] > float(ny) - 0.5 or g[2] > float(nz) - 0.5:
        return 0.0
    i0 = int(wp.floor(g[0]))
    j0 = int(wp.floor(g[1]))
    k0 = int(wp.floor(g[2]))
    fx = g[0] - float(i0)
    fy = g[1] - float(j0)
    fz = g[2] - float(k0)
    c000 = ldc(rho, i0, j0, k0)
    c100 = ldc(rho, i0 + 1, j0, k0)
    c010 = ldc(rho, i0, j0 + 1, k0)
    c110 = ldc(rho, i0 + 1, j0 + 1, k0)
    c001 = ldc(rho, i0, j0, k0 + 1)
    c101 = ldc(rho, i0 + 1, j0, k0 + 1)
    c011 = ldc(rho, i0, j0 + 1, k0 + 1)
    c111 = ldc(rho, i0 + 1, j0 + 1, k0 + 1)
    c00 = c000 + (c100 - c000) * fx
    c10 = c010 + (c110 - c010) * fx
    c01 = c001 + (c101 - c001) * fx
    c11 = c011 + (c111 - c011) * fx
    c0 = c00 + (c10 - c00) * fy
    c1 = c01 + (c11 - c01) * fy
    return c0 + (c1 - c0) * fz


@wp.func
def box_range(o: wp.vec3, d: wp.vec3, bmax: wp.vec3):
    """slab test against [0, bmax]; returns (t_enter, t_exit), t_exit < t_enter means miss"""
    t0 = float(0.0)
    t1 = float(1e9)
    for a in range(3):
        da = d[a]
        if wp.abs(da) < 1e-9:
            if o[a] < 0.0 or o[a] > bmax[a]:
                t1 = -1.0
        else:
            ta = (0.0 - o[a]) / da
            tb = (bmax[a] - o[a]) / da
            t0 = wp.max(t0, wp.min(ta, tb))
            t1 = wp.min(t1, wp.max(ta, tb))
    return wp.vec2(t0, t1)


@wp.kernel
def cast_rays(rho: wp.array3d(dtype=float), star: wp.vec3, Q: float, dr: float, nsteps: int,
              dx: float, rcap: float, bmax: wp.vec3, rfront: wp.array2d(dtype=float)):
    iu, ip = wp.tid()
    nu = rfront.shape[0]
    nphi = rfront.shape[1]
    u = -1.0 + (float(iu) + 0.5) * 2.0 / float(nu)
    s = wp.sqrt(wp.max(1.0 - u * u, 0.0))
    phi = (float(ip) + 0.5) * 2.0 * wp.pi / float(nphi)
    d = wp.vec3(s * wp.cos(phi), s * wp.sin(phi), u)
    col = float(0.0)
    rf = rcap
    tr = box_range(star, d, bmax)
    n0 = int(tr[0] / dr)
    n1 = wp.min(nsteps, int(tr[1] / dr) + 2)
    for n in range(n0, n1):
        r = (float(n) + 0.5) * dr
        nv = sample_n(rho, star + d * r, dx)
        inc = nv * nv * r * r * dr
        if col + inc > Q:
            rf = r - 0.5 * dr + dr * (Q - col) / wp.max(inc, 1e-20)
            break
        col = col + inc
    rfront[iu, ip] = rf


@wp.func
def rf_at(rfront: wp.array2d(dtype=float), iu: int, ip: int):
    nu = rfront.shape[0]
    nphi = rfront.shape[1]
    ipw = ((ip % nphi) + nphi) % nphi
    return rfront[wp.clamp(iu, 0, nu - 1), ipw]


@wp.kernel
def apply_front(rfront: wp.array2d(dtype=float), star: wp.vec3, dx: float, width: float,
                xi: wp.array3d(dtype=float), dfront: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    p = (wp.vec3(float(i), float(j), float(k)) + wp.vec3(0.5)) * dx
    rel = p - star
    r = wp.max(wp.length(rel), 1e-6)
    u = rel[2] / r
    phi = wp.atan2(rel[1], rel[0])
    if phi < 0.0:
        phi = phi + 2.0 * wp.pi
    nu = rfront.shape[0]
    nphi = rfront.shape[1]
    fu = (u + 1.0) * 0.5 * float(nu) - 0.5
    fp = phi / (2.0 * wp.pi) * float(nphi) - 0.5
    iu = int(wp.floor(fu))
    ip = int(wp.floor(fp))
    tu = fu - float(iu)
    tp = fp - float(ip)
    a = rf_at(rfront, iu, ip) * (1.0 - tp) + rf_at(rfront, iu, ip + 1) * tp
    b = rf_at(rfront, iu + 1, ip) * (1.0 - tp) + rf_at(rfront, iu + 1, ip + 1) * tp
    rf = a * (1.0 - tu) + b * tu
    val = wp.clamp((rf - r) / width + 0.5, 0.0, 1.0)
    if val > xi[i, j, k]:
        xi[i, j, k] = val
    # signed distance to this star's front along the ray (<0 inside the HII region); keep the min
    dfront[i, j, k] = wp.min(dfront[i, j, k], r - rf)


@wp.kernel
def smooth_xi(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    m = (ldc(src, i - 1, j, k) + ldc(src, i + 1, j, k) + ldc(src, i, j - 1, k) + ldc(src, i, j + 1, k)
         + ldc(src, i, j, k - 1) + ldc(src, i, j, k + 1)) / 6.0
    dst[i, j, k] = 0.5 * src[i, j, k] + 0.5 * m



# ----------------------------------------------------------------------------------- FUV field
@wp.kernel
def cast_fuv(rho: wp.array3d(dtype=float), xi: wp.array3d(dtype=float), star: wp.vec3, kappa: float,
             dr: float, nsteps: int, dx: float, binr: float, bmax: wp.vec3, att: wp.array3d(dtype=float)):
    """Dust attenuation exp(-tau_FUV(r)) along rays from a star, stored on a coarse spherical
    grid att[ir, iu, iphi] (bin width binr). Dust = neutral gas + a little in ionized gas."""
    iu, ip = wp.tid()
    nr = att.shape[0]
    nu = att.shape[1]
    nphi = att.shape[2]
    u = -1.0 + (float(iu) + 0.5) * 2.0 / float(nu)
    s = wp.sqrt(wp.max(1.0 - u * u, 0.0))
    phi = (float(ip) + 0.5) * 2.0 * wp.pi / float(nphi)
    d = wp.vec3(s * wp.cos(phi), s * wp.sin(phi), u)
    tau = float(0.0)
    ib = int(0)
    tr = box_range(star, d, bmax)
    n0 = int(tr[0] / dr)
    n1 = wp.min(nsteps, int(tr[1] / dr) + 2)
    for n in range(n0, n1):
        r = (float(n) + 0.5) * dr
        q = star + d * r
        n_tot = sample_n(rho, q, dx)
        x_i = sample_n(xi, q, dx)
        tau = tau + kappa * n_tot * (1.0 - 0.92 * x_i) * dr
        nb = int(r / binr)
        if nb > ib:
            for b in range(ib, wp.min(nb, nr)):
                att[b, iu, ip] = wp.exp(-tau)
            ib = nb
        if ib >= nr:
            break
    for b in range(ib, nr):
        att[b, iu, ip] = wp.exp(-tau)


@wp.kernel
def apply_fuv(att: wp.array3d(dtype=float), star: wp.vec3, lum: float, dx: float, binr: float,
              g0: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    p = (wp.vec3(float(i), float(j), float(k)) + wp.vec3(0.5)) * dx
    rel = p - star
    r = wp.max(wp.length(rel), 1e-6)
    u = rel[2] / r
    phi = wp.atan2(rel[1], rel[0])
    if phi < 0.0:
        phi = phi + 2.0 * wp.pi
    nr = att.shape[0]
    nu = att.shape[1]
    nphi = att.shape[2]
    fr = r / binr - 0.5
    fu = (u + 1.0) * 0.5 * float(nu) - 0.5
    fp = phi / (2.0 * wp.pi) * float(nphi) - 0.5
    ir = int(wp.floor(fr))
    iu = int(wp.floor(fu))
    ip = int(wp.floor(fp))
    tr = fr - float(ir)
    tu = fu - float(iu)
    tp = fp - float(ip)
    acc = float(0.0)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                w = wp.where(a == 0, 1.0 - tr, tr) * wp.where(b == 0, 1.0 - tu, tu) * wp.where(c == 0, 1.0 - tp, tp)
                rr = wp.clamp(ir + a, 0, nr - 1)
                uu = wp.clamp(iu + b, 0, nu - 1)
                pp = (((ip + c) % nphi) + nphi) % nphi
                acc = acc + w * att[rr, uu, pp]
    r2 = wp.max(r * r, 0.0004)
    g0[i, j, k] = g0[i, j, k] + lum * acc / r2

# ----------------------------------------------------------------------------------- setup
def grf3(shape, slope, rng):
    kx = np.fft.fftfreq(shape[0])[:, None, None]
    ky = np.fft.fftfreq(shape[1])[None, :, None]
    kz = np.fft.rfftfreq(shape[2])[None, None, :]
    k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    k[0, 0, 0] = 1.0
    amp = k ** (-slope / 2.0)
    amp[0, 0, 0] = 0.0
    ph = rng.standard_normal(amp.shape) + 1j * rng.standard_normal(amp.shape)
    f = np.fft.irfftn(amp * ph, s=shape, axes=(0, 1, 2)).astype(np.float32)
    return (f - f.mean()) / f.std()


def grf2(shape, slope, rng):
    kx = np.fft.fftfreq(shape[0])[:, None]
    ky = np.fft.rfftfreq(shape[1])[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2); k[0, 0] = 1.0
    amp = k ** (-slope / 2.0); amp[0, 0] = 0.0
    ph = rng.standard_normal(amp.shape) + 1j * rng.standard_normal(amp.shape)
    f = np.fft.irfftn(amp * ph, s=shape, axes=(0, 1)).astype(np.float32)
    return (f - f.mean()) / f.std()


def initial_conditions(nx, ny, nz, args, rng):
    """'cliff': a molecular cloud wall in the bottom half with a ragged top, a cluster above."""
    dx = 1.0 / nz
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    turb = grf3((nx, ny, nz), args.turb_slope, rng)
    top = args.cloud_top + args.edge_amp * grf2((nx, ny), 2.8, rng)[:, :, None]
    # large-scale tilt: the wall bulges toward the viewer (low y) like the Cosmic Cliffs
    top = top + 0.06 * np.sin(np.pi * Y / Y.max()) - 0.03
    body = 1.0 / (1.0 + np.exp((Z - top) / 0.012))
    sig = args.turb_sigma
    rho = np.exp(sig * turb - 0.5 * sig * sig) * (0.02 + 1.5 * body)
    # dense globules near the surface: pillar seeds (evaluated in local boxes only)
    for _ in range(args.globules):
        gx, gy = rng.uniform(0.05, 0.95) * x.max(), rng.uniform(0.1, 0.9) * y.max()
        ix, iy = min(int(gx / dx), nx - 1), min(int(gy / dx), ny - 1)
        gz = float(top[ix, iy, 0]) + rng.uniform(-0.08, 0.03)
        gr = rng.uniform(0.008, 0.02)
        amp = rng.uniform(3.0, 8.0)
        h = int(4 * gr / dx) + 1
        sl = tuple(slice(max(c - h, 0), min(c + h + 1, n)) for c, n in
                   ((ix, nx), (iy, ny), (int(gz / dx), nz)))
        rr = (X[sl] - gx) ** 2 + (Y[sl] - gy) ** 2 + (Z[sl] - gz) ** 2
        rho[sl] += (np.exp(-rr / (2 * gr * gr)) * amp).astype(np.float32)
    rho = np.maximum(rho, args.floor).astype(np.float32)
    return rho, body.astype(np.float32), top


def initial_conditions_bubbles(nx, ny, nz, args, rng):
    """'bubbles': a thick, warped turbulent sheet seen face-on (we look along +y)."""
    dx = 1.0 / nz
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    Ly = ny * dx
    turb = grf3((nx, ny, nz), args.turb_slope, rng)
    warp = 0.05 * grf2((nx, nz), 3.2, rng)                       # sheet mid-plane warp (x,z)
    mod = 1.0 + 0.45 * grf2((nx, nz), 3.6, rng)                  # large-scale column variations
    y0 = 0.5 * Ly + warp[:, None, :]
    half = args.sheet_w * Ly
    prof = 1.0 / (1.0 + np.exp((np.abs(y[None, :, None] - y0) - half) / 0.025))
    sig = args.turb_sigma
    rho = np.exp(sig * turb - 0.5 * sig * sig) * (0.02 + 1.2 * prof * np.clip(mod, 0.3, None)[:, None, :])
    rho = np.maximum(rho, args.floor).astype(np.float32)
    return rho, prof.astype(np.float32), None


def place_bubble_stars(args, rng, Lx, Ly):
    """Poisson-disk positions in the sheet so every star gets room to carve its own void."""
    pts = []
    tries = 0
    while len(pts) < args.n_bubbles and tries < 20000:
        tries += 1
        p = np.array([rng.uniform(0.15, 0.85) * Lx, 0.5 * Ly + rng.uniform(-0.04, 0.04),
                      rng.uniform(0.22, 0.78)])
        if all(np.hypot(p[0] - q[0], p[2] - q[2]) > args.min_sep for q in pts):
            pts.append(p)
    stars = []
    for i, p in enumerate(pts):
        Q = args.q_main * (1.0 if i == 0 else rng.uniform(0.35, 0.9))
        ton = 0.0 if i == 0 else float(rng.uniform(0, 0.15 * args.t_end))
        stars.append((float(p[0]), float(p[1]), float(p[2]), float(Q), ton))
    return stars


def find_triggered(rho_np, xi_np, df_np, sm0, stars, args, dx, t):
    """Collect & collapse: gas swept into a shell just outside an ionization front that has been
    compressed >= trigger x its initial local density (and is dense in absolute terms) forms a
    new star. Works on a 2x-downsampled grid for speed."""
    from scipy import ndimage
    r = rho_np[::2, ::2, ::2]
    sm = ndimage.gaussian_filter(r, 1.0)
    x_ = xi_np[::2, ::2, ::2]
    d_ = df_np[::2, ::2, ::2]
    comp = sm / np.maximum(sm0, 0.05)
    cand = (comp > args.trigger) & (x_ < 0.05) & (d_ > 0.0) & (d_ < args.shell_w) & (sm > args.trigger_min)
    if not cand.any():
        return []
    born = []
    idx = np.argwhere(cand)
    score = (comp * sm)[cand]
    L = np.array(rho_np.shape) * dx
    m = 0.06
    # pick among the best candidates at random (weighted) so births spread over many shells
    top = np.argsort(-score)[:4000]
    w = score[top] / score[top].sum()
    rng_t = np.random.default_rng(int(t * 1e6) % (2 ** 31))
    order = rng_t.choice(top, size=min(len(top), 400), replace=False, p=w)
    for o in order:
        p = (idx[o] * 2 + 1.0) * dx
        if np.any(p[[0, 2]] < m) or np.any(p[[0, 2]] > L[[0, 2]] - m):
            continue                      # stay away from the (outflow) domain boundaries
        if all(np.linalg.norm(p - np.array(q[:3])) > args.trigger_sep for q in stars + born):
            born.append((float(p[0]), float(p[1]), float(p[2]), args.q_trig, float(t)))
            break                         # at most one birth per check
    return born


def solenoidal(shape, rng, kmin=1.0, kmax=6.0):
    """unit-RMS divergence-free random vector field on a small grid (large-scale stirring)"""
    ks = np.meshgrid(*[np.fft.fftfreq(n) * n for n in shape], indexing="ij")
    K = np.sqrt(sum(k * k for k in ks))
    amp = np.where((K >= kmin) & (K <= kmax), 1.0 / np.maximum(K, 1e-6), 0.0)
    F = [amp * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) for _ in range(3)]
    kd = sum(k * f for k, f in zip(ks, F)) / np.maximum(K * K, 1e-12)
    F = [f - k * kd for k, f in zip(ks, F)]
    v = [np.fft.ifftn(f).real for f in F]
    rms = np.sqrt(sum((c * c).mean() for c in v))
    return [np.ascontiguousarray(c / rms, dtype=np.float32) for c in v]


def initial_conditions_bipolar(nx, ny, nz, args, rng):
    """'bipolar': one star wrapped in a dense torus + equatorially concentrated envelope.
    Isotropic ionization/wind can only escape along the poles -> hourglass lobes (the same
    physics shapes protostellar outflow cavities and bipolar planetary nebulae like NGC 6302)."""
    dx = 1.0 / nz
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    c = np.array([nx * dx * 0.5, ny * dx * 0.5, 0.5])
    tl = np.radians(args.tilt)
    n = np.array([np.sin(tl), 0.0, np.cos(tl)])
    RX, RY, RZ = X - c[0], Y - c[1], Z - c[2]
    r = np.sqrt(RX ** 2 + RY ** 2 + RZ ** 2) + 1e-6
    h = RX * n[0] + RY * n[1] + RZ * n[2]                   # height along the axis
    rc = np.sqrt(np.maximum(r * r - h * h, 0))             # cylindrical radius
    turb = grf3((nx, ny, nz), args.turb_slope, rng)
    sig = args.turb_sigma
    env = np.clip((r / args.env_r0) ** -1.5, 0.05, 6.0) * np.exp(sig * turb - 0.5 * sig * sig)
    eq = 1.0 + args.eq_contrast * np.exp(-(h / (0.35 * r + 0.01)) ** 2)
    torus = args.torus_rho * np.exp(-((rc - args.torus_R) ** 2 + h ** 2) / (2 * args.torus_a ** 2))
    rho = np.maximum(0.3 * env * eq, args.floor).astype(np.float32)
    return rho, torus.astype(np.float32), None, (tuple(float(v) for v in c), tuple(float(v) for v in n))


def pick_embedded_stars(rho, body, top, n, rng, dx, t_range):
    """New stars form in dense cores inside the cloud: pick high-density peaks a bit below
    the surface, spaced apart."""
    from scipy import ndimage
    sm = ndimage.gaussian_filter(rho * (body > 0.98), 2.0)
    peaks = (sm == ndimage.maximum_filter(sm, size=15)) & (sm > np.percentile(sm[body > 0.98], 97))
    idx = np.argwhere(peaks)
    nx, ny, nz = rho.shape
    m = 12
    keep = []
    for p in idx:
        if not (m <= p[0] < nx - m and m <= p[1] < ny - m):
            continue
        depth = float(top[p[0], p[1], 0]) - (p[2] + 0.5) * dx
        if 0.02 < depth < 0.15:
            keep.append(p)
    idx = np.array(keep).reshape(-1, 3)
    rng.shuffle(idx)
    out = []
    for p in idx:
        if len(out) >= n:
            break
        if all(np.linalg.norm(p - q) > 30 for q, _ in out):
            out.append((p, rng.uniform(*t_range)))
    return [((p + 0.5) * dx, t) for p, t in out]


# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cliff_a")
    ap.add_argument("--nx", type=int, default=288)
    ap.add_argument("--ny", type=int, default=160)
    ap.add_argument("--nz", type=int, default=192)
    ap.add_argument("--t_end", type=float, default=0.25)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--ci_over_cn", type=float, default=12.0)
    ap.add_argument("--floor", type=float, default=2e-3)
    ap.add_argument("--cloud_top", type=float, default=0.55)
    ap.add_argument("--edge_amp", type=float, default=0.05)
    ap.add_argument("--turb_sigma", type=float, default=1.4)
    ap.add_argument("--turb_slope", type=float, default=3.8)
    ap.add_argument("--globules", type=int, default=40)
    ap.add_argument("--q_main", type=float, default=2.5e-3)
    ap.add_argument("--q_embedded", type=float, default=6e-5)
    ap.add_argument("--n_embedded", type=int, default=4)
    ap.add_argument("--wind", type=float, default=0.0)
    ap.add_argument("--ion_every", type=int, default=2)
    ap.add_argument("--nu", type=int, default=768)
    ap.add_argument("--nphi", type=int, default=1536)
    ap.add_argument("--kappa_fuv", type=float, default=25.0, help="FUV dust opacity per unit rho")
    ap.add_argument("--fuv_grid", type=int, nargs=3, default=[192, 256, 512], help="nr nu nphi")
    ap.add_argument("--star_z", type=float, default=0.86)
    ap.add_argument("--star_y", type=float, default=0.62, help="cluster depth (fraction of Ly); low = in front of the wall")
    ap.add_argument("--scenario", default="cliff", choices=["cliff", "bubbles", "bipolar"])
    ap.add_argument("--sheet_w", type=float, default=0.28, help="bubbles: sheet half-thickness (fraction of Ly)")
    ap.add_argument("--n_bubbles", type=int, default=6)
    ap.add_argument("--min_sep", type=float, default=0.42, help="bubbles: min star separation (x-z)")
    ap.add_argument("--trigger", type=float, default=0.0,
                    help="triggered star formation: local compression factor rho/rho_initial in a shell (0=off)")
    ap.add_argument("--trigger_min", type=float, default=2.0, help="min absolute (smoothed) density to form a star")
    ap.add_argument("--shell_w", type=float, default=0.05, help="shell = neutral gas within this distance of a front")
    ap.add_argument("--trigger_every", type=int, default=40)
    ap.add_argument("--trigger_sep", type=float, default=0.12)
    ap.add_argument("--q_trig", type=float, default=2e-4)
    ap.add_argument("--max_stars", type=int, default=32)
    ap.add_argument("--max_births", type=int, default=1000)
    ap.add_argument("--pre_t", type=float, default=0.0, help="supersonic turbulence phase before stars (time)")
    ap.add_argument("--pre_amp", type=float, default=1.5, help="stirring acceleration")
    ap.add_argument("--pre_corr", type=float, default=0.06, help="forcing correlation time")
    ap.add_argument("--pre_kmax", type=float, default=6.0, help="largest stirred wavenumber (lower = bigger eddies)")
    ap.add_argument("--drive_main", type=float, default=0.0, help="keep stirring during the main phase (x pre_amp)")
    ap.add_argument("--jet", type=float, default=0.0, help="jet strength of the main star (bipolar)")
    ap.add_argument("--jet_proto", type=float, default=0.0, help="jet strength for newborn/embedded stars")
    ap.add_argument("--jet_angle", type=float, default=10.0, help="jet cone half-angle (deg)")
    ap.add_argument("--jet_r", type=float, default=0.04, help="jet launching radius")
    ap.add_argument("--jet_pulse", type=float, default=0.6, help="jet velocity variability -> knots")
    ap.add_argument("--jet_period", type=float, default=0.02)
    ap.add_argument("--tilt", type=float, default=25.0, help="bipolar axis tilt from +z (deg, in x-z)")
    ap.add_argument("--torus_rho", type=float, default=60.0)
    ap.add_argument("--torus_R", type=float, default=0.07)
    ap.add_argument("--torus_a", type=float, default=0.025)
    ap.add_argument("--eq_contrast", type=float, default=6.0)
    ap.add_argument("--env_r0", type=float, default=0.12, help="bipolar: envelope core radius")
    ap.add_argument("--birth_gap", type=float, default=0.0, help="min sim time between triggered births")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim3d", args.run)
    os.makedirs(os.path.join(root, "frames"), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    nx, ny, nz = args.nx, args.ny, args.nz
    dx = 1.0 / nz
    t0 = time.time()
    Lx, Ly = nx * dx, ny * dx
    # stars: (x, y, z, Q, t_on)
    jets = []      # per-star (ax, ay, az, strength), parallel to `stars`
    torus_np = None
    if args.scenario == "bubbles":
        rho_np, body, top = initial_conditions_bubbles(nx, ny, nz, args, rng)
        stars = place_bubble_stars(args, rng, Lx, Ly)
    elif args.scenario == "bipolar":
        rho_np, torus_np, top, (c, n) = initial_conditions_bipolar(nx, ny, nz, args, rng)
        body = None
        if args.pre_t <= 0:            # no stirring phase: the torus is there from the start
            rho_np = rho_np + torus_np
            torus_np = None
        stars = [(c[0], c[1], c[2], args.q_main, 0.0)]
        jets = [(n[0], n[1], n[2], args.jet)]
    else:
        rho_np, body, top = initial_conditions(nx, ny, nz, args, rng)
        zs = args.star_z
        ys = args.star_y
        stars = [(0.50 * Lx, ys * Ly, zs, args.q_main, 0.0),
                 (0.30 * Lx, (ys + 0.08) * Ly, zs + 0.06, 0.5 * args.q_main, 0.0),
                 (0.74 * Lx, (ys - 0.07) * Ly, zs + 0.03, 0.4 * args.q_main, 0.02)]
        for p, ton in pick_embedded_stars(rho_np, body, top, args.n_embedded, rng, dx,
                                          (0.15 * args.t_end, 0.6 * args.t_end)):
            stars.append((float(p[0]), float(p[1]), float(p[2]), args.q_embedded, float(ton)))
    def rand_axis():
        v = rng.standard_normal(3)
        return v / np.linalg.norm(v)
    for i in range(len(jets), len(stars)):     # main stars: no jets; embedded protostars: jets
        is_proto = stars[i][3] < 0.3 * args.q_main
        a_ = rand_axis()
        jets.append((float(a_[0]), float(a_[1]), float(a_[2]), args.jet_proto if is_proto else 0.0))
    sm0 = None
    if args.trigger > 0:
        from scipy import ndimage
        sm0 = ndimage.gaussian_filter(rho_np[::2, ::2, ::2], 1.0)
    print(f"setup {time.time()-t0:.1f}s  grid {nx}x{ny}x{nz}  stars:")
    for s_ in stars:
        print("   ", tuple(round(v, 4) for v in s_))

    dev = "cuda:0"
    zeros = lambda: wp.zeros((nx, ny, nz), dtype=float, device=dev)
    rho = wp.array(rho_np, dtype=float, device=dev)
    mx, my, mz = zeros(), zeros(), zeros()
    xi, xi_tmp, dfront = zeros(), zeros(), zeros()
    r1, mx1, my1, mz1 = zeros(), zeros(), zeros(), zeros()
    r2, mx2, my2, mz2 = zeros(), zeros(), zeros(), zeros()
    rfront = wp.zeros((args.nu, args.nphi), dtype=float, device=dev)
    smax = wp.zeros(1, dtype=float, device=dev)
    fnr, fnu, fnp = args.fuv_grid
    att = wp.zeros((fnr, fnu, fnp), dtype=float, device=dev)
    g0 = zeros()
    cn2 = (1.0 / args.ci_over_cn) ** 2
    ci2 = 1.0
    diag = math.sqrt(Lx ** 2 + Ly ** 2 + 1.0) + 0.6
    dr = 0.5 * dx
    nsteps = int(diag / dr) + 1
    cap = max(len(stars), args.max_stars)
    star_arr = wp.zeros(cap, dtype=wp.vec4, device=dev)
    jet_arr = wp.zeros(cap, dtype=wp.vec4, device=dev)
    jet_cos = math.cos(math.radians(args.jet_angle))
    shock = zeros()
    bmax = wp.vec3(Lx, Ly, 1.0)

    def ionize(t):
        xi.zero_()
        dfront.fill_(1e3)
        for (sx, sy, sz, Q, ton) in stars:
            if t < ton:
                continue
            sv = wp.vec3(sx, sy, sz)
            wp.launch(cast_rays, dim=(args.nu, args.nphi),
                      inputs=[rho, sv, Q, dr, nsteps, dx, diag, bmax, rfront], device=dev)
            wp.launch(apply_front, dim=(nx, ny, nz),
                      inputs=[rfront, sv, dx, 1.2 * dx, xi, dfront], device=dev)
        wp.launch(smooth_xi, dim=(nx, ny, nz), inputs=[xi, xi_tmp], device=dev)
        wp.copy(xi, xi_tmp)

    def set_star_weights(t):
        w = [(sx, sy, sz, (Q / args.q_main) if t >= ton else 0.0) for (sx, sy, sz, Q, ton) in stars]
        w += [(0.0, 0.0, 0.0, 0.0)] * (cap - len(w))
        star_arr.assign(np.array(w, dtype=np.float32))
        jw = list(jets) + [(0.0, 0.0, 1.0, 0.0)] * (cap - len(jets))
        jet_arr.assign(np.array(jw, dtype=np.float32))

    def fuv_field(t):
        """G0 = sum_star L exp(-tau_FUV) / r^2 ; L ~ Q (same massive stars dominate both)."""
        g0.zero_()
        binr = diag / fnr
        for (sx, sy, sz, Q, ton) in stars:
            if t < ton:
                continue
            sv = wp.vec3(sx, sy, sz)
            wp.launch(cast_fuv, dim=(fnu, fnp), inputs=[rho, xi, sv, args.kappa_fuv, dr, nsteps, dx, binr, bmax, att],
                      device=dev)
            wp.launch(apply_fuv, dim=(nx, ny, nz), inputs=[att, sv, Q / args.q_main, dx, binr, g0], device=dev)

    def save(frame, t):
        fuv_field(t)
        np.save(os.path.join(root, "frames", f"f{frame:04d}_g0.npy"), np.minimum(g0.numpy(), 6e4).astype(np.float16))
        np.save(os.path.join(root, "frames", f"f{frame:04d}_rho.npy"), rho.numpy().astype(np.float16))
        np.save(os.path.join(root, "frames", f"f{frame:04d}_xi.npy"),
                np.round(np.clip(xi.numpy(), 0, 1) * 255).astype(np.uint8))
        np.save(os.path.join(root, "frames", f"f{frame:04d}_df.npy"),
                np.clip(dfront.numpy(), -4, 4).astype(np.float16))
        wp.launch(shock_field, dim=(nx, ny, nz), inputs=[rho, mx, my, mz, dx, shock], device=dev)
        np.save(os.path.join(root, "frames", f"f{frame:04d}_sh.npy"),
                np.minimum(shock.numpy(), 6e4).astype(np.float16))
        times.append(t)

    meta = dict(vars(args), dx=dx, shape=[nx, ny, nz], stars=stars)
    times = []
    t, it, frame = 0.0, 0, 0
    n_born, last_birth = 0, -1e9
    dt_frame = args.t_end / max(args.frames - 1, 1)
    next_frame = 0.0
    # stirring fields (low-res, solenoidal), cross-faded every pre_corr
    sfac = 6
    shp = (nx // sfac + 2, ny // sfac + 2, nz // sfac + 2)
    dxs = sfac * dx
    mkF = lambda: [wp.array(c_, dtype=float, device=dev) for c_ in solenoidal(shp, rng, 1.0, args.pre_kmax)]
    drv = {"A": mkF(), "B": mkF(), "clock": 0.0}

    def stir(dt, amp):
        blend = min(drv["clock"] / args.pre_corr, 1.0)
        wp.launch(drive, dim=(nx, ny, nz),
                  inputs=[rho, mx, my, mz] + drv["A"] + drv["B"] + [blend, amp, dt, dx, dxs], device=dev)
        drv["clock"] += dt
        if drv["clock"] >= args.pre_corr:
            drv["A"], drv["B"], drv["clock"] = drv["B"], mkF(), 0.0

    if args.pre_t > 0:
        tp, ip = 0.0, 0
        tpre = time.time()
        while tp < args.pre_t:
            smax.zero_()
            wp.launch(max_speed, dim=(nx, ny, nz), inputs=[rho, mx, my, mz, xi, cn2, ci2, smax], device=dev)
            dt = args.cfl * dx / float(smax.numpy()[0])
            common = [star_arr, 0, 0.0, jet_arr, jet_cos, args.jet_r, 0.0, dt, dx]
            wp.launch(hydro_stage, dim=(nx, ny, nz),
                      inputs=[rho, mx, my, mz, xi, rho, mx, my, mz, r1, mx1, my1, mz1] + common
                      + [0.0, 1.0, args.floor, cn2, ci2, 6.0], device=dev)
            wp.launch(hydro_stage, dim=(nx, ny, nz),
                      inputs=[r1, mx1, my1, mz1, xi, rho, mx, my, mz, r2, mx2, my2, mz2] + common
                      + [0.5, 0.5, args.floor, cn2, ci2, 6.0], device=dev)
            rho, r2 = r2, rho
            mx, mx2 = mx2, mx
            my, my2 = my2, my
            mz, mz2 = mz2, mz
            stir(dt, args.pre_amp)
            tp += dt
            ip += 1
        r_ = rho.numpy()
        v_ = np.sqrt(mx.numpy() ** 2 + my.numpy() ** 2 + mz.numpy() ** 2) / r_
        mach = float(np.sqrt((r_ * v_ ** 2).sum() / r_.sum())) * args.ci_over_cn
        print(f"pre-turbulence: {ip} steps t={tp:.3f} mass-weighted Mach ~{mach:.1f} "
              f"rho max {r_.max():.1f} ({time.time()-tpre:.0f}s)", flush=True)
        if torus_np is not None:      # the disk/torus forms after the turbulent phase, at rest
            frac = torus_np / (torus_np + r_)
            rho = wp.array((r_ + torus_np).astype(np.float32), dtype=float, device=dev)
            for m_ in (mx, my, mz):
                m_.assign((m_.numpy() * (1.0 - frac)).astype(np.float32))
            r_ = rho.numpy()
        if sm0 is not None:
            from scipy import ndimage
            sm0 = ndimage.gaussian_filter(r_[::2, ::2, ::2], 1.0)   # compression is measured from here

    ionize(0.0)
    tl = time.time()
    while True:
        if t >= next_frame - 1e-9:
            save(frame, t)
            frame += 1
            next_frame += dt_frame
            if frame >= args.frames:
                break
        if (args.trigger > 0 and it > 0 and it % args.trigger_every == 0 and len(stars) < cap
                and n_born < args.max_births and t - last_birth >= args.birth_gap):
            born = find_triggered(rho.numpy(), xi.numpy(), dfront.numpy(), sm0, stars, args, dx, t)
            for b in born[: cap - len(stars)]:
                stars.append(b)
                a_ = rand_axis()
                jets.append((float(a_[0]), float(a_[1]), float(a_[2]), args.jet_proto))
                n_born += 1
                last_birth = t
                print(f"   * star born t={t:.4f} at {tuple(round(v, 3) for v in b[:3])}", flush=True)
        if it % args.ion_every == 0:
            ionize(t)
        set_star_weights(t)
        smax.zero_()
        wp.launch(max_speed, dim=(nx, ny, nz), inputs=[rho, mx, my, mz, xi, cn2, ci2, smax], device=dev)
        dt = args.cfl * dx / float(smax.numpy()[0])
        dt = min(dt, next_frame - t + 1e-7)
        pulse = max(0.0, 1.0 + args.jet_pulse * math.sin(2 * math.pi * t / args.jet_period))
        common = [star_arr, len(stars), args.wind, jet_arr, jet_cos, args.jet_r, pulse, dt, dx]
        wp.launch(hydro_stage, dim=(nx, ny, nz),
                  inputs=[rho, mx, my, mz, xi, rho, mx, my, mz, r1, mx1, my1, mz1] + common
                  + [0.0, 1.0, args.floor, cn2, ci2, 6.0], device=dev)
        wp.launch(hydro_stage, dim=(nx, ny, nz),
                  inputs=[r1, mx1, my1, mz1, xi, rho, mx, my, mz, r2, mx2, my2, mz2] + common
                  + [0.5, 0.5, args.floor, cn2, ci2, 6.0], device=dev)
        rho, r2 = r2, rho
        mx, mx2 = mx2, mx
        my, my2 = my2, my
        mz, mz2 = mz2, mz
        if args.drive_main > 0:
            stir(dt, args.pre_amp * args.drive_main)
        t += dt
        it += 1
        if it % 50 == 0:
            wp.synchronize()
            print(f"it {it:5d} t={t:.4f} dt={dt:.2e} frame {frame}/{args.frames} "
                  f"ion={float(xi.numpy().mean()):.3f} {time.time()-tl:.0f}s", flush=True)
    meta["times"] = times
    meta["stars"] = stars
    meta["jets"] = jets
    meta["steps"] = it
    meta["wall_s"] = time.time() - t0
    json.dump(meta, open(os.path.join(root, "meta.json"), "w"), indent=1)
    print(f"done: {it} steps, {frame} frames, {time.time()-t0:.0f}s total -> {root}")


if __name__ == "__main__":
    main()
