"""NebulaViewer GPU renderer: emission + absorption ray-marcher for the nebula3d fields (NVIDIA Warp).

Reproduces the Blender "Nebula Volume" shader (no scattering):
  dust = rho (1 - xi) + 0.08 rho xi ;  ha = (rho xi)^2
  oiii = ha * smoothstep(-0.03, -0.12, dfront) ;  sii = ha * smoothstep(-0.05, -0.005, dfront)
  g0eff = g0 / (1 + g0/Gs) + ambient
  emission = Escale (Ha c_ha ha + OIII c_o3 oiii + SII c_s2 sii + (PAH c_pah + Glow c_glow) dust g0eff + Shock c_sh shock)
  absorption = dust * DustOpacity * reddening
Blender coefficients are per BU (10 BU per domain unit) -> x10 here, so radiance matches Cycles.

Pipeline per frame (all on the GPU, nothing touches the disk):
  pack_volume  : sim fields -> one RGBA16F voxel grid (emission rgb, dust) incl. edge fade
                 (pack_volume_lm: float16 dfront / g0, shock proxy computed here; it packs into the sim's scratch
                 block and the texture belongs to the sim, shared by every Renderer: 8 B/cell)
  Texture3D    : hardware trilinear filtering of that grid
  raymarch     : orbit camera, slab test, exact per-segment emission/absorption, early out
  post         : background stars x transmittance, ionizing stars (core + 6 JWST spikes),
                 exposure, AgX-like tone map, dither -> RGBA8 (also straight into a GL buffer)
"""
import math
import time
from typing import Any

import numpy as np
import warp as wp

# ----------------------------------------------------------------------------------- look presets
PALETTES = {
    # JWST NIRCam-like: hot ionized cavity blue, PDR skins orange/gold, dust brown
    "jwst": dict(ha=(0.22, 0.45, 1.0), o3=(0.30, 0.75, 1.0), s2=(1.0, 0.55, 0.25), pah=(1.0, 0.50, 0.15),
                 glow=(0.75, 0.42, 0.22), shock=(1.0, 0.35, 0.45), w={}),
    # Hubble SHO: SII red, Ha green, OIII blue
    "sho": dict(ha=(0.15, 1.0, 0.25), o3=(0.10, 0.45, 1.0), s2=(1.0, 0.25, 0.10), pah=(0.9, 0.6, 0.2),
                glow=(0.5, 0.4, 0.3), shock=(1.0, 0.35, 0.45), w={"sii": 3.0, "pah": 0.04}),
    # 'natural' colour: Ha deep red-pink, OIII teal
    "natural": dict(ha=(1.0, 0.18, 0.28), o3=(0.15, 0.85, 0.80), s2=(1.0, 0.10, 0.10), pah=(0.6, 0.35, 0.2),
                    glow=(0.35, 0.5, 0.95), shock=(1.0, 0.35, 0.45), w={"pah": 0.01}),
}
PALETTE_NAMES = list(PALETTES)

# Blender "Nebula Volume" values per BU (Felipe's lookdev 2026-10-01); converted x10 at pack time
LOOK_BASE = dict(dust_opacity=4.0, emission=2.0, ha=1.5, oiii=1.5, sii=1.2, pah=0.07, glow=0.002, shock=0.002,
                 gs=12.0, ambient=0.1, reddening=(0.55, 0.85, 1.35), edge_fade=0.08, detail=0.0, detail_scale=2.5,
                 exposure=-0.8, star_gain=0.5, star_size=1.0, bg_stars=1.0, spikes=True, tonemap="agx", palette="jwst",
                 # AgX 'Punchy' fitted to Cycles renders of the same data (bip_test3_f032: RMS 7.9/255)
                 agx_pow=1.5, agx_sat=1.1,
                 filter="smooth",    # 'linear' | 'smooth' ([1 4 1] prefilter + trilinear) | 'cubic' (tricubic B-spline)
                 clusters=0)         # decorative (non-ionizing) young stars around each main star, as in Blender
LOOK_PRESETS = {
    # G0 Saturation 1e4 (= off) as in build_nebula_scene.py for bipolar; bubbles use 12
    "bipolar": dict(dust_opacity=1.5, emission=6.0, ambient=0.3, gs=1e4, star_gain=0.3, detail=0.0),
    "bubbles": dict(shock=0.0, ambient=0.1, star_gain=0.5, clusters=22),
    "cliff": dict(dust_opacity=0.25, emission=1.0, ha=0.6, glow=0.0015, shock=0.0, gs=1e4, ambient=0.0,
                  exposure=0.0, star_gain=1.0),
}


def make_look(scenario, overrides=None):
    lk = dict(LOOK_BASE)
    lk.update(LOOK_PRESETS.get(scenario, {}))
    if overrides:
        lk.update(overrides)
    return lk


# ----------------------------------------------------------------------------------- structs
@wp.struct
class Camera:
    pos: wp.vec3
    fwd: wp.vec3
    right: wp.vec3
    up: wp.vec3
    tan_x: float
    tan_y: float


@wp.struct
class PackParams:
    c_ha: wp.vec3
    c_o3: wp.vec3
    c_s2: wp.vec3
    c_dg: wp.vec3       # (PAH c_pah + Glow c_glow) -> multiplies dust * g0eff
    c_sh: wp.vec3
    gs: float
    ambient: float
    edge: float
    inv_L: wp.vec3
    dx: float


@wp.struct
class MarchParams:
    bmax: wp.vec3
    inv_dx: float
    step: float
    ext: wp.vec3        # dust opacity * reddening (per domain unit)
    detail: float
    detail_freq: float
    jitter_seed: float
    cubic: int          # 1 = tricubic B-spline sampling (8 hardware-trilinear fetches)


@wp.struct
class PostParams:
    exposure: float     # linear multiplier
    bg_gain: float
    bg_density: float
    pix_ang: float      # radians per pixel (vertical)
    star_px: float      # star sprite scale in pixels
    spikes: float
    space: wp.vec3      # faint sky colour
    tonemap: int        # 0 = AgX punchy, 1 = AgX base, 2 = filmic (nebula_preview)
    agx_pow: float      # punchy look: CDL power
    agx_sat: float      # punchy look: saturation
    flip: int
    dither: float


# ----------------------------------------------------------------------------------- kernels
@wp.func
def sstep(e0: float, e1: float, x: float):
    t = wp.clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def fade_axis(u: float, e: float):
    if e <= 0.0:
        return 1.0
    return sstep(0.0, e, u) * sstep(0.0, e, 1.0 - u)


@wp.kernel
def pack_volume(rho: wp.array3d(dtype=float), xi: wp.array3d(dtype=float), dfront: wp.array3d(dtype=float),
                g0: wp.array3d(dtype=float), shock: wp.array3d(dtype=float), pp: PackParams,
                out: wp.array3d(dtype=wp.vec4h)):
    i, j, k = wp.tid()
    r = rho[i, j, k]
    x = wp.clamp(xi[i, j, k], 0.0, 1.0)
    df = dfront[i, j, k]
    ni = r * x
    ha = ni * ni
    oiii = ha * sstep(-0.03, -0.12, df)
    sii = ha * sstep(-0.05, -0.005, df)
    dust = r * (1.0 - x) + 0.08 * ni
    g = g0[i, j, k]
    geff = g / (1.0 + g / pp.gs) + pp.ambient
    em = ha * pp.c_ha + oiii * pp.c_o3 + sii * pp.c_s2 + (dust * geff) * pp.c_dg + shock[i, j, k] * pp.c_sh
    # soft fade near the box faces (hides the domain edges), normalized box coordinates
    p = wp.cw_mul((wp.vec3(float(i), float(j), float(k)) + wp.vec3(0.5)) * pp.dx, pp.inv_L)
    f = fade_axis(p[0], pp.edge) * fade_axis(p[1], pp.edge) * fade_axis(p[2], pp.edge)
    em = em * f
    dust = dust * f
    m = 60000.0
    out[i, j, k] = wp.vec4h(wp.float16(wp.min(em[0], m)), wp.float16(wp.min(em[1], m)),
                            wp.float16(wp.min(em[2], m)), wp.float16(wp.min(dust, m)))


@wp.func
def _ldc(a: wp.array3d(dtype=float), i: int, j: int, k: int):
    """a[i, j, k] with the indices clamped to the grid (nebula3d.ldc)"""
    return a[wp.clamp(i, 0, a.shape[0] - 1), wp.clamp(j, 0, a.shape[1] - 1), wp.clamp(k, 0, a.shape[2] - 1)]


@wp.func
def shock_at(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
             mz: wp.array3d(dtype=float), i: int, j: int, k: int, dx: float):
    """nebula3d.shock_field at one cell: max(0, -div v) rho"""
    du = _ldc(mx, i + 1, j, k) / _ldc(rho, i + 1, j, k) - _ldc(mx, i - 1, j, k) / _ldc(rho, i - 1, j, k)
    dv = _ldc(my, i, j + 1, k) / _ldc(rho, i, j + 1, k) - _ldc(my, i, j - 1, k) / _ldc(rho, i, j - 1, k)
    dw = _ldc(mz, i, j, k + 1) / _ldc(rho, i, j, k + 1) - _ldc(mz, i, j, k - 1) / _ldc(rho, i, j, k - 1)
    div = (du + dv + dw) / (2.0 * dx)
    return wp.max(-div, 0.0) * rho[i, j, k]


@wp.kernel
def pack_volume_lm(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
                   mz: wp.array3d(dtype=float), xi: wp.array3d(dtype=float),
                   dfront: wp.array3d(dtype=Any), g0: wp.array3d(dtype=Any), pp: PackParams,
                   shock_on: int, out: wp.array3d(dtype=wp.vec4h)):
    """pack_volume with float16 (or float32) dfront / g0 and the shock proxy computed here"""
    i, j, k = wp.tid()
    r = rho[i, j, k]
    x = wp.clamp(xi[i, j, k], 0.0, 1.0)
    df = float(dfront[i, j, k])
    ni = r * x
    ha = ni * ni
    oiii = ha * sstep(-0.03, -0.12, df)
    sii = ha * sstep(-0.05, -0.005, df)
    dust = r * (1.0 - x) + 0.08 * ni
    g = float(g0[i, j, k])
    geff = g / (1.0 + g / pp.gs) + pp.ambient
    sh = float(0.0)
    if shock_on != 0:
        sh = shock_at(rho, mx, my, mz, i, j, k, pp.dx)
    em = ha * pp.c_ha + oiii * pp.c_o3 + sii * pp.c_s2 + (dust * geff) * pp.c_dg + sh * pp.c_sh
    # soft fade near the box faces (hides the domain edges), normalized box coordinates
    p = wp.cw_mul((wp.vec3(float(i), float(j), float(k)) + wp.vec3(0.5)) * pp.dx, pp.inv_L)
    f = fade_axis(p[0], pp.edge) * fade_axis(p[1], pp.edge) * fade_axis(p[2], pp.edge)
    em = em * f
    dust = dust * f
    m = 60000.0
    out[i, j, k] = wp.vec4h(wp.float16(wp.min(em[0], m)), wp.float16(wp.min(em[1], m)),
                            wp.float16(wp.min(em[2], m)), wp.float16(wp.min(dust, m)))


@wp.func
def box_hit(o: wp.vec3, d: wp.vec3, bmax: wp.vec3):
    """slab test against [0, bmax]; returns (t_enter >= 0, t_exit); miss if t_exit <= t_enter"""
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


@wp.func
def cam_ray(cam: Camera, x: float, y: float, W: int, H: int):
    sx = (2.0 * x / float(W) - 1.0) * cam.tan_x
    sy = (1.0 - 2.0 * y / float(H)) * cam.tan_y
    return wp.normalize(cam.fwd + sx * cam.right + sy * cam.up)


@wp.func
def ign(x: float, y: float):
    """interleaved gradient noise in [0,1) (cheap blue-ish per-pixel jitter)"""
    v = 52.9829189 * (0.06711056 * x + 0.00583715 * y - wp.floor(0.06711056 * x + 0.00583715 * y))
    return v - wp.floor(v)


@wp.func
def seg_weight(tau: float, sig: float, h: float):
    if tau > 1e-4:
        return (1.0 - wp.exp(-tau)) / sig
    return h * (1.0 - 0.5 * tau)


@wp.func
def bspline_axis(x: float):
    """fast tricubic B-spline (Sigg & Hadwiger 2005): two linear fetch positions + weight, per axis.
    x in texel units (texel centres at i + 0.5)"""
    c = x - 0.5
    i = wp.floor(c)
    f = c - i
    f2 = f * f
    f3 = f2 * f
    w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0
    w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0
    w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0
    w3 = f3 / 6.0
    g0 = w0 + w1
    return wp.vec3(i - 0.5 + w1 / g0, i + 1.5 + w3 / (w2 + w3), g0)


@wp.func
def sample_vol(vol: wp.Texture3D, g: wp.vec3, cubic: int):
    """packed volume at grid position g = (x, y, z) in voxel units (texture is [x][y][z] -> w, v, u)"""
    if cubic == 0:
        return wp.texture_sample(vol, wp.vec3(g[2], g[1], g[0]), dtype=wp.vec4f)
    ax = bspline_axis(g[0])
    ay = bspline_axis(g[1])
    az = bspline_axis(g[2])
    acc = wp.vec4f(0.0)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                w = wp.where(a == 0, ax[2], 1.0 - ax[2]) * wp.where(b == 0, ay[2], 1.0 - ay[2])                     * wp.where(c == 0, az[2], 1.0 - az[2])
                q = wp.vec3(az[c], ay[b], ax[a])
                acc = acc + w * wp.texture_sample(vol, q, dtype=wp.vec4f)
    return acc


@wp.kernel
def smooth_axis(src: wp.array3d(dtype=wp.vec4h), dst: wp.array3d(dtype=wp.vec4h), axis: int):
    """[1 4 1]/6 along one axis: B-spline-like prefilter (Cycles renders these volumes with cubic
    interpolation; this removes the voxel terracing of sub-voxel-thin emitting shells)"""
    i, j, k = wp.tid()
    oi = wp.where(axis == 0, 1, 0)
    oj = wp.where(axis == 1, 1, 0)
    ok = wp.where(axis == 2, 1, 0)
    n0 = src.shape[0] - 1
    n1 = src.shape[1] - 1
    n2 = src.shape[2] - 1
    a = src[wp.max(i - oi, 0), wp.max(j - oj, 0), wp.max(k - ok, 0)]
    b = src[i, j, k]
    c = src[wp.min(i + oi, n0), wp.min(j + oj, n1), wp.min(k + ok, n2)]
    r = wp.vec4h()
    for ch in range(4):
        r[ch] = wp.float16((float(a[ch]) + 4.0 * float(b[ch]) + float(c[ch])) / 6.0)
    dst[i, j, k] = r


@wp.kernel
def raymarch(vol: wp.Texture3D, det: wp.Texture3D, cam: Camera, mp: MarchParams,
             L_out: wp.array2d(dtype=wp.vec3), T_out: wp.array2d(dtype=wp.vec3)):
    y, x = wp.tid()
    H = L_out.shape[0]
    W = L_out.shape[1]
    d = cam_ray(cam, float(x) + 0.5, float(y) + 0.5, W, H)
    tr = box_hit(cam.pos, d, mp.bmax)
    L = wp.vec3(0.0)
    T = wp.vec3(1.0)
    if tr[1] > tr[0]:
        h = mp.step
        j = ign(float(x), float(y)) + mp.jitter_seed
        j = j - wp.floor(j)
        t = tr[0] + h * j
        while t < tr[1]:
            p = cam.pos + d * t
            s = sample_vol(vol, p * mp.inv_dx, mp.cubic)
            if s[3] > 1e-6 or s[0] + s[1] + s[2] > 1e-9:
                em = wp.vec3(s[0], s[1], s[2])
                dust = s[3]
                if mp.detail > 0.0:
                    q = p * mp.detail_freq
                    n = wp.texture_sample(det, wp.vec3(q[2], q[1], q[0]), dtype=float)
                    md = wp.max(1.0 + mp.detail * n, 0.0)
                    em = em * md
                    dust = dust * md
                sig = mp.ext * dust
                tau = sig * h
                w = wp.vec3(seg_weight(tau[0], sig[0], h), seg_weight(tau[1], sig[1], h),
                            seg_weight(tau[2], sig[2], h))
                L = L + wp.cw_mul(T, wp.cw_mul(em, w))
                T = wp.cw_mul(T, wp.vec3(wp.exp(-tau[0]), wp.exp(-tau[1]), wp.exp(-tau[2])))
                if wp.max(T[0], wp.max(T[1], T[2])) < 1e-3:
                    T = wp.vec3(0.0)
                    break
            t = t + h
    L_out[y, x] = L
    T_out[y, x] = T


@wp.kernel
def star_transmittance(vol: wp.Texture3D, cam_pos: wp.vec3, stars: wp.array(dtype=wp.vec4), mp: MarchParams,
                       out: wp.array(dtype=wp.vec3)):
    """transmittance from the camera to each star through the dust (reddened)"""
    s = wp.tid()
    sp = wp.vec3(stars[s][0], stars[s][1], stars[s][2])
    rel = sp - cam_pos
    dist = wp.length(rel)
    d = rel / wp.max(dist, 1e-6)
    tr = box_hit(cam_pos, d, mp.bmax)
    tau = wp.vec3(0.0)
    t1 = wp.min(tr[1], dist)
    if t1 > tr[0]:
        h = 0.5 / mp.inv_dx
        t = tr[0] + 0.5 * h
        while t < t1:
            p = cam_pos + d * t
            v = sample_vol(vol, p * mp.inv_dx, 0)
            tau = tau + mp.ext * (v[3] * h)
            t = t + h
    out[s] = wp.vec3(wp.exp(-tau[0]), wp.exp(-tau[1]), wp.exp(-tau[2]))


@wp.func
def bg_stars(d: wp.vec3, pix_ang: float, density: float):
    """procedural star field fixed on the sky (cube map cells, one candidate star per cell)"""
    ax = wp.abs(d[0])
    ay = wp.abs(d[1])
    az = wp.abs(d[2])
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    if ax >= ay and ax >= az:
        face = wp.where(d[0] > 0.0, 0, 1)
        u = d[1] / ax
        v = d[2] / ax
    elif ay >= az:
        face = wp.where(d[1] > 0.0, 2, 3)
        u = d[0] / ay
        v = d[2] / ay
    else:
        face = wp.where(d[2] > 0.0, 4, 5)
        u = d[0] / az
        v = d[1] / az
    G = 384
    fu = (u * 0.5 + 0.5) * float(G)
    fv = (v * 0.5 + 0.5) * float(G)
    cu = int(wp.floor(fu))
    cv = int(wp.floor(fv))
    col = wp.vec3(0.0)
    for du in range(-1, 2):
        for dv in range(-1, 2):
            a = cu + du
            b = cv + dv
            if a >= 0 and a < G and b >= 0 and b < G:
                st = wp.rand_init(1337, face * G * G + a * G + b)
                if wp.randf(st) < density:
                    su = (float(a) + wp.randf(st)) / float(G) * 2.0 - 1.0
                    sv = (float(b) + wp.randf(st)) / float(G) * 2.0 - 1.0
                    sd = wp.vec3(0.0)
                    if face < 2:
                        sd = wp.vec3(wp.where(face == 0, 1.0, -1.0), su, sv)
                    elif face < 4:
                        sd = wp.vec3(su, wp.where(face == 2, 1.0, -1.0), sv)
                    else:
                        sd = wp.vec3(su, sv, wp.where(face == 4, 1.0, -1.0))
                    sd = wp.normalize(sd)
                    r = wp.length(d - sd) / pix_ang
                    if r < 4.0:
                        uu = wp.randf(st)
                        br = 0.35 * wp.min(wp.pow(1.0 - uu * 0.9999, -1.0 / 1.6), 60.0)   # pareto tail (Blender field stars)
                        tsel = wp.randf(st)
                        tint = wp.vec3(1.0, 1.0, 1.0)
                        if tsel < 0.45:
                            tint = wp.vec3(1.0, 0.82, 0.62)
                        elif tsel < 0.8:
                            tint = wp.vec3(0.78, 0.88, 1.0)
                        col = col + tint * (br * wp.exp(-r * r / (2.0 * 0.55 * 0.55)))
    return col


@wp.func
def agx_contrast(x: float):
    x2 = x * x
    x4 = x2 * x2
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232


@wp.func
def agx(c: wp.vec3, look: int, pw: float, sat: float):
    """AgX (Blender 4+ view transform) analytic approximation; look 0 = Punchy, 1 = base"""
    # inset matrix (rows)
    r = 0.842479062253094 * c[0] + 0.0784335999999992 * c[1] + 0.0792237451477643 * c[2]
    g = 0.0423282422610123 * c[0] + 0.878468636469772 * c[1] + 0.0791661274605434 * c[2]
    b = 0.0423756549057051 * c[0] + 0.0784336 * c[1] + 0.879142973793104 * c[2]
    mn = -12.47393
    mx = 4.026069
    v = wp.vec3(0.0)
    for ch in range(3):
        val = r
        if ch == 1:
            val = g
        if ch == 2:
            val = b
        e = wp.clamp(wp.log2(wp.max(val, 1e-10)), mn, mx)
        v[ch] = agx_contrast((e - mn) / (mx - mn))
    if look == 0:
        luma = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
        for ch in range(3):
            v[ch] = wp.pow(wp.max(v[ch], 0.0), pw)
        luma2 = luma
        for ch in range(3):
            v[ch] = luma2 + sat * (v[ch] - luma2)
    # outset matrix
    o0 = 1.19687900512017 * v[0] - 0.0980208811401368 * v[1] - 0.0990297440797205 * v[2]
    o1 = -0.0528968517574562 * v[0] + 1.15190312990417 * v[1] - 0.0989611768448433 * v[2]
    o2 = -0.0529716355144438 * v[0] - 0.0980434501171241 * v[1] + 1.15107367264116 * v[2]
    return wp.vec3(wp.clamp(o0, 0.0, 1.0), wp.clamp(o1, 0.0, 1.0), wp.clamp(o2, 0.0, 1.0))


@wp.func
def filmic(c: wp.vec3):
    """nebula_preview.py curve: soft exponential shoulder + mild gamma (display referred)"""
    o = wp.vec3(0.0)
    for ch in range(3):
        o[ch] = wp.pow(wp.clamp(1.0 - wp.exp(-c[ch] * 1.6), 0.0, 1.0), 1.0 / 1.25)
    return o


@wp.func
def srgb_oetf(x: float):
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * wp.pow(x, 1.0 / 2.4) - 0.055


@wp.kernel
def post(L_in: wp.array2d(dtype=wp.vec3), T_in: wp.array2d(dtype=wp.vec3), cam: Camera, pp: PostParams,
         sprites: wp.array(dtype=wp.vec4), strans: wp.array(dtype=wp.vec3), nspr: int,
         out: wp.array(dtype=wp.uint32)):
    y, x = wp.tid()
    H = L_in.shape[0]
    W = L_in.shape[1]
    c = L_in[y, x]
    T = T_in[y, x]
    if pp.bg_gain > 0.0:
        d = cam_ray(cam, float(x) + 0.5, float(y) + 0.5, W, H)
        c = c + wp.cw_mul(T, bg_stars(d, pp.pix_ang, pp.bg_density) * pp.bg_gain + pp.space)
    # ionizing stars: gaussian core + halo + 6 diffraction spikes (JWST hexagonal pupil)
    fx = float(x) + 0.5
    fy = float(y) + 0.5
    s_px = pp.star_px
    for s in range(nspr):
        sp = sprites[s]
        ddx = fx - sp[0]
        ddy = fy - sp[1]
        r2 = ddx * ddx + ddy * ddy
        reach = 40.0 * s_px * wp.sqrt(wp.max(sp[2], 0.01))
        if r2 < reach * reach:
            sc = 0.8 * s_px * sp[3]
            core = wp.exp(-r2 / (2.0 * sc * sc)) + 0.12 * wp.exp(-r2 / (2.0 * 9.0 * sc * sc))
            spk = float(0.0)
            if pp.spikes > 0.0:
                for a in range(3):
                    ang = (90.0 + 60.0 * float(a)) * 0.017453292
                    dpar = ddx * wp.cos(ang) + ddy * wp.sin(ang)
                    dperp = -ddx * wp.sin(ang) + ddy * wp.cos(ang)
                    spk = spk + wp.exp(-dperp * dperp / (2.0 * 0.45 * 0.45)) * wp.exp(-wp.abs(dpar) / (5.0 * s_px))
                # faint short horizontal spike pair (JWST strut)
                spk = spk + 0.4 * wp.exp(-ddy * ddy / (2.0 * 0.4 * 0.4)) * wp.exp(-wp.abs(ddx) / (2.5 * s_px))
            c = c + strans[s] * (sp[2] * (core + pp.spikes * 0.08 * spk))
    c = c * pp.exposure
    o = wp.vec3(0.0)
    if pp.tonemap == 2:
        o = filmic(c)
    else:
        o = agx(c, pp.tonemap, pp.agx_pow, pp.agx_sat)
    # dither (removes banding in the smooth dark gradients)
    n = ign(float(x) + 17.0, float(y) + 31.0) - 0.5
    rr = wp.uint32(wp.clamp(o[0] * 255.0 + n * pp.dither + 0.5, 0.0, 255.0))
    gg = wp.uint32(wp.clamp(o[1] * 255.0 + n * pp.dither + 0.5, 0.0, 255.0))
    bb = wp.uint32(wp.clamp(o[2] * 255.0 + n * pp.dither + 0.5, 0.0, 255.0))
    row = y
    if pp.flip != 0:
        row = H - 1 - y
    out[row * W + x] = rr | (gg << wp.uint32(8)) | (bb << wp.uint32(16)) | (wp.uint32(255) << wp.uint32(24))


# ----------------------------------------------------------------------------------- host side
# default shots = the cameras of build_nebula_scene.py (domain units; Blender BU / 10)
BLENDER_VIEWS = {
    # whole hourglass in view (bipolar_long: 13 -> 16 BU from the centre)
    # (aspect = W/H of the Blender render: the vertical extent is kept for other window shapes)
    "bipolar": dict(target=None, dist=1.45, pitch=0.0, aspect=1.0, box_scale=True),
    # face-on to the sheet, camera at y = -1.8 -> 2.175 from the box centre (y = 0.375)
    "bubbles": dict(target=None, dist=2.2, pitch=0.0, aspect=16 / 9),
    # cliff: camera at y ~ -1.4, z = 0.57, tilted 2 deg down
    "cliff": dict(target=None, dist=1.85, pitch=2.0, tz=0.57, aspect=16 / 9),
}


class OrbitCamera:
    """yaw/pitch/distance around a target. yaw=0, pitch=0 looks along +y with z up (Blender front view).
    fov = field of view of the LARGER image side, like a Blender camera with sensor fit 'Auto'
    (default 39.6 deg = 50 mm lens on a 36 mm sensor, the camera of build_nebula_scene.py)."""

    def __init__(self, target=(0.5, 0.5, 0.5), dist=1.6, yaw=0.0, pitch=0.0, fov=39.6):
        self.target = np.array(target, dtype=np.float64)
        self.dist, self.yaw, self.pitch, self.fov = dist, yaw, pitch, fov

    def basis(self):
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        v = np.array([sy * cp, -cy * cp, sp])
        pos = self.target + self.dist * v
        fwd = -v
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, fwd)
        return pos, fwd, right, up

    def tans(self, W, H):
        t = math.tan(math.radians(self.fov) * 0.5)
        return (t, t * H / W) if W >= H else (t * W / H, t)

    def to_struct(self, W, H, tans=None):
        pos, fwd, right, up = self.basis()
        c = Camera()
        c.pos, c.fwd, c.right, c.up = (wp.vec3(*map(float, a)) for a in (pos, fwd, right, up))
        c.tan_x, c.tan_y = tans if tans is not None else self.tans(W, H)
        return c

    def project(self, pts, W, H, tans=None):
        """world points (n,3) -> pixel coords (n,2) and a visibility mask"""
        pos, fwd, right, up = self.basis()
        tx, ty = tans if tans is not None else self.tans(W, H)
        rel = np.asarray(pts, dtype=np.float64).reshape(-1, 3) - pos
        z = rel @ fwd
        ok = z > 1e-4
        zs = np.where(ok, z, 1.0)
        xn = (rel @ right) / (zs * tx)
        yn = (rel @ up) / (zs * ty)
        px = (xn + 1.0) * 0.5 * W
        py = (1.0 - yn) * 0.5 * H
        return np.stack([px, py], -1), ok

    def blender_view(self, scenario, L, aspect=None):
        """the default shot of build_nebula_scene.py for this scenario (same vertical extent as the
        Blender render whatever the window aspect)"""
        v = BLENDER_VIEWS.get(scenario, BLENDER_VIEWS["bipolar"])
        L = np.asarray(L, dtype=np.float64)
        tgt = np.array(v["target"] if v["target"] is not None else L * 0.5, dtype=np.float64)
        if "tz" in v:
            tgt[2] = v["tz"]
        dist = v["dist"]
        if v.get("box_scale"):                   # bipolar: frame the cloud (~0.7 radius), not the whole box
            dist = dist * (1.0 + 0.6 * (L[2] - 1.0))
        if "tz" in v:                            # cliff: tz is in content units (the box may be taller)
            tgt[2] = v["tz"] + 0.5 * (L[2] - 1.0)
        if aspect is not None:
            ty_blender = self.tans(v["aspect"], 1.0)[1]
            ty_window = self.tans(aspect, 1.0)[1]
            dist = dist * ty_blender / ty_window
        self.target, self.dist, self.pitch, self.yaw = tgt, dist, v["pitch"], 0.0

    def frame_box(self, L, aspect=16 / 9):
        """centre the box and pick a distance that fits all of it in view"""
        L = np.asarray(L, dtype=np.float64)
        self.target = L * 0.5
        tx, ty = self.tans(aspect, 1.0)
        half_w, half_h, depth = 0.5 * L[0], 0.5 * L[2], 0.5 * L[1]
        self.dist = max(half_h / ty, half_w / tx) * 1.05 + depth

    def blender_location(self):
        """camera location in Blender units (10 BU per domain unit) + rotation (deg) for a 50 mm lens"""
        pos, fwd, right, up = self.basis()
        return (pos * 10.0).tolist(), [90.0 - self.pitch, 0.0, self.yaw]


def detail_noise(n=96, slope=4.5, seed=11):
    """tileable fBm-like Gaussian noise (periodic GRF), unit std. slope 4.5 ~ Blender noise, roughness 0.6"""
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(n)[:, None, None]
    ky = np.fft.fftfreq(n)[None, :, None]
    kz = np.fft.rfftfreq(n)[None, None, :]
    k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    k[0, 0, 0] = 1.0
    amp = k ** (-slope / 2.0) * (k > 1.5 / n)
    ph = rng.standard_normal(amp.shape) + 1j * rng.standard_normal(amp.shape)
    f = np.fft.irfftn(amp * ph, s=(n, n, n)).astype(np.float32)
    return ((f - f.mean()) / f.std()).astype(np.float32)


class Renderer:
    """Owns the packed volume texture and the image buffers; render(sim, cam, look) -> RGBA8 on GPU."""

    def __init__(self, width, height, device="cuda:0"):
        self.device = device
        self.vol = None
        self.vol_shape = None
        self.packed_key = None
        self.prof = {}
        self.accum = 0
        self.resize(width, height)
        dn = detail_noise()
        self.det = wp.Texture3D(dn, filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.WRAP,
                                normalized_coords=True, device=device)
        self.max_sprites = 512
        self.sprites = wp.zeros(self.max_sprites, dtype=wp.vec4, device=device)
        self.star_pos = wp.zeros(self.max_sprites, dtype=wp.vec4, device=device)
        self.strans = wp.zeros(self.max_sprites, dtype=wp.vec3, device=device)
        self._cluster_cache = {}

    def resize(self, width, height):
        self.W, self.H = int(width), int(height)
        self.L = wp.zeros((self.H, self.W), dtype=wp.vec3, device=self.device)
        self.T = wp.zeros((self.H, self.W), dtype=wp.vec3, device=self.device)
        self.rgba = wp.zeros(self.H * self.W, dtype=wp.uint32, device=self.device)

    # ------------------------------------------------------------------ packing
    def _ensure_volume(self, shape):
        if self.vol_shape == shape:
            return
        nx, ny, nz = shape
        self.packed = wp.zeros(shape, dtype=wp.vec4h, device=self.device)
        self.packed2 = wp.zeros(shape, dtype=wp.vec4h, device=self.device)
        self.vol = wp.Texture3D(width=nz, height=ny, depth=nx, num_channels=4, dtype=wp.float16,
                                filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.CLAMP,
                                normalized_coords=False, device=self.device)
        self.vol_shape = shape
        self.packed_key = None

    @staticmethod
    def pack_params(look, sim):
        pal = PALETTES[look["palette"]]
        w = dict(look)
        for k_, v_ in pal["w"].items():        # palette-specific weight overrides (Blender presets)
            w[k_] = v_ if look.get("_pal_override", True) else w[k_]
        s = 10.0 * w["emission"]               # per BU -> per domain unit
        v3 = lambda c, f: wp.vec3(*(float(f * ci) for ci in c))
        pp = PackParams()
        pp.c_ha = v3(pal["ha"], s * w["ha"])
        pp.c_o3 = v3(pal["o3"], s * w["oiii"])
        pp.c_s2 = v3(pal["s2"], s * w["sii"])
        pp.c_dg = wp.vec3(*(float(s * (w["pah"] * a + w["glow"] * b)) for a, b in zip(pal["pah"], pal["glow"])))
        pp.c_sh = v3(pal["shock"], s * w["shock"])
        pp.gs = float(w["gs"])
        pp.ambient = float(w["ambient"])
        pp.edge = float(w["edge_fade"])
        pp.inv_L = wp.vec3(*(1.0 / v for v in sim.L))
        pp.dx = float(sim.dx)
        return pp

    PACK_KEYS = ("palette", "emission", "ha", "oiii", "sii", "pah", "glow", "shock", "gs", "ambient", "edge_fade",
                 "filter")

    def pack(self, sim, look):
        """sim fields -> RGBA16F texture (only when the sim or the look changed). The packing grids are views of
        the sim's scratch block (free between steps) and the texture is the sim's own (sim.volume), shared by
        every Renderer that draws this sim."""
        if not hasattr(sim, "render_scratch"):
            return self._pack_own(sim, look)
        gen = getattr(sim, "generation", 0)
        lk = tuple(tuple(v) if isinstance(v, list) else v for v in (look.get(k) for k in self.PACK_KEYS))
        v = sim.__dict__.get("volume")
        if v is not None and v["key"] == (gen, sim.version, sim.rf_version, lk):
            self.vol, self.vol_shape = v["tex"], sim.shape
            return False
        t0 = time.perf_counter()
        sim.update_render_fields()
        if v is None or v["shape"] != sim.shape:
            sim.volume = None
            nx, ny, nz = sim.shape
            tex = wp.Texture3D(width=nz, height=ny, depth=nx, num_channels=4, dtype=wp.float16,
                               filter_mode=wp.TextureFilterMode.LINEAR, address_mode=wp.TextureAddressMode.CLAMP,
                               normalized_coords=False, device=self.device)
            v = sim.volume = dict(tex=tex, shape=sim.shape, key=None)
        self.packed, self.packed2 = sim.render_scratch()
        self.pack_kernel(sim, look, self.pack_params(look, sim), self.packed)
        if look["filter"] == "smooth":
            for ax in range(3):
                src, dst = (self.packed, self.packed2) if ax != 1 else (self.packed2, self.packed)
                wp.launch(smooth_axis, dim=sim.shape, inputs=[src, dst, ax], device=self.device)
            v["tex"].copy_from(self.packed2)
        else:
            v["tex"].copy_from(self.packed)
        self.packed = self.packed2 = None          # scratch views: the sim reuses that memory on its next step
        v["key"] = (gen, sim.version, sim.rf_version, lk)
        self.vol, self.vol_shape = v["tex"], sim.shape
        self.packed_key = None
        self.prof["pack"] = time.perf_counter() - t0
        return True

    def pack_kernel(self, sim, look, pp, out):
        """sim fields -> `out` (RGBA16F grid, emission rgb + dust). A subclass with its own pack kernel overrides
        this (and adds its look keys to PACK_KEYS); smoothing, texture and caching stay as in pack()."""
        shock_on = 1 if any(c != 0.0 for c in (pp.c_sh[0], pp.c_sh[1], pp.c_sh[2])) else 0
        wp.launch(pack_volume_lm, dim=sim.shape, inputs=[sim.rho, sim.mx, sim.my, sim.mz, sim.xi, sim.dfront,
                                                         sim.g0, pp, shock_on, out], device=self.device)

    def _pack_own(self, sim, look):
        """pack for a sim without a scratch block (float32 fields + shock grid): own grids + texture"""
        gen = getattr(sim, "generation", 0)
        lk = tuple(tuple(v) if isinstance(v, list) else v for v in (look.get(k) for k in self.PACK_KEYS))
        key = (id(sim), gen, sim.version, sim.rf_version, lk)
        if key == self.packed_key and self.vol_shape == sim.shape:
            return False
        t0 = time.perf_counter()
        sim.update_render_fields()
        self._ensure_volume(sim.shape)
        wp.launch(pack_volume, dim=sim.shape, inputs=[sim.rho, sim.xi, sim.dfront, sim.g0, sim.shock,
                                                      self.pack_params(look, sim), self.packed], device=self.device)
        if look["filter"] == "smooth":
            for ax in range(3):
                src, dst = (self.packed, self.packed2) if ax != 1 else (self.packed2, self.packed)
                wp.launch(smooth_axis, dim=sim.shape, inputs=[src, dst, ax], device=self.device)
            self.vol.copy_from(self.packed2)
        else:
            self.vol.copy_from(self.packed)
        self.packed_key = (id(sim), gen, sim.version, sim.rf_version, key[4])
        self.prof["pack"] = time.perf_counter() - t0
        return True

    # ------------------------------------------------------------------ frame
    def render(self, sim, cam, look, step_vox=0.75, flip=False, jitter=0.0, tans=None):
        """tans: (tan_x, tan_y) override, e.g. a crop of the viewport camera for video recording"""
        self.pack(sim, look)
        W, H = self.W, self.H
        mp = MarchParams()
        mp.bmax = wp.vec3(*sim.L)
        mp.inv_dx = float(1.0 / sim.dx)
        mp.step = float(step_vox * sim.dx)
        red = look["reddening"]
        op = 10.0 * look["dust_opacity"]
        mp.ext = wp.vec3(*(float(op * r) for r in red))
        mp.detail = float(look["detail"] * 0.3)
        mp.detail_freq = float(look["detail_scale"])
        mp.jitter_seed = float(jitter)
        mp.cubic = 1 if look["filter"] == "cubic" else 0
        cs = cam.to_struct(W, H, tans)
        wp.launch(raymarch, dim=(H, W), inputs=[self.vol, self.det, cs, mp, self.L, self.T], device=self.device)
        # stars: screen position, brightness, transmittance camera->star (+ decorative clusters)
        pts, br, size = self.star_list(sim, look)
        n = len(pts)
        if n:
            pix, ok = cam.project(pts, W, H, (cs.tan_x, cs.tan_y))
            spr = np.zeros((n, 4), np.float32)
            spr[:, 0], spr[:, 1] = pix[:, 0], pix[:, 1]
            spr[:, 2] = np.where(ok, br, 0.0)
            spr[:, 3] = size
            self.sprites.assign(np.vstack([spr, np.zeros((self.max_sprites - n, 4), np.float32)]))
            sp4 = np.zeros((self.max_sprites, 4), np.float32)
            sp4[:n, :3] = pts
            self.star_pos.assign(sp4)
            wp.launch(star_transmittance, dim=n, inputs=[self.vol, cs.pos, self.star_pos, mp, self.strans],
                      device=self.device)
        pp = PostParams()
        pp.exposure = float(2.0 ** look["exposure"])
        pp.bg_gain = float(look["bg_stars"])
        pp.bg_density = float(look.get("bg_density", 0.08))
        pp.pix_ang = float(2.0 * cs.tan_y / H)  # radians per pixel
        pp.star_px = float(max(H, 360) / 480.0) * float(look.get("star_size", 1.0))
        pp.spikes = 1.0 if look["spikes"] else 0.0
        pp.space = wp.vec3(0.0015, 0.002, 0.004)
        pp.tonemap = {"agx": 0, "agx_base": 1, "filmic": 2}[look["tonemap"]]
        pp.agx_pow = float(look.get("agx_pow", 1.35))
        pp.agx_sat = float(look.get("agx_sat", 1.4))
        pp.flip = 1 if flip else 0
        pp.dither = 1.0
        wp.launch(post, dim=(H, W), inputs=[self.L, self.T, cs, pp, self.sprites, self.strans, n, self.rgba],
                  device=self.device)
        return self.rgba

    def star_list(self, sim, look):
        """(points (n,3), brightness (n,), size (n,)) of ionizing stars and optional cluster members"""
        stars = sim.active_stars()
        pts, br, size = [], [], []
        g = look["star_gain"] * 60.0
        for s in stars:
            ramp = min(1.0, s[4] / 0.01 + 0.05)
            pts.append(s[:3])
            br.append(g * math.sqrt(max(s[3], 0.0)) * ramp)
            size.append(1.0 if s[3] >= 0.1 else 0.7)
        nc = int(look.get("clusters", 0))
        if nc:
            for s in stars:
                if s[3] < 0.3:                      # only around the main (massive) stars
                    continue
                key = (round(s[0], 5), round(s[1], 5), round(s[2], 5), nc)
                if key not in self._cluster_cache:
                    rnd = np.random.default_rng(abs(hash(key)) % (2 ** 32))
                    off = rnd.normal(0.0, 0.055, (nc, 3))
                    m = rnd.pareto(1.35, nc) + 1.0
                    self._cluster_cache[key] = (off, np.clip(4.0 * m, 4.0, 200.0) / 1500.0)
                off, rel = self._cluster_cache[key]
                ramp = min(1.0, s[4] / 0.01 + 0.05)
                for o, r_ in zip(off, rel):
                    pts.append((s[0] + o[0], s[1] + o[1], s[2] + o[2]))
                    br.append(g * r_ * ramp)
                    size.append(0.6)
        n = min(len(pts), self.max_sprites)
        return np.asarray(pts[:n], np.float64).reshape(-1, 3), np.asarray(br[:n], np.float32),             np.asarray(size[:n], np.float32)

    def write_into(self, dst):
        """copy the last RGBA8 frame into another device buffer (e.g. a mapped GL PBO)"""
        wp.copy(dst, self.rgba)

    def image(self):
        """last frame as a (H, W, 4) uint8 numpy array (top row first unless rendered with flip)"""
        return self.rgba.numpy().view(np.uint8).reshape(self.H, self.W, 4)
