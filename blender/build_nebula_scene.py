"""Blender 5.0 scene for nebula3d runs.
blender -b --factory-startup --python build_nebula_scene.py -- <run> [--palette jwst|sho|natural]

* Volume object = VDB sequence from nebula_to_vdb.py (one file per sim frame)
* 'Nebula Volume' node group: emission (ionized lines + PAH/PDR + baked dust glow) and
  dust absorption with spectral reddening. No real scattering by default (render cost ~ emission only).
* Stars as a point cloud (GN Mesh to Points) with per-star blackbody temperature/strength;
  embedded protostars only appear after their t_on (driven by the scene frame).
* JWST-like 6-point diffraction spikes + bloom in the compositor.
"""
import bpy, sys, os, json, math, random
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
run = argv[0]
palette = argv[argv.index("--palette") + 1] if "--palette" in argv else "jwst"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIM = os.path.join(ROOT, "sim3d", run)
meta = json.load(open(os.path.join(SIM, "meta.json")))
vdb_dir = os.path.join(SIM, "vdb")
vdbs = sorted(f for f in os.listdir(vdb_dir) if f.endswith(".vdb"))
SCALE = 10.0
nx, ny, nz = meta["shape"]
dx = meta["dx"]
LX, LY, LZ = nx * dx * SCALE, ny * dx * SCALE, nz * dx * SCALE
times = meta["times"]
SCEN = meta.get("scenario", "cliff")

PALETTES = {
    # colours + strengths per preset (lookdev n005)
    "jwst":    dict(ha=(0.22, 0.45, 1.0), o3=(0.30, 0.75, 1.0), s2=(1.0, 0.55, 0.25),
                    pah=(1.0, 0.50, 0.15), glow=(0.75, 0.42, 0.22),
                    w={"H-alpha": 0.6, "PAH": 0.07, "Dust Glow": 0.0015}),
    "sho":     dict(ha=(0.15, 1.0, 0.25), o3=(0.10, 0.45, 1.0), s2=(1.0, 0.25, 0.10),
                    pah=(0.9, 0.6, 0.2), glow=(0.5, 0.4, 0.3),
                    w={"SII": 3.0, "PAH": 0.04}),
    "natural": dict(ha=(1.0, 0.18, 0.28), o3=(0.15, 0.85, 0.80), s2=(1.0, 0.10, 0.10),
                    pah=(0.6, 0.35, 0.2), glow=(0.35, 0.5, 0.95),
                    w={"PAH": 0.01, "Dust Glow": 0.0015}),
}
PAL = PALETTES[palette]

bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene


# ------------------------------------------------------------------------------------ helpers
def sock(t, io, name, typ, default=None, mn=None, mx=None, desc=""):
    s = t.interface.new_socket(name=name, in_out=io, socket_type=typ, description=desc)
    if default is not None and hasattr(s, "default_value"):
        s.default_value = default
    if mn is not None and hasattr(s, "min_value"):
        s.min_value = mn
    if mx is not None and hasattr(s, "max_value"):
        s.max_value = mx
    return s


def node(t, idn, loc, **kw):
    n = t.nodes.new(idn)
    n.location = loc
    for k, v in kw.items():
        setattr(n, k, v)
    return n


# ------------------------------------------------------------------------------------ shader
def build_nebula_shader():
    t = bpy.data.node_groups.new("Nebula Volume", "ShaderNodeTree")
    I = "INPUT"
    sock(t, I, "Dust Opacity", "NodeSocketFloat", 0.25, 0, 50, "absorption per unit dust per BU")
    sock(t, I, "Reddening", "NodeSocketColor", (0.55, 0.85, 1.35, 1), desc="relative extinction R,G,B (~1/lambda)")
    sock(t, I, "Dust Albedo", "NodeSocketFloat", 0.0, 0, 1, "real scattering (needs lights; expensive)")
    sock(t, I, "Emission Scale", "NodeSocketFloat", 1.0, 0, 1e4)
    sock(t, I, "H-alpha", "NodeSocketFloat", 1.0, 0, 100)
    sock(t, I, "H-alpha Color", "NodeSocketColor", (*PAL["ha"], 1))
    sock(t, I, "OIII", "NodeSocketFloat", 1.5, 0, 100)
    sock(t, I, "OIII Color", "NodeSocketColor", (*PAL["o3"], 1))
    sock(t, I, "SII", "NodeSocketFloat", 1.2, 0, 100)
    sock(t, I, "SII Color", "NodeSocketColor", (*PAL["s2"], 1))
    sock(t, I, "PAH", "NodeSocketFloat", 0.02, 0, 100, "PDR / PAH glow ~ dust * G0")
    sock(t, I, "PAH Color", "NodeSocketColor", (*PAL["pah"], 1))
    sock(t, I, "Dust Glow", "NodeSocketFloat", 0.0005, 0, 100, "baked single scattering ~ dust * G0")
    sock(t, I, "Dust Glow Color", "NodeSocketColor", (*PAL["glow"], 1))
    sock(t, I, "Shock", "NodeSocketFloat", 0.002 if SCEN == "bipolar" else 0.0, 0, 100,
         "shock-excited emission (jets / HH knots / turbulent shocks)")
    sock(t, I, "Shock Color", "NodeSocketColor", (1.0, 0.35, 0.45, 1))
    sock(t, I, "G0 Saturation", "NodeSocketFloat", 12.0 if SCEN == "bubbles" else 1e4, 0.01, 1e5,
         "soft cap of the FUV field: g/(1+g/Gs). Tames the glow right next to stars")
    sock(t, I, "Ambient G0", "NodeSocketFloat", 0.3 if SCEN == "bubbles" else 0.0, 0, 1000,
         "diffuse interstellar radiation field: lights the dust between bubbles")
    sock(t, I, "Edge Fade", "NodeSocketFloat", 0.08, 0.0, 0.5,
         "soft fade of every field near the simulation box faces (hides the domain edges)")
    sock(t, I, "Detail", "NodeSocketFloat", 0.5, 0, 2, "sub-voxel fBm modulation of dust & emission")
    sock(t, I, "Detail Scale", "NodeSocketFloat", 12.0, 0.01, 1000, "noise frequency per BU (voxel ~0.04 BU)")
    sock(t, "OUTPUT", "Volume", "NodeSocketShader")
    gi = node(t, "NodeGroupInput", (-1400, 0))
    go = node(t, "NodeGroupOutput", (900, 0))
    L = t.links.new
    att = {}
    for i, nm in enumerate(("dust", "ha", "oiii", "sii", "g0", "shock")):
        att[nm] = node(t, "ShaderNodeAttribute", (-1150, 400 - i * 180), attribute_name=nm).outputs["Fac"]

    def mul(a, b, loc, op="MULTIPLY"):
        n = node(t, "ShaderNodeMath", loc, operation=op)
        L(a, n.inputs[0]) if not isinstance(a, float) else None
        L(b, n.inputs[1]) if not isinstance(b, float) else None
        return n.outputs[0]

    def vscale(vec, f, loc):
        n = node(t, "ShaderNodeVectorMath", loc, operation="SCALE")
        L(vec, n.inputs[0]); L(f, n.inputs[3])
        return n.outputs[0]

    def vadd(a, b, loc):
        n = node(t, "ShaderNodeVectorMath", loc, operation="ADD")
        L(a, n.inputs[0]); L(b, n.inputs[1])
        return n.outputs[0]

    gi_o = gi.outputs
    # sub-voxel detail: m = max(0, 1 + Detail * (fbm - 0.5) * 2), applied to every field
    tc = node(t, "ShaderNodeTexCoord", (-1700, -1200))
    nz = node(t, "ShaderNodeTexNoise", (-1500, -1200), noise_dimensions="3D")
    nz.inputs["Detail"].default_value = 5.0
    nz.inputs["Roughness"].default_value = 0.6
    nz.inputs["Distortion"].default_value = 0.0
    L(tc.outputs["Object"], nz.inputs["Vector"])
    L(gi_o["Detail Scale"], nz.inputs["Scale"])
    c = node(t, "ShaderNodeMath", (-1300, -1200), operation="SUBTRACT"); c.inputs[1].default_value = 0.5
    L(nz.outputs["Fac"], c.inputs[0])
    c2 = node(t, "ShaderNodeMath", (-1150, -1200), operation="MULTIPLY"); c2.inputs[1].default_value = 2.0
    L(c.outputs[0], c2.inputs[0])
    c3 = node(t, "ShaderNodeMath", (-1000, -1200), operation="MULTIPLY_ADD", use_clamp=False)
    L(c2.outputs[0], c3.inputs[0]); L(gi_o["Detail"], c3.inputs[1]); c3.inputs[2].default_value = 1.0
    dm = node(t, "ShaderNodeMath", (-850, -1200), operation="MAXIMUM"); dm.inputs[1].default_value = 0.0
    L(c3.outputs[0], dm.inputs[0])
    # normalized box coords from Object space (the VDB's index space starts at the object origin)
    nrm = node(t, "ShaderNodeVectorMath", (-1650, -1450), operation="DIVIDE")
    L(tc.outputs["Object"], nrm.inputs[0])
    nrm.inputs[1].default_value = (LX, LY, LZ)
    sep = node(t, "ShaderNodeSeparateXYZ", (-1500, -1450))
    L(nrm.outputs[0], sep.inputs[0])
    fade = None
    for ax_i in range(3):
        lo = node(t, "ShaderNodeMapRange", (-1300, -1400 - ax_i * 120), interpolation_type="SMOOTHSTEP")
        L(sep.outputs[ax_i], lo.inputs["Value"])
        lo.inputs["From Min"].default_value = 0.0
        L(gi_o["Edge Fade"], lo.inputs["From Max"])
        inv = node(t, "ShaderNodeMath", (-1300, -1460 - ax_i * 120), operation="SUBTRACT")
        inv.inputs[0].default_value = 1.0
        L(sep.outputs[ax_i], inv.inputs[1])
        hi = node(t, "ShaderNodeMapRange", (-1150, -1400 - ax_i * 120), interpolation_type="SMOOTHSTEP")
        L(inv.outputs[0], hi.inputs["Value"])
        hi.inputs["From Min"].default_value = 0.0
        L(gi_o["Edge Fade"], hi.inputs["From Max"])
        f_ax = mul(lo.outputs[0], hi.outputs[0], (-1000, -1400 - ax_i * 120))
        fade = f_ax if fade is None else mul(fade, f_ax, (-900, -1400 - ax_i * 120))
    detail = mul(dm.outputs[0], fade, (-800, -1250))
    for nm in list(att):
        if nm != "g0":
            att[nm] = mul(att[nm], detail, (-1000, 400 - list(att).index(nm) * 180))
    gs1 = node(t, "ShaderNodeMath", (-1300, -650), operation="DIVIDE")
    L(att["g0"], gs1.inputs[0]); L(gi_o["G0 Saturation"], gs1.inputs[1])
    gs2 = node(t, "ShaderNodeMath", (-1200, -650), operation="ADD"); gs2.inputs[1].default_value = 1.0
    L(gs1.outputs[0], gs2.inputs[0])
    gs3 = node(t, "ShaderNodeMath", (-1100, -650), operation="DIVIDE")
    L(att["g0"], gs3.inputs[0]); L(gs2.outputs[0], gs3.inputs[1])
    g0a = node(t, "ShaderNodeMath", (-1000, -600), operation="ADD")
    L(gs3.outputs[0], g0a.inputs[0]); L(gi_o["Ambient G0"], g0a.inputs[1])
    dg = mul(att["dust"], g0a.outputs[0], (-900, -500))
    terms = [
        vscale(gi_o["H-alpha Color"], mul(att["ha"], gi_o["H-alpha"], (-900, 350)), (-650, 350)),
        vscale(gi_o["OIII Color"], mul(att["oiii"], gi_o["OIII"], (-900, 200)), (-650, 200)),
        vscale(gi_o["SII Color"], mul(att["sii"], gi_o["SII"], (-900, 50)), (-650, 50)),
        vscale(gi_o["PAH Color"], mul(dg, gi_o["PAH"], (-900, -300)), (-650, -300)),
        vscale(gi_o["Dust Glow Color"], mul(dg, gi_o["Dust Glow"], (-900, -700)), (-650, -700)),
        vscale(gi_o["Shock Color"], mul(att["shock"], gi_o["Shock"], (-900, -850)), (-650, -850)),
    ]
    acc = terms[0]
    for i, tm in enumerate(terms[1:]):
        acc = vadd(acc, tm, (-400 + i * 120, 0))
    em = vscale(acc, gi_o["Emission Scale"], (200, 0))
    k = mul(att["dust"], gi_o["Dust Opacity"], (-650, -900))
    ext = vscale(gi_o["Reddening"], k, (-400, -900))
    one_m = node(t, "ShaderNodeMath", (-400, -1050), operation="SUBTRACT")
    one_m.inputs[0].default_value = 1.0
    L(gi_o["Dust Albedo"], one_m.inputs[1])
    absorb = vscale(ext, one_m.outputs[0], (-150, -900))
    scat = vscale(ext, gi_o["Dust Albedo"], (-150, -1050))
    coef = node(t, "ShaderNodeVolumeCoefficients", (500, -300), phase="HENYEY_GREENSTEIN")
    coef.inputs["Anisotropy"].default_value = 0.6
    L(absorb, coef.inputs["Absorption Coefficients"])
    L(scat, coef.inputs["Scatter Coefficients"])
    L(em, coef.inputs["Emission Coefficients"])
    L(coef.outputs[0], go.inputs["Volume"])
    return t


def nebula_material():
    g = build_nebula_shader()
    m = bpy.data.materials.new("M_Nebula")
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (300, 0)
    gn = nt.nodes.new("ShaderNodeGroup"); gn.node_tree = g; gn.width = 260
    nt.links.new(gn.outputs["Volume"], out.inputs["Volume"])
    for k, v in PAL["w"].items():
        gn.inputs[k].default_value = v
    if SCEN in ("bubbles", "bipolar"):   # Felipe's lookdev (2026-10-01): thick high-contrast dust
        for k, v in {"Dust Opacity": 4.0, "Emission Scale": 2.0, "H-alpha": 1.5, "Dust Glow": 0.002,
                     "Ambient G0": 0.1 if SCEN == "bubbles" else 0.3,
                     "Detail": 4.0, "Detail Scale": 1.0}.items():
            gn.inputs[k].default_value = v
    if SCEN == "bipolar":   # lobes are evacuated by the wind: push emission, keep dust thinner
        for k, v in {"Emission Scale": 6.0, "Dust Opacity": 1.5, "Detail": 2.0}.items():
            gn.inputs[k].default_value = v
    m.cycles.volume_interpolation = "CUBIC"
    return m


# ------------------------------------------------------------------------------------ volume
def nebula_volume(mat):
    vol = bpy.data.volumes.new("Nebula")
    vol.filepath = os.path.join(vdb_dir, vdbs[0])
    vol.is_sequence = True
    vol.frame_start = 1
    vol.frame_duration = len(times)
    vol.frame_offset = -1            # scene frame 1 -> neb_0000.vdb
    vol.sequence_mode = "CLIP"
    vol.render.precision = "HALF"
    vol.materials.append(mat)
    ob = bpy.data.objects.new("Nebula", vol)
    scn.collection.objects.link(ob)
    return ob


# ------------------------------------------------------------------------------------ stars
def star_cloud(name, pts, radius, temp, strength, t_on_frame):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in pts], [], [])
    for nm, vals in (("radius", radius), ("temp", temp), ("strength", strength), ("t_on", t_on_frame)):
        a = me.attributes.new(nm, "FLOAT", "POINT")
        a.data.foreach_set("value", np.asarray(vals, dtype=np.float32))
    ob = bpy.data.objects.new(name, me)
    scn.collection.objects.link(ob)
    # GN: points appear when scene frame >= t_on, Mesh -> Points with per-point radius
    t = bpy.data.node_groups.get("StarPoints")
    if t is None:
        t = bpy.data.node_groups.new("StarPoints", "GeometryNodeTree")
        t.is_modifier = True
        sock(t, "INPUT", "Geometry", "NodeSocketGeometry")
        sock(t, "OUTPUT", "Geometry", "NodeSocketGeometry")
        gi = node(t, "NodeGroupInput", (-800, 0)); go = node(t, "NodeGroupOutput", (600, 0))
        st = node(t, "GeometryNodeInputSceneTime", (-800, -250))
        ton = node(t, "GeometryNodeInputNamedAttribute", (-800, -400), data_type="FLOAT")
        ton.inputs["Name"].default_value = "t_on"
        cmp = node(t, "FunctionNodeCompare", (-550, -300), data_type="FLOAT", operation="LESS_THAN")
        t.links.new(st.outputs["Frame"], cmp.inputs["A"])
        t.links.new(ton.outputs["Attribute"], cmp.inputs["B"])
        dele = node(t, "GeometryNodeDeleteGeometry", (-300, 0), domain="POINT")
        t.links.new(gi.outputs[0], dele.inputs["Geometry"])
        t.links.new(cmp.outputs[0], dele.inputs["Selection"])
        rad = node(t, "GeometryNodeInputNamedAttribute", (-300, -300), data_type="FLOAT")
        rad.inputs["Name"].default_value = "radius"
        m2p = node(t, "GeometryNodeMeshToPoints", (0, 0))
        t.links.new(dele.outputs[0], m2p.inputs["Mesh"])
        t.links.new(rad.outputs["Attribute"], m2p.inputs["Radius"])
        sm = node(t, "GeometryNodeSetMaterial", (300, 0))
        sm.inputs["Material"].default_value = star_material()
        t.links.new(m2p.outputs[0], sm.inputs["Geometry"])
        t.links.new(sm.outputs[0], go.inputs[0])
    mod = ob.modifiers.new("StarPoints", "NODES")
    mod.node_group = t
    ob.visible_shadow = False
    return ob


def star_material():
    m = bpy.data.materials.get("M_Star")
    if m:
        return m
    m = bpy.data.materials.new("M_Star")
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
    em = nt.nodes.new("ShaderNodeEmission"); em.location = (200, 0)
    bb = nt.nodes.new("ShaderNodeBlackbody"); bb.location = (0, 100)
    at = nt.nodes.new("ShaderNodeAttribute"); at.location = (-200, 100); at.attribute_name = "temp"
    st = nt.nodes.new("ShaderNodeAttribute"); st.location = (-200, -100); st.attribute_name = "strength"
    gain = nt.nodes.new("ShaderNodeMath"); gain.location = (0, -100); gain.operation = "MULTIPLY"
    gain.label = "Star Gain"
    gain.inputs[1].default_value = {"bubbles": 0.5, "bipolar": 0.3}.get(SCEN, 1.0)
    nt.links.new(at.outputs["Fac"], bb.inputs[0])
    nt.links.new(bb.outputs[0], em.inputs["Color"])
    nt.links.new(st.outputs["Fac"], gain.inputs[0])
    nt.links.new(gain.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


def time_to_frame(t):
    # frame f (1-based) shows sim time times[f-1]
    return 1 + int(np.searchsorted(times, t))


def make_stars():
    rnd = np.random.default_rng(5)
    # 1) ionizing stars from the sim (hot O/B stars) + embedded protostars
    P, R, T, S, F = [], [], [], [], []
    for (sx, sy, sz, Q, ton) in meta["stars"]:
        P.append((sx * SCALE, sy * SCALE, sz * SCALE))
        main = Q >= 0.1 * meta["q_main"]
        R.append(0.035 if main else 0.02)
        T.append(30000.0 if main else 4500.0)          # embedded protostars: reddened, cool-looking
        S.append(1500.0 * (Q / meta["q_main"]) ** 0.5 if main else 120.0)
        F.append(time_to_frame(ton))
    star_cloud("Stars_Ionizing", P, R, T, S, F)
    # 2) young clusters around the main stars (non-ionizing in the sim, visual only)
    if SCEN == "bubbles":
        prim = [st for st in meta["stars"] if st[3] >= 0.3 * meta["q_main"]]
        pts, ton = [], []
        for st in prim:
            k = 22
            pts.append(np.array(st[:3]) * SCALE + rnd.normal(0, 0.55, (k, 3)))
            ton.append(np.full(k, time_to_frame(st[4])))
        pts, ton = np.vstack(pts), np.concatenate(ton)
    elif SCEN == "bipolar":
        pts, ton = np.zeros((0, 3)), np.zeros(0)
    else:
        n = 160
        c = np.array(meta["stars"][0][:3]) * SCALE
        pts = c + rnd.normal(0, 1, (n, 3)) * np.array([2.6, 1.6, 0.9])
        pts[:, 2] = np.maximum(pts[:, 2], LZ * 0.62)
        ton = np.zeros(n)
    n = len(pts)
    m = rnd.pareto(1.35, n) + 1
    if n:
      star_cloud("Stars_Cluster", pts, np.clip(0.008 * m ** 0.4, 0.008, 0.03), rnd.uniform(7000, 25000, n),
               np.clip(4 * m, 4, 200), ton)
    # 3) field stars far behind and a few in front of the nebula
    n = 4000
    far = np.column_stack([rnd.uniform(-25, LX + 25, n), rnd.uniform(LY + 30, LY + 60, n),
                           rnd.uniform(-15, LZ + 20, n)])
    nf = 140
    near = np.column_stack([rnd.uniform(-4, LX + 4, nf), rnd.uniform(-6, -1, nf), rnd.uniform(0, LZ + 3, nf)])
    pts = np.vstack([far, near])
    m = rnd.pareto(1.6, n + nf) + 1
    temp = np.where(rnd.random(n + nf) < 0.6, rnd.uniform(3200, 6000, n + nf), rnd.uniform(6000, 15000, n + nf))
    rad = np.concatenate([np.full(n, 0.03), np.full(nf, 0.012)]) * np.clip(m ** 0.3, 1, 3)
    star_cloud("Stars_Field", pts, rad, temp, np.clip(1.2 * m, 1.2, 80), np.zeros(n + nf))


# ------------------------------------------------------------------------------------ camera etc
def camera():
    cd = bpy.data.cameras.new("Cam")
    cd.lens = 50
    cd.sensor_width = 36
    cd.clip_end = 1000
    ob = bpy.data.objects.new("Cam", cd)
    scn.collection.objects.link(ob)
    ob.rotation_euler = (math.radians(90 + 2.0), 0, 0)
    scn.camera = ob
    # slow dolly + truck over the timelapse so the 3D parallax reads
    n = len(times)
    if SCEN == "bipolar":      # whole hourglass in view, slow push-in
        ob.rotation_euler = (math.radians(90), 0, 0)
        # pull back as the lobes grow (long runs); short tests keep a tighter framing
        d0, d1 = (13.0, 16.0) if n > 48 else (13.5, 12.0)
        ob.location = (LX * 0.5, LY * 0.5 - d0, LZ * 0.5); ob.keyframe_insert("location", frame=1)
        ob.location = (LX * 0.5, LY * 0.5 - d1, LZ * 0.5); ob.keyframe_insert("location", frame=n)
    elif SCEN == "bubbles":      # face-on to the sheet, whole field in view
        ob.rotation_euler = (math.radians(90), 0, 0)
        ob.location = (LX * 0.5 - 0.5, -19.0, LZ * 0.5); ob.keyframe_insert("location", frame=1)
        ob.location = (LX * 0.5 + 0.5, -17.0, LZ * 0.5); ob.keyframe_insert("location", frame=n)
    else:
        ob.location = (LX * 0.5 - 0.8, -15.5, LZ * 0.58); ob.keyframe_insert("location", frame=1)
        ob.location = (LX * 0.5 + 0.8, -13.0, LZ * 0.56); ob.keyframe_insert("location", frame=n)
    for fc in ob.animation_data.action.fcurves if hasattr(ob.animation_data.action, "fcurves") else []:
        for kp in fc.keyframe_points:
            kp.interpolation = "SINE"
    scn.frame_set(1)


def world():
    w = bpy.data.worlds.new("Space")
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    scn.world = w


def compositor():
    t = bpy.data.node_groups.new("NebulaComp", "CompositorNodeTree")
    sock(t, "OUTPUT", "Image", "NodeSocketColor")
    rl = node(t, "CompositorNodeRLayers", (-600, 0))
    spikes = node(t, "CompositorNodeGlare", (-300, 0))
    spikes.inputs["Type"].default_value = "Streaks"
    spikes.inputs["Threshold"].default_value = 3.0
    spikes.inputs["Streaks"].default_value = 6
    spikes.inputs["Streaks Angle"].default_value = math.radians(90)
    spikes.inputs["Strength"].default_value = 0.35
    spikes.inputs["Fade"].default_value = 0.82
    bloom = node(t, "CompositorNodeGlare", (0, 0))
    bloom.inputs["Type"].default_value = "Fog Glow"
    bloom.inputs["Threshold"].default_value = 1.5
    bloom.inputs["Strength"].default_value = 0.25
    out = node(t, "NodeGroupOutput", (300, 0))
    t.links.new(rl.outputs["Image"], spikes.inputs["Image"])
    t.links.new(spikes.outputs["Image"], bloom.inputs["Image"])
    t.links.new(bloom.outputs["Image"], out.inputs[0])
    scn.compositing_node_group = t


def render_settings():
    scn.render.engine = "CYCLES"
    c = scn.cycles
    c.device = "GPU"
    c.samples = 4          # emission + absorption only: converges in a few samples (+ denoiser)
    c.use_adaptive_sampling = True
    c.use_denoising = True
    c.volume_biased = True          # unbiased emission volumes are noisy in Blender 5.0
    c.volume_step_rate = 1.0
    c.volume_bounces = 0            # emission + absorption only
    c.max_bounces = 4
    c.transparent_max_bounces = 8
    scn.render.resolution_x, scn.render.resolution_y = (1440, 1440) if SCEN == "bipolar" else (1920, 1080)
    scn.render.resolution_percentage = 50
    scn.render.fps = 12                 # 60 sim frames -> 5 s timelapse
    scn.frame_start, scn.frame_end = 1, len(times)
    scn.render.use_persistent_data = True
    # EEVEE lookdev preset (switch the render engine to EEVEE for a near-real-time viewport):
    # froxel volumes blur fine detail, so use small froxels and a depth range hugging the nebula
    e = scn.eevee
    e.volumetric_tile_size = "2"        # "1" for final-quality EEVEE renders (slower)
    e.volumetric_samples = 192
    e.volumetric_sample_distribution = 0.6
    e.use_volume_custom_range = True
    e.volumetric_start, e.volumetric_end = (7.0, 26.0) if SCEN == "bipolar" else (2.0, 45.0)
    e.use_volumetric_shadows = False    # emission + absorption only
    e.taa_samples = 8
    e.taa_render_samples = 16
    vs = scn.view_settings
    vs.view_transform = "AgX"
    vs.look = "AgX - Punchy"
    vs.exposure = -0.8 if SCEN in ("bubbles", "bipolar") else 0.0


mat = nebula_material()
nebula_volume(mat)
make_stars()
camera()
world()
compositor()
render_settings()
scn.frame_set(46)
out = os.path.join(ROOT, f"nebula_{run}.blend")
bpy.ops.wm.save_as_mainfile(filepath=out)
print("saved", out, "frames", len(vdbs), "domain BU", (round(LX, 2), round(LY, 2), round(LZ, 2)))
