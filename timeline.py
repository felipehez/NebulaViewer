"""Timeline cache: snapshots of the simulation kept on the GPU, for scrubbing, playback and branching.

Each snapshot stores everything needed to show the frame AND to resume the simulation from it:
  rho, mx, my, mz (float16)  -> hydro state (resume)
  xi (uint8), dfront, g0 (float16) -> ionization + FUV (render without re-casting rays)
plus the small python state (clocks, stars, live parameters, rng, stirring fields) from Sim.get_state_meta().
13 bytes per cell: 27 MB at 128^3, 92 MB at 192^3, 218 MB at 256^3.

Snapshots are taken every `dtau` of sim time (tau < 0 = pre-turbulence phase). When the memory budget
is full, every other snapshot is dropped and dtau doubles, so the whole history stays reachable.
"""
import math
import warp as wp
import sim as simmod

N = simmod.N


@wp.kernel
def k_pack(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
           mz: wp.array3d(dtype=float), xi: wp.array3d(dtype=float), df: wp.array3d(dtype=float),
           g0: wp.array3d(dtype=float),
           o_r: wp.array3d(dtype=wp.float16), o_mx: wp.array3d(dtype=wp.float16), o_my: wp.array3d(dtype=wp.float16),
           o_mz: wp.array3d(dtype=wp.float16), o_xi: wp.array3d(dtype=wp.uint8), o_df: wp.array3d(dtype=wp.float16),
           o_g0: wp.array3d(dtype=wp.float16)):
    i, j, k = wp.tid()
    m = 65000.0
    o_r[i, j, k] = wp.float16(wp.min(rho[i, j, k], m))
    o_mx[i, j, k] = wp.float16(wp.clamp(mx[i, j, k], -m, m))
    o_my[i, j, k] = wp.float16(wp.clamp(my[i, j, k], -m, m))
    o_mz[i, j, k] = wp.float16(wp.clamp(mz[i, j, k], -m, m))
    o_xi[i, j, k] = wp.uint8(int(wp.clamp(xi[i, j, k], 0.0, 1.0) * 255.0 + 0.5))
    o_df[i, j, k] = wp.float16(wp.clamp(df[i, j, k], -m, m))
    o_g0[i, j, k] = wp.float16(wp.min(g0[i, j, k], m))


@wp.kernel
def k_unpack(s_r: wp.array3d(dtype=wp.float16), s_mx: wp.array3d(dtype=wp.float16),
             s_my: wp.array3d(dtype=wp.float16), s_mz: wp.array3d(dtype=wp.float16),
             s_xi: wp.array3d(dtype=wp.uint8), s_df: wp.array3d(dtype=wp.float16),
             s_g0: wp.array3d(dtype=wp.float16), floor: float,
             rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
             mz: wp.array3d(dtype=float), xi: wp.array3d(dtype=float), df: wp.array3d(dtype=float),
             g0: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    rho[i, j, k] = wp.max(float(s_r[i, j, k]), floor)
    mx[i, j, k] = float(s_mx[i, j, k])
    my[i, j, k] = float(s_my[i, j, k])
    mz[i, j, k] = float(s_mz[i, j, k])
    xi[i, j, k] = float(int(s_xi[i, j, k])) / 255.0
    df[i, j, k] = float(s_df[i, j, k])
    g0[i, j, k] = float(s_g0[i, j, k])


@wp.kernel
def k_pack_lm(rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
              mz: wp.array3d(dtype=float), xi: wp.array3d(dtype=float), df: wp.array3d(dtype=wp.float16),
              g0: wp.array3d(dtype=wp.float16),
              o_r: wp.array3d(dtype=wp.float16), o_mx: wp.array3d(dtype=wp.float16), o_my: wp.array3d(dtype=wp.float16),
              o_mz: wp.array3d(dtype=wp.float16), o_xi: wp.array3d(dtype=wp.uint8), o_df: wp.array3d(dtype=wp.float16),
              o_g0: wp.array3d(dtype=wp.float16)):
    """k_pack for float16 dfront / g0 (copied: the same values k_pack rounds to)"""
    i, j, k = wp.tid()
    m = 65000.0
    o_r[i, j, k] = wp.float16(wp.min(rho[i, j, k], m))
    o_mx[i, j, k] = wp.float16(wp.clamp(mx[i, j, k], -m, m))
    o_my[i, j, k] = wp.float16(wp.clamp(my[i, j, k], -m, m))
    o_mz[i, j, k] = wp.float16(wp.clamp(mz[i, j, k], -m, m))
    o_xi[i, j, k] = wp.uint8(int(wp.clamp(xi[i, j, k], 0.0, 1.0) * 255.0 + 0.5))
    o_df[i, j, k] = wp.float16(wp.clamp(float(df[i, j, k]), -m, m))
    o_g0[i, j, k] = wp.float16(wp.min(float(g0[i, j, k]), m))


@wp.kernel
def k_unpack_lm(s_r: wp.array3d(dtype=wp.float16), s_mx: wp.array3d(dtype=wp.float16),
                s_my: wp.array3d(dtype=wp.float16), s_mz: wp.array3d(dtype=wp.float16),
                s_xi: wp.array3d(dtype=wp.uint8), s_df: wp.array3d(dtype=wp.float16),
                s_g0: wp.array3d(dtype=wp.float16), floor: float,
                rho: wp.array3d(dtype=float), mx: wp.array3d(dtype=float), my: wp.array3d(dtype=float),
                mz: wp.array3d(dtype=float), xi: wp.array3d(dtype=float), df: wp.array3d(dtype=wp.float16),
                g0: wp.array3d(dtype=wp.float16)):
    i, j, k = wp.tid()
    rho[i, j, k] = wp.max(float(s_r[i, j, k]), floor)
    mx[i, j, k] = float(s_mx[i, j, k])
    my[i, j, k] = float(s_my[i, j, k])
    mz[i, j, k] = float(s_mz[i, j, k])
    xi[i, j, k] = float(int(s_xi[i, j, k])) / 255.0
    df[i, j, k] = s_df[i, j, k]
    g0[i, j, k] = s_g0[i, j, k]


BYTES_PER_CELL = 13


class Timeline:
    def __init__(self, budget_gb=8.0, device="cuda:0"):
        self.device = device
        self.budget = budget_gb * 1e9
        self.snaps = []          # sorted by tau: dict(tau, arrays, meta)
        self.dtau = 0.002
        self.next_tau = -1e9
        self.ncells = 0
        self.thinned = 0

    # ------------------------------------------------------------------ bookkeeping
    def clear(self, sim=None):
        self.snaps = []
        self.thinned = 0
        if sim is not None:
            self.ncells = sim.shape[0] * sim.shape[1] * sim.shape[2]
            # ~200 snapshots over the main phase (and as many over the pre-turbulence)
            self.dtau = max(sim.P["t_end"] / 200.0, 1e-4)
            self.next_tau = sim.tau()

    def bytes_used(self):
        return len(self.snaps) * self.ncells * BYTES_PER_CELL

    def capacity(self):
        # the budget is honoured (at least one snapshot): with the big grids the low-memory sim allows, the old
        # minimum of 4 could hold more GPU memory than the sim itself (4 x 2.7 GB at a 640 crop window)
        return max(1, int(self.budget // max(self.ncells * BYTES_PER_CELL, 1)))

    def set_budget(self, gb):
        self.budget = gb * 1e9
        while len(self.snaps) > self.capacity():
            self._thin()

    def taus(self):
        return [s["tau"] for s in self.snaps]

    def nearest(self, tau):
        if not self.snaps:
            return None
        return min(range(len(self.snaps)), key=lambda i: abs(self.snaps[i]["tau"] - tau))

    def index_at_or_before(self, tau):
        idx = None
        for i, s in enumerate(self.snaps):
            if s["tau"] <= tau + 1e-12:
                idx = i
        return idx

    # ------------------------------------------------------------------ capture / restore
    def maybe_capture(self, sim):
        """call after sim steps: takes a snapshot when sim time crosses the next grid point"""
        tau = sim.tau()
        if tau + 1e-12 < self.next_tau:
            return False
        if self.snaps and tau <= self.snaps[-1]["tau"] + 1e-12:
            return False
        self.capture(sim)
        self.next_tau = (math.floor(tau / self.dtau + 1e-9) + 1) * self.dtau
        return True

    def capture(self, sim):
        sh = sim.shape
        dev = self.device
        h = lambda: wp.empty(sh, dtype=wp.float16, device=dev)
        arrs = [h(), h(), h(), h(), wp.empty(sh, dtype=wp.uint8, device=dev), h(), h()]
        sim.update_render_fields()           # g0 up to date with this state
        kp = k_pack_lm if sim.dfront.dtype == wp.float16 else k_pack
        wp.launch(kp, dim=sh, inputs=[sim.rho, sim.mx, sim.my, sim.mz, sim.xi, sim.dfront, sim.g0] + arrs,
                  device=dev)
        self.snaps.append(dict(tau=sim.tau(), arrays=arrs, meta=sim.get_state_meta()))
        while len(self.snaps) > self.capacity():
            self._thin()

    def _thin(self):
        """drop every other snapshot (keep first and last), double the spacing"""
        if len(self.snaps) < 3:
            self.snaps = self.snaps[-1:]
            return
        keep = self.snaps[::2]
        if keep[-1] is not self.snaps[-1]:
            keep.append(self.snaps[-1])
        self.snaps = keep
        self.dtau *= 2.0
        self.thinned += 1
        if self.snaps:
            self.next_tau = (math.floor(self.snaps[-1]["tau"] / self.dtau + 1e-9) + 1) * self.dtau

    def restore(self, sim, i):
        s = self.snaps[i]
        a = s["arrays"]
        if a[0].shape != sim.shape:
            return False
        ku = k_unpack_lm if sim.dfront.dtype == wp.float16 else k_unpack
        wp.launch(ku, dim=sim.shape, inputs=a + [float(sim.P["floor"]), sim.rho, sim.mx, sim.my, sim.mz,
                                               sim.xi, sim.dfront, sim.g0], device=self.device)
        sim.set_state_meta(s["meta"])
        return True

    def truncate_after(self, tau):
        """branch: forget everything later than tau (the sim continues from there with new parameters)"""
        self.snaps = [s for s in self.snaps if s["tau"] <= tau + 1e-12]
        self.next_tau = (math.floor(tau / self.dtau + 1e-9) + 1) * self.dtau

    def status(self):
        n = len(self.snaps)
        if not n:
            return "cache empty"
        return (f"{n} frames  {self.bytes_used()/1e9:.1f}/{self.budget/1e9:.0f} GB  "
                f"every {self.dtau:.4g}{'  (thinned x' + str(2 ** self.thinned) + ')' if self.thinned else ''}")
