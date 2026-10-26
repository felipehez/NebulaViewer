# GPU memory, resolution and cropped domains

Resolution is what makes the biggest difference in NebulaViewer: higher resolution brings out the filaments, the globules with tails and the thin fronts. This page explains how much GPU memory a grid needs, how far each scenario goes on a given card, and how to spend the cells only where the camera looks.

## Bytes per cell

The simulation and the renderer keep **48 bytes per cell** on the GPU:

| Buffer | Bytes/cell | Notes |
|---|---|---|
| `rho`, `mx`, `my`, `mz` | 16 | gas state, float32 |
| `xi` | 4 | ionized fraction, float32 (it sets the pressure) |
| `stage` | 16 | the RK2 stage. Between steps it is scratch: smoothing, accumulators, the renderer's packing grids |
| `dfront`, `g0` | 4 | distance to the ionization front and FUV field, float16 (render, timeline, star formation) |
| RGBA16F texture | 8 | the packed volume. It belongs to the sim and every renderer shares it |

Notes:
- The second-order time step writes its second stage in place, so it needs one stage copy, not two.
- The shock-emission term is computed while packing.
- Results are the same as the original 92–96 B/cell layout: the hydrodynamics match bit for bit, and images differ by at most 1/255 because of the float16 fields.
- Code that writes `dfront` / `g0` with float32 kernels can pass `--set fp32_fields=true` (+4 B/cell).
- A custom renderer overrides `Renderer.pack_kernel(sim, look, pp, out)` and adds its look keys to `PACK_KEYS`.

On top of the grid: ~0.4 GB of fixed buffers (FUV grid, rays, 4K frame buffers), plus the timeline.

## How far each scenario goes

Largest resolution (`--res`, cells per nebula unit) whose sim and renderer fit in a given amount of free GPU memory, with the default box of each scenario. In parentheses: the same budget with a **9:16 vertical crop** of the default camera (see below).

| Scenario | 12 GB | 16 GB | 21 GB |
|---|---|---|---|
| bubbles | 496 (664) | 544 (736) | 600 (800) |
| cliff | 472 (656) | 520 (736) | 576 (800) |
| bipolar | 384 (416) | 424 (464) | 472 (512) |

- `--max_grid_gb` (14 by default) caps the grid and leaves room for the timeline.
- Cost grows with the cells **and** with the number of steps, which is proportional to the resolution (CFL).
- On an RTX 3090 a step takes ~1–1.5 ns per cell: 125 M cells ≈ 130–150 ms, 324 M cells ≈ 0.4 s.
- At the same memory, a 1.25× sharper shot takes ~2.4× longer to simulate.

## Cropped domains

```bat
python viewer.py --scenario bubbles --res 640 --set "crop=[0.29,0.71,0,1,0.01,0.99]"
```

`crop = [x0, x1, y0, y1, z0, z1]` is a window of the full box, as fractions of each axis (z is up, the default camera looks along +y). Only the window is simulated, so it can run at a much higher resolution: a vertical 9:16 strip of a 16:9 view keeps ~40–50% of the width.

The window is the same nebula, cut out:
- **Initial conditions:** the large scales of the `matched` noise are exactly those of the full box. The smaller scales are another realization with the same spectrum, as between two resolutions.
- **Edges:**
  - the cloud keeps its organic edge;
  - every cut face gets its own organic fade, `crop_edge` wide (0.1 by default), so the walls do not show;
  - no stars are triggered inside that fade.
- **Stars:** they stay at their full-box positions.
  - In `bubbles`, a star outside the window still lights it in the FUV, but does not ionize or push the gas.
  - The `cliff` main stars are always kept.
- **Camera:** the viewer frames the default view of the **full** box (`sim.crop_origin` gives the window's corner), so camera offsets keep their meaning.
- **Helper:** `crop.crop_for_shot(scenario, res, box, cameras, (tan_x, tan_y), margin=0.12)` returns the crop that contains everything a set of cameras sees.

Limits:
- `bipolar` windows are centred on the star.
- The depth axis (y) of `bubbles` / `cliff` is cropped symmetrically.
- Whatever would flow in from outside the window, such as a neighbour's bubble, is missing.
- A cropped run is only valid for the framing it was cut for.

## Timeline

Each snapshot keeps 13 B/cell (float16 state, uint8 ionization), enough to show the frame and to resume the simulation from it. The budget (`--cache_gb`, "Timeline memory" slider) is honoured with a minimum of one snapshot, so big grids keep fewer frames.

## RAM at reset, and Windows

Building the initial cloud (NumPy, on the CPU) briefly needs about **32 B/cell** of RAM for `cliff` and `bubbles` and **68 B/cell** for `bipolar` (float64 intermediates). This happens before the GPU grids are allocated, so it does not add to them, but it must fit.

On Windows (WDDM drivers), **GPU allocations also count against the system commit limit** (RAM + page file). A 20 GB grid needs 20 GB of free commit on top of what the rest of the system uses. If large grids fail with out-of-memory errors while the GPU still has room, enlarge the page file.
