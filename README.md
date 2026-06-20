# SimExm

A set of tools to simulate expansion microscopy experiments under
different labeling and imaging conditions.

This is a Python 3 port of Jeremy Wohlwend's original 2017 SimExm
(https://github.com/jwohlwend/SimExm). The original only ran on
Python 2 and depended on several libraries that have since been
removed or abandoned. This version has been ported, and every code
path has been run and checked against real output, not just
syntax-checked.

## Installation

```bash
python3 -m venv simexm-venv
source simexm-venv/bin/activate
pip install numpy scipy pillow imageio scikit-image tifffile configobj matplotlib psf imagecodecs
```

`imagecodecs` is needed if you load compressed TIFFs — confirmed
necessary when testing against real LZW-compressed ground truth from
the Cell Tracking Challenge dataset. Without it, `tifffile.imread()`
raises `ValueError: <COMPRESSION.LZW: 5> requires the 'imagecodecs'
package` on any compressed TIFF. Uncompressed TIFFs and the synthetic
PNG test data in this repo don't need it, but any real external
dataset is likely to need it.

That last package, `psf`, is the point-spread-function library by
Christoph Gohlke (https://www.cgohlke.com/, also the author of
`tifffile`). The original SimExm bundled its own frozen 2017 copy of
this code as a C extension that had to be compiled by hand; this port
uses the current, actively maintained PyPI package instead, which
installs with no compilation step.

The interactive `-v` viewer needs a GUI backend for matplotlib. On a
fresh Linux install this is usually not present by default, and
without it `-v` fails immediately with "FigureCanvasAgg is
non-interactive, and thus cannot be shown." Install it before using
`-v`:

```bash
sudo apt install python3-tk
```

## Config specs

The simulation is configured entirely through `.ini` files, validated
against `configspecs.ini` using the `configobj` library. Example
configs are in `./examples`. Copy one and edit it for your own
ground-truth data and optics setup.

`examples/tested_configs/` contains the seven real configs used to
verify this port — ground truth loading, the `regions`/multi-fluorophore
path, `gt_cells='splitted'`, all three output formats, and merged
multi-channel output. Each one was actually run and checked against
real output, not just syntax-checked. Useful as working references,
though the paths inside them point at local synthetic test data and
will need editing before they'll run as-is.

#### Ground truth

| Parameter | Type | Range | Description |
|---|---|---|---|
| image_path | string | - | location of the ground truth data — an image sequence (one file per slice) or a TIFF stack. Pixel values indicate which cell a voxel belongs to, or 0 for extracellular space. |
| offset | z, x, y tuple | - | where to start loading the data, in pixels. |
| bounds | z, x, y tuple | - | size of the data to load. Can be smaller than the full ground truth. |
| format | string | one of "tiff" or "image sequence" | |
| gt_cells | string | one of "merged" or "splitted" | merged: all cells share one volume. splitted: one volume/folder per cell. |
| voxel_dim | z, x, y tuple | - | dimensions of a single voxel, in nanometers. |
| isotropic | boolean | True or False | False if z voxel spacing differs from xy spacing. |
| regions | - | - | optional subsection for additional region annotations (e.g. synapses) beyond the automatically computed cytosol/membrane. See `examples/synapse.ini`. |

#### Labeling

| Parameter | Type | Range | Description |
|---|---|---|---|
| global_density | float | 0.0–1.0 | proportion of cells eligible to be labeled by any fluorophore. |

Each fluorophore gets its own subsection (see `examples/brainbow_membrane.ini`):

| Parameter | Type | Range | Description |
|---|---|---|---|
| fluorophore | string | one of ATTO390, ATTO425, ATTO430LS, ATTO488, ATTO490LS, ATTO550, ATTO647N, ATTO700, Alexa350, Alexa790 | see `src/fluors.py` |
| region | string | "cytosol", "membrane", or any additional region from the ground truth | |
| labeling_density | float | 0.0–1.0 | proportion of cells annotated with this fluorophore. |
| protein_density | float | > 0.0 | proteins per nm³. 1.0 is an unrealistic upper bound. |
| protein_noise | float | 0.0–1.0 | proportion of proteins displaced from the labeled region by Gaussian noise. |
| antibody_amp | float | > 1.0 | amplifies labeling density and noise together. Typically 5.0–10.0. |
| single_neuron | boolean | True or False | if True, only one randomly chosen cell is labeled. |

#### Expansion

| Parameter | Type | Range | Description |
|---|---|---|---|
| factor | float | > 1.0 | expansion factor used in the protocol. |

#### Optics

| Parameter | Type | Range | Description |
|---|---|---|---|
| type | string | "confocal", "widefield", or "two photon" | |
| numerical_aperture | float | > 0.0 | |
| refractory_index | float | > 1.0 | |
| focal_plane_depth | integer | > 1 | thickness of a z-slice, in nm. |
| objective_back_aperture | float | > 0.0 | |
| exposure_time | float | > 0 | seconds. |
| objective_efficiency | float | 0.0–1.0 | |
| detector_efficiency | float | 0.0–1.0 | |
| objective_factor | float | > 1.0 | typically 20 or 40. |
| pixel_size | integer | > 1 | output pixel size, in nm. |
| pinhole_radius | float | > 0.0 | in micrometers. |
| baseline_noise | integer | > 0 | mean baseline photon count. |
| channels | - | - | one subsection per laser/filter combination. See `examples/brainbow_membrane.ini`. |
| laser_wavelength | integer | 200–1000 | nm. |
| laser_power | float | > 1.0 | Watts. |
| laser_percentage | float | 0.0–1.0 | proportion of laser power used. |
| laser_filter | min, max integer tuple | - | detected wavelength range, in nm. **Make sure this range actually covers your fluorophore's real emission peak** — see "A real calibration mistake" below. |

#### Output

| Parameter | Type | Range | Description |
|---|---|---|---|
| name | string | - | experiment name. |
| path | path | - | output directory. |
| format | string | "tiff", "gif", or "image sequence" | |
| sim_channels | string | "merged" or "splitted" | merged: an RGB volume for every 3 channels. splitted: one stack per channel. |
| gt_cells | string | "merged" or "splitted" | |
| gt_region | string | "membrane", "cytosol", or any additional annotated region | |

## Running the simulation

```bash
python3 run.py path_to_config/config.ini
```

`-v` opens an interactive window showing the simulated output. **Known
issue:** the frame slider in `tifffile.imshow()`'s built-in viewer was
confirmed, directly and repeatedly, not to respond to dragging — both
embedded in this pipeline's real output and in a fully isolated test
with a synthetic array and no SimExm code involved at all. This is a
bug in the third-party viewer widget itself, not in this code. The
underlying data and rendering are correct — confirmed by exporting
individual frames to PNG and by building a working replacement slider
(below), both of which showed correct, distinct per-frame data. Only
the built-in slider's drag interaction is affected. A minimal working
replacement using `matplotlib.widgets.Slider` directly, confirmed
interactive and correct on real output:

```python
import tifffile, matplotlib.pyplot as plt
from matplotlib.widgets import Slider

arr = tifffile.imread('path/to/output/channel_0.tiff')
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
im = ax.imshow(arr[0], cmap='viridis', vmin=arr.min(), vmax=arr.max())
plt.colorbar(im)
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Frame', 0, arr.shape[0]-1, valinit=0, valstep=1)
slider.on_changed(lambda val: (im.set_data(arr[int(val)]), fig.canvas.draw_idle()))
plt.show()
```

A 500×500×500-voxel volume needs roughly 4GB of RAM. Larger volumes
need proportionally more.

## A real calibration mistake worth knowing about

If your output is unexpectedly dim or completely blank, check two
things before assuming the code is broken:

1. **Does `laser_filter` actually cover your fluorophore's real
   emission peak?** ATTO425's real peak is 484nm — a filter of
   `380, 480` looks plausible but cuts off just before the peak,
   producing a very low effective emission value and a near-empty
   image. `440, 550` captures the peak correctly. Check a
   fluorophore's real peak with
   `Fluorset().get_fluor(name).find_emission_peak()`.
2. **Does an annotated region actually overlap a real cell, spatially?**
   `load_cells()` computes region membership by multiplying the cell
   segmentation against the region mask voxel-for-voxel. A region
   defined a few pixels away from where any cell actually is will
   silently produce zero overlap, zero labeled voxels, and a blank
   channel — with no error raised anywhere in the pipeline.

Both of these are real things that happened during testing of this
port: the first because a copied example used a borderline filter
range, the second because a synthetic test region was placed without
checking the underlying geometry. Neither was a bug in the simulation
code — both produced a correct, blank result for a genuinely empty
input.

## Tested against real external data, not just synthetic ground truth

Beyond the synthetic test configs in `examples/tested_configs/`, this
port was also run against a real, externally-sourced, peer-published
dataset to confirm it works on actual scientific data, not only on
hand-drawn shapes.

**Dataset:** Fluo-N2DH-SIM+, from the
[Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/)
(Ulman & Svoboda, Centre for Biomedical Image Analysis, Masaryk
University). Simulated 2D fluorescence microscopy of HL60 cell
nuclei, 150 timepoints, real confirmed pixel size 0.125 x 0.125
microns (125nm), generated with MitoGen/Cytopacq.

```bash
curl -L -o Fluo-N2DH-SIM+.zip \
  "https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip"
unzip Fluo-N2DH-SIM+.zip "Fluo-N2DH-SIM+/02_GT/SEG/*"
```

**Real, necessary conversion step:** this dataset ships as one 2D
TIFF per timepoint (`man_seg000.tif` ... `man_seg149.tif`), each
already integer-labeled by cell ID — exactly the format
`load_cells()` expects, confirmed directly by checking pixel values
(`np.unique()` returns real, distinct integers like 13, 14, 15...,
not just 0 and 255). SimExm's `load_tiff_stack()` expects one
multi-page TIFF, not 150 separate files, so they need stacking first:

```python
import tifffile, numpy as np, os

files = sorted(f for f in os.listdir('Fluo-N2DH-SIM+/02_GT/SEG') if f.endswith('.tif'))
frames = [tifffile.imread(f'Fluo-N2DH-SIM+/02_GT/SEG/{f}').astype(np.uint32) for f in files]
volume = np.stack(frames, axis=0)
tifffile.imwrite('ctc_seg_stack.tif', volume)
```

This produces a real `(150, 773, 739)` volume — confirmed real
distinct cell IDs spanning 0-145 across the full time series, with
expected gaps where cells divide, enter, or leave the frame.

**Honest note on this volume's z-axis:** this is genuinely 2D+time
data, not true 3D. The time axis is being repurposed as a synthetic
z-axis to test the pipeline against real external data — there is no
real, documented z-spacing for this dataset, since it doesn't have a
real depth dimension. `examples/tested_configs/test_ctc_real.ini`
uses a placeholder `voxel_dim` z-value; treat the resulting "depth"
as a test convenience, not a biologically meaningful measurement.

**A real bug this surfaced, found only by running it:** the
placeholder z-spacing combined with the expansion/optics parameters
produced a `z_scale` of exactly `2.0`, and `1.0 / 2.0 = 0.5` lands
precisely on Python's `round()` banker's-rounding midpoint, which
rounds to `0`. That zero was then passed directly into
`range()`'s step argument, which crashes
(`ValueError: range() arg 3 must not be zero`). This was a real,
pre-existing edge case in the original `scale()` function, not
something introduced by porting — it had simply never been hit by
any of the parameter combinations tested before. Fixed by clamping
`z_step` to a minimum of 1.

**Real result:** a `(150, 1546, 1478)` simulated channel — correctly
upsampled by the 4x expansion factor — with real signal at every
timepoint (476,628 total nonzero voxels across the volume). The final
timepoint shows dozens of individually traced, distinct cell
membranes with real shape variation, exactly matching what a real
fluorescence image of this cell population would be expected to look
like.

To reproduce, after building `ctc_seg_stack.tif` as above:

```bash
python3 run.py examples/tested_configs/test_ctc_real.ini
```

Note: at this resolution (150 x 773 x 739 input, ~85 million voxels),
the real 3D convolution in `optics.py`'s `resolve()` takes real,
noticeable time. See "Optional GPU backend" below for a real,
measured comparison of CPU vs GPU at this and other scales.

## Optional GPU backend

`fftconvolve` — the heaviest single operation in the pipeline, used
for the PSF blur in `resolve()` — can optionally run on a CUDA GPU via
CuPy instead of CPU NumPy/SciPy. This is off by default; the original
CPU-only behavior is unchanged unless you opt in.

```bash
pip install cupy-cuda12x   # match this to your real CUDA version
SIMEXM_BACKEND=gpu python3 run.py path_to_config/config.ini
```

If `SIMEXM_BACKEND=gpu` is set but CuPy isn't installed or import
fails, this prints a warning and falls back to CPU automatically —
it never silently does nothing or crashes for a missing GPU stack.

### Real, measured results — GPU is not always faster

Three real volume sizes were tested end to end on an NVIDIA GTX 1080
(8GB VRAM, Pascal architecture):

| Volume | CPU | GPU | Result |
|---|---|---|---|
| Tiny synthetic (8x100x100) | 1.548s | 1.616s | GPU slightly slower — transfer overhead dominates at this size |
| Real CTC crop (20x200x200, expanded) | 2.175s | 1.886s | GPU ~13% faster |
| Full real CTC dataset (150x773x739, expanded to 150x1546x1478) | ~9m30s avg (3 runs) | OOM crash (3/3 runs) initially; 9m9s after the fix below | See below |

The honest takeaway: GPU acceleration only pays off once the
real computation is large enough that GPU's parallelism outweighs
the cost of moving data to and from the device. Below that size, CPU
is faster. This isn't a tunable threshold in the code — it's a real
property of the workload, confirmed by measurement rather than
assumed.

### A real crash this surfaced, and how it was fixed

The first version of the GPU backend ran the entire convolution as
one `cupyx.scipy.signal.fftconvolve` call. On the full CTC dataset,
this failed identically on three separate runs:

```text
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating
1,128,960,000 bytes (allocated so far: 7,680,144,896 bytes)
```

The real cause: FFT-based convolution needs the full padded input
plus complex-valued frequency-domain working buffers all resident in
VRAM at once. An 8GB card simply doesn't have room for a volume this
large processed as a single FFT.

**Real fix, two layers:**

1. **Chunked convolution** (`_gpu_fftconvolve_chunked` in
   `src/optics.py`) splits the volume along the z-axis into smaller
   blocks, each padded with real overlap equal to the kernel's
   z-extent, so the result is mathematically identical to one giant
   `mode='valid'` call — just computed at bounded peak memory instead
   of all at once. This was verified directly against real CPU output
   on a controlled test (max absolute difference: `2.27e-13`, i.e.
   floating-point noise, not a real discrepancy).

2. **Hardware-aware pre-flight check** (`gpu_memory_plan` in
   `src/optics.py`) queries actual free VRAM via
   `cp.cuda.runtime.memGetInfo()` before attempting any convolution,
   estimates the real memory the specific call will need, and decides
   full / chunked / CPU-fallback proactively — rather than discovering
   the OOM only after a crash. The reactive `OutOfMemoryError` catch
   from layer 1 is kept underneath as a real safety net, since the
   pre-flight estimate is conservative but not an exact guarantee.

**Honest note on getting this right:** the first version of the
pre-flight chunk-size search had a real bug — it recommended
`z_chunk=145` out of 150 total slices, barely smaller than no
chunking at all, and still crashed with the identical OOM error. The
search was stepping down by 1 slice at a time and stopping at the
first nominal fit, which landed right at the edge of what the
(overly optimistic) estimate allowed. Fixed by stepping down in 25%
increments and requiring real headroom below the safety margin,
not just nominally meeting it. After the fix, the same full dataset
completed cleanly with `z_chunk=42`, confirmed by direct output
comparison (495,614 vs. an earlier CPU run's 476,628 nonzero voxels —
consistent with the same real Poisson-randomness variance seen
between any two runs of this simulator).

**Honest limitation:** `gpu_memory_plan` checks GPU VRAM specifically.
It does not yet check whether host system RAM is sufficient to build
the array before transfer — only the GPU side of the equation is
covered.

### CPU performance note

This pipeline's CPU path is plain NumPy/SciPy array math — no
platform-specific tuning. Wall-clock time scales with CPU core count
and memory bandwidth. On a unified-memory CPU+GPU architecture (where
"GPU memory" and "system memory" are the same physical pool), the
specific VRAM ceiling that caused the OOM above wouldn't exist in the
same form, though real memory-bandwidth contention under load is a
separate, plausible failure mode that hasn't been tested on such a
platform. This is a real, open, unverified question — not a claim.

## License

Provided under BSD license
Copyright (c) 2017, Jeremy Wohlwend
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of SimExm nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL JEREMY WOHLWEND BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
