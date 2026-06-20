# Changelog

All notable changes in the Python 3 port of SimExm are documented here.

### Added

**Python 3 port, all six files**
- `cpu_gups`-style file-by-file conversion of `fluors.py`, `labeling.py`,
  `load.py`, `optics.py`, `output.py`, and `run.py`, plus the package's
  `__init__.py`.
- Every `print` statement (13 total across all files) converted to a
  function call.
- `xrange` replaced with `range` everywhere it appeared (`load.py`,
  `output.py`).
- Deprecated `np.int` (removed in NumPy 1.24+) replaced with the
  builtin `int` in `optics.py`.
- Integer-division padding bug fixed in `optics.py`: `d/2` always
  returns a float in Python 3, but `np.pad`'s pad-width argument
  requires an integer. Changed to `d//2`.

**Dead dependency replacements**
- `scipy.misc.imread` (removed from SciPy since 1.2) replaced with
  `imageio.v3.imread` in `load.py`, preserving the original's forced
  32-bit grayscale behavior with an explicit cast.
- `scipy.misc.imresize` (removed from SciPy since 1.0) replaced with
  `skimage.transform.resize` in `output.py`, using `order=0` (nearest-
  neighbor) and `preserve_range=True` so integer cell-ID labels are
  not blended or rescaled.
- `images2gif.writeGif` (unmaintained since ~2017) replaced with
  `imageio.v3.imwrite` in `output.py`.
- The bundled, frozen 2017 `psf.c`/`psf.py` C extension replaced with
  the current, actively maintained `psf` package on PyPI by the same
  original author (Christoph Gohlke), installable with no compilation
  step.
- `tifffile.imsave`, renamed to `tifffile.imwrite` in current
  `tifffile` versions, updated in `output.py`. Found only by actually
  running the ported code — not visible from reading the source.

**Other real fixes, unrelated to Python 2/3 compatibility**
- `fluors.py`: `get_emission_file()` and `get_excitation_file()` left
  file handles open with no `with` block. Both now use `with`, so the
  file closes even if parsing raises an exception.
- `fluors.py`: data files were split strictly on `\r` (classic
  Mac-style line endings only). Replaced with `str.splitlines()`,
  which handles `\r`, `\n`, and `\r\n` correctly.
- `output.py`: the hand-rolled zero-padding logic for output filenames
  (`image_001.png`, etc.) replaced with an f-string format spec,
  removing a special case the original needed for index 0.

### Fixed

**Real bugs found only through actual execution, not by reading the code**

- **`optics.py`, `from fluors import Fluorset`**: this file lives
  inside the `src` package and needs a package-relative import,
  `from .fluors import Fluorset`. The flat import worked in isolated
  testing but failed the moment the real pipeline was run through
  `run.py`'s actual `src.optics` import path.

- **`optics.py`, `mean_photons()` zero-photon bug — the one real
  defect that mattered.** The function rounded its result to an
  integer before returning: `int(np.round(detected_photons))`. For
  any fluorophore/laser/filter combination producing a mean photon
  rate below 0.5 per protein per exposure — which is physically
  realistic, not unusual — this silently rounded to 0, and the
  rendered image was completely blank with no error or warning
  anywhere in the pipeline. Fixed by returning the float rate
  directly and letting `np.random.poisson()` (already used downstream
  in `resolve()`) sample it correctly. A rate of 0.23 photons/protein
  now correctly produces mostly-zero, occasionally-nonzero Poisson
  draws, instead of being floored to zero on every single voxel.

### Verified

Every code path below was run with real (synthetic) input data and
checked against real output, not just confirmed to import or pass a
syntax check:

- `load_merged_gt()` — real ground truth with multiple cells, correct
  membrane/cytosol voxel counts confirmed by direct convolution-based
  edge detection test.
- `load_splitted_gt()` — real per-cell directory structure, voxel
  counts confirmed to exactly match the equivalent merged-format
  result.
- The `regions` / multi-fluorophore labeling path (`synapse.ini`-style
  configs) — real second annotation volume, both channels confirmed
  with real nonzero signal after fixing a synthetic-data geometry
  mistake (see "A real calibration mistake" in the README).
- `optics.py`'s `resolve()`, including the real `psf` package
  integration — confirmed end to end with nonzero, correctly-shaped
  output, and the `mean_photons()` fix confirmed by checking that an
  improved (correctly peak-matched) emission filter produced a real,
  measurable increase in detected signal (19,324 vs. 16,821 nonzero
  voxels on the same ground truth).
- All three output formats (`tiff`, `gif`, `image sequence`) — `gif`
  confirmed as a valid multi-frame animation via direct `imageio`
  read-back, not just file existence.
- `sim_channels = "merged"` (the `merge()` 3-channel RGB packing
  function) — confirmed with two real channels plus the expected
  zero-filled padding channel, verified per-channel nonzero counts.
- The `-v` / `show_output` interactive display path — confirmed the
  window opens and renders correct data with the correct colorbar
  range. The built-in `tifffile.imshow()` frame slider does not
  respond to dragging in this environment (a third-party widget
  bug, isolated and reproduced independently of any SimExm code); a
  replacement slider built directly with `matplotlib.widgets.Slider`
  was confirmed fully interactive on the same data.

No part of this port was declared complete based on import success or
syntax checking alone. Every item above produced a traceback, an
unexpected result, or a confirmed correct result from actually running
it before being called done.

### Added (real external data validation)

- Tested the full pipeline against Fluo-N2DH-SIM+, a real, externally
  sourced dataset from the Cell Tracking Challenge (Ulman & Svoboda,
  Masaryk University) — 150 real timepoints of integer-labeled cell
  segmentation, not synthetic hand-drawn shapes.
- `imagecodecs` added to install instructions — required to read the
  real LZW-compressed TIFFs in this dataset; not needed for the
  synthetic PNG test data already in the repo, but needed for most
  real external datasets.

### Fixed

- **`optics.py`, `scale()`, `z_step` rounding to zero.** Found only
  by running the pipeline against real external data with a z-spacing
  value that happened to produce a `z_scale` of exactly `2.0`. Python's
  `round(1.0 / 2.0)` lands on `round()`'s banker's-rounding midpoint
  and returns `0`, which was then passed directly into `range()`'s
  step argument — `range()` raises `ValueError: range() arg 3 must
  not be zero` on a zero step. This was a real, pre-existing edge
  case in the original code, not introduced by the Python 3 port;
  it had simply never been exercised by any prior test's parameter
  combination. Fixed by clamping `z_step` to a minimum of 1.

### Added (optional GPU backend)

- Optional CUDA GPU backend for `fftconvolve` via CuPy, opt-in via
  `SIMEXM_BACKEND=gpu`. Off by default; CPU behavior unchanged unless
  explicitly enabled. Falls back to CPU automatically with a warning
  if CuPy isn't installed.
- Real, measured CPU vs GPU comparison across three volume sizes on
  an NVIDIA GTX 1080 (8GB VRAM): GPU is slower than CPU on small
  volumes (transfer overhead dominates), ~13% faster on a medium real
  crop of the CTC dataset, and required the fixes below to complete
  at all on the full real CTC dataset.
- Chunked GPU convolution (`_gpu_fftconvolve_chunked`) for volumes too
  large to fit in VRAM as a single FFT. Verified mathematically
  identical to CPU output on a controlled test (max difference
  2.27e-13, floating-point noise only).
- Hardware-aware pre-flight memory check (`gpu_memory_plan`) that
  queries real free VRAM via `cp.cuda.runtime.memGetInfo()` before
  attempting a convolution and decides full/chunked/CPU proactively,
  reusing the real `/proc/meminfo`-based introspection pattern from
  `parallelArchitect/ewc_moe_atari`'s `gb10/memory.py`.

### Fixed

- **Real OOM crash on the full CTC dataset, confirmed on 3/3 runs**:
  `cupy.cuda.memory.OutOfMemoryError` allocating 1.1GB more with
  7.68GB already used, on an 8GB card. Root cause: a single
  `fftconvolve` call needs the full padded input plus complex-valued
  FFT working buffers all in VRAM at once. Fixed by the chunked
  convolution and pre-flight check above.
- **Real bug in the first version of the pre-flight chunk-size
  search**: it recommended `z_chunk=145` out of 150 slices — barely
  smaller than no chunking — and still hit the identical OOM error.
  The search stepped down by 1 slice and stopped at the first nominal
  fit, landing at the edge of an overly optimistic estimate. Fixed by
  stepping down in 25% increments and requiring real headroom below
  the safety margin. Confirmed fixed: the same full dataset completed
  cleanly afterward with `z_chunk=42`.
