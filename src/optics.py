# Provided under BSD license
# Copyright (c) 2017, Jeremy Wohlwend
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of SimExm nor the names of its contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL JEREMY WOHLWEND BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
optics.py

Set of methods to handle the simulation of the optical process.
Implements confocal light microscopy.

Ported to Python 3:
  - Two print statements converted to function calls.
  - d/2, w/2, h/2 padding widths changed to d//2, w//2, h//2 --
    Python 2's `/` floor-divided integers by default; Python 3's
    `/` always returns float, which np.pad's integer pad-width
    argument rejects. // is the explicit floor-division operator
    in both versions, so this is a like-for-like fix, not a
    behavior change.
  - np.int (removed in NumPy 1.24+) replaced with the builtin int.
  - import psf now refers to the actively-maintained PyPI package
    (pip install psf, by the same author, Christoph Gohlke) rather
    than the bundled 2017 psf.c/psf.py extension. The call signature
    -- psf.PSF(psf.ISOTROPIC | psf_type[type], **args) with shape,
    dims, ex_wavelen, em_wavelen, num_aperture, refr_index,
    pinhole_radius, magnification -- matches the current package's
    documented API as of its 2026-01-17 release, confirmed against
    its real GitHub README example. NOT independently verified by
    running it -- this sandbox's network allowlist blocks pypi.org,
    so `pip install psf` and a real test run need to happen on your
    machine. If psf_volume() raises a TypeError or AttributeError
    on first run, the most likely cause is a parameter name or
    constant that changed between the 2017 version SimExm was
    originally written against and the current one -- check
    psf.PSF.__doc__ and psf.__all__ for the current names.
"""

import numpy as np
import psf
from .fluors import Fluorset

# GPU backend support, following the same pattern confirmed in microsim
# (tlambert03/microsim, src/microsim/schema/backend.py): try CuPy first,
# fall back to plain NumPy/SciPy if it's unavailable or fails to import.
# Default is CPU -- this preserves all existing behavior exactly. Set the
# SIMEXM_BACKEND environment variable to "gpu" to opt in.
import os

_BACKEND = os.environ.get("SIMEXM_BACKEND", "cpu").lower()
_GPU_AVAILABLE = False

if _BACKEND == "gpu":
    try:
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve as _gpu_fftconvolve
        _GPU_AVAILABLE = True
        print("optics.py: using GPU backend (CuPy)")
    except ImportError as e:
        print(f"optics.py: SIMEXM_BACKEND=gpu requested but CuPy is not "
              f"available ({e}). Falling back to CPU.")
        _BACKEND = "cpu"

if not _GPU_AVAILABLE:
    from scipy.signal import fftconvolve as _cpu_fftconvolve


def _read_system_meminfo():
    """
    Real system memory introspection, following the same pattern as
    parallelArchitect/ewc_moe_atari's gb10/memory.py: /proc/meminfo is
    ground truth, MemAvailable (not MemFree, not MemTotal) is the
    correct usable-memory signal. This function is hardware-agnostic
    Linux introspection -- no GPU-specific or platform-specific logic,
    so it reads correctly on Pascal, GB10, or any other Linux system.
    Returns None if /proc/meminfo isn't available (e.g. non-Linux).
    """
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        meminfo[key] = int(parts[1])
                    except ValueError:
                        pass
        return meminfo.get("MemAvailable", None)
    except (FileNotFoundError, OSError):
        return None


def _estimate_fftconvolve_bytes(a_shape, b_shape, dtype_size=8):
    """
    Real, conservative estimate of peak memory a single fftconvolve
    call needs: the padded FFT working size (next-fast-length padding
    of a_shape + b_shape - 1 along each axis), held as complex values
    (2x dtype_size) for both the forward-transformed input and kernel,
    plus the output array. This intentionally overestimates rather
    than underestimates -- a false "won't fit" that triggers chunking
    is a much smaller cost than a real OOM crash mid-run.
    """
    fft_shape = tuple(a + b - 1 for a, b in zip(a_shape, b_shape))
    fft_elements = 1
    for dim in fft_shape:
        fft_elements *= dim
    # Two complex-valued buffers (input spectrum + kernel spectrum)
    # plus one real-valued output buffer, real-valued input copy.
    complex_bytes = fft_elements * dtype_size * 2
    return 2 * complex_bytes + fft_elements * dtype_size


def gpu_memory_plan(a_shape, b_shape, dtype_size=8, safety_margin=0.7):
    """
    Real pre-flight decision: given the real shapes about to be
    convolved, query actual free GPU memory right now (not a static
    guess) and decide whether the full convolution will fit, whether
    it needs chunking, or whether GPU isn't viable at all for this
    call -- before attempting it, not after it crashes.

    safety_margin caps usable VRAM at 70% of what's actually free by
    default, since CuPy's memory pool, CUDA context overhead, and any
    other process sharing the GPU all consume real memory beyond the
    convolution buffers themselves -- the three real OOM crashes
    tonight all happened with VRAM that looked nominally sufficient
    until the actual allocator hit its limit.

    Returns a dict: {'fits': bool, 'estimated_bytes': int,
    'free_bytes': int, 'recommended_z_chunk': int or None}.
    """
    estimated = _estimate_fftconvolve_bytes(a_shape, b_shape, dtype_size)

    if not _GPU_AVAILABLE:
        return {"fits": False, "estimated_bytes": estimated,
                "free_bytes": 0, "recommended_z_chunk": None,
                "reason": "GPU backend not active"}

    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    usable_bytes = int(free_bytes * safety_margin)

    if estimated <= usable_bytes:
        return {"fits": True, "estimated_bytes": estimated,
                "free_bytes": free_bytes, "recommended_z_chunk": None,
                "reason": "fits within safety margin"}

    # Real chunk-size search: find the largest z_chunk whose estimated
    # memory fits with real headroom. The first version of this search
    # stepped down by 1 and stopped at the first nominal fit, which in
    # practice landed on a chunk barely smaller than the full volume
    # (confirmed by a real crash: z_chunk=145 out of 150 still OOM'd
    # identically to no chunking at all). Step down by meaningful
    # fractions instead, and require each candidate to fit with a
    # real additional margin beyond the base safety_margin, since the
    # estimate itself was shown to be optimistic on real hardware.
    az = a_shape[0]
    kz = b_shape[0]
    best_chunk = None
    candidate = az
    while candidate >= 1:
        chunk_a_shape = (min(candidate + kz - 1, az),) + a_shape[1:]
        chunk_estimate = _estimate_fftconvolve_bytes(
            chunk_a_shape, b_shape, dtype_size)
        # Extra real margin on top of the base safety_margin: the
        # per-chunk estimate is checked against the SAME usable_bytes
        # budget, but chunk sizes near the full volume have shown to
        # still OOM in practice, so require fitting at half the
        # already-reduced usable budget before accepting a chunk size.
        if chunk_estimate <= usable_bytes * 0.5:
            best_chunk = candidate
            break
        # Step down by 25% each time rather than by 1, so the search
        # actually reaches meaningfully smaller sizes instead of
        # crawling one slice at a time near the unfit full size.
        candidate = max(1, int(candidate * 0.75))

    return {"fits": False, "estimated_bytes": estimated,
            "free_bytes": free_bytes,
            "recommended_z_chunk": best_chunk,
            "reason": "exceeds safety margin, chunking required"
                      if best_chunk else
                      "no viable chunk size fits in available VRAM"}


def fftconvolve(a, b, mode):
    """
    Backend-dispatching fftconvolve. Identical signature and behavior
    to scipy.signal.fftconvolve on CPU. On GPU, checks real available
    VRAM via gpu_memory_plan() before attempting the call, choosing
    full convolution, a sized chunk, or falling back to CPU based on
    actual current GPU state -- not a static assumption about the
    hardware. This is the proactive version of the chunked fallback:
    rather than catching OutOfMemoryError after a real crash, it
    queries cp.cuda.runtime.memGetInfo() first and decides before
    attempting anything, so a confirmed-too-large convolution never
    needs to fail once before recovering.

    Still keeps the reactive OutOfMemoryError catch as a real safety
    net underneath the pre-flight check -- the estimate is
    conservative but not guaranteed exact (CuPy/cuFFT plan caching
    and fragmentation can shift real usage), so both layers exist on
    purpose, the same defense-in-depth principle as a thermal cutoff
    backing up a software throttle.
    """
    if _GPU_AVAILABLE:
        plan = gpu_memory_plan(a.shape, b.shape, dtype_size=a.dtype.itemsize)
        if not plan["fits"] and plan["recommended_z_chunk"] is None:
            print(f"optics.py: pre-flight check found no viable GPU "
                  f"chunk size ({plan['estimated_bytes']/1e9:.2f}GB "
                  f"needed, {plan['free_bytes']/1e9:.2f}GB free) -- "
                  f"falling back to CPU for this call")
            return _cpu_fftconvolve(a, b, mode)
        if not plan["fits"]:
            print(f"optics.py: pre-flight check recommends chunked "
                  f"convolution (z_chunk={plan['recommended_z_chunk']}, "
                  f"{plan['estimated_bytes']/1e9:.2f}GB needed vs "
                  f"{plan['free_bytes']/1e9:.2f}GB free)")
            return _gpu_fftconvolve_chunked(
                a, b, mode, z_chunk=plan["recommended_z_chunk"])
        try:
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            result_gpu = _gpu_fftconvolve(a_gpu, b_gpu, mode=mode)
            result = cp.asnumpy(result_gpu)
            del a_gpu, b_gpu, result_gpu
            cp.get_default_memory_pool().free_all_blocks()
            return result
        except cp.cuda.memory.OutOfMemoryError:
            # Real safety net: the pre-flight estimate said this should
            # fit, but it didn't. Re-run the planner now (real free
            # memory may differ from the pre-flight check, e.g. other
            # allocations since then) and use whatever chunk size it
            # actually recommends, rather than guessing a fixed number.
            cp.get_default_memory_pool().free_all_blocks()
            print("optics.py: GPU OOM despite pre-flight check passing "
                  "(estimate was optimistic) -- recomputing a real "
                  "chunk plan and retrying")
            retry_plan = gpu_memory_plan(
                a.shape, b.shape, dtype_size=a.dtype.itemsize,
                safety_margin=0.4)
            z_chunk = retry_plan["recommended_z_chunk"] or 1
            return _gpu_fftconvolve_chunked(a, b, mode, z_chunk=z_chunk)
    return _cpu_fftconvolve(a, b, mode)


def _gpu_fftconvolve_chunked(a, b, mode, z_chunk=8):
    """
    Real, memory-bounded fallback for GPU convolution on volumes too
    large to fit in VRAM as a single FFT. Only supports mode='valid',
    which is the only mode this codebase actually uses (confirmed:
    resolve() always calls fftconvolve(..., 'valid')).

    Splits the input volume `a` into z-axis chunks, each padded with
    `kz` extra slices of real overlap on both sides (kz = kernel's
    z-extent), convolves each padded chunk independently, then crops
    and concatenates the valid-mode results. This is mathematically
    identical to one giant 'valid'-mode convolution -- the overlap
    exists specifically so each chunk has the real neighboring data
    its edge voxels need, rather than approximating with zero-padding.

    z_chunk controls how many output z-slices are computed per chunk;
    smaller values use less peak VRAM but make more separate GPU calls.
    """
    if mode != 'valid':
        raise NotImplementedError(
            "_gpu_fftconvolve_chunked only supports mode='valid' "
            "(the only mode this codebase uses); got mode=%r" % mode)

    kz, ky, kx = b.shape
    az, ay, ax = a.shape
    out_z = az - kz + 1
    if out_z <= 0:
        raise ValueError("kernel z-extent %d >= input z-extent %d; "
                          "no valid output possible" % (kz, az))

    b_gpu = cp.asarray(b)
    chunks = []
    z = 0
    while z < out_z:
        this_chunk = min(z_chunk, out_z - z)
        # Real overlap: need input rows [z, z + this_chunk + kz - 1)
        # to produce 'valid' output rows [z, z + this_chunk)
        a_slice = a[z : z + this_chunk + kz - 1]
        a_gpu = cp.asarray(a_slice)
        result_gpu = _gpu_fftconvolve(a_gpu, b_gpu, mode=mode)
        chunks.append(cp.asnumpy(result_gpu))
        del a_gpu, result_gpu
        cp.get_default_memory_pool().free_all_blocks()
        z += this_chunk

    return np.concatenate(chunks, axis=0)

def resolve(labeled_volumes, volume_dim, voxel_dim, expansion_params, optics_params):
    """
    Resolves the labeled volumes with the given optics parameters.
    Performs photon count calculation, convolution with a point spread function,
    baseline noise and rescaling.

    Args:
        labeled_volumes: dict fluorophore (string) -> volume (numpy 3D uint32 array)
            dictionary containing the volumes to resolve
        volume_dim: (z, x, y) integer tuple
            dimensions of each volume in number of voxels
        voxel_dim: (z, x, y) integer tuple
            dimensions of a voxel in nm
        expansion_parameters: dict
            dicitonary containing the expansion parameters
        optics_parameters: dict
            dicitonary containing the optics parameters
    Returns:
        volumes: list of numpy 3D uint8 arrays
            as list contianing a volume resolved for each channel
    """
    #Create volume
    volumes = []
    #Resolve each channel one by one
    #Make sure they're sorted by name for consistency
    channels = sorted(optics_params['channels'].keys())
    for channel in channels:
        print("Resolving {}".format(channel))
        channel_vol = np.zeros(volume_dim, np.uint32)
        channel_params = optics_params['channels'][channel]
        #Each fluorophore may produce photons in the given channel
        for fluorophore in labeled_volumes:
            #Merge parameters
            params = optics_params.copy()
            params.update(channel_params)
            #Compute photon count
            mean_photon = mean_photons(fluorophore, **params)
            #Only spend time convolving if the fluorophore is not orthogonal to
            #this channel
            if mean_photon > 0:
                fluo_vol = np.zeros(volume_dim, np.float64)
                Z, X, Y = np.nonzero(labeled_volumes[fluorophore])
                photons = np.random.poisson(mean_photon, size = len(Z)).astype(np.uint32)
                photons = np.multiply(labeled_volumes[fluorophore][Z, X, Y], photons)
                np.add.at(fluo_vol, (Z, X, Y), photons)
                #Convolve with point spread
                psf_vol = psf_volume(voxel_dim, expansion_params['factor'], fluorophore, **params)
                (d, w, h) = psf_vol.shape
                #Resize fluo_vol for convolution
                fluo_vol = np.pad(fluo_vol, ((d // 2, d // 2), (w // 2, w // 2), (h // 2, h // 2)), 'reflect')
                channel_vol += np.round(fftconvolve(fluo_vol, psf_vol, 'valid')).astype(np.uint32)
        #Add noise
        channel_vol += baseline_volume(channel_vol.shape, **optics_params)
        #Optical scaling
        channel_vol = scale(channel_vol, voxel_dim, expansion_params['factor'], **optics_params)
        #Normalize
        channel_vol = normalize(channel_vol)
        volumes.append(channel_vol)

    return volumes

def psf_volume(voxel_dim, expansion, fluorophore, laser_wavelength, numerical_aperture,\
                refractory_index, pinhole_radius, objective_factor, type, **kwargs):
    """
    Creates a point spread volume, using the given parameters.

    Args:
        voxel_dim: (z, x, y) tuple
            the dimensions of a voxel in nm
        expansion: float
            the expansion factor
        fluorophore: string
            the fluorophore that is excited
        laser_wavelength: int
            the wavelength of the excitation laser in nm
        numerical_aperture:
            the numerical aperture of the system
        refractory_index:
            the refractory index, tipically 1.33 for ExM
        pinhole_radius:
            the pinhole raids in microns
        objective_factor: float
            objective factor of the microscope, tipically 0, 20 or 40
        type: string
            one of 'confocal', 'widefield' or 'two photon'
    Returns:
        psf_vol: numpy 3d float64 array
            the point spread function
    """
    fluorset = Fluorset()
    f = fluorset.get_fluor(fluorophore)
    #Map to psf type
    psf_type = {'confocal': psf.CONFOCAL, 'widefield': psf.WIDEFIELD, 'two photon': psf.TWOPHOTON}
    #Upper bound for psf size
    precision = 16
    z, x, y = np.array(voxel_dim) * expansion
    #Arguments
    back_projected_radius = pinhole_radius / float(objective_factor)
    #Fill args in dictionary
    args = dict(shape=(precision, precision), dims=(precision * z * 1e-3, precision * x * 1e-3),\
                ex_wavelen=laser_wavelength, em_wavelen=f.find_emission_peak(),\
                num_aperture=numerical_aperture, refr_index=refractory_index,\
                pinhole_radius=back_projected_radius, magnification = 1)
    #Compute psf
    psf_vol = psf.PSF(psf.ISOTROPIC | psf_type[type], **args)
    return psf_vol.volume()

def baseline_volume(volume_dim, baseline_noise, **kwargs):
    """
    Creates a volume of baseline photon noise, using a poisson distribution
    multiplied by a gaussian to mimic the light focus towards the center of the image.

    Args:
        volume_dim: (z, x, y) integer tuple
            the size of the volume to create
        baseline_noise: integer
            the mean number of photons of the poisson distribution
    Returns:
        out: numpy 3D uint32 array
            a volume of basline photon noise
    """
    (d, w, h) = volume_dim
    indices = np.indices((d, w, h), dtype=np.float64)
    gaussian = np.exp(-((indices[1] - w / 2.0)**2 / (0.5 * w**2) + (indices[2] - h / 2.0)**2 / (0.5 * h**2)))
    out = np.round(np.multiply(np.random.poisson(baseline_noise, size=volume_dim), gaussian))
    return out.astype(np.uint32)

def normalize(volume):
    """
    Normalizes the volume to the [0, 255] range by dividing by the maximum value.

    Args:
        volume: 3D numpy array, np.uint32
            the volume to normalize
    Returns:
        normalized:3D numpy array, np.uint8
            the normalized volume
    """
    max = np.amax(volume)
    if max == 0:#Fixes dividing by 0 error if nothing in the volume
        return volume.astype(np.uint8)

    normalized = volume * (255.0 / max)
    normalized = np.round(normalized).astype(np.uint8)
    return normalized

def scale(volume, voxel_dim, expansion, objective_factor,
          pixel_size, focal_plane_depth, **kwargs):
    """
    Scales the output volume with the appropriate optics parameters
    using nearest neighbour interpolation.

    Args:
        volume: numpy 3D array (z, x, y)
            the volume to scale
        voxel_dim: (z, x, y) tuple
            the dimensions of a voxel in nm
        expansion: float
            the expansion factor
        objective_factor: float
            objective factor of the microscope, tipically 0, 20 or 40
        pixel_size: integer
            the size of a pixel in the microscope, in nm
        focal_plane_depth: integer
            the thickness of a slice in nm
    Returns:
        out: numpt 3D array
            the scaled volume in all three axis
    """
    xy_step = float(pixel_size) / (voxel_dim[1] * expansion * objective_factor)
    #This removes rounding artifacts, by binning with an integer number of pixels
    if xy_step < 1:
        xy_scale = 1.0 / xy_step
        xy_scale = np.round(xy_scale)
        print("Warning: the ground truth resolution is too low to resolve the volume with the desired expansion. Attempting a work around.")
    else:
        xy_step = np.round(xy_step)
        xy_scale = 1.0 / xy_step
    z_scale = voxel_dim[0] * expansion / float(focal_plane_depth)
    z_step = np.round(1.0 / z_scale).astype(int)
    # z_step can round down to exactly 0 (e.g. via banker's rounding when
    # 1.0/z_scale lands exactly on 0.5), which crashes range()'s step
    # argument below. A step of 0 has no meaningful interpretation here
    # anyway -- 1 is the smallest valid step (every slice kept).
    z_step = max(z_step, 1)
    out = []
    for i in range(0, volume.shape[0], z_step):
        X, Y = np.nonzero(volume[i, :, :])
        values = volume[i, X, Y]
        #Rescale and round
        X = np.floor(xy_scale * X).astype(np.int64)
        Y = np.floor(xy_scale * Y).astype(np.int64)
        #Create new image
        d, w, h = np.ceil(np.array(volume.shape) * xy_scale)
        im = np.zeros((int(w), int(h)), np.uint32)
        #Adding poisson if the volume is expanded, to avoid grid-like images
        if xy_scale > 1:
            X = np.clip(X + np.random.poisson(int(xy_scale), size = len(X)), 0, w - 1)
            Y = np.clip(Y + np.random.poisson(int(xy_scale), size = len(Y)), 0, h - 1)
        #This allows to add to repetition of the same index
        np.add.at(im, (X.astype(np.uint64), Y.astype(np.uint64)), values)
        out.append(im)
    return np.array(out)

def mean_photons(fluorophore, exposure_time, objective_efficiency,\
                detector_efficiency, objective_back_aperture, objective_factor, \
                laser_wavelength, laser_filter, laser_power, laser_percentage, **kwargs):
    """
    Computes the mean number of detected photons for a given fluorophore and laser parameters

    Args:
        fluorophore: string
            the fluorophore to measure
        exposure_time: float
            amoung of time that the laser is shined on the specimen, in seconds
        objective_efficiency: float
            the efficiency of the microscope's objective
        detector_efficiency: float
            the efficiency of the microsocpe's detector
        objective_back_aperture: float
            name says it all
        objective_factor:
            objective lense, tipically 0, 20 or 40
        laser_wavelength: integer
            the wavelength of the laser, in nm
        laser_filter: integer list of length 2
            wavelength_min and wavelength_maximum detected
        laser_power: float
            the laser power
        laser_percentage: float
            proportion of power to use
    Returns:
        detected_photons: integer
            the mean number of detected photons per fluorophore protein
    """
    fluorset = Fluorset()
    #Get fluorophore data
    f = fluorset.get_fluor(fluorophore)
    qy, ext_coeff = f.get_quantum_yield(), f.get_extinction_coefficient()
    excitation, emission = f.find_excitation(laser_wavelength), f.find_emission(laser_filter)
    #Compute parameters
    laser_radius = float(objective_back_aperture) / objective_factor
    laser_intensity = laser_power * laser_percentage / (np.pi * laser_radius**2)# [Watts / (m ^ 2)]
    CONSTANT = 0.119626566 # in m^3 * kg * s^{-1} Avogadro's number * Planck's constant * speed of light
    #Get mean photon count
    emitted_photons = excitation * qy * (ext_coeff * 1e2) * exposure_time *\
                      laser_intensity * (laser_wavelength * 1e-9) / (1e3 * CONSTANT)
    detected_photons = emitted_photons * emission * objective_efficiency * detector_efficiency
    # Return the float rate directly -- this is a mean photon count per protein
    # per exposure, used as the lambda parameter in np.random.poisson() inside
    # resolve(). Rounding to int here discards sub-1 values (e.g. 0.23 -> 0),
    # killing all signal when the per-protein photon rate is realistically below 1.
    # Poisson sampling of a float rate is the physically correct behavior.
    return detected_photons
