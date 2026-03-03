#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pinocchio painting module (MAP mode or PARTICLE mode) + HOD galaxies + (particle) miscentering + profiles.

This script is part of CosmoPostProcess and has been written by Roberto Ingrao.

================================================================================
READ ME (high-level mental model)
================================================================================

This script takes a Pinocchio halo catalogue (NPZ) and does *one of two workflows*:

1) MAP mode (if you pass --massmap-file):
   - Reads a HEALPix map of Pinocchio *particle counts* (a density / count map).
   - Converts those counts to mass (Msun/h) using the same particle-mass formula
     used in the original pipeline (p_mass_h).
   - "Paints" each halo onto a HEALPix map as a projected surface-density profile,
     out to paint_rvir_factor * R200m.
     - If CosmoPostProcess baryonification tools are available, halos above
       --m-bary-min get a baryonified profile.
     - Otherwise, halos are painted with a pure NFW profile (fallback).
   - Optional: paint halos at higher NSIDE (--halo-nside), then *mass-preserving*
     degrade back to the base NSIDE of the Pinocchio map before adding them.
   - Outputs:
     - halo-only map (bnfw / NFW / baryonified) at base NSIDE
     - composite map = Pinocchio mass map + halo-painted mass map
     - galaxy catalogue NPZ

2) PARTICLE mode (if you do NOT pass --massmap-file):
   - For each halo, samples a 3D NFW distribution of particles (in comoving Mpc/h)
     out to --particle-rmax-com.
   - Ensures total particle mass equals M200m*h using the same p_mass_h.
   - Optional boundary buffers:
     - you can load --npz-prev and/or --npz-next
     - buffer halos are only painted if they are within ±neighbor_buffer_com (cMpc/h)
       of the target shell boundaries (from filename z-range if available).
   - Paint galaxies for *all painted halos* (target + buffer).
   - Miscentering (particle mode only, NO P_mem):
     - For each target halo, selects galaxies within miscenter_radius_factor * R200_com
       and finds a miscentered position via a smoothed RA/Dec histogram threshold.
   - Profiles (particle mode only):
     - Only target halos above --profile-mass-cut are profiled.
     - Builds Σ(R) using *all painted particles* in a cylinder of radius --profile-rmax
       and depth --depth-cyl.
   - Outputs:
     - galaxies catalogue NPZ
     - miscentered profiles NPZ
     - miscentered halo catalogue NPZ

================================================================================
Implementation notes (important for new users)
================================================================================

- The script is optimized for HPC usage:
  - it forces BLAS/OpenMP to 1 thread per process, because multiprocessing is used.
  - it prints periodic "[Progress]" lines so SLURM .out logs show activity.
  - it supports adjustable pool chunksizes.

- A lot of data are held in global variables (prefixed _G_ or similar). This is
  deliberate:
  - Python multiprocessing needs worker processes to access large arrays without
    repeatedly pickling/copying them.
  - By setting globals once in the parent, then forking, workers can see them
    cheaply (especially with start_method=fork).

- Colossus cosmology is configured per-process because:
  - workers may run in separate processes and need cosmology initialized there too.
  - some environments crash if $HOME is not set (fix included).

Requested behaviors implemented
-------------------------------
✔ Two modes
  - MAP mode if --massmap-file is provided (healpy FITS):
      * Read Pinocchio particle count map, convert to Msun/h using original p_mass_h
      * Paint halo mass onto map out to paint_rvir_factor * R200m (default 3)
        using baryonification above --m-bary-min (if CosmoPostProcess available) else NFW fallback
      * NEW: optional high-resolution halo painting (--halo-nside) then mass-preserving
        degradation back to base NSIDE before adding to Pinocchio.
      * Save: bnfw map (base NSIDE), composite map, galaxy catalog

  - PARTICLE mode if --massmap-file is NOT provided:
      * Paint each halo as 3D particles in comoving Mpc/h out to --particle-rmax-com (default 30)
        conserving total mass to M200m*h using the SAME p_mass_h formula as the original code
      * Extra boundary buffer (particle mode only):
          - Optionally load --npz-prev and --npz-next
          - Paint buffer halos only if they lie within ±neighbor_buffer_com (default 50 cMpc/h)
            of the target shell radial boundaries (from filename z-range if available, else from z in target file)
      * Paint galaxies for ALL painted halos (target + buffer)
      * Miscentering (particle mode only, NO P_mem):
          - Use galaxies within miscenter_radius_factor * R200_com (default 1.0)
          - Keep smoothed RA/Dec histogram bins above (miscenter_nmult * median) (n from sbatch)
      * Profiles (particle mode only):
          - Apply mass cut ONLY for profiles: --profile-mass-cut
          - Σ(R) computed from *all painted particles* inside a cylinder:
              radius = --profile-rmax (default 10 cMpc/h)
              depth  = --depth-cyl   (default 100 cMpc/h)
          - Parallelized over selected halos
      * Save: profiles NPZ, galaxies catalog NPZ, miscentered halo catalog NPZ

✔ Outputs are labeled with shell redshift range (e.g. z1.000_1.200) to avoid overwriting.

✔ SLURM / monitoring compatibility:
  - Accepts: --progress-to-stdout, --no-progress, --pool-chunksize
  - Uses tqdm by default (to stdout) + periodic "[Progress]" lines (so it shows in .out)

✔ Colossus HOME crash fix:
  - If HOME environment variable is missing (seen on some batch setups), it is set to SLURM_SUBMIT_DIR or /tmp
"""

import os

# ----------------------------------------------------------
# Force single-threaded BLAS / OpenMP inside each process
# ----------------------------------------------------------
# Why this exists:
# - We use multiprocessing (many Python processes).
# - If NumPy/SciPy/BLAS tries to use multiple threads per process, you can
#   massively oversubscribe CPU cores and slow everything down or crash nodes.
# - Setting these to "1" ensures each process stays single-threaded internally.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Colossus can crash if HOME is missing (observed on some batch nodes).
# Some HPC environments do not define HOME for compute jobs.
# We set it to something safe and writable to avoid Colossus persistence issues.
if "HOME" not in os.environ or not os.environ["HOME"]:
    os.environ["HOME"] = os.environ.get("SLURM_SUBMIT_DIR", "/tmp")

import re
import sys
import time
import argparse
import warnings
import numpy as np
import multiprocessing as mp

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u_ast

from PinocchioParamfile import params_file

from scipy.ndimage import gaussian_filter
from scipy import special as spec
from scipy.spatial import cKDTree
from scipy import integrate

from tqdm.auto import tqdm

# healpy is only required for MAP mode
# We try-import it so that particle mode can still run without healpy installed.
try:
    import healpy as hp
except Exception:
    hp = None

from colossus.cosmology import cosmology as col_cosmo
from colossus.halo import concentration, profile_nfw


# ==========================================================
# Global constants & cosmology
# ==========================================================

# Default simulation / catalog constants matching the Pinocchio setup (elb).
# These are later changed in the main() function with the parsed values 
BoxSize = 3380.0  # Mpc/h
Om0 = 0.32
H0 = 67.0
GridSize = 3072.

# Astropy cosmology object for distances
cosmo = FlatLambdaCDM(Om0=Om0, H0=H0)
h = H0 / 100.0

# Default scratch base for outputs on G100
DEFAULT_BASE_OUTDIR = "/g100_scratch/userexternal/ringrao0/Pinocchio"


# ==========================================================
# Optional baryonification (MAP mode)
# ==========================================================
# These imports may fail if CosmoPostProcess is not available.
# If they fail, we simply fall back to pure NFW painting in MAP mode.
try:
    from CosmoPostProcess.Baryonification.models_and_displacement import (
        contract_dmo_profile_with_ac,
        build_sigmaR_provider,
    )
except Exception:
    contract_dmo_profile_with_ac = None
    build_sigmaR_provider = None


# ==========================================================
# Progress / monitoring
# ==========================================================
# _NO_PROGRESS:
#   - if True: disables tqdm entirely (useful in very noisy log environments)
# _PROGRESS_TO_STDOUT:
#   - if True: tqdm writes to stdout explicitly (SLURM logs typically capture stdout)
_NO_PROGRESS = False
_PROGRESS_TO_STDOUT = True


def _tqdm(it, total=None, desc="", unit="it"):
    """
    Wrapper around tqdm so we can globally enable/disable progress bars.
    Returns:
      - original iterator if _NO_PROGRESS True
      - tqdm-wrapped iterator otherwise
    """
    if _NO_PROGRESS:
        return it
    return tqdm(
        it,
        total=total,
        desc=desc,
        unit=unit,
        file=sys.stdout if _PROGRESS_TO_STDOUT else None,
        dynamic_ncols=True,
        mininterval=0.5,
    )


def _maybe_print_progress(tag, done, total, t_last, every_sec=20.0):
    """
    Periodic "heartbeat" printer for SLURM .out logs.

    Why not rely only on tqdm?
    - Some clusters buffer output or don't show tqdm nicely.
    - Printing a simple line periodically makes it obvious the job is alive.

    Parameters
    ----------
    tag : str
        Name of the phase, e.g. "Loop1: paint".
    done : int
        Units completed.
    total : int
        Total units.
    t_last : float
        Last time we printed.
    every_sec : float
        Print interval in seconds.
    """
    now = time.time()
    if now - t_last >= every_sec:
        frac = (done / total) if total else 0.0
        print(f"[Progress] {tag}: {done}/{total} ({frac*100:.2f}%)", flush=True)
        return now
    return t_last


# ==========================================================
# HOD (unchanged physics)
# ==========================================================
# The HOD functions implement the same behavior as in the original code.
# We keep them exactly the same; we only add explanation comments.

def hod_centrals(m, params_hod):
    """
    Probability of having a central galaxy as a function of halo mass.

    Notes:
    - Uses an error function "soft step" in log10(M).
    - EPS_CEN_MIN + log10M_cut logic avoids tiny, numerically irrelevant tails.

    Parameters
    ----------
    m : array-like or float
        Halo mass (assumed in Msun, consistent with rest of script).
    params_hod : dict
        HOD parameters.

    Returns
    -------
    pcen : ndarray
        Probability in [0,1] for each halo mass.
    """
    EPS_CEN_MIN = 1e-2
    m = np.atleast_1d(m).astype(np.float64, copy=False)
    M_min_cen = float(params_hod["M_min_cen"])
    sigma_log_M = float(params_hod["sigma_log_M"])
    if sigma_log_M <= 0:
        raise ValueError("sigma_log_M must be positive.")
    log10M_cut = M_min_cen + sigma_log_M * spec.erfinv(2.0 * EPS_CEN_MIN - 1.0)
    log10m = np.log10(np.clip(m, 1e-300, None))
    mask = log10m >= log10M_cut
    pcen = np.zeros_like(m, dtype=np.float64)
    pcen[mask] = 0.5 * (1.0 + spec.erf((log10m[mask] - M_min_cen) / sigma_log_M))
    return pcen


def hod_satelites(m, z_cl, params_hod):
    """
    Expected satellite occupation and a stochastic "intrinsic" scatter.

    Returns:
      psat : mean satellite expectation (pre-scatter)
      noise: Gaussian perturbation with std = sigm_intr_sat * psat
    The calling code then draws Ns ~ Poisson(max(psat+noise, 0)).
    """
    M_min_sat = params_hod["M_min_sat"]
    M_1_sat = params_hod["M_1_sat"]
    alpha = params_hod["alpha"]
    epsilon = params_hod["epsilon"]
    pivot_z0 = params_hod["pivot_z0"]
    sigm_intr_sat = params_hod["sigm_intr_sat"]

    m = np.atleast_1d(m)
    z_cl = np.atleast_1d(z_cl)

    mask = m > 10 ** M_min_sat
    psat = np.zeros_like(m)
    psat[mask] = (
        ((m[mask] - 10**M_min_sat) / (10**M_1_sat - 10**M_min_sat)) ** alpha
        * ((1 + z_cl[mask]) / (1 + pivot_z0)) ** epsilon
    )
    noise = np.random.normal(0, sigm_intr_sat * psat)
    return psat, noise


# Default HOD parameters (frozen values).
# If you want to explore HOD variations, change these numbers (but that changes physics).
params_hod = {
    "M_min_cen": 11.07316643,
    "sigma_log_M": 0.6,
    "M_min_sat": 11.07316643,
    "M_1_sat": 12.19094465,
    "alpha": 0.87877214,
    "epsilon": 0.95529752,
    "pivot_z0": 1.25,
    "sigm_intr_sat": 0.20782678,
}


# ==========================================================
# Label + shell boundary helpers
# ==========================================================
def shell_label_from_npz_path(npz_path: str) -> str:
    """
    Extract label like 'z1.000_1.200' from filename '..._z1.000_1.200.npz'.

    Why this matters:
    - Many jobs run multiple shells; label prevents output overwriting.
    - If the pattern isn't present, we fallback to the file basename to still
      produce unique-ish names.

    Returns
    -------
    label : str
    """
    base = os.path.basename(npz_path)
    m = re.search(r"_z([0-9.]+)_([0-9.]+)\.npz$", base)
    if m:
        z1, z2 = m.group(1), m.group(2)
        return f"z{z1}_{z2}"
    return os.path.splitext(base)[0]


def shell_edges_from_filename(npz_path: str):
    """
    Return (z_lo, z_hi) if filename has _zX_Y.npz, else None.

    Used to:
    - determine comoving boundary distances for buffer selection in particle mode.

    Returns
    -------
    (z_lo, z_hi) : tuple[float,float] or None
        Sorted so z_lo <= z_hi.
    """
    base = os.path.basename(npz_path)
    m = re.search(r"_z([0-9.]+)_([0-9.]+)\.npz$", base)
    if not m:
        return None
    z1, z2 = float(m.group(1)), float(m.group(2))
    return (min(z1, z2), max(z1, z2))


def resolve_outdir(base_outdir: str, outdir: str) -> str:
    """
    Resolve output directory:
    - if outdir is absolute: use it directly
    - if outdir is relative: place it under base_outdir

    This lets you pass --outdir outputs and still write to scratch
    without hardcoding full paths.
    """
    if outdir is None:
        outdir = "outputs"
    outdir = str(outdir)
    if os.path.isabs(outdir):
        return outdir
    return os.path.join(str(base_outdir), outdir)


def comoving_distance_Mpch(z_val: float) -> float:
    """
    Comoving distance in Mpc/h.

    Astropy returns comoving distance in Mpc (physical units of Mpc).
    We multiply by h to convert Mpc -> Mpc/h.
    """
    return float(cosmo.comoving_distance(float(z_val)).to(u_ast.Mpc).value) * h


# ==========================================================
# Geometry + miscentering helpers
# ==========================================================
def wrap_ra_deg(ra):
    """
    Force RA into [0, 360) degrees.
    Helpful after adding/subtracting offsets.
    """
    return (np.asarray(ra) + 360.0) % 360.0


def unwrap_ra_around_center(ra_deg, ra_center_deg):
    """
    Shift RA values so they're continuous around ra_center_deg.

    This is essential when the halo is near RA=0/360:
      - e.g. halo at 359.9 deg and galaxy at 0.1 deg are close on sky,
        but numerically they appear far if you do naive averaging.
    The trick:
      - compute delta in (-180, 180], then reconstruct around center.
    """
    ra = np.asarray(ra_deg, dtype=float)
    d = (ra - ra_center_deg + 180.0) % 360.0 - 180.0
    return ra_center_deg + d


def unitvec_from_radec_deg(ra_deg, dec_deg):
    """
    Convert RA/Dec (deg) to 3D unit vectors on the sphere.

    Supports scalars or arrays:
      - scalar inputs return shape (3,)
      - array inputs return shape (N,3)

    Coordinate convention:
      x = cos(dec) cos(ra)
      y = cos(dec) sin(ra)
      z = sin(dec)
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    if x.ndim == 0:
        return np.array([float(x), float(y), float(z)])
    return np.column_stack([x, y, z])


def tangent_basis_from_unit(u):
    """
    Build an orthonormal basis (e1, e2, e3) where:
      - e3 is the line-of-sight direction u (unit vector)
      - e1,e2 span the tangent plane perpendicular to u

    Used in particle painting:
      - we sample offsets (dx,dy,dz) in a coordinate frame aligned with u,
        then convert them to global Cartesian coordinates.

    Steps:
      - choose a reference vector "ref" not too parallel to u
      - e1 = normalize(cross(ref, u))
      - e2 = normalize(cross(u, e1))
      - e3 = u
    """
    u = np.asarray(u, dtype=float)
    u /= np.linalg.norm(u)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, u)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])

    e1 = np.cross(ref, u)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2, u


def sigma_z_photo(z_true):
    """
    Photometric redshift scatter model sigma_z(z).

    Used when generating observed galaxy redshifts around halo redshift.
    """
    z_true = float(z_true)
    return 0.041 - 0.046 * z_true + 0.034 * z_true**2


def compute_miscentered_center_threshold(
    ra_sel_deg,
    dec_sel_deg,
    ra_center_deg,
    bins=60,
    sigma=3.0,
    n_mult=1.0,
):
    """
    Miscentering estimator (no P_mem version):

    Algorithm:
      1) Unwrap RA around the halo center to avoid 0/360 discontinuity.
      2) Make a 2D histogram in (RA, Dec).
      3) Smooth the histogram with a Gaussian filter.
      4) Compute median of smoothed map.
      5) Select "active" bins where smoothed value > n_mult * median.
      6) Return weighted average of bin centers (weights = smoothed values).

    Returns
    -------
    (ra_mis, dec_mis) : float, float
        If insufficient points or no bins pass threshold, returns (nan, nan).
    """
    ra_sel = np.asarray(ra_sel_deg, dtype=float)
    dec_sel = np.asarray(dec_sel_deg, dtype=float)

    m = np.isfinite(ra_sel) & np.isfinite(dec_sel)
    if np.count_nonzero(m) < 3:
        return np.nan, np.nan

    ra_u = unwrap_ra_around_center(ra_sel[m], float(ra_center_deg))
    dec_u = dec_sel[m]

    hist, xedges, yedges = np.histogram2d(ra_u, dec_u, bins=int(bins))
    hist_s = gaussian_filter(hist, sigma=float(sigma))
    hist_s[hist_s <= 0] = 1e-12

    med = float(np.median(hist_s))
    thr = float(n_mult) * med
    aoi = hist_s > thr
    if not np.any(aoi):
        return np.nan, np.nan

    idx = np.argwhere(aoi)
    x_cent = 0.5 * (xedges[idx[:, 0]] + xedges[idx[:, 0] + 1])
    y_cent = 0.5 * (yedges[idx[:, 1]] + yedges[idx[:, 1] + 1])
    w = hist_s[aoi]

    ra_mis_u = float(np.average(x_cent, weights=w))
    dec_mis = float(np.average(y_cent, weights=w))
    return float(wrap_ra_deg(ra_mis_u)), float(dec_mis)


# ==========================================================
# Log-log interpolation + projection (MAP mode baryonification)
# ==========================================================
def interp_loglog(x_eval, x_grid, y_grid):
    """
    Log-log interpolation:
      - takes x_grid, y_grid defined on positive domain
      - interpolates y(x) in log-space for smoother power-law behavior

    Used to evaluate Σ(R) at per-pixel radii in MAP mode.
    """
    xg = np.asarray(x_grid, dtype=float)
    yg = np.asarray(y_grid, dtype=float)
    xe = np.asarray(x_eval, dtype=float)
    logx = np.log(np.clip(xg, 1e-300, None))
    logy = np.log(np.clip(yg, 1e-300, None))
    lx = np.log(np.clip(xe, xg[0], xg[-1]))
    return np.exp(np.interp(lx, logx, logy))


def project_sigma_from_rho_vec(r3d_kpch, rho3d, R_eval_kpch):
    """
    Project 3D density rho(r) to surface density Σ(R):

      Σ(R) = 2 ∫_R^∞ rho(r) r dr / sqrt(r^2 - R^2)

    Units:
      r and R are in kpc/h physical
      rho is in (Msun/h) / (kpc/h)^3
      => Σ is in (Msun/h) / (kpc/h)^2 (physical)

    Implementation:
      - vectorized integrand for multiple R values at once
      - Simpson integration over r-grid
    """
    r = np.asarray(r3d_kpch, dtype=float)
    rho = np.asarray(rho3d, dtype=float)
    R = np.asarray(R_eval_kpch, dtype=float)

    rr = r[None, :]
    RR = R[:, None]

    diff = rr * rr - RR * RR
    mask = diff > 0.0
    denom = np.sqrt(np.where(mask, diff, 1.0))

    integrand = (rho[None, :] * rr) / np.maximum(denom, 1e-300)
    integrand[~mask] = 0.0

    return 2.0 * integrate.simpson(integrand, x=r, axis=1)


# ==========================================================
# NEW: Mass-preserving degrade from high-res to base NSIDE
# ==========================================================
def degrade_halo_mass_map_ring(bnfw_high, nside_high, nside_low):
    """
    Degrade a *mass* map from nside_high -> nside_low by summing child pixels.

    Important:
      - This is NOT an "average" or "intensity" degrade.
      - Pixel values represent *mass per pixel*, so to preserve total mass we SUM.

    Assumes:
      - Input and output are RING ordering.
      - nside_high is a power-of-two multiple of nside_low (HEALPix hierarchy).

    Implementation:
      - reorder RING -> NESTED (children contiguous)
      - reshape to (npix_low, children_per_parent)
      - sum along children axis
      - reorder back to RING
    """
    if nside_high == nside_low:
        return np.asarray(bnfw_high, dtype=np.float64, copy=True)

    if nside_high % nside_low != 0:
        raise ValueError(f"nside_high ({nside_high}) must be a multiple of nside_low ({nside_low}).")
    factor = nside_high // nside_low
    # factor must be a power of two (HEALPix hierarchy)
    if factor & (factor - 1):
        raise ValueError(f"nside_high/nside_low = {factor} must be a power of 2.")

    npix_low = hp.nside2npix(nside_low)
    npix_high = hp.nside2npix(nside_high)
    children_per_parent = factor * factor
    if npix_high != npix_low * children_per_parent:
        raise ValueError("Inconsistent npix for given nside_high/nside_low.")

    # Work in NESTED, where children of a parent are contiguous
    m_nested = hp.reorder(np.asarray(bnfw_high, dtype=np.float64), r2n=True)
    m_nested = m_nested.reshape(npix_low, children_per_parent)
    m_low_nested = np.sum(m_nested, axis=1)
    m_low_ring = hp.reorder(m_low_nested, n2r=True)
    return m_low_ring


# ==========================================================
# Colossus cosmology setup (avoid persistence if supported)
# ==========================================================
def prepare_colossus_cosmology():
    """
    Configure Colossus cosmology used to compute NFW properties:
      - concentration
      - R200m via NFWProfile

    Why this is separated:
      - In multiprocessing each worker process must run this once, since
        Colossus has per-process global state.
      - Also ensures HOME is set (crash fix).

    The persistence handling is defensive across Colossus versions:
      - newer versions accept persistence="" to disable persistence
      - some accept None
      - some accept no arg at all
    """
    # Colossus can crash if HOME is missing; ensure it's set to something writable.
    if "HOME" not in os.environ or not os.environ["HOME"]:
        os.environ["HOME"] = os.environ.get("SLURM_SUBMIT_DIR", "/tmp")

    cosmo_pars = dict(flat=True, H0=H0, Om0=Om0, Ob0=0.049, sigma8=0.811, ns=0.965)
    if "myCosmo" not in col_cosmo.cosmologies:
        col_cosmo.addCosmology("myCosmo", cosmo_pars)

    # Robust across Colossus versions:
    try:
        col_cosmo.setCosmology("myCosmo", persistence="")
    except Exception:
        try:
            col_cosmo.setCosmology("myCosmo", persistence=None)
        except Exception:
            col_cosmo.setCosmology("myCosmo")


def compute_c200m(M200m, z_i):
    """
    Compute NFW concentration c200m using Colossus (Duffy08 model).

    Note:
      - Colossus expects masses in Msun/h.
      - M200m in this script is "Msun" (not divided by h), so we convert:
          M_halo_h = M200m * h

    Fallback:
      - if Colossus returns non-finite or non-positive, we use c=4.0.
    """
    M_halo_h = float(M200m) * h  # Msun/h
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = float(concentration.concentration(M_halo_h, mdef="200m", z=float(z_i), model="duffy08"))
    if (not np.isfinite(c)) or (c <= 0):
        c = 4.0
    return c


def halo_R200_comoving_Mpch(M200m, z_i, c200m):
    """
    Compute halo R200m and an NFWProfile object.

    Colossus radii are returned in kpc/h *physical*.
    We convert as:
      R200_kpch = physical kpc/h
      R200_phys_Mpch = R200_kpch / 1000
      R200_com_Mpch = R200_phys_Mpch * (1+z)

    Returns
    -------
    R200_kpch : float
        kpc/h physical
    R200_com_Mpch : float
        Mpc/h comoving
    prof : colossus.halo.profile_nfw.NFWProfile
        Colossus NFW profile instance for this halo.
    """
    M_halo_h = float(M200m) * h
    prof = profile_nfw.NFWProfile(M=M_halo_h, c=float(c200m), z=float(z_i), mdef="200m")
    R200_kpch = float(prof.RDelta(float(z_i), "200m"))         # kpc/h physical
    R200_phys_Mpch = R200_kpch / 1000.0                        # Mpc/h physical
    R200_com_Mpch = R200_phys_Mpch * (1.0 + float(z_i))        # comoving Mpc/h
    return R200_kpch, R200_com_Mpch, prof


# ==========================================================
# NFW-as-particles sampling (PARTICLE mode)
# ==========================================================
def nfw_inverse_cdf_radii_phys_Mpch(rs_phys_Mpch, rmax_phys_Mpch, n, rng):
    """
    Sample radii r from an NFW mass distribution truncated at rmax (3D).

    Theory:
      M(<r) ∝ g(x), where x=r/rs and g(x)=ln(1+x) - x/(1+x).
    So the CDF is g(x)/g(x_max).

    Practical approach:
      - build a dense x-grid (geometric spacing)
      - compute g(x), normalize to CDF
      - invert CDF via 1D interpolation

    Units:
      input rs_phys_Mpch and rmax_phys_Mpch are physical Mpc/h
      output radii are physical Mpc/h
    """
    rs = float(rs_phys_Mpch)
    rmax = float(rmax_phys_Mpch)
    n = int(n)
    if rs <= 0 or rmax <= 0 or n <= 0:
        return np.empty(0, dtype=float)

    x_max = rmax / rs
    if not np.isfinite(x_max) or x_max <= 0:
        return np.empty(0, dtype=float)

    # If extremely small, approximate uniform-in-volume within rmax
    if x_max <= 1e-6:
        u = rng.random(n)
        return (rmax * u ** (1.0 / 3.0)).astype(float)

    x_grid = np.geomspace(1e-6, x_max, 4096)
    g = np.log1p(x_grid) - x_grid / (1.0 + x_grid)
    g_max = float(g[-1])
    if not np.isfinite(g_max) or g_max <= 0:
        return np.empty(0, dtype=float)

    cdf = g / g_max
    u = rng.random(n)
    x = np.interp(u, cdf, x_grid)
    return x * rs


# ==========================================================
# Global state for multiprocessing workers
# ==========================================================
# This script relies heavily on module-level globals so worker processes can
# access large arrays and shared config without passing them every call.
#
# Naming convention:
#   _G_*    : globals for loop1 (painting) stage
#   _PART_* : globals for particle KDTree/profile stage
#   _GAL_*  : globals for galaxy KDTree/miscentering stage
#
# This is a common HPC Python pattern.

# Painting (loop1) globals
_G_MODE = None  # "map" or "particle"

# Per-halo catalog arrays (combined target+buffer in particle mode)
_G_RA = None
_G_DEC = None
_G_Z = None
_G_MASS = None
_G_IS_TARGET = None
_G_N_TARGET = 0

# Particle mass (Msun/h) computed from box/grid and cosmology
_G_P_MASS_H = None

# MAP mode globals (only defined/used in MAP mode workers)
_G_NSIDE = None         # halo painting NSIDE (could be higher than base nside)
_G_MASK_MAP = None      # halo painting mask (at halo NSIDE)
_G_OMEGA_PIX = None     # halo pixel solid angle (at halo NSIDE)
_G_BNFW_MAP = None      # unused placeholder; kept for completeness

# MAP bary globals
_G_SIGMAR_PROVIDER = None
_G_DISABLE_BARYONIFICATION = False  # map mode toggle
_G_COSMO_BARY = None

# Numeric config used by workers
_G_PAINT_RVIR_FACTOR = 1.0
_G_M_BARY_MIN = 1e13
_G_N_SIGMA_GRID = 256
_G_N_R3D = 256
_G_RMIN_SAT = 1e-4
_G_SEED_BASE = 12345

# Halo properties (filled in parent from worker returns)
# We allocate arrays for ALL painted halos, even though some later steps apply
# only to target halos. This keeps the code simpler and avoids special cases.


def _loop1_init_worker():
    """
    Pool initializer for loop1 workers.
    Runs once per worker process.

    Purpose:
      - Ensure Colossus cosmology is configured in each process.
    """
    prepare_colossus_cosmology()


def _paint_galaxies_for_halo(i_halo, ra_c, dec_c, z_i, M200m, R200_com_Mpch, prof, rng):
    """
    Generate a simple HOD galaxy population for a halo.

    Output arrays are appended and returned as NumPy arrays.

    Implementation details:
      - Central galaxy:
          - included with probability pcen(M)
          - placed exactly at halo center
          - observed redshift includes photo-z scatter
      - Satellite galaxies:
          - expected mean psat(M,z) + Gaussian "intrinsic" scatter
          - Ns ~ Poisson(max(psat + noise, 0))
          - satellite radii are drawn with weight proportional to Σ(R)*2πR dR
            using the NFW surface density from Colossus
          - satellites placed around the halo center with small-angle approximation
            (theta = R/Dc)
          - observed redshifts include photo-z scatter
    """
    gal_hidx = []
    gal_ra = []
    gal_dec = []
    gal_zobs = []

    # -------------------------
    # Central galaxy
    # -------------------------
    pcen = float(hod_centrals(M200m, params_hod)[0])
    if rng.random() < pcen:
        z_obs = float(z_i + rng.normal(0.0, sigma_z_photo(z_i)))
        gal_hidx.append(i_halo)
        gal_ra.append(float(ra_c))
        gal_dec.append(float(dec_c))
        gal_zobs.append(z_obs)

    # -------------------------
    # Satellite galaxies
    # -------------------------
    psat, noise = hod_satelites(M200m, z_i, params_hod)
    psat_eff = max(float(psat[0] + noise[0]), 0.0)
    Ns = int(rng.poisson(psat_eff))
    if Ns <= 0 or (not np.isfinite(R200_com_Mpch)) or R200_com_Mpch <= 0:
        return (
            np.asarray(gal_hidx, dtype=np.int64),
            np.asarray(gal_ra, dtype=np.float64),
            np.asarray(gal_dec, dtype=np.float64),
            np.asarray(gal_zobs, dtype=np.float64),
        )

    Dc_com = comoving_distance_Mpch(z_i)
    if Dc_com <= 0:
        return (
            np.asarray(gal_hidx, dtype=np.int64),
            np.asarray(gal_ra, dtype=np.float64),
            np.asarray(gal_dec, dtype=np.float64),
            np.asarray(gal_zobs, dtype=np.float64),
        )

    # sampling grid from rmin to R200_com
    # Note: R200_com is comoving; we use it as a max projected radius scale.
    Rmax_com = float(R200_com_Mpch)
    ngrid = 256
    R_edges_com = np.geomspace(max(_G_RMIN_SAT, 1e-8), Rmax_com, ngrid + 1)
    R_cent_com = np.sqrt(R_edges_com[:-1] * R_edges_com[1:])
    dR_com = np.diff(R_edges_com)

    # Convert to physical kpc/h for Colossus Sigma call:
    # Colossus surface density expects physical kpc/h.
    zf = float(z_i)
    R_phys_Mpch = R_cent_com / (1.0 + zf)
    R_kpch = R_phys_Mpch * 1000.0  # kpc/h physical
    Sigma_kpc = prof.surfaceDensityInner(R_kpch)  # (Msun/h)/(kpc/h)^2 physical

    # weight ∝ Σ(R_phys) * 2π R_phys dR_phys
    # We compute weights in physical coordinates (consistent with Σ).
    R_phys = R_cent_com / (1.0 + zf)
    dR_phys = dR_com / (1.0 + zf)
    w = np.clip(Sigma_kpc, 0.0, None) * (2.0 * np.pi * R_phys * dR_phys)

    # If weights are pathological (all zeros / nan), fallback to uniform-in-area
    if (not np.isfinite(np.sum(w))) or np.sum(w) <= 0:
        R_samp = Rmax_com * np.sqrt(rng.random(Ns))
    else:
        pdf = w / np.sum(w)
        cdf = np.cumsum(pdf)
        u = rng.random(Ns)
        k = np.searchsorted(cdf, u, side="right")
        k = np.clip(k, 0, ngrid - 1)
        R_low = R_edges_com[k]
        R_high = R_edges_com[k + 1]
        R_samp = R_low + rng.random(Ns) * np.maximum(R_high - R_low, 0.0)

    # Random azimuthal angles
    phi = 2.0 * np.pi * rng.random(Ns)

    # Convert projected comoving radii to small-angle offsets on the sky.
    # theta ~ R / Dc
    theta = R_samp / max(Dc_com, 1e-12)
    dec_c_rad = np.deg2rad(dec_c)
    ddec = theta * np.sin(phi)
    dra = theta * np.cos(phi) / max(np.cos(dec_c_rad), 1e-12)

    ra_sat = wrap_ra_deg(float(ra_c) + np.rad2deg(dra))
    dec_sat = float(dec_c) + np.rad2deg(ddec)

    # Observed redshifts with photo-z scatter
    z_obs = (zf + rng.normal(0.0, sigma_z_photo(zf), size=Ns)).astype(np.float64)

    # Append satellites to output lists
    gal_hidx.extend([i_halo] * Ns)
    gal_ra.extend(ra_sat.tolist())
    gal_dec.extend(np.asarray(dec_sat).tolist())
    gal_zobs.extend(z_obs.tolist())

    return (
        np.asarray(gal_hidx, dtype=np.int64),
        np.asarray(gal_ra, dtype=np.float64),
        np.asarray(gal_dec, dtype=np.float64),
        np.asarray(gal_zobs, dtype=np.float64),
    )


def _paint_particles_for_halo(
    i_halo,
    ra_c,
    dec_c,
    z_i,
    M200m,
    c200m,
    R200_kpch,
    Dc_com_Mpch,
    particle_rmax_com,
    rng,
):
    """
    PARTICLE mode halo painting: represent a halo as discrete particles in 3D.

    Key requirement:
      - Total mass of particles must equal M200m*h (Msun/h).
      - Particle mass is fixed to p_mass_h (global _G_P_MASS_H), except the final
        particle may have a "remainder" mass.

    Geometry:
      - We place the halo center at comoving distance Dc_com along its LOS unit vector.
      - Sample NFW radii in *physical* coordinates then convert to comoving:
          r_com = r_phys * (1+z)
      - Sample isotropic direction (mu, phi), build offsets in LOS-aligned basis,
        and add to the center position.
    """
    M_halo_h = float(M200m) * h  # Msun/h
    if (not np.isfinite(M_halo_h)) or M_halo_h <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float64)

    p_mass_h = float(_G_P_MASS_H)

    # Number of full-mass particles, and a possible remainder particle.
    n_full = int(np.floor(M_halo_h / p_mass_h))
    rem = float(M_halo_h - n_full * p_mass_h)
    n_tot = n_full + (1 if rem > 0 else 0)
    if n_tot <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float64)

    masses = np.full(n_tot, p_mass_h, dtype=np.float64)
    if rem > 0:
        masses[-1] = rem

    # Convert R200 to physical Mpc/h, compute scale radius rs = R200/c
    R200_phys_Mpch = float(R200_kpch) / 1000.0  # Mpc/h physical
    if (not np.isfinite(R200_phys_Mpch)) or R200_phys_Mpch <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float64)

    rs_phys = R200_phys_Mpch / max(float(c200m), 1e-8)

    # Truncation radius given in comoving; convert to physical for sampling.
    rmax_com = float(particle_rmax_com)
    rmax_phys = rmax_com / (1.0 + float(z_i))

    # Sample physical radii from truncated NFW
    r_phys = nfw_inverse_cdf_radii_phys_Mpch(rs_phys, rmax_phys, n_tot, rng)
    if r_phys.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float64)

    # Convert to comoving radii
    r_com = r_phys * (1.0 + float(z_i))

    # Sample isotropic directions:
    #   mu = cos(theta) uniform in [-1,1]
    #   phi uniform in [0,2pi]
    mu = 2.0 * rng.random(n_tot) - 1.0
    phi = 2.0 * np.pi * rng.random(n_tot)
    sin_t = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))

    # Offsets in the local LOS-aligned coordinate system:
    dx = r_com * sin_t * np.cos(phi)
    dy = r_com * sin_t * np.sin(phi)
    dz = r_com * mu

    # Convert local offsets into global Cartesian coordinates
    u = unitvec_from_radec_deg(ra_c, dec_c)
    e1, e2, e3 = tangent_basis_from_unit(u)
    off = (dx[:, None] * e1[None, :] + dy[:, None] * e2[None, :] + dz[:, None] * e3[None, :]).astype(np.float32)

    # Halo center in comoving Cartesian coordinates:
    center = (float(Dc_com_Mpch) * u).astype(np.float32)
    pos = center[None, :] + off
    return pos, masses


def _loop1_chunk_worker(chunk_indices):
    """
    LOOP1 worker: paints a chunk of halos.

    Input
    -----
    chunk_indices : array-like
        Halo indices (into the combined arrays _G_RA, _G_DEC, ...).

    Output (depends on mode)
    ------------------------
    Always returns:
      - idxs: the halo indices processed in this chunk
      - Dc_out, R200com_out, c_out, R200kpch_out: per-halo derived properties
      - galaxy arrays for halos in chunk

    Additionally:
      - PARTICLE mode returns particle positions and masses
      - MAP mode returns pixel indices and masses to add to the halo mass map

    Notes:
      - We use a per-chunk RNG seeded by _G_SEED_BASE + first halo index
        to give deterministic behavior for a given input order.
    """
    idxs = np.asarray(chunk_indices, dtype=np.int64)
    if idxs.size == 0:
        return None

    rng = np.random.default_rng(int(_G_SEED_BASE) + int(idxs[0]))

    # Allocate per-halo output arrays for derived properties
    Dc_out = np.zeros(idxs.size, dtype=np.float64)
    R200com_out = np.zeros(idxs.size, dtype=np.float64)
    c_out = np.zeros(idxs.size, dtype=np.float64)
    R200kpch_out = np.zeros(idxs.size, dtype=np.float64)

    # Accumulators for variable-length outputs (galaxies, particles, pixels)
    gal_hidx_all, gal_ra_all, gal_dec_all, gal_zobs_all = [], [], [], []
    part_pos_list, part_mass_list = [], []
    pix_list, pix_mass_list = [], []

    for j, i in enumerate(idxs):
        ra_c = float(_G_RA[i])
        dec_c = float(_G_DEC[i])
        z_i = float(_G_Z[i])
        M200m = float(_G_MASS[i])

        # Comoving distance to halo redshift
        Dc_com = comoving_distance_Mpch(z_i)
        Dc_out[j] = Dc_com

        # Skip degenerate halos
        if (not np.isfinite(M200m)) or M200m <= 0 or (not np.isfinite(Dc_com)) or Dc_com <= 0:
            continue

        # NFW parameters from Colossus
        c200m = compute_c200m(M200m, z_i)
        R200_kpch, R200_com, prof = halo_R200_comoving_Mpch(M200m, z_i, c200m)

        c_out[j] = c200m
        R200kpch_out[j] = R200_kpch
        R200com_out[j] = R200_com

        # -------------------------
        # Galaxies for ALL halos
        # -------------------------
        gh, gra, gdec, gz = _paint_galaxies_for_halo(
            i_halo=int(i),
            ra_c=ra_c,
            dec_c=dec_c,
            z_i=z_i,
            M200m=M200m,
            R200_com_Mpch=R200_com,
            prof=prof,
            rng=rng,
        )
        if gh.size:
            gal_hidx_all.append(gh)
            gal_ra_all.append(gra)
            gal_dec_all.append(gdec)
            gal_zobs_all.append(gz)

        # -------------------------
        # PARTICLE mode painting
        # -------------------------
        if _G_MODE == "particle":
            pos, m = _paint_particles_for_halo(
                i_halo=int(i),
                ra_c=ra_c,
                dec_c=dec_c,
                z_i=z_i,
                M200m=M200m,
                c200m=c200m,
                R200_kpch=R200_kpch,
                Dc_com_Mpch=Dc_com,
                particle_rmax_com=float(_ARGS.particle_rmax_com),
                rng=rng,
            )
            if pos.shape[0] > 0:
                part_pos_list.append(pos)
                part_mass_list.append(m)

        # -------------------------
        # MAP mode painting
        # -------------------------
        elif _G_MODE == "map":
            if hp is None:
                raise RuntimeError("healpy is required for map mode but could not be imported.")

            # paint radius in physical kpc/h:
            #   Rmax = paint_rvir_factor * R200m
            Rmax_kpch = float(_G_PAINT_RVIR_FACTOR) * float(R200_kpch)
            if (not np.isfinite(Rmax_kpch)) or Rmax_kpch <= 0:
                continue

            # Build a radius grid where we compute Σ(R).
            # This grid is in physical kpc/h (because Colossus uses physical).
            Rmin_kpch = 1e-3
            R_grid_kpch = np.geomspace(Rmin_kpch, Rmax_kpch, int(_G_N_SIGMA_GRID))

            # Decide whether to use baryonification or pure NFW
            use_bary = (
                (not _G_DISABLE_BARYONIFICATION)
                and (contract_dmo_profile_with_ac is not None)
                and (_G_SIGMAR_PROVIDER is not None)
                and (M200m >= float(_G_M_BARY_MIN))
            )

            if use_bary:
                # Build 3D density profile rho(r) on r3d grid, baryonify it,
                # then project to surface density Σ(R).
                r3d_kpch = np.geomspace(Rmin_kpch, Rmax_kpch, int(_G_N_R3D))
                rho_dmo_kpc = prof.density(r3d_kpch)

                # Convert kpc/h -> Mpc/h for the baryonification function
                r3d_mpch = r3d_kpch / 1000.0
                rho_dmo_mpc = rho_dmo_kpc * (1000.0**3)

                M_halo_h = float(M200m) * h
                rvir_mpch = float(R200_kpch) / 1000.0

                rho_contr_mpc, _, _ = contract_dmo_profile_with_ac(
                    rf=r3d_mpch,
                    rho_dmo=rho_dmo_mpc,
                    M_vir=M_halo_h,
                    rvir=rvir_mpch,
                    z=z_i,
                    frh=0.075,
                    cosmo=_G_COSMO_BARY,
                    sigma_provider=_G_SIGMAR_PROVIDER,
                )
                rho_contr_kpc = rho_contr_mpc / (1000.0**3)
                Sigma_kpc = project_sigma_from_rho_vec(r3d_kpch, rho_contr_kpc, R_grid_kpch)
            else:
                # Pure NFW projected surface density from Colossus
                Sigma_kpc = prof.surfaceDensityInner(R_grid_kpch)

            # Query which HEALPix pixels lie within the halo painting radius.
            # Convert physical Rmax to comoving Mpc/h first:
            Rmax_com = (Rmax_kpch / 1000.0) * (1.0 + z_i)  # comoving Mpc/h
            theta_max = Rmax_com / max(Dc_com, 1e-12)

            vec_c = unitvec_from_radec_deg(ra_c, dec_c)
            ipix = hp.query_disc(int(_G_NSIDE), vec_c, theta_max)

            # Apply the (halo-nside) mask, if provided.
            if _G_MASK_MAP is not None:
                ipix = ipix[np.asarray(_G_MASK_MAP, dtype=bool)[ipix]]

            if ipix.size == 0:
                continue

            # Compute angular separation for each pixel, then projected radius.
            xv, yv, zv = hp.pix2vec(int(_G_NSIDE), ipix)
            dot = vec_c[0] * xv + vec_c[1] * yv + vec_c[2] * zv
            th = np.arccos(np.clip(dot, -1.0, 1.0))

            # Rpix_com = Dc * theta  (small-angle approximation)
            Rpix_com = Dc_com * th
            # Convert comoving -> physical and then to kpc/h
            Rpix_phys = Rpix_com / (1.0 + z_i)
            Rpix_kpch = Rpix_phys * 1000.0

            # Evaluate Σ at each pixel radius using log-log interpolation.
            Sigma_pix = interp_loglog(Rpix_kpch, R_grid_kpch, Sigma_kpc)

            # Pixel physical transverse area at lens redshift:
            # A = Ω * (Dc/(1+z))^2  in (Mpc/h)^2
            # Convert to (kpc/h)^2 by multiplying (1000^2)
            A_pix_phys_kpc2 = float(_G_OMEGA_PIX) * (Dc_com / (1.0 + z_i)) ** 2 * (1000.0**2)

            # Mass per pixel = Σ * area
            m_pix = Sigma_pix * A_pix_phys_kpc2  # Msun/h per pixel

            pix_list.append(ipix.astype(np.int64))
            pix_mass_list.append(m_pix.astype(np.float64))
        else:
            raise ValueError(f"Unknown mode: {_G_MODE}")

    # Concatenate galaxy outputs for this chunk
    gal_hidx = np.concatenate(gal_hidx_all) if gal_hidx_all else np.empty(0, dtype=np.int64)
    gal_ra = np.concatenate(gal_ra_all) if gal_ra_all else np.empty(0, dtype=np.float64)
    gal_dec = np.concatenate(gal_dec_all) if gal_dec_all else np.empty(0, dtype=np.float64)
    gal_zobs = np.concatenate(gal_zobs_all) if gal_zobs_all else np.empty(0, dtype=np.float64)

    if _G_MODE == "particle":
        part_pos = np.concatenate(part_pos_list, axis=0) if part_pos_list else np.zeros((0, 3), dtype=np.float32)
        part_mass = np.concatenate(part_mass_list) if part_mass_list else np.zeros((0,), dtype=np.float64)
        return (idxs, Dc_out, R200com_out, c_out, R200kpch_out,
                gal_hidx, gal_ra, gal_dec, gal_zobs, part_pos, part_mass)

    # map mode: concatenate pixel contributions
    if pix_list:
        ipix_all = np.concatenate(pix_list)
        mpix_all = np.concatenate(pix_mass_list)
    else:
        ipix_all = np.empty(0, dtype=np.int64)
        mpix_all = np.empty(0, dtype=np.float64)

    return (idxs, Dc_out, R200com_out, c_out, R200kpch_out,
            gal_hidx, gal_ra, gal_dec, gal_zobs, ipix_all, mpix_all)


# ==========================================================
# Particle mode: profile globals + workers
# ==========================================================
# These globals are populated after Loop1 in particle mode only.
# They support fast spatial queries for profiles and miscentering.

_PART_POS = None
_PART_MASS = None
_PART_TREE = None

_GAL_RA2 = None
_GAL_DEC2 = None
_GAL_ZOBS2 = None
_GAL_UNIT2 = None
_GAL_TREE2 = None

_HALO_Dc = None
_HALO_R200_COM = None

_PROFILE_R_EDGES = None
_PROFILE_R_CENTERS = None


def _get_galaxies_within_radius(ra_c, dec_c, Dc_com, Rmax_com):
    """
    Find galaxies within an angular radius corresponding to comoving Rmax_com.

    Trick:
      - We build a KDTree in 3D unit-vector space (on the unit sphere).
      - For a given angular radius theta_max, points within theta_max are within
        chord distance r_chord = 2 sin(theta/2).

    Returns
    -------
    idx : ndarray of int
        Indices of galaxies within that chord radius.
    vec_c : ndarray shape (3,)
        Unit vector for the halo center.
    theta_max : float
        Angular search radius in radians.
    """
    if _GAL_TREE2 is None or _GAL_UNIT2 is None or _GAL_UNIT2.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.array([0.0, 0.0, 1.0]), 0.0
    vec_c = unitvec_from_radec_deg(ra_c, dec_c)
    theta_max = float(Rmax_com) / max(float(Dc_com), 1e-12)
    r_chord = 2.0 * np.sin(theta_max / 2.0)
    idx = _GAL_TREE2.query_ball_point(vec_c, r_chord)
    return np.asarray(idx, dtype=np.int64), vec_c, theta_max


def _compute_halo_miscenter_no_pmem(i_halo):
    """
    Compute miscentered halo center in particle mode without P_mem.

    Steps:
      - Start from true halo center (ra0, dec0, z0).
      - Select galaxies within Rsel = miscenter_radius_factor * R200_com.
      - Apply additional selection:
          |z_gal - z_halo| <= sigma_z_photo(z_halo)
      - Run histogram+smooth threshold algorithm to get (ra_mis, dec_mis).
      - Return miscentered center and miscentering distance Rmis in comoving Mpc/h.
    """
    ra0 = float(_G_RA[i_halo])
    dec0 = float(_G_DEC[i_halo])
    z0 = float(_G_Z[i_halo])
    Dc0 = float(_HALO_Dc[i_halo])
    R200 = float(_HALO_R200_COM[i_halo])

    if (not np.isfinite(R200)) or R200 <= 0 or (not np.isfinite(Dc0)) or Dc0 <= 0:
        return ra0, dec0, 0.0

    Rsel = float(_ARGS.miscenter_radius_factor) * R200
    idx, vec_true, _ = _get_galaxies_within_radius(ra0, dec0, Dc0, Rsel)
    if idx.size < 3:
        return ra0, dec0, 0.0

    ra_g = _GAL_RA2[idx]
    dec_g = _GAL_DEC2[idx]
    z_g = _GAL_ZOBS2[idx]
    vec_g = _GAL_UNIT2[idx]

    # Projected comoving separation: Rproj = Dc * theta
    dot = np.clip(np.sum(vec_g * vec_true[None, :], axis=1), -1.0, 1.0)
    theta = np.arccos(dot)
    Rproj = Dc0 * theta

    # photo-z selection (simple)
    dz = z_g - z0
    sigz = sigma_z_photo(z0)

    m = (Rproj <= Rsel) & (np.abs(dz) <= sigz)
    if np.count_nonzero(m) < 3:
        return ra0, dec0, 0.0

    ra_m, dec_m = compute_miscentered_center_threshold(
        ra_g[m],
        dec_g[m],
        ra_center_deg=ra0,
        bins=int(_ARGS.miscenter_bins),
        sigma=float(_ARGS.miscenter_sigma),
        n_mult=float(_ARGS.miscenter_nmult),
    )
    if not np.isfinite(ra_m) or not np.isfinite(dec_m):
        return ra0, dec0, 0.0

    # Convert miscenter angle to comoving distance:
    v0 = unitvec_from_radec_deg(ra0, dec0)
    vm = unitvec_from_radec_deg(ra_m, dec_m)
    th = float(np.arccos(np.clip(np.dot(v0, vm), -1.0, 1.0)))
    Rmis = Dc0 * th
    return ra_m, dec_m, Rmis


def _profile_worker(i_halo):
    """
    Worker to compute a miscentered Σ(R) profile for one halo (particle mode).

    Outline:
      1) Compute miscentered center (ra_m, dec_m, Rmis).
      2) Define a cylinder aligned with LOS u at comoving distance Dc:
           - radius = profile_rmax
           - depth  = depth_cyl
      3) Use particle KDTree to quickly retrieve candidates in a bounding sphere.
      4) Filter candidates to those inside the cylinder:
           - |LOS distance| <= depth/2
           - projected radius <= profile_rmax (implicitly by binning)
      5) Bin particle masses in annuli and divide by area -> Σ(R)

    Returns:
      (halo_index, ra_m, dec_m, Rmis, Sigma_array)
    """
    ra_m, dec_m, Rmis = _compute_halo_miscenter_no_pmem(int(i_halo))

    Dc0 = float(_HALO_Dc[i_halo])
    u = unitvec_from_radec_deg(ra_m, dec_m)
    center = Dc0 * u

    depth = float(_ARGS.depth_cyl)
    half = 0.5 * depth
    Rmax = float(_ARGS.profile_rmax)

    # Bounding sphere radius for initial KDTree query:
    # any point in cylinder lies within sqrt(half^2 + Rmax^2) of center.
    r_search = float(np.sqrt(half * half + Rmax * Rmax))

    cand = _PART_TREE.query_ball_point(center, r_search)
    if len(cand) == 0:
        Sigma = np.zeros_like(_PROFILE_R_CENTERS)
        return (int(i_halo), ra_m, dec_m, Rmis, Sigma)

    idx = np.asarray(cand, dtype=np.int64)

    # Vector from center to particles
    dp = _PART_POS[idx] - center[None, :]

    # LOS component = projection onto u
    los = dp @ u
    m_los = np.abs(los) <= half
    if not np.any(m_los):
        Sigma = np.zeros_like(_PROFILE_R_CENTERS)
        return (int(i_halo), ra_m, dec_m, Rmis, Sigma)

    dp = dp[m_los]
    idx = idx[m_los]
    los = los[m_los]

    # Transverse component in plane perpendicular to u
    proj = dp - los[:, None] * u[None, :]
    Rproj = np.sqrt(np.sum(proj * proj, axis=1))

    # Bin projected radii into annuli
    bin_idx = np.digitize(Rproj, _PROFILE_R_EDGES) - 1
    ok = (bin_idx >= 0) & (bin_idx < _PROFILE_R_CENTERS.size)
    if not np.any(ok):
        Sigma = np.zeros_like(_PROFILE_R_CENTERS)
        return (int(i_halo), ra_m, dec_m, Rmis, Sigma)

    bin_idx = bin_idx[ok]
    mass_in = _PART_MASS[idx[ok]]

    # Accumulate masses per annulus
    mass_bins = np.zeros_like(_PROFILE_R_CENTERS, dtype=np.float64)
    np.add.at(mass_bins, bin_idx, mass_in)

    # Σ = mass / area
    r1 = _PROFILE_R_EDGES[:-1]
    r2 = _PROFILE_R_EDGES[1:]
    area = np.pi * (r2 * r2 - r1 * r1)
    Sigma = mass_bins / np.maximum(area, 1e-300)

    return (int(i_halo), ra_m, dec_m, Rmis, Sigma)


# ==========================================================
# main
# ==========================================================
def main():
    """
    Main program entry point.
    Handles:
      - argument parsing
      - reading input data
      - selecting mode + buffer halos
      - loop1 painting in parallel
      - saving galaxy outputs
      - map-mode outputs OR particle-mode profiles + miscentering outputs
    """
    global _NO_PROGRESS, _PROGRESS_TO_STDOUT, _ARGS
    global _G_MODE, _G_RA, _G_DEC, _G_Z, _G_MASS, _G_IS_TARGET, _G_N_TARGET
    global _G_P_MASS_H, _G_NSIDE, _G_MASK_MAP, _G_OMEGA_PIX, _G_SIGMAR_PROVIDER, _G_COSMO_BARY
    global _G_PAINT_RVIR_FACTOR, _G_M_BARY_MIN, _G_N_SIGMA_GRID, _G_N_R3D, _G_RMIN_SAT, _G_SEED_BASE

    parser = argparse.ArgumentParser()

    # ---------------------------------------------------------------------
    # Core inputs
    # ---------------------------------------------------------------------
    parser.add_argument("--npz-file", required=True)
    parser.add_argument("--base-outdir", default=DEFAULT_BASE_OUTDIR,
                        help="Base directory for outputs when --outdir is relative (default: G100 scratch).")
    parser.add_argument("--outdir", default="outputs",
                        help="Output directory. If relative, it is resolved under --base-outdir.")

    # ---------------------------------------------------------------------
    # MAP mode controls
    # ---------------------------------------------------------------------
    parser.add_argument("--massmap-file", default=None,
                        help="If provided, run MAP mode. If omitted, PARTICLE mode.")
    parser.add_argument("--paint-rvir-factor", type=float, default=3.0,
                        help="MAP mode: paint radius = factor * R200m.")
    parser.add_argument("--halo-nside", type=int, default=None,
                        help=("MAP mode: NSIDE for halo painting. "
                              "If None, use the NSIDE of --massmap-file. "
                              "Must be >= base NSIDE and a power-of-two multiple."))

    # ---------------------------------------------------------------------
    # PARTICLE buffer shells (particle mode only)
    # ---------------------------------------------------------------------
    parser.add_argument("--npz-prev", default=None,
                        help="PARTICLE mode only: previous shell catalogue for buffer.")
    parser.add_argument("--npz-next", default=None,
                        help="PARTICLE mode only: next shell catalogue for buffer.")
    parser.add_argument("--neighbor-buffer-com", type=float, default=50.0,
                        help="Buffer half-thickness (cMpc/h).")

    # ---------------------------------------------------------------------
    # Mass thresholds
    # ---------------------------------------------------------------------
    parser.add_argument("--m-bary-min", type=float, default=1e13,
                        help="MAP mode baryonification threshold.")
    parser.add_argument("--disable-baryonification", action="store_true",
                        help="MAP mode: disable baryonification and paint pure NFW for all halos.")
    parser.add_argument("--profile-mass-cut", type=float, default=1e13,
                        help="PARTICLE mode: apply ONLY to profiles.")

    # ---------------------------------------------------------------------
    # PARAMETER FILE for setup/cosmo
    # ---------------------------------------------------------------------
    parser.add_argument("--paramfile-path", required=True,
                        help="Path to the Pinocchio parameter file.")

    # ---------------------------------------------------------------------
    # Parallel / chunking / monitoring
    # ---------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Loop1 chunk size (bigger reduces overhead).")
    parser.add_argument("--procs", type=int, default=0,
                        help="Loop1 processes (0=auto).")
    parser.add_argument("--profile-procs", type=int, default=0,
                        help="Profile processes (0=auto).")
    parser.add_argument("--pool-chunksize", type=int, default=8,
                        help="imap_unordered chunksize for pools.")
    parser.add_argument("--progress-to-stdout", action="store_true",
                        help="Compatibility flag; stdout progress is default.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm.")
    parser.add_argument("--monitor-every", type=float, default=20.0,
                        help="Seconds between progress log lines.")

    # ---------------------------------------------------------------------
    # PARTICLE mode parameters
    # ---------------------------------------------------------------------
    parser.add_argument("--particle-rmax-com", type=float, default=30.0,
                        help="NFW-as-particles truncation (cMpc/h).")

    # ---------------------------------------------------------------------
    # Profile cylinder + bins (particle)
    # ---------------------------------------------------------------------
    parser.add_argument("--depth-cyl", type=float, default=100.0,
                        help="Cylinder depth along LOS (cMpc/h).")
    parser.add_argument("--profile-rmax", type=float, default=10.0,
                        help="Max projected radius for profiles (cMpc/h).")
    parser.add_argument("--nbins", type=int, default=20)
    parser.add_argument("--profile-rmin", type=float, default=0.01,
                        help="Min profile radius (cMpc/h).")

    # ---------------------------------------------------------------------
    # Miscentering (particle, no P_mem)
    # ---------------------------------------------------------------------
    parser.add_argument("--miscenter-nmult", type=float, default=1.0,
                        help="Threshold = n_mult * median(smooth_hist).")
    parser.add_argument("--miscenter-bins", type=int, default=60)
    parser.add_argument("--miscenter-sigma", type=float, default=3.0)
    parser.add_argument("--miscenter-radius-factor", type=float, default=1.0,
                        help="Selection radius = factor * R200_com.")

    # ---------------------------------------------------------------------
    # MAP mode bary grids
    # ---------------------------------------------------------------------
    parser.add_argument("--n-sigma-grid", type=int, default=256)
    parser.add_argument("--n-r3d", type=int, default=256)

    # ---------------------------------------------------------------------
    # Satellite sampling small-r floor
    # ---------------------------------------------------------------------
    parser.add_argument("--rmin-sat", type=float, default=1e-4)

    args = parser.parse_args()
    _ARGS = args

    # Progress settings
    _NO_PROGRESS = bool(args.no_progress)
    _PROGRESS_TO_STDOUT = True  # default ON (compat with .out monitoring)

    # Resolve output directory under base scratch (unless absolute)
    args.outdir = resolve_outdir(args.base_outdir, args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[Output] outdir: {args.outdir}")

    # Use fork if possible to reduce pickling / speed up for huge arrays
    # Note: on some platforms (macOS, some python builds) fork may not exist.
    try:
        if "fork" in mp.get_all_start_methods():
            mp.set_start_method("fork", force=False)
    except RuntimeError:
        pass

    # ---------------------------------------------------------------------
    # Pinocchio Cosmology and setup
    # ---------------------------------------------------------------------
    # this reads the Pinocchio parameter file and extracts:
    # - Hubble100
    # - Omega0
    # - GridSize
    # - BoxSize
    P = params_file()
    P.load(_ARGS.paramfile_path)
    global H0, h, Om0, GridSize, BoxSize

    h = P.cosmo['Hubble100']
    Om0 = P.cosmo['Omega0']
    GridSize = P.setup['GridSize']
    BoxSize = P.setup['BoxSize']
    H0 = h * 100.0
    # Astropy cosmology object for distances
    cosmo = FlatLambdaCDM(Om0=Om0, H0=H0)

    # ---------------------------------------------------------------------
    # Particle mass (same formula as original code)
    # ---------------------------------------------------------------------
    # p_mass is computed from:
    #   - box volume
    #   - grid resolution
    #   - critical density
    #   - Omega_m
    #
    # p_mass_h = p_mass * h converts Msun -> Msun/h.
    p_mass = (BoxSize / H0 * 100.0) ** 3 / GridSize**3 * cosmo.critical_density0.to("Msun/Mpc^3").value * cosmo.Om0
    p_mass_h = float(p_mass * h)
    print(f"[Config] p_mass_h = {p_mass_h:.6e} Msun/h")

    # ---------------------------------------------------------------------
    # Load target halo catalogue
    # ---------------------------------------------------------------------
    label = shell_label_from_npz_path(args.npz_file)
    edges = shell_edges_from_filename(args.npz_file)

    cat_t = np.load(args.npz_file)
    theta_t = np.asarray(cat_t["theta"], dtype=float)
    phi_t = np.asarray(cat_t["phi"], dtype=float)
    z_t = np.asarray(cat_t["z"], dtype=float)
    m_t = np.asarray(cat_t["Mass"], dtype=float)

    # Convert from spherical angles to RA/Dec:
    #   theta is polar angle from +z (colatitude)
    #   dec = 90 - theta(deg)
    #   phi is azimuth = RA
    ra_t = np.rad2deg(phi_t)
    dec_t = 90.0 - np.rad2deg(theta_t)

    N_target = ra_t.size
    if N_target == 0:
        raise RuntimeError("Empty target halo catalogue.")

    print(f"[Input] NPZ target: {os.path.basename(args.npz_file)}  label={label}  "
          f"file_z={list(edges) if edges else 'None'}")
    print(f"[Input] target halos: N={N_target}  log10M=[{np.log10(m_t).min():.3f}, {np.log10(m_t).max():.3f}]")

    # ---------------------------------------------------------------------
    # Determine mode
    # ---------------------------------------------------------------------
    massmap_file = args.massmap_file
    if isinstance(massmap_file, str) and massmap_file.strip() == "":
        massmap_file = None
    map_mode = massmap_file is not None

    if map_mode:
        if hp is None:
            raise RuntimeError("healpy is required for MAP mode but could not be imported.")
        _G_MODE = "map"
        print("[Mode] MAP mode")
    else:
        _G_MODE = "particle"
        print("[Mode] PARTICLE mode")

    # ---------------------------------------------------------------------
    # Select buffer halos (particle mode only)
    # ---------------------------------------------------------------------
    # Start with target-only arrays
    ra_all, dec_all, z_all, m_all = ra_t, dec_t, z_t, m_t
    is_target_all = np.ones(N_target, dtype=bool)

    # Only if particle mode AND user provided prev/next files:
    if (not map_mode) and (args.npz_prev or args.npz_next):
        # Determine shell boundaries in comoving distance:
        # - Prefer filename edges if present
        # - Otherwise derive from z range in the target file
        if edges is not None:
            z_lo, z_hi = edges
        else:
            z_lo, z_hi = float(np.min(z_t)), float(np.max(z_t))

        Dc_lo = comoving_distance_Mpch(z_lo)
        Dc_hi = comoving_distance_Mpch(z_hi)
        if Dc_hi < Dc_lo:
            Dc_lo, Dc_hi = Dc_hi, Dc_lo

        buf = float(args.neighbor_buffer_com)

        def load_and_filter(npz_path, side):
            """
            Load a neighboring shell NPZ and select halos close to the boundary.

            side="prev":
              keep halos with Dc in [Dc_lo - buf, Dc_lo)
            side="next":
              keep halos with Dc in (Dc_hi, Dc_hi + buf]
            """
            if npz_path is None:
                return (np.empty(0), np.empty(0), np.empty(0), np.empty(0))
            cat = np.load(npz_path)
            th = np.asarray(cat["theta"], dtype=float)
            ph = np.asarray(cat["phi"], dtype=float)
            zz = np.asarray(cat["z"], dtype=float)
            mm = np.asarray(cat["Mass"], dtype=float)
            rr = np.rad2deg(ph)
            dd = 90.0 - np.rad2deg(th)

            # Convert all redshifts to comoving distances
            Dc = np.array([comoving_distance_Mpch(v) for v in zz], dtype=float)

            if side == "prev":
                sel = (Dc >= (Dc_lo - buf)) & (Dc < Dc_lo)
            else:
                sel = (Dc <= (Dc_hi + buf)) & (Dc > Dc_hi)

            return rr[sel], dd[sel], zz[sel], mm[sel]

        ra_p, dec_p, z_p, m_p = load_and_filter(args.npz_prev, "prev")
        ra_n, dec_n, z_n, m_n = load_and_filter(args.npz_next, "next")

        nbuf = ra_p.size + ra_n.size
        if nbuf > 0:
            # Append buffer halos to the combined arrays
            ra_all = np.concatenate([ra_all, ra_p, ra_n])
            dec_all = np.concatenate([dec_all, dec_p, dec_n])
            z_all = np.concatenate([z_all, z_p, z_n])
            m_all = np.concatenate([m_all, m_p, m_n])

            # Target flag: target halos True, buffer halos False
            is_target_all = np.concatenate([is_target_all, np.zeros(nbuf, dtype=bool)])

            print(f"[Buffer] Added buffer halos: N={nbuf} within ±{buf:.1f} cMpc/h of shell boundaries")
        else:
            print("[Buffer] No buffer halos selected (or none provided).")

    N_all = ra_all.size
    _G_N_TARGET = int(N_target)
    print(f"[Loop1] painting halos total (target + buffer, if any): N={N_all}  (target={N_target})")

    # ---------------------------------------------------------------------
    # Set globals for workers
    # ---------------------------------------------------------------------
    _G_RA = ra_all
    _G_DEC = dec_all
    _G_Z = z_all
    _G_MASS = m_all
    _G_IS_TARGET = is_target_all
    _G_P_MASS_H = p_mass_h

    _G_PAINT_RVIR_FACTOR = float(args.paint_rvir_factor)
    _G_M_BARY_MIN = float(args.m_bary_min)
    _G_DISABLE_BARYONIFICATION = bool(args.disable_baryonification)
    _G_N_SIGMA_GRID = int(args.n_sigma_grid)
    _G_N_R3D = int(args.n_r3d)
    _G_RMIN_SAT = float(args.rmin_sat)
    _G_SEED_BASE = int(args.seed) if args.seed is not None else 12345

    # Prepare colossus cosmology in parent too (safe)
    prepare_colossus_cosmology()

    # ---------------------------------------------------------------------
    # MAP inputs + bary provider
    # ---------------------------------------------------------------------
    # These variables are only meaningful if map_mode is True.
    bnfw_map = None           # halo mass map at halo_nside
    mask_map = None           # base mask (Pinocchio)
    Omega_pix = None          # base pixel solid angle
    mass_map_pin = None       # Pinocchio mass map (Msun/h per base pixel)

    _G_SIGMAR_PROVIDER = None
    _G_COSMO_BARY = {"Om0": Om0, "h": h, "f_b": 0.049 / Om0}

    if map_mode:
        # Base Pinocchio count map (HEALPix), read as floats.
        # UNSEEN pixels are masked.
        m_counts = hp.read_map(massmap_file)
        nside_base = int(hp.get_nside(m_counts))
        npix_base = int(hp.nside2npix(nside_base))
        mask_map_base = (m_counts != hp.UNSEEN)
        Omega_pix_base = 4.0 * np.pi / float(npix_base)

        # Convert counts -> mass by multiplying by particle mass
        mass_map_pin = np.where(mask_map_base, m_counts * p_mass_h, 0.0)

        # Decide halo painting NSIDE:
        # - default = base NSIDE
        # - user can request higher NSIDE, but it must be a power-of-two multiple
        halo_nside = nside_base
        if args.halo_nside is not None:
            requested = int(args.halo_nside)
            if requested < nside_base:
                print(f"[Map] Requested halo_nside={requested} < base nside={nside_base}; using base nside.",
                      flush=True)
            else:
                factor = requested // nside_base
                if (requested % nside_base != 0) or (factor & (factor - 1)):
                    print(f"[Map] halo_nside={requested} is not a power-of-two multiple of base nside={nside_base}; "
                          f"using base nside.", flush=True)
                else:
                    halo_nside = requested

        npix_halo = int(hp.nside2npix(halo_nside))
        bnfw_map = np.zeros(npix_halo, dtype=np.float64)

        # Build halo-mode mask:
        # - if halo_nside==base: use base mask directly
        # - else: upsample base mask to halo_nside using ud_grade
        if halo_nside == nside_base:
            mask_map_halo = mask_map_base
        else:
            mask_float = mask_map_base.astype(np.float64)
            mask_hi = hp.ud_grade(
                mask_float,
                nside_out=halo_nside,
                order_in="RING",
                order_out="RING",
                power=0,
            )
            mask_map_halo = mask_hi > 0.5

        Omega_pix_halo = 4.0 * np.pi / float(npix_halo)

        # Globals used inside workers (MAP mode)
        _G_NSIDE = halo_nside
        _G_MASK_MAP = mask_map_halo
        _G_OMEGA_PIX = Omega_pix_halo

        # For later outputs
        mask_map = mask_map_base
        Omega_pix = Omega_pix_base

        print(f"[Map] base nside={nside_base} npix={npix_base} masked={int(np.count_nonzero(~mask_map_base))}")
        print(f"[Map] halo painting nside={halo_nside} npix={npix_halo}")

        if _G_DISABLE_BARYONIFICATION:
            print("[Setup] baryonification: DISABLED (painting pure NFW)")

        # Attempt to build sigmaR provider for baryonification if available.
        # If it fails, baryonification is silently disabled and NFW is used.
        if (not _G_DISABLE_BARYONIFICATION) and (build_sigmaR_provider is not None):
            try:
                zmin, zmax = float(np.min(z_all)), float(np.max(z_all))
                z_grid = np.linspace(zmin, zmax, 20)
                R_grid = np.geomspace(0.02, 10.0, 128)
                _G_SIGMAR_PROVIDER = build_sigmaR_provider(
                    {"Om0": Om0, "h": h, "Ob0": 0.049, "ns": 0.965, "sigma8": 0.811},
                    z_grid,
                    R_grid,
                )
                print("[Setup] sigmaR_provider: OK")
            except Exception as e:
                _G_SIGMAR_PROVIDER = None
                print(f"[Setup] sigmaR_provider: FAILED ({e}); fallback to NFW.")
        else:
            print("[Setup] sigmaR_provider: unavailable; fallback to NFW.")
    else:
        _G_NSIDE = None
        _G_MASK_MAP = None
        _G_OMEGA_PIX = None

    # ---------------------------------------------------------------------
    # Allocate arrays for halo properties (for ALL painted halos)
    # ---------------------------------------------------------------------
    halo_Dc = np.zeros(N_all, dtype=np.float64)
    halo_R200_com = np.zeros(N_all, dtype=np.float64)
    halo_c200m = np.zeros(N_all, dtype=np.float64)
    halo_R200_kpch = np.zeros(N_all, dtype=np.float64)

    # ---------------------------------------------------------------------
    # Loop1 parallel setup
    # ---------------------------------------------------------------------
    # CPU selection:
    # - if running under SLURM, honor SLURM_CPUS_PER_TASK
    # - otherwise use mp.cpu_count()
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or "0")
    n_cores = slurm_cpus if slurm_cpus > 0 else mp.cpu_count()
    procs = int(args.procs) if int(args.procs) > 0 else min(n_cores, max(1, n_cores))
    procs = max(1, procs)

    print(f"[Loop1] using {procs}/{n_cores} processes  chunk_size={int(args.chunk_size)} "
          f"pool_chunksize={int(args.pool_chunksize)}")

    # Generator producing chunks of halo indices without building a huge list
    def chunk_iter(n, chunk_size):
        cs = int(chunk_size)
        for start in range(0, n, cs):
            yield np.arange(start, min(start + cs, n), dtype=np.int64)

    # Collect outputs
    gal_hidx_list, gal_ra_list, gal_dec_list, gal_zobs_list = [], [], [], []
    part_pos_list, part_mass_list = [], []

    t0 = time.time()
    t_last = t0
    done = 0
    total_chunks = int((N_all + int(args.chunk_size) - 1) // int(args.chunk_size))

    # Run Loop1 painting
    with mp.Pool(processes=procs, initializer=_loop1_init_worker) as pool:
        it = pool.imap_unordered(_loop1_chunk_worker,
                                 chunk_iter(N_all, args.chunk_size),
                                 chunksize=int(args.pool_chunksize))
        for out in _tqdm(it, total=total_chunks, desc="Loop1: paint", unit="chunk"):
            if out is None:
                continue

            if _G_MODE == "particle":
                (idxs, Dc_out, R200com_out, c_out, R200kpch_out,
                 gh, gra, gdec, gz, part_pos, part_mass) = out

                halo_Dc[idxs] = Dc_out
                halo_R200_com[idxs] = R200com_out
                halo_c200m[idxs] = c_out
                halo_R200_kpch[idxs] = R200kpch_out

                if gh.size:
                    gal_hidx_list.append(gh)
                    gal_ra_list.append(gra)
                    gal_dec_list.append(gdec)
                    gal_zobs_list.append(gz)

                if part_pos.shape[0] > 0:
                    part_pos_list.append(part_pos)
                    part_mass_list.append(part_mass)

            else:
                (idxs, Dc_out, R200com_out, c_out, R200kpch_out,
                 gh, gra, gdec, gz, ipix, mpix) = out

                halo_Dc[idxs] = Dc_out
                halo_R200_com[idxs] = R200com_out
                halo_c200m[idxs] = c_out
                halo_R200_kpch[idxs] = R200kpch_out

                if gh.size:
                    gal_hidx_list.append(gh)
                    gal_ra_list.append(gra)
                    gal_dec_list.append(gdec)
                    gal_zobs_list.append(gz)

                # Add this chunk's halo-painted pixel masses into the global halo map.
                # np.add.at handles repeated indices.
                if ipix.size > 0:
                    np.add.at(bnfw_map, ipix, mpix)

            done += 1
            t_last = _maybe_print_progress("Loop1: paint", done, total_chunks,
                                            t_last, every_sec=float(args.monitor_every))

    t1 = time.time()
    print(f"[Loop1] done in {(t1 - t0)/60:.2f} min", flush=True)

    # ---------------------------------------------------------------------
    # Finalize galaxies
    # ---------------------------------------------------------------------
    # Concatenate all per-chunk galaxy arrays into one catalog.
    if gal_hidx_list:
        gal_hidx = np.concatenate(gal_hidx_list).astype(np.int64)
        gal_ra = np.concatenate(gal_ra_list).astype(np.float64)
        gal_dec = np.concatenate(gal_dec_list).astype(np.float64)
        gal_zobs = np.concatenate(gal_zobs_list).astype(np.float64)
    else:
        gal_hidx = np.empty(0, dtype=np.int64)
        gal_ra = np.empty(0, dtype=np.float64)
        gal_dec = np.empty(0, dtype=np.float64)
        gal_zobs = np.empty(0, dtype=np.float64)

    # Add target/buffer tagging for galaxies:
    # - gal_hidx points into combined halo list [0..N_all-1]
    # - we record whether that halo was target, and (if not) store -1 in target index
    if gal_hidx.size > 0:
        gal_halo_is_target = _G_IS_TARGET[np.clip(gal_hidx, 0, N_all - 1)].astype(np.int8)
        gal_halo_idx_target = np.where(gal_halo_is_target == 1, gal_hidx, -1).astype(np.int64)
    else:
        gal_halo_is_target = np.empty(0, dtype=np.int8)
        gal_halo_idx_target = np.empty(0, dtype=np.int64)

    gal_out_path = os.path.join(args.outdir, f"galaxies_catalog_{label}.npz")
    np.savez(
        gal_out_path,
        gal_halo_idx=gal_hidx,
        gal_halo_is_target=gal_halo_is_target,
        gal_halo_idx_target=gal_halo_idx_target,
        gal_ra=gal_ra,
        gal_dec=gal_dec,
        gal_z_obs=gal_zobs,
    )
    print(f"[Output] saved galaxies: {gal_out_path}", flush=True)

    # ---------------------------------------------------------------------
    # MAP mode outputs
    # ---------------------------------------------------------------------
    if map_mode:
        nside_base = int(hp.get_nside(mass_map_pin))
        nside_halo = int(_G_NSIDE)

        # If we painted halos at higher nside, degrade to base nside by summing masses.
        if nside_halo == nside_base:
            bnfw_low = bnfw_map.astype(np.float64, copy=True)
        else:
            print(f"[Map] Degrading halo map from nside={nside_halo} -> base nside={nside_base}", flush=True)
            bnfw_low = degrade_halo_mass_map_ring(bnfw_map, nside_halo, nside_base)

        # Apply base mask and write halo-only map
        bnfw_out = bnfw_low.astype(np.float64, copy=True)
        bnfw_out[~mask_map] = hp.UNSEEN

        did_bary = (not _G_DISABLE_BARYONIFICATION) and (_G_SIGMAR_PROVIDER is not None)
        bnfw_fname = (f"baryonified_NFW_global_massmap_{label}.fits"
                      if did_bary else f"NFW_global_massmap_{label}.fits")
        bnfw_path = os.path.join(args.outdir, bnfw_fname)
        hp.write_map(bnfw_path, bnfw_out, overwrite=True)

        # Composite: Pinocchio (base nside) + degraded halo mass (same nside)
        tot_out = np.where(mask_map, mass_map_pin + bnfw_low, hp.UNSEEN)
        tot_fname = (f"pinocchio_plus_baryonified_massmap_{label}.fits"
                     if did_bary else f"pinocchio_plus_NFW_massmap_{label}.fits")
        tot_path = os.path.join(args.outdir, tot_fname)
        hp.write_map(tot_path, tot_out, overwrite=True)

        # Mass check:
        # - compare sum of halo-painted map with sum of halo masses (M*h)
        valid = mask_map & np.isfinite(bnfw_low)
        sum_bnfw = float(np.sum(bnfw_low[valid]))
        sum_halos = float(np.sum(m_all * h))
        print("[Check] halo-painted map vs halo mass (Msun/h):")
        print(f"  sum(bnfw_map_low) = {sum_bnfw:.6e}")
        print(f"  sum(mass*h)       = {sum_halos:.6e}")
        if sum_halos > 0:
            print(f"  ratio             = {sum_bnfw/sum_halos:.6f}")

        print(f"[Output] saved bnfw map (base nside): {bnfw_path}", flush=True)
        print(f"[Output] saved composite map: {tot_path}", flush=True)
        return

    # ---------------------------------------------------------------------
    # PARTICLE mode: build particle KDTree + run profiles ONLY for target halos
    # ---------------------------------------------------------------------
    if part_pos_list:
        part_pos = np.concatenate(part_pos_list, axis=0).astype(np.float32)
        part_mass = np.concatenate(part_mass_list).astype(np.float64)
    else:
        part_pos = np.zeros((0, 3), dtype=np.float32)
        part_mass = np.zeros((0,), dtype=np.float64)

    print(f"[Particles] N={part_pos.shape[0]} total_mass={np.sum(part_mass):.6e} Msun/h", flush=True)
    sum_halos = float(np.sum(m_all * h))
    if sum_halos > 0:
        print(f"[Check] particles vs sum(mass*h): ratio = {float(np.sum(part_mass))/sum_halos:.6f}", flush=True)

    if part_pos.shape[0] == 0:
        print("[Profiles] no particles to profile; exiting.", flush=True)
        return

    # Particle KDTree for fast spatial queries in 3D
    global _PART_POS, _PART_MASS, _PART_TREE
    _PART_POS = part_pos
    _PART_MASS = part_mass
    _PART_TREE = cKDTree(_PART_POS)

    # Galaxy KDTree on unit sphere (for miscentering selection)
    global _GAL_RA2, _GAL_DEC2, _GAL_ZOBS2, _GAL_UNIT2, _GAL_TREE2
    _GAL_RA2 = gal_ra
    _GAL_DEC2 = gal_dec
    _GAL_ZOBS2 = gal_zobs
    if gal_ra.size > 0:
        _GAL_UNIT2 = unitvec_from_radec_deg(gal_ra, gal_dec).astype(np.float64)
        _GAL_TREE2 = cKDTree(_GAL_UNIT2)
    else:
        _GAL_UNIT2 = np.zeros((0, 3), dtype=np.float64)
        _GAL_TREE2 = None

    # Halo property globals used by miscentering/profile workers
    global _HALO_Dc, _HALO_R200_COM
    _HALO_Dc = halo_Dc
    _HALO_R200_COM = halo_R200_com

    # Profile radial bins (log-spaced)
    global _PROFILE_R_EDGES, _PROFILE_R_CENTERS
    nb = int(args.nbins)
    rmin = float(args.profile_rmin)
    rmax = float(args.profile_rmax)
    if rmin <= 0 or rmax <= rmin:
        raise ValueError("Invalid profile radii; require 0 < rmin < rmax.")
    _PROFILE_R_EDGES = np.logspace(np.log10(rmin), np.log10(rmax), nb + 1)
    _PROFILE_R_CENTERS = np.sqrt(_PROFILE_R_EDGES[:-1] * _PROFILE_R_EDGES[1:])

    # Apply mass cut ONLY for profiles, and ONLY on TARGET halos
    prof_cut = float(args.profile_mass_cut)
    target_mass = m_all[:N_target]
    prof_idx = np.where(target_mass >= prof_cut)[0].astype(np.int64)
    print(f"[Profiles] target halos above cut ({prof_cut:.3e} Msun): N={prof_idx.size}", flush=True)
    if prof_idx.size == 0:
        print("[Profiles] none selected; exiting.", flush=True)
        return

    # Profile parallelism
    profile_procs = int(args.profile_procs)
    if profile_procs <= 0:
        profile_procs = max(1, min(n_cores, prof_idx.size))
    print(f"[Profiles] using {profile_procs} processes  pool_chunksize={int(args.pool_chunksize)}", flush=True)

    t2 = time.time()
    t_last = t2
    out_rows = []
    done = 0
    total = int(prof_idx.size)

    # Run profiles in parallel
    with mp.Pool(processes=profile_procs) as pool:
        itp = pool.imap_unordered(_profile_worker, prof_idx,
                                  chunksize=int(args.pool_chunksize))
        for row in _tqdm(itp, total=total, desc="Particle profiles", unit="halo"):
            out_rows.append(row)
            done += 1
            t_last = _maybe_print_progress("Particle profiles", done, total,
                                            t_last, every_sec=float(args.monitor_every))

    t3 = time.time()
    print(f"[Profiles] done in {(t3 - t2)/60:.2f} min", flush=True)

    # Sort by halo index to keep outputs stable and easy to interpret
    out_rows.sort(key=lambda x: x[0])
    halo_idx_out = np.array([r[0] for r in out_rows], dtype=np.int64)
    ra_mis = np.array([r[1] for r in out_rows], dtype=np.float64)
    dec_mis = np.array([r[2] for r in out_rows], dtype=np.float64)
    R_mis = np.array([r[3] for r in out_rows], dtype=np.float64)
    Sigma = np.vstack([r[4] for r in out_rows]).astype(np.float64)

    # Save profiles NPZ
    prof_out_path = os.path.join(args.outdir, f"miscentered_particle_profiles_{label}.npz")
    np.savez(
        prof_out_path,
        r_edges=_PROFILE_R_EDGES,
        r_centers=_PROFILE_R_CENTERS,
        halo_indices=halo_idx_out,
        halo_mass=target_mass[halo_idx_out],
        halo_ra=ra_t[halo_idx_out],
        halo_dec=dec_t[halo_idx_out],
        halo_z=z_t[halo_idx_out],
        halo_Dc_com=halo_Dc[halo_idx_out],
        halo_R200_com=halo_R200_com[halo_idx_out],
        ra_mis=ra_mis,
        dec_mis=dec_mis,
        R_mis_com=R_mis,
        Sigma_mis=Sigma,             # Msun/h / (Mpc/h)^2 (comoving)
        depth_cyl=float(args.depth_cyl),
        profile_rmax=float(args.profile_rmax),
        miscenter_nmult=float(args.miscenter_nmult),
        miscenter_radius_factor=float(args.miscenter_radius_factor),
        particle_rmax_com=float(args.particle_rmax_com),
        neighbor_buffer_com=float(args.neighbor_buffer_com),
        p_mass_h=float(p_mass_h),
    )
    print(f"[Output] saved profiles: {prof_out_path}", flush=True)

    # Save miscentered halo catalogue NPZ:
    # We store miscentered theta/phi in radians to match Pinocchio-style catalogs.
    theta_mis = np.deg2rad(90.0 - dec_mis)
    phi_mis = np.deg2rad(ra_mis)

    halo_mis_cat_path = os.path.join(args.outdir, f"halo_catalog_miscentered_{label}.npz")
    np.savez(
        halo_mis_cat_path,
        theta=theta_mis.astype(np.float64),
        phi=phi_mis.astype(np.float64),
        z=z_t[halo_idx_out].astype(np.float64),
        Mass=target_mass[halo_idx_out].astype(np.float64),

        halo_indices=halo_idx_out.astype(np.int64),
        ra_true=ra_t[halo_idx_out].astype(np.float64),
        dec_true=dec_t[halo_idx_out].astype(np.float64),
        ra_mis=ra_mis.astype(np.float64),
        dec_mis=dec_mis.astype(np.float64),
        R_mis_com=R_mis.astype(np.float64),
        halo_Dc_com=halo_Dc[halo_idx_out].astype(np.float64),
        halo_R200_com=halo_R200_com[halo_idx_out].astype(np.float64),
        label=label,
    )
    print(f"[Output] saved miscentered halo catalogue: {halo_mis_cat_path}", flush=True)


if __name__ == "__main__":
    # Standard Python entry-point guard:
    # - ensures main() runs only when script executed directly
    # - avoids accidental execution when imported as a module
    main()
