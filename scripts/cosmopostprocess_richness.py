#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute global richness (lambda_obs) over multiple redshift shells.

This script is designed to:
  - Read many halo-shell NPZ catalogues (each shell has theta/phi/z/Mass for halos)
  - For each shell, read the *galaxy catalogue* previously generated for that shell
  - Merge all halos and galaxies from all shells into ONE global dataset
  - Compute (for *massive halos only*):
      * R200_comoving radius (via Colossus + NFW)
      * f_bkg (background fraction) using an *overlap annulus* method
      * lambda_obs (observed richness) using CosmoPostProcess's lambda_observed (P_mem model)
      * major contributor mapping: which galaxies contribute strongly (P_mem above a threshold)
      * lambda_obs_cut_pmem: richness using only major contributors (P_mem above threshold)

Key outputs:
  - One merged halo NPZ with lambda_true (from HOD), lambda_obs, lambda_obs_cut_pmem, fbkg, etc.
  - One merged galaxy NPZ plus a mapping of major (halo, galaxy, P_mem) pairs.

Important: This file is intended to be run after you have already created galaxy catalogs
for each shell (e.g. via your earlier galaxy painting script).
"""

import os
import re
import glob
import argparse
import warnings
import multiprocessing as mp

import numpy as np
from tqdm.auto import tqdm

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u_ast

from scipy.spatial import cKDTree

import torch

from colossus.cosmology import cosmology as col_cosmo
from colossus.halo import concentration, profile_nfw

from CosmoPostProcess.GalaxyPainting_RichProcessing.galaxy_functions import (
    lambda_observed,
    Model,
    calculate_total_overlap_fractions_batch,
)

# ----------------------------------------------------------
# Performance / stability knobs (important on HPC / clusters)
# ----------------------------------------------------------
# Many scientific Python stacks can try to use multiple threads internally
# (MKL/OpenBLAS/NumExpr/OpenMP). But THIS script also uses multiprocessing.
# If you allow each process to use multiple BLAS threads, you can oversubscribe
# cores badly (e.g. 64 processes × 16 threads = 1024 threads).
# So we force each worker process to be effectively single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# Torch by default can also use multiple threads; do the same for torch.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ==========================================================
# Cosmology configuration used throughout
# ==========================================================
# These constants determine distances, conversions, halo sizes, etc.
# These are later changed in the main() function with the parsed values 
Om0 = 0.32
H0 = 67.0
cosmo = FlatLambdaCDM(Om0=Om0, H0=H0)
h = H0 / 100.0

# ==========================================================
# P_mem neural model config
# ==========================================================
# P_mem is computed by lambda_observed using a trained neural model.
# The defaults here are specific to your CosmoPostProcess setup.
DEFAULT_PMEM_MODEL_PATH = (
    "/g100/home/userexternal/ringrao0/CosmoPostProcess/CosmoPostProcess/"
    "AssignationModels/best_model_opt_370deg2_cpz_pzwav_bkgf.pt"
)

# These kwargs must match how the model was trained.
DEFAULT_PMEM_MODEL_KWARGS = dict(
    input_size=4,
    hidden_size=100,
    output_size=1,
    dropout_rate=0.13884307002455876,
)
#DEFAULT_PMEM_MODEL_KWARGS = dict(
#    input_size=3,
#    hidden_size=100,
#    output_size=1,
#    dropout_rate=0.13884307002455876,
#)

# Feature scaling: (mean, std) or (shift, scale) arrays used by the model
# to normalize input features (depends on how CosmoPostProcess expects it).
DEFAULT_FEATURE_SCALING = (
    np.array([-0.4188571, 0.00528203, 0.1, 0.29777926]),
    np.array([0.41599655, 0.99999779, 2.47849679, 1.0]),
)

#DEFAULT_FEATURE_SCALING = (
#    np.array([0.        , 0.00470999, 0.1  ]),
#    np.array([1.37514035, 0.99999903, 2.48933601])
#)

# ==========================================================
# Colossus cosmology setup
# ==========================================================
# Colossus may attempt filesystem writes for persistence; on some batch nodes,
# HOME can be missing or unwritable. We robustly set HOME if needed, and try
# to disable persistence across different Colossus versions.
def prepare_colossus_cosmology():
    # Ensure HOME exists so Colossus doesn't crash when trying to access it.
    if "HOME" not in os.environ or not os.environ["HOME"]:
        os.environ["HOME"] = os.environ.get("SLURM_SUBMIT_DIR", "/tmp")

    # Define cosmology in Colossus and activate it.
    cosmo_pars = dict(flat=True, H0=args['H0'], Om0=args['Omega0'], Ob0=0.049, sigma8=0.811, ns=0.965)
    if "myCosmo" not in col_cosmo.cosmologies:
        col_cosmo.addCosmology("myCosmo", cosmo_pars)

    # Try to disable persistence (varies by Colossus version).
    try:
        col_cosmo.setCosmology("myCosmo", persistence="")
    except Exception:
        try:
            col_cosmo.setCosmology("myCosmo", persistence=None)
        except Exception:
            col_cosmo.setCosmology("myCosmo")


def compute_c200m(M200m, z_i):
    """
    Compute halo concentration c200m using a Colossus concentration model.

    Inputs:
      M200m : halo mass (Msun, *not* Msun/h)
      z_i   : halo redshift

    Returns:
      c200m : concentration (dimensionless)

    Notes:
      - The concentration model expects mass in Msun/h, so we multiply by h.
      - If concentration fails, we fall back to a typical value 4.0.
    """
    M_halo_h = float(M200m) * h
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = float(
            concentration.concentration(
                M_halo_h, mdef="200m", z=float(z_i), model="duffy08"
            )
        )
    if (not np.isfinite(c)) or (c <= 0):
        c = 4.0
    return c


def halo_R200_comoving_Mpc(M200m, z_i, c200m):
    """
    Compute R200m in *comoving* Mpc (not Mpc/h).

    Steps:
      1) Build an NFW profile in Colossus (expects mass in Msun/h).
      2) Ask Colossus for the physical radius RDelta (kpc/h physical).
      3) Convert kpc/h physical -> Mpc physical.
      4) Convert physical -> comoving by multiplying by (1+z).

    Returns:
      R200_com_Mpc : R200m in comoving Mpc
    """
    M_halo_h = float(M200m) * h
    prof = profile_nfw.NFWProfile(M=M_halo_h, c=float(c200m), z=float(z_i), mdef="200m")

    # Colossus gives RDelta in kpc/h physical:
    R200_kpch = float(prof.RDelta(float(z_i), "200m"))

    # Convert to Mpc physical (divide by 1000 kpc/Mpc, and also divide by h because it's kpc/h):
    R200_phys_Mpc = R200_kpch / (1000.0 * h)

    # Comoving Mpc = physical Mpc × (1+z):
    R200_com_Mpc = R200_phys_Mpc * (1.0 + float(z_i))
    return R200_com_Mpc


# ==========================================================
# Shell label & edge parsing helpers
# ==========================================================
# These are used to:
#  - match halos to galaxies (galaxy catalogs are stored per shell label)
#  - optionally filter shells by z_max_shell
def shell_label_from_npz_path(npz_path: str) -> str:
    """
    If filename contains pattern _z<z1>_<z2>.npz, return label "z<z1>_<z2>".
    Otherwise return basename without extension.
    """
    base = os.path.basename(npz_path)
    m = re.search(r"_z([0-9.]+)_([0-9.]+)\.npz$", base)
    if m:
        z1, z2 = m.group(1), m.group(2)
        return f"z{z1}_{z2}"
    return os.path.splitext(base)[0]


def shell_edges_from_npz_path(npz_path: str):
    """
    If filename contains _z<z1>_<z2>.npz, return numeric (z_lo, z_hi).
    Otherwise return None.
    """
    base = os.path.basename(npz_path)
    m = re.search(r"_z([0-9.]+)_([0-9.]+)\.npz$", base)
    if not m:
        return None
    z1 = float(m.group(1))
    z2 = float(m.group(2))
    return (min(z1, z2), max(z1, z2))


# ==========================================================
# Geometry helpers: unit vectors + 3D halo positions
# ==========================================================
def unitvec_from_radec_deg(ra_deg, dec_deg):
    """
    Convert RA/Dec in degrees -> 3D unit vector on the sphere.

    Supports scalars or arrays:
      - If scalar input, returns shape (3,)
      - If array input, returns shape (N,3)
    """
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    if x.ndim == 0:
        return np.array([float(x), float(y), float(z)])
    return np.column_stack([x, y, z])


def halo_pos3d_com_mpc_over_h(ra_deg, dec_deg, Dc_Mpc):
    """
    Convert (RA, Dec, Dc[Mpc]) -> comoving Cartesian position in cMpc/h.

    Why this function exists:
      - f_bkg overlap code expects positions in comoving Mpc/h.
      - We have comoving distances Dc in Mpc from astropy.
      - Convert to Mpc/h by multiplying by h.

    Output:
      - array shape (3,) or (N,3) of XYZ positions in Mpc/h
    """
    u = unitvec_from_radec_deg(ra_deg, dec_deg).astype(float)
    Dc_mpc_over_h = np.asarray(Dc_Mpc, dtype=float) * h
    if u.ndim == 1:
        return u * float(Dc_mpc_over_h)
    return u * Dc_mpc_over_h[:, None]


# ==========================================================
# GLOBALS used by multiprocessing workers
# ==========================================================
# These are set in main() before creating mp.Pool.
# Workers access them read-only (safe under fork on Linux HPC).
halo_ra = None
halo_dec = None
halo_z = None
halo_mass = None
halo_Dc_Mpc = None
halo_R200_com_Mpc = None

# For f_bkg we use a 3D KDTree over halo positions (Mpc/h) and store radii (Mpc/h).
halo_pos3d_mpc_over_h = None
halo_radii_mpc_over_h = None
halo_tree_3d = None
halo_is_massive = None
_max_halo_radius_mpc_over_h = None

# Galaxies: we build a KDTree on the *unit sphere* to query nearby galaxies fast.
gal_ra = None
gal_dec = None
gal_z_obs = None
gal_Dc_Mpc = None
gal_unit = None
gal_tree = None

# P_mem model and config in each worker
_pmem_model = None
_pmem_model_kwargs = None
_pmem_feature_scaling = None
_pmem_device = torch.device("cpu")

# lambda_observed can itself parallelize internally using "num_workers"
# (typically torch multiprocessing/threads). We control it here.
_LAMBDA_NUM_WORKERS = 4

# threshold used to define "major contributor" galaxies
_PMEM_MAJOR_MIN = 0.05


# ==========================================================
# P_mem model loading (per worker)
# ==========================================================
def _load_pmem_model(model_path: str):
    """
    Load the trained P_mem model once in each worker process.

    Why per worker?
      - Torch models are not trivially shared across processes.
      - Each worker must have its own model instance.
    """
    global _pmem_model, _pmem_model_kwargs, _pmem_feature_scaling
    _pmem_model_kwargs = dict(DEFAULT_PMEM_MODEL_KWARGS)
    _pmem_feature_scaling = DEFAULT_FEATURE_SCALING

    _pmem_model = Model(**_pmem_model_kwargs).to(_pmem_device)
    state = torch.load(model_path, map_location=_pmem_device)
    _pmem_model.load_state_dict(state)
    _pmem_model.eval()


# ==========================================================
# P_mem computation wrapper
# ==========================================================
def compute_pmem_for_center(
    ra_c,
    dec_c,
    z_cl,
    R200_com,
    gal_ra_sel,
    gal_dec_sel,
    z_gal_obs_sel,
    f_bkg=1.0,
    D_cl=None,
    D_gal=None,
):
    """
    Compute membership probabilities P_mem for a set of candidate galaxies
    around a single halo center.

    Inputs:
      ra_c, dec_c : halo center in degrees
      z_cl        : halo redshift
      R200_com    : halo R200 in comoving Mpc
      gal_*_sel   : arrays for candidate galaxies (subset)
      f_bkg       : background fraction for this halo (computed via annulus overlap)
      D_cl        : optional precomputed halo comoving distance (Mpc)
      D_gal       : optional precomputed galaxy comoving distance array (Mpc)

    Output:
      pm : array of P_mem in [0,1], aligned with the input candidate galaxy arrays.

    Notes on geometry:
      - lambda_observed expects "gal_pos" in a specific coordinate convention:
          gal_pos = [D_gal, x_t, y_t] where x_t,y_t are transverse distances
          computed from small-angle approximations around the center.
      - center_vec = [D_cl, 0, 0] is the halo in that coordinate system.
    """
    gal_ra_sel = np.asarray(gal_ra_sel)
    gal_dec_sel = np.asarray(gal_dec_sel)
    z_gal_obs_sel = np.asarray(z_gal_obs_sel)
    if gal_ra_sel.size == 0:
        return np.array([])

    # Distances in Mpc (NOT Mpc/h). Astropy uses the cosmology defined above.
    if D_cl is None:
        D_cl = cosmo.comoving_distance(z_cl).to(u_ast.Mpc).value
    if D_gal is None:
        D_gal = cosmo.comoving_distance(z_gal_obs_sel).to(u_ast.Mpc).value
    else:
        D_gal = np.asarray(D_gal, dtype=float)

    # Convert angles to radians.
    ra_c_rad = np.deg2rad(ra_c)
    dec_c_rad = np.deg2rad(dec_c)
    ra_rad = np.deg2rad(gal_ra_sel)
    dec_rad = np.deg2rad(gal_dec_sel)

    # Small-angle approximation:
    #   dx_ang ~ ΔRA*cos(Dec_center)
    #   dy_ang ~ ΔDec
    d_alpha = ra_rad - ra_c_rad
    d_delta = dec_rad - dec_c_rad
    dx_ang = d_alpha * np.cos(dec_c_rad)
    dy_ang = d_delta

    # Transverse physical/comoving offsets in Mpc (since D_cl is Mpc).
    x_t = D_cl * dx_ang
    y_t = D_cl * dy_ang

    # Build galaxy positions in the coordinate system expected by lambda_observed.
    gal_pos = np.column_stack([D_gal, x_t, y_t])
    center_vec = np.array([D_cl, 0.0, 0.0])

    # Halo radius scale used internally by lambda_observed.
    r0 = float(R200_com)

    # z_cl_obs is an array; here we just set it equal to z_cl for all candidates.
    z_cl_obs = np.full_like(z_gal_obs_sel, z_cl, dtype=float)

    # lambda_observed expects fbkg as an array aligned to "targets"; here we have 1 target.
    single_fbkg = np.array([float(f_bkg)], dtype=float)

    # Call the CosmoPostProcess pipeline:
    # It returns (lambda_obs, ...) and P_mem values pm.
    _, _, pm = lambda_observed(
        center_vec,
        single_fbkg,
        z_cl,
        z_cl_obs,
        z_gal_obs_sel,
        gal_pos,
        r0,
        np.ones(len(gal_pos)),   # weights (all 1 here)
        axis=0,
        boxsize=1e5,             # large box to avoid boundary artifacts
        model_class=Model,
        model_kwargs=_pmem_model_kwargs,
        model_state_dict=_pmem_model.state_dict(),
        Omega_m=args['Omega0'],
        h=h,
        feature_scaling=_pmem_feature_scaling,
        num_workers=_LAMBDA_NUM_WORKERS,
    )

    # Ensure numpy array output and clip to [0,1].
    if isinstance(pm, torch.Tensor):
        pm = pm.cpu().numpy()
    pm = np.asarray(pm, dtype=float).ravel()
    return np.clip(pm, 0.0, 1.0)


def get_galaxies_within_radius(ra_c, dec_c, Dc_Mpc_val, Rmax_com):
    """
    Fast angular query: find galaxies within projected radius Rmax_com (Mpc)
    around a halo at comoving distance Dc_Mpc_val.

    Implementation details:
      - We store galaxy positions as unit vectors on the sphere in gal_tree.
      - We convert the halo center to a unit vector vec_c.
      - We approximate a spherical cap query using chord distance in 3D unit-vector space:
          chord = 2 sin(theta/2)
        where theta is the angular radius corresponding to Rmax_com at distance Dc.

    Returns:
      idx : indices into the global galaxy arrays (gal_ra, gal_dec, gal_z_obs, ...)
    """
    if gal_tree is None or gal_unit is None or gal_unit.shape[0] == 0:
        return np.asarray([], dtype=int)

    ra_c_rad = np.deg2rad(ra_c)
    dec_c_rad = np.deg2rad(dec_c)
    vec_c = np.array(
        [
            np.cos(dec_c_rad) * np.cos(ra_c_rad),
            np.cos(dec_c_rad) * np.sin(ra_c_rad),
            np.sin(dec_c_rad),
        ]
    )

    # theta_max is small-angle: R = D * theta  -> theta ~ R/D
    theta_max = float(Rmax_com) / float(Dc_Mpc_val)

    # Convert angular radius to chord distance on unit sphere.
    r_tree = 2.0 * np.sin(theta_max / 2.0)

    idx = gal_tree.query_ball_point(vec_c, r_tree)
    return np.asarray(idx, dtype=int)


# ==========================================================
# f_bkg computation (requested "fixed annulus 3–5 cMpc/h")
# ==========================================================
# These define the annulus in comoving Mpc/h, not scaled by halo size.
_ANNULUS_RMIN = 3.0  # cMpc/h
_ANNULUS_RMAX = 5.0  # cMpc/h

# We clip fbkg to avoid zeros/nans that might break later computations.
_FBKG_CLIP_MIN = 1e-6
_FBKG_CLIP_MAX = 1.0


def compute_fbkg_for_halo(i_halo: int) -> float:
    """
    Compute f_bkg for one halo using overlap accounting in a fixed annulus.

    Concept:
      - f_bkg here represents "fraction of annulus area not overlapped"
        by other halos' projected footprints (implementation depends on CosmoPostProcess).
      - We use calculate_total_overlap_fractions_batch(..., overlap_type="annulus")
        which internally uses a fixed annulus with r_min=3 and r_max=5 cMpc/h.
      - IMPORTANT: annulus does NOT scale with halo R200 (per your request).

    Implementation strategy:
      - Build a 3D KDTree over halo positions (Mpc/h).
      - For halo i, only consider neighbors within:
            r_search = ANNULUS_RMAX + max_halo_radius
        because only those can overlap the annulus region.
      - Call overlap function on this small subset for speed.

    Returns:
      fbkg in [~0,1]. Defaults to 1.0 on failures / for non-massive halos.
    """
    i = int(i_halo)

    # By design: only compute fbkg for "massive" halos.
    if not bool(halo_is_massive[i]):
        return 1.0

    # If trees not built, fall back.
    if (halo_tree_3d is None) or (halo_pos3d_mpc_over_h is None) or (halo_radii_mpc_over_h is None):
        return 1.0

    pos_i = halo_pos3d_mpc_over_h[i]
    if not np.isfinite(pos_i).all():
        return 1.0

    # Neighbor search radius: annulus outer edge + maximum halo footprint radius.
    r_search = float(_ANNULUS_RMAX + _max_halo_radius_mpc_over_h)

    # Candidate neighbors in 3D.
    idx = halo_tree_3d.query_ball_point(pos_i, r=r_search)
    if not idx:
        return 1.0
    idx = np.asarray(idx, dtype=int)

    # Only keep valid massive halos with valid radii/positions.
    valid = (
        halo_is_massive[idx]
        & np.isfinite(halo_radii_mpc_over_h[idx])
        & (halo_radii_mpc_over_h[idx] > 0)
        & np.isfinite(halo_pos3d_mpc_over_h[idx]).all(axis=1)
    )
    idx = idx[valid]
    if idx.size == 0:
        return 1.0

    # Ensure the target halo itself is included in the set passed to the function.
    if i not in idx:
        idx = np.concatenate([idx, np.array([i], dtype=int)])

    # Coordinates and radii arrays for the subset (Mpc/h).
    coords = halo_pos3d_mpc_over_h[idx].astype(float)
    radii = halo_radii_mpc_over_h[idx].astype(float)

    # Target is a single halo position.
    target = halo_pos3d_mpc_over_h[i].reshape(1, 3).astype(float)

    # target_radii is required by API but overlap_type="annulus" ignores it;
    # kept for compatibility.
    target_r = np.array([halo_radii_mpc_over_h[i]], dtype=float)

    # Call CosmoPostProcess overlap computation.
    try:
        fbkg = calculate_total_overlap_fractions_batch(
            coordinates=coords,
            radii=radii,
            targets=target,
            target_radii=target_r,
            overlap_type="annulus",
        )
        fbkg = float(np.asarray(fbkg, dtype=float).ravel()[0])
    except Exception:
        return 1.0

    if not np.isfinite(fbkg):
        fbkg = 1.0

    # Clip to safe range.
    fbkg = float(np.clip(fbkg, _FBKG_CLIP_MIN, _FBKG_CLIP_MAX))
    return fbkg


# ==========================================================
# Multiprocessing worker initialization
# ==========================================================
def _richness_init_worker(model_path, lambda_workers, pmem_major_min):
    """
    This runs once in each worker process when the Pool starts.

    It:
      - configures lambda_observed's internal parallelism (num_workers)
      - sets the P_mem major threshold
      - loads the neural P_mem model into that worker
    """
    global _LAMBDA_NUM_WORKERS, _PMEM_MAJOR_MIN
    _LAMBDA_NUM_WORKERS = int(lambda_workers)
    _PMEM_MAJOR_MIN = float(pmem_major_min)
    _load_pmem_model(model_path)


def richness_worker(i_halo):
    """
    Compute richness (lambda_obs) and major-contributor mapping for ONE halo.

    Returned tuple:
      (i_halo,
       lambda_obs,
       lambda_obs_cut_pmem,
       fbkg,
       major_gal_indices,
       major_pmem_values)

    Notes:
      - Candidate galaxies are selected by angular/projected distance <= R200_com.
      - P_mem is computed for all candidates.
      - Major contributors are those with P_mem >= _PMEM_MAJOR_MIN.
      - lambda_obs is sum(P_mem) over all candidates.
      - lambda_obs_cut_pmem is sum(P_mem) only over major contributors.
    """
    i = int(i_halo)

    ra_c = float(halo_ra[i])
    dec_c = float(halo_dec[i])
    z_cl = float(halo_z[i])
    R200_com = float(halo_R200_com_Mpc[i])
    Dc_i = float(halo_Dc_Mpc[i])

    # If halo has invalid geometry, we cannot compute anything meaningful.
    if (R200_com <= 0) or (not np.isfinite(R200_com)) or (Dc_i <= 0) or (not np.isfinite(Dc_i)):
        return (i, 0.0, 0.0, 1.0, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

    # Compute local background fraction for this halo
    fbkg_i = compute_fbkg_for_halo(i)

    # Candidate galaxies within projected R200
    idx_cand = get_galaxies_within_radius(ra_c, dec_c, Dc_i, R200_com)
    if idx_cand.size == 0:
        return (i, 0.0, 0.0, fbkg_i, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

    # Extract candidate galaxy observables
    ra_all = gal_ra[idx_cand]
    dec_all = gal_dec[idx_cand]
    z_all = gal_z_obs[idx_cand]

    # Compute P_mem for those candidates
    pm = compute_pmem_for_center(
        ra_c, dec_c, z_cl, R200_com,
        ra_all, dec_all, z_all,
        f_bkg=fbkg_i,
        D_cl=Dc_i,                    # pass precomputed halo Dc
        D_gal=gal_Dc_Mpc[idx_cand],    # pass precomputed galaxy Dc
    )

    if pm.size == 0:
        return (i, 0.0, 0.0, fbkg_i, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

    # Richness definition: sum of membership probabilities
    lam = float(np.sum(pm))

    # Major contributors: P_mem above threshold
    major_mask = pm >= _PMEM_MAJOR_MIN
    lam_cut = float(np.sum(pm[major_mask])) if np.any(major_mask) else 0.0

    # If none are major, return empty arrays for mappings
    if not np.any(major_mask):
        return (i, lam, lam_cut, fbkg_i, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

    # Convert back to GLOBAL galaxy indices (not local candidate indices)
    major_idx_global = idx_cand[major_mask].astype(np.int64)
    major_pmem = pm[major_mask].astype(np.float32)

    return (i, lam, lam_cut, fbkg_i, major_idx_global, major_pmem)


# ==========================================================
# main program
# ==========================================================
def main():
    """
    High-level pipeline:

    1) Discover halo shell files (NPZ) under --halo-root using --halo-pattern
       Optionally filter shells by --z-max-shell based on filename.
    2) For each shell:
         - Load halo catalogue (theta, phi, z, Mass)
         - Load the corresponding galaxy catalogue under --galaxy-root/<label>/galaxies_catalog_<label>.npz
         - Derive lambda_true per halo from galaxy->halo mapping if available
    3) Merge all shells into global arrays
    4) Identify "massive halos" by --m-threshold
    5) Compute Dc for all halos; compute R200 only for massive halos
    6) Build:
         - global galaxy KDTree on unit vectors (sky queries)
         - global halo 3D KDTree for f_bkg overlap calculation
    7) Parallelize richness calculation across massive halos only
    8) Save merged halo NPZ and merged galaxy NPZ
    """
    parser = argparse.ArgumentParser(description="Compute global richness (lambda_obs) over all shells.")

    # ----------------------------
    # Required filesystem inputs
    # ----------------------------
    parser.add_argument("--halo-root", required=True)
    parser.add_argument("--halo-pattern", default="plc_shell_thetaphizmass_z*.npz")
    parser.add_argument("--galaxy-root", required=True)

    # ----------------------------
    # Required outputs
    # ----------------------------
    parser.add_argument("--out-halo-npz", required=True)
    parser.add_argument("--out-galaxy-npz", required=True)

    # ----------------------------
    # Model / runtime controls
    # ----------------------------
    parser.add_argument("--pmem-model-path", default=DEFAULT_PMEM_MODEL_PATH)
    parser.add_argument("--lambda-num-workers", type=int, default=4)
    parser.add_argument("--procs", type=int, default=0)

    # ----------------------------
    # PARAMETER FILE for setup/cosmo
    # ----------------------------
    parser.add_argument("--paramfile-path", required=True,
                        help="Path to the Pinocchio parameter file.")

    # ----------------------------
    # Scientific selection controls
    # ----------------------------
    parser.add_argument("--z-max-shell", type=float, default=None)
    parser.add_argument("--m-threshold", type=float, default=1e13)
    parser.add_argument("--pmem-major-threshold", type=float, default=0.05)

    # ----------------------------
    # UI/verbosity controls
    # ----------------------------
    parser.add_argument("--no-progress", action="store_true")

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Pinocchio Cosmology 
    # ---------------------------------------------------------------------
    # this reads the Pinocchio parameter file and extracts:
    # - Hubble100
    # - Omega0
    global H0, h, Om0, cosmo

    P = params_file()
    P.load(args.paramfile_path)

    h = P.cosmo['Hubble100']
    Om0 = P.cosmo['Omega0']
    H0 = h * 100.0

    cosmo = FlatLambdaCDM(Om0=Om0, H0=H0)

    # Prefer "fork" on Linux HPC for speed / memory sharing when possible.
    try:
        if "fork" in mp.get_all_start_methods():
            mp.set_start_method("fork", force=False)
    except RuntimeError:
        pass

    # Progress-bar wrapper: if --no-progress, we use identity wrapper.
    use_tqdm = not args.no_progress
    twrap = (lambda it, **kw: tqdm(it, **kw)) if use_tqdm else (lambda it, **kw: it)

    # -------------------------------
    # 1. Find halo shells
    # -------------------------------
    # Find halo NPZ files under halo-root matching halo-pattern.
    halo_glob = os.path.join(args.halo_root, args.halo_pattern)
    halo_files_all = sorted(glob.glob(halo_glob))

    # Some file variants (tidal_proxy) are explicitly excluded from "mass only" processing.
    halo_files_mass = [f for f in halo_files_all if "tidal_proxy" not in os.path.basename(f)]

    # Optional: filter shells by the z upper bound from filename.
    if args.z_max_shell is not None:
        z_max = float(args.z_max_shell)
        halo_files = []
        skipped = 0
        for f in halo_files_mass:
            edges = shell_edges_from_npz_path(f)
            if edges is None:
                skipped += 1
                continue
            _, z_hi = edges
            if z_hi <= z_max:
                halo_files.append(f)
            else:
                skipped += 1
        print(f"[Input] z_max_shell = {z_max:.3f} -> using {len(halo_files)} shells, skipped {skipped}.")
    else:
        halo_files = halo_files_mass
        print("[Input] z_max_shell not set -> using all shells.")

    if not halo_files:
        raise RuntimeError(f"No halo files remain after filtering (pattern {halo_glob}).")

    print(f"[Input] Found {len(halo_files)} halo shells (mass only, after cuts).")
    print(f"[Input] halo-root    = {args.halo_root}")
    print(f"[Input] galaxy-root  = {args.galaxy_root}")

    # -------------------------------
    # 2. Load & merge halo + galaxy per shell (compute lambda_true from HOD)
    # -------------------------------
    # We accumulate per-shell arrays, then concatenate all shells at once.
    halo_theta_list = []
    halo_phi_list = []
    halo_z_list = []
    halo_mass_list = []
    halo_label_list = []
    halo_local_index_list = []
    lambda_true_list = []

    gal_ra_list = []
    gal_dec_list = []
    gal_z_obs_list = []
    gal_Dc_Mpc_list = []
    gal_parent_shell_label_list = []
    gal_parent_halo_local_list = []

    print("[Step] Reading shells and building lambda_true per halo...")

    for ih, halo_path in enumerate(twrap(halo_files, desc="Shells", unit="shell")):
        base = os.path.basename(halo_path)
        label = shell_label_from_npz_path(halo_path)
        print(f"[Shell] {ih}: {base} -> label={label}")

        # Load halo shell catalogue
        hcat = np.load(halo_path)
        theta = np.asarray(hcat["theta"], dtype=float)
        phi = np.asarray(hcat["phi"], dtype=float)
        z = np.asarray(hcat["z"], dtype=float)
        mass = np.asarray(hcat["Mass"], dtype=float)

        N_halo_shell = theta.size
        if N_halo_shell == 0:
            print("  [Warn] Empty halo shell, skipping.")
            continue

        # Galaxy catalog must be found in:
        #   galaxy_root/<label>/galaxies_catalog_<label>.npz
        gal_dir = os.path.join(args.galaxy_root, label)
        gal_path = os.path.join(gal_dir, f"galaxies_catalog_{label}.npz")
        if not os.path.isfile(gal_path):
            print(f"  [Warn] Galaxy catalog not found for shell {label}: {gal_path} (skipping shell)")
            continue

        # Load galaxy shell catalogue
        gcat = np.load(gal_path)
        gal_ra_shell = np.asarray(gcat["gal_ra"], dtype=float)
        gal_dec_shell = np.asarray(gcat["gal_dec"], dtype=float)
        gal_z_obs_shell = np.asarray(gcat["gal_z_obs"], dtype=float)

        # Precompute comoving distance for each galaxy once (used by P_mem computation).
        gal_Dc_shell = cosmo.comoving_distance(gal_z_obs_shell).to(u_ast.Mpc).value.astype(float)

        # lambda_true is derived from how many *target* galaxies are assigned to each halo.
        # This requires gal_halo_idx_target (local halo index in the shell) to exist.
        if "gal_halo_idx_target" in gcat:
            gal_halo_idx_target = np.asarray(gcat["gal_halo_idx_target"], dtype=int)
            mask_target_gal = gal_halo_idx_target >= 0

            # bincount yields counts per halo local index -> lambda_true per halo.
            lambda_true_shell = np.bincount(
                gal_halo_idx_target[mask_target_gal],
                minlength=N_halo_shell,
            ).astype(float)

            # record for each galaxy its parent halo local index (or -1 if not a target)
            gal_parent_halo_local = np.where(mask_target_gal, gal_halo_idx_target, -1).astype(int)
        else:
            print("  [Warn] gal_halo_idx_target not in galaxy catalog; setting lambda_true=0.")
            lambda_true_shell = np.zeros(N_halo_shell, dtype=float)
            gal_parent_halo_local = -np.ones(gal_ra_shell.size, dtype=int)

        # Append halo shell data for later concatenation.
        halo_theta_list.append(theta)
        halo_phi_list.append(phi)
        halo_z_list.append(z)
        halo_mass_list.append(mass)
        halo_label_list.append(np.full(N_halo_shell, label, dtype=object))
        halo_local_index_list.append(np.arange(N_halo_shell, dtype=int))
        lambda_true_list.append(lambda_true_shell)

        # Append galaxy shell data for later concatenation.
        gal_ra_list.append(gal_ra_shell)
        gal_dec_list.append(gal_dec_shell)
        gal_z_obs_list.append(gal_z_obs_shell)
        gal_Dc_Mpc_list.append(gal_Dc_shell)
        gal_parent_shell_label_list.append(np.full(gal_ra_shell.size, label, dtype=object))
        gal_parent_halo_local_list.append(gal_parent_halo_local)

    # -------------------------------
    # 3. Merge all shells
    # -------------------------------
    if not halo_theta_list:
        raise RuntimeError("No halos collected from any shell.")

    theta_all = np.concatenate(halo_theta_list)
    phi_all = np.concatenate(halo_phi_list)
    z_all = np.concatenate(halo_z_list)
    mass_all = np.concatenate(halo_mass_list)
    shell_label_all = np.concatenate(halo_label_list)
    shell_local_index_all = np.concatenate(halo_local_index_list)
    lambda_true_all = np.concatenate(lambda_true_list).astype(np.float32)

    # Convert (theta,phi) radians to RA/Dec degrees.
    ra_all = np.rad2deg(phi_all)
    dec_all = 90.0 - np.rad2deg(theta_all)

    N_halo_tot = ra_all.size
    halo_index = np.arange(N_halo_tot, dtype=int)

    print(f"[Merge] Total halos: {N_halo_tot}")

    gal_ra_all = np.concatenate(gal_ra_list) if gal_ra_list else np.empty(0, dtype=float)
    gal_dec_all = np.concatenate(gal_dec_list) if gal_dec_list else np.empty(0, dtype=float)
    gal_z_obs_all = np.concatenate(gal_z_obs_list) if gal_z_obs_list else np.empty(0, dtype=float)
    gal_Dc_Mpc_all = np.concatenate(gal_Dc_Mpc_list) if gal_Dc_Mpc_list else np.empty(0, dtype=float)
    gal_parent_shell_label_all = (
        np.concatenate(gal_parent_shell_label_list)
        if gal_parent_shell_label_list else np.empty(0, dtype=object)
    )
    gal_parent_halo_local_all = (
        np.concatenate(gal_parent_halo_local_list)
        if gal_parent_halo_local_list else np.empty(0, dtype=int)
    )

    N_gal_tot = gal_ra_all.size
    gal_index = np.arange(N_gal_tot, dtype=int)

    print(f"[Merge] Total galaxies: {N_gal_tot}")

    # -----------------------------------------
    # 3b. Mass threshold: pick "massive halos"
    # -----------------------------------------
    m_thresh = float(args.m_threshold)
    massive_mask = mass_all >= m_thresh
    massive_indices = np.where(massive_mask)[0]
    N_massive = massive_indices.size

    print(f"[Mass cut] m_threshold = {m_thresh:.3e}")
    print(f"[Mass cut] massive halos = {N_massive} / {N_halo_tot}")

    # -------------------------------
    # 4. Distances and R200
    # -------------------------------
    # Dc in Mpc for all halos: used for sky-to-transverse conversions and P_mem model inputs.
    print("[Step] Computing halo comoving distances (Dc_Mpc) for all halos...")
    halo_Dc_all = cosmo.comoving_distance(z_all).to(u_ast.Mpc).value.astype(float)

    # R200 is only needed for massive halos; compute it using Colossus.
    print("[Step] Computing halo R200_com_Mpc via Colossus (massive halos only)...")
    prepare_colossus_cosmology()

    halo_R200_com_all = np.zeros(N_halo_tot, dtype=float)
    if N_massive == 0:
        print("[R200] No halos above mass threshold; all R200_com_Mpc = 0.")
    else:
        for i in twrap(massive_indices, desc="R200 (massive)", unit="halo"):
            M200m_i = float(mass_all[i])
            z_i = float(z_all[i])
            if (M200m_i <= 0) or (not np.isfinite(M200m_i)):
                halo_R200_com_all[i] = 0.0
                continue
            c200m = compute_c200m(M200m_i, z_i)
            halo_R200_com_all[i] = halo_R200_comoving_Mpc(M200m_i, z_i, c200m)

    # -------------------------------
    # 5. Build global galaxy KDTree (sky)
    # -------------------------------
    # We store galaxies as unit vectors and query them by angular radius efficiently.
    print("[Step] Building global galaxy KDTree (sky unit vectors)...")
    global gal_ra, gal_dec, gal_z_obs, gal_Dc_Mpc, gal_unit, gal_tree
    gal_ra = gal_ra_all
    gal_dec = gal_dec_all
    gal_z_obs = gal_z_obs_all
    gal_Dc_Mpc = gal_Dc_Mpc_all

    if N_gal_tot > 0:
        gal_unit = unitvec_from_radec_deg(gal_ra, gal_dec).astype(float)
        gal_tree = cKDTree(gal_unit)
    else:
        gal_unit = np.zeros((0, 3), dtype=float)
        gal_tree = None

    # -------------------------------
    # 6. Build halo 3D KDTree for f_bkg (Mpc/h)
    # -------------------------------
    # Overlap function uses 3D halo positions (Mpc/h) and halo footprint radii (Mpc/h).
    print("[Step] Building halo 3D KDTree for f_bkg (comoving Mpc/h)...")
    global halo_pos3d_mpc_over_h, halo_radii_mpc_over_h, halo_tree_3d, halo_is_massive, _max_halo_radius_mpc_over_h

    halo_is_massive = massive_mask.astype(bool)
    halo_pos3d_mpc_over_h = halo_pos3d_com_mpc_over_h(ra_all, dec_all, halo_Dc_all).astype(float)

    # radii in Mpc/h: R200_com_Mpc × h
    halo_radii_mpc_over_h = (halo_R200_com_all.astype(float) * h)

    # Determine which massive halos are valid for tree + radius computations.
    idx_massive_valid = massive_indices[
        np.isfinite(halo_pos3d_mpc_over_h[massive_indices]).all(axis=1)
        & np.isfinite(halo_radii_mpc_over_h[massive_indices])
        & (halo_radii_mpc_over_h[massive_indices] > 0)
    ]

    if idx_massive_valid.size == 0:
        halo_tree_3d = None
        _max_halo_radius_mpc_over_h = 0.0
        print("[f_bkg] No valid massive halos for KDTree; f_bkg will default to 1.")
    else:
        # Note: the tree contains ALL halos, but compute_fbkg_for_halo filters by massive_mask.
        halo_tree_3d = cKDTree(halo_pos3d_mpc_over_h.astype(float))
        _max_halo_radius_mpc_over_h = float(np.nanmax(halo_radii_mpc_over_h[idx_massive_valid]))
        print(f"[f_bkg] annulus fixed: [{_ANNULUS_RMIN:.1f},{_ANNULUS_RMAX:.1f}] cMpc/h")
        print(f"[f_bkg] max halo radius (Mpc/h) among valid massive = {_max_halo_radius_mpc_over_h:.3f}")

    # -------------------------------
    # 7. Publish halo arrays as globals for workers
    # -------------------------------
    global halo_ra, halo_dec, halo_z, halo_mass, halo_Dc_Mpc, halo_R200_com_Mpc
    halo_ra = ra_all
    halo_dec = dec_all
    halo_z = z_all
    halo_mass = mass_all
    halo_Dc_Mpc = halo_Dc_all
    halo_R200_com_Mpc = halo_R200_com_all

    # -------------------------------
    # 8. Parallel richness computation (massive halos only)
    # -------------------------------
    print("[Step] Computing lambda_obs for massive halos (parallel)...")

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or "0")
    n_cores = slurm_cpus if slurm_cpus > 0 else mp.cpu_count()
    procs = int(args.procs) if int(args.procs) > 0 else n_cores
    procs = max(1, procs)

    print(f"[Richness] Using {procs} processes on {n_cores} cores.")
    print(f"[Richness] lambda-num-workers (inner P_mem workers) = {args.lambda_num_workers}")
    print(f"[Richness] P_mem major threshold = {float(args.pmem_major_threshold):.3f}")
    print(f"[Richness] f_bkg: fixed annulus 3–5 cMpc/h (no R200 scaling)")

    # Arrays sized for ALL halos; we fill only massive ones and leave defaults for others.
    lambda_obs_all = np.zeros(N_halo_tot, dtype=np.float32)
    lambda_obs_cut_all = np.zeros(N_halo_tot, dtype=np.float32)
    fbkg_all = np.ones(N_halo_tot, dtype=np.float32)

    # For major contributor mapping we store sparse pair lists:
    # (halo_index, gal_index, pmem_value) for each major contributor.
    major_halo_lists = []
    major_gal_lists = []
    major_pmem_lists = []

    if N_massive == 0:
        print("[Richness] No massive halos; all lambda_obs = 0.")
    else:
        with mp.Pool(
            processes=procs,
            initializer=_richness_init_worker,
            initargs=(args.pmem_model_path, int(args.lambda_num_workers), float(args.pmem_major_threshold)),
        ) as pool:
            it = pool.imap_unordered(richness_worker, massive_indices, chunksize=32)

            for i_halo, lam, lam_cut, fbkg_i, major_idx_g, major_p in twrap(
                it, total=N_massive, desc="Richness+f_bkg (massive)", unit="halo"
            ):
                lambda_obs_all[i_halo] = float(lam)
                lambda_obs_cut_all[i_halo] = float(lam_cut)
                fbkg_all[i_halo] = float(fbkg_i)

                # If there are major contributors, store the mapping as sparse lists.
                if major_idx_g.size > 0:
                    major_halo_lists.append(np.full(major_idx_g.size, int(i_halo), dtype=np.int64))
                    major_gal_lists.append(major_idx_g.astype(np.int64))
                    major_pmem_lists.append(major_p.astype(np.float32))

    # Concatenate sparse mapping lists into final arrays.
    if major_halo_lists:
        major_pair_halo_index = np.concatenate(major_halo_lists).astype(np.int64)
        major_pair_gal_index = np.concatenate(major_gal_lists).astype(np.int64)
        major_pair_pmem = np.concatenate(major_pmem_lists).astype(np.float32)
    else:
        major_pair_halo_index = np.empty(0, dtype=np.int64)
        major_pair_gal_index = np.empty(0, dtype=np.int64)
        major_pair_pmem = np.empty(0, dtype=np.float32)

    print(f"[Richness] Major-contributor pairs: {major_pair_halo_index.size}")

    # -------------------------------
    # Monitoring: quick sanity summaries
    # -------------------------------
    if N_massive > 0:
        fb = fbkg_all[massive_indices]
        fb = fb[np.isfinite(fb)]
        if fb.size > 0:
            q = np.quantile(fb, [0.01, 0.1, 0.5, 0.9, 0.99])
            print("[f_bkg] massive-halo quantiles [1%,10%,50%,90%,99%] =", q)
            print("[f_bkg] min/max =", float(np.min(fb)), float(np.max(fb)))

        l0 = lambda_obs_all[massive_indices]
        l1 = lambda_obs_cut_all[massive_indices]
        ok = np.isfinite(l0) & np.isfinite(l1)
        if np.any(ok):
            frac = np.divide(l1[ok], np.maximum(l0[ok], 1e-30))
            fq = np.quantile(frac, [0.01, 0.1, 0.5, 0.9, 0.99])
            print("[lambda_cut/lambda] massive-halo quantiles [1%,10%,50%,90%,99%] =", fq)

    # -------------------------------
    # 9. Save outputs (merged halo and galaxy catalogues)
    # -------------------------------
    print(f"[Output] Saving merged halo catalogue to: {args.out_halo_npz}")
    halo_out = dict(
        # Identifiers / geometry
        halo_index=halo_index,
        theta=theta_all,
        phi=phi_all,
        ra=ra_all,
        dec=dec_all,
        z=z_all,
        Mass=mass_all,
        Dc_Mpc=halo_Dc_all,
        R200_com_Mpc=halo_R200_com_all,

        # Shell bookkeeping
        shell_label=shell_label_all,
        shell_local_index=shell_local_index_all,

        # Richness measures
        lambda_true=lambda_true_all.astype(np.float32),
        lambda_obs=lambda_obs_all,
        lambda_obs_cut_pmem=lambda_obs_cut_all,

        # Metadata for reproducibility
        m_threshold=float(args.m_threshold),
        pmem_major_threshold=float(args.pmem_major_threshold),

        # f_bkg output (the annulus 3–5 cMpc/h method)
        fbkg_annulus_3_5_cMpc_over_h=fbkg_all,
    )
    np.savez(args.out_halo_npz, **halo_out)

    print(f"[Output] Saving merged galaxy catalogue to: {args.out_galaxy_npz}")
    gal_out = dict(
        # Identifiers + observables
        gal_index=gal_index,
        gal_ra=gal_ra_all,
        gal_dec=gal_dec_all,
        gal_z_obs=gal_z_obs_all,
        gal_Dc_Mpc=gal_Dc_Mpc_all,

        # Shell bookkeeping and "true" host mapping (if available)
        gal_parent_shell_label=gal_parent_shell_label_all,
        gal_parent_halo_local=gal_parent_halo_local_all,

        # Major-contributor sparse mapping: each row is one (halo, galaxy, P_mem) pair
        major_pair_halo_index=major_pair_halo_index,
        major_pair_gal_index=major_pair_gal_index,
        major_pair_pmem=major_pair_pmem,

        # Metadata
        pmem_major_threshold=float(args.pmem_major_threshold),
    )
    np.savez(args.out_galaxy_npz, **gal_out)

    print("[Done] Global richness + major-contributor pairs computation finished.")


if __name__ == "__main__":
    main()
