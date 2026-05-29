import argparse
import os
import numpy as np

from astropy.cosmology import FlatLambdaCDM
from cctoolkit.cosmology import CosmologyCalculator
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator

from ReadPinocchio5 import plc as pin_plc
from PinocchioParamfile import params_file

"""
This script recalibrates the HMF of the Pinocchio light cones to match Castro22 HMF

Written by Tiago Castro and re-adapted for PinocchioSBI
"""
# ============================================================
# Global defaults
# ============================================================

# Sky geometry
# If 70 deg is the radius of the sky cap, keep this.
# If 70 deg is the full opening angle, use 35.0 instead.
THETA_DEG = 70.0
 
# Theoretical mass grid
MMIN = 1e13
MMAX = 5e16
NM = 250
 
# Redshift grid for HMF interpolation
Z_MIN = 0.0
Z_MAX = 2.0
NZ = 200
 
# Redshift shells where calibration is done independently
N_ZSHELLS = 11
 
# Optional read-time completeness cut (set to None to keep all objects)
READ_MASS_MIN = None
 
 
# ============================================================
# Theory
# ============================================================

def build_theory(param_path, pk_path, fsky):
    """
    Load the power spectrum and build all theory objects needed for
    abundance matching:  hmf_interp, background, Marr, lnMarr, zarr, zshells.
 
    Parameters
    ----------
    param_file : str
        Path to the Pinocchio parameter file (with specified sigma8 !!!).
    pk_path : str
        Path to the CAMB P(k) file (two columns: k, P(k)).
    fsky : float
        Sky fraction covered by the light-cone.
 
    Returns
    -------
    dict with keys: hmf_interp, background, Marr, lnMarr, zarr, zshells, fsky
    """
    k, Pk = np.loadtxt(pk_path, unpack=True)
 
    # Read the cosmology from the Pinocchio parameter file
    P = params_file()
    P.load(param_path)

    # Fiducial cosmology parameters
    COSMO_PARAMS = {
            'H0': P.cosmo['Hubble100'] * 100,
            'Ob0': P.cosmo['OmegaBaryon'],
            'Om0': P.cosmo['Omega0'],
            'sigma8': P.cosmo['Sigma8'],
            'ns': P.cosmo['PrimordialIndex'],
            'mnu': 0.0,
            'num_massive_neutrinos': 0,
            'TCMB': 0.5,
            }

    cosmo_calc = CosmologyCalculator(COSMO_PARAMS, power_spectrum=[k, Pk])
 
    Marr = np.geomspace(MMIN, MMAX, NM)
    lnMarr = np.log(Marr)
    zarr = np.linspace(Z_MIN, Z_MAX, NZ)
    zshells = np.linspace(Z_MIN, Z_MAX, N_ZSHELLS)
 
    # dndlnM(z, M) grid, shape = (NZ, NM)
    dndlnM_grid = np.array([cosmo_calc.dndlnM(Marr, z) for z in zarr])
 
    # Interpolator expects (z, lnM)
    hmf_interp = RegularGridInterpolator(
        (zarr, lnMarr),
        dndlnM_grid,
        bounds_error=False,
        fill_value=np.nan,
    )
 
    # H0=100 to keep distances/volumes in h-units numerically
    background = FlatLambdaCDM(Om0=COSMO_PARAMS["Om0"], H0=100.0)
 
    return {
            'hmf_interp': hmf_interp,
            'background': background,
            'Marr': Marr,
            'lnMarr': lnMarr,
            'zarr': zarr,
            'zshells': zshells,
            'fsky': fsky,
            }

# ============================================================
# Helper functions
# ============================================================

def theoretical_cumulative_counts(z1, z2, theory_dict):
    """
    Returns N_theory(>M) in shell [z1, z2), for a lightcone.
    """
    hmf_interp = theory_dict['hmf_interp']
    bkg = theory_dict['background']
    Marr = theory_dict['Marr']
    lnMarr = theory_dict['lnMarr']
    fsky = theory_dict['fsky']

    zint = np.linspace(z1, z2, 256)

    Z, LNM = np.meshgrid(zint, lnMarr, indexing="ij")
    pts = np.column_stack([Z.ravel(), LNM.ravel()])
    dndlnM = hmf_interp(pts).reshape(len(zint), len(Marr))

    if np.isnan(dndlnM).any():
        raise RuntimeError(f"HMF interpolation failed in shell [{z1}, {z2}).")

    dVdz = 4.0 * np.pi * fsky * bkg.differential_comoving_volume(zint).value

    dN_dlnM = np.trapezoid(dndlnM * dVdz[:, None], x=zint, axis=0)

    cum_low_to_high = cumulative_trapezoid(dN_dlnM, x=lnMarr, initial=0.0)
    Ncum = cum_low_to_high[-1] - cum_low_to_high

    return Ncum


def invert_cumulative_counts(Ncum, ranks, Marr):
    """
    Invert N(>M) -> M for the requested ranks.
    """
    valid = np.isfinite(Ncum) & (Ncum > 0)

    counts_inc = Ncum[valid][::-1]
    masses_inc = Marr[valid][::-1]

    counts_inc, unique_idx = np.unique(counts_inc, return_index=True)
    masses_inc = masses_inc[unique_idx]

    if len(counts_inc) < 2:
        raise RuntimeError("Could not invert theoretical cumulative counts.")

    if ranks.min() < counts_inc.min():
        raise RuntimeError("MMAX is too small. Increase the theoretical mass range.")
    if ranks.max() > counts_inc.max():
        raise RuntimeError("MMIN is too large. Decrease the theoretical mass range.")

    return np.interp(ranks, counts_inc, masses_inc)


def abundance_match_with_limit(Ncum, nobs, Marr):
    """
    Match as many objects as the theory supports.
    Remaining objects are assigned the limiting mass Mlim = Marr[0].

    Returns
    -------
    masses_calib_sorted : array
        Calibrated masses in descending-rank order.
    below_limit_sorted : bool array
        True for objects pushed to the limiting mass.
    mass_limit_sorted : array
        The limiting mass recorded for each object in this shell.
    Nmatch : int
        Number of objects directly matched to theory.
    """
    Mlim = Marr[0]

    # For ranks = n - 0.5, largest supported object count is floor(Ncum[0] + 0.5)
    Nmatch = int(np.floor(Ncum[0] + 0.5))
    Nmatch = min(Nmatch, nobs)

    masses_calib_sorted = np.empty(nobs, dtype=np.float32)
    below_limit_sorted = np.zeros(nobs, dtype=bool)
    mass_limit_sorted = np.full(nobs, Mlim, dtype=np.float32)

    if Nmatch > 0:
        ranks = np.arange(1, Nmatch + 1, dtype=float) - 0.5
        masses_calib_sorted[:Nmatch] = invert_cumulative_counts(Ncum, ranks, Marr)

    if nobs > Nmatch:
        masses_calib_sorted[Nmatch:] = Mlim
        below_limit_sorted[Nmatch:] = True

    return masses_calib_sorted, below_limit_sorted, mass_limit_sorted, Nmatch

# ============================================================
# Single lightcone re-calibration
# ============================================================

def calibration(plc_path, theory_dict):
    """
    Read a Pinocchio lightcone, recalibrate the masses and save the new lightcone
    as a .fits file.


    Parameters
    ----------
        plc_path : str
            Path to the Pinocchio past light cone.
        theory_dict : dict
            Output dict of the `build_theory` function
    """

    zshells = theory_dict['zshells']
    Marr = theory_dict['Marr']

    # Reading the PLC
    print(f"Reading {plc_path}")
    cat = pin_plc(plc_path, silent=True)
    data = cat.data

    # Selection
    mass = data['Mass'].astype(np.float32)
    z = data['truez'].astype(np.float32)

    if READ_MASS_MIN is not None:
        keep = (mass >= READ_MASS_MIN) & (z <= zshells.max()) & (z >= zshells.min())
    else:
        keep = (z <= zshells.max()) & (z >= zshells.min())

    mass = mass[keep]
    z = z[keep]
    valid = np.isfinite(mass) & np.isfinite(z)

    mass_calib  = mass.copy()
    below_limit = np.zeros(len(mass), dtype=bool)
    mass_limit  = np.full(len(mass), np.nan, dtype=np.float32)
 
    # shell-by-shell calibration
    for z1, z2 in zip(zshells[:-1], zshells[1:]):
        Ncum = theoretical_cumulative_counts(z1, z2, theory)
 
        sel = valid & (z >= z1) & (z < z2)
        idx = np.flatnonzero(sel)
 
        if len(idx) == 0:
            print(f'  [{z1:.2f}, {z2:.2f}) : no halos')
            continue
 
        masses_obs = mass[idx]
        order      = np.argsort(masses_obs)[::-1]
        idx_sorted = idx[order]
 
        masses_calib_sorted, below_limit_sorted, mass_limit_sorted, Nmatch = (
            abundance_match_with_limit(Ncum, len(masses_obs), Marr)
        )
 
        mass_calib[idx_sorted]  = masses_calib_sorted
        below_limit[idx_sorted] = below_limit_sorted
        mass_limit[idx_sorted]  = mass_limit_sorted
 
        print(
            f'  [{z1:.2f}, {z2:.2f}) : '
            f'Nobs = {len(masses_obs):8d}, '
            f'Nth(>MMIN) = {Ncum[0]:10.2f}, '
            f'Nmatch = {Nmatch:8d}, '
            f'Mlim = {Marr[0]:.3e}'
        )
 
    # Write the calibrated PLC in .fits
    cat.data['Mass'][keep] = mass_calib
    outname = plc_path.replace('.out', '.masscalib.fits')
    cat.write_fits(outname)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=("Recalibrate the Pinocchio HMF to match Castro22 HMF."))
    parser.add_argument('--main-dir', type=str, required=True)
    parser.add_argument('--base-name', type=str, default='model')
    parser.add_argument('--total-runs', type=int, required=True)
    parser.add_argument('--start-run', type=int, default=0)

    args = parser.parse_args()

    print('Pinocchio Mass Calibration')
    print(f'Recalibrating {args.base_name}_{args.start_run}-{args.total_runs-1} from {args.main_dir}\n')
    for name in ['THETA_DEG', 'MMIN', 'MMAX', 'NM', 'Z_MIN', 'Z_MAX', 'NZ', 'N_ZSHELLS', 'READ_MASS_MIN']:
        print(f'Using {name}={globals()[name]}')
    print('Imposing m_nu=0')

    # Sky fraction
    fsky = 0.5 * (1.0 - np.cos(np.deg2rad(THETA_DEG)))
 
    for run_number in range(args.start_run, args.total_runs):

        print('\n')

        run_name  = f'{args.base_name}_{run_number}'
        run_dir   = os.path.join(args.main_dir, run_name)
        camb_name = f'pk_camb_z0_{run_name}.dat'
        paramfile = f'parameter_file_{run_name}_sig8'

        plc_path   = os.path.join(run_dir, f'pinocchio.{run_name}.plc.out')
        pk_path    = os.path.join(run_dir, camb_name)
        param_path = os.path.join(run_dir, paramfile) 
 
        theory = build_theory(param_path, pk_path, fsky)
        calibration(plc_path, theory)
 
    print('Mass calibration completed for all the light cones.')
