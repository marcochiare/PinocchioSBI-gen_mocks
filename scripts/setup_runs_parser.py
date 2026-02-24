import argparse
import camb
import numpy as np
import os

import time
from datetime import datetime

from PinocchioParamfile import params_file

"""
Script to setup multiple Pinocchio runs from the Sobol sequence
"""

def printlog(text: str):
    now = datetime.now().strftime('%H:%M:%S')
    print(f'\033[92m[SETUP: {now}]>\033[0m ' + text)

def format_timedelta_dhms(diff):
    total_s = int(diff.total_seconds())
    days = total_s // 86400
    res = total_s % 86400
    return f'{days}-{time.strftime("%H:%M:%S", time.gmtime(res))}'

def camb_linear_powerspectrum(Omega_m, sigma_8, h, Omega_b, n_s, mnu = 0., kmax = 1e3,
                              minkh = 1e-4, maxkh = 1e2, npoints = 200, unscaled = False,
                              **kwargs):
    """
    Compute the linear power spectrum at z=0.0 for the given set of
    cosmological parameters using CAMB. For flat LambdaCDM.

    Args:
        Omega_m, sigma_8, Omega_b, n_s: cosmological parameters
        unscaled (default=False): if True, the power spectrum has amplitude A_s=2e-9
        mnu (default=0.): sum of neutrino masses
        kmax (default=1e3): max k for calculating the matter power spectrum
        minkh (default=1e-4): minimum k (h/Mpc) to return the matter power spectrum
        maxkh (default=1e2): maximum k (h/Mpc) to return the matter power spectrum
        npoints (default=200): number of points k points for the matter power spectrum
        **kwargs: extra keyword argument to pass to camb.CAMBparams

    Return:
        kh: k (h/Mpc) for the matter power spectrum at z=0.0
        pk_z0: P(k) ((h^-1 Mpc)^3) at z=0.0
    """
    ombh2 = Omega_b * h ** 2
    omch2 = (Omega_m - Omega_b) * h ** 2

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100*h, ombh2=ombh2, omch2=omch2, mnu=mnu, **kwargs)
    pars.InitPower.set_params(ns=n_s, As=2e-9)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)

    res = camb.get_results(pars) 
    kh, z, pk = res.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=npoints)

    if not unscaled: 
        # Scale to match sigma8
        s8_unscaled = res.get_sigma8_0()
        rescale_fact = (sigma_8 / s8_unscaled) ** 2
        pk *= rescale_fact

    pk_z0 = pk[0, :]

    return kh, pk_z0 

def read_cosmo_params(cosmo_params_file: str, params_idx: dict[str, int], params_names = None):
    """
    Returns the array of requested cosmo parameters for the Pinocchio runs contained in cosmo_params_file
    """
    data = np.genfromtxt(cosmo_params_file)

    if params_names is None:
        return data

    cosmo_params_names = [name for name in params_names if name in params_idx]

    cosmo_params = np.zeros((data.shape[0], len(cosmo_params_names)))

    for i, name in enumerate(cosmo_params_names):

        # NOTE: column's order follows the order in params_names
        idx = params_idx.get(name)
        cosmo_params[:, i] = data[:, idx]

    return cosmo_params

def setup_pinocchio_runs(dir_path: str, base_run_name: str, cosmo_params_file: str, params: list[str],
                         total_runs: int, z_out: list[float], seed: int = 1, **kwargs
                         ):
    """
    Setup multiple Pinocchio runs. assuming Omega_m + Omega_lambda = 1
    Currently, only Omega_m, Sigma8, h, w0, wa can be read from the cosmo_params_file

    Args:
        dir_path (str): working directory containing all runs
        base_run_name (str): run template name. run number added at the end (e.g "base_run_name=example"-> "example_1")
        cosmo_params_file (str): path to the file containing the cosmological parameters 
        params (list[str]): list of the cosmological parameters to read from the file
        total_runs (int): total number of Pinocchio runs to setup
        z_out (list[float]): list of redshifts for the output file
        seed (int, default=1): seed for the random generation (seed*run_number)
        **kwargs: common keyword arguments to pass to the parameter file for ALL RUNS
    """
    allowed_param_names = ['Omega_m', 'sigma_8', 'h', 'w0', 'wa']

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        printlog(f'Created directory {dir_path}')

    params_idx = {}
    for i, param_name in enumerate(allowed_param_names): # extend this list for generalization
        params_idx[param_name] = i
    cosmo_params = read_cosmo_params(cosmo_params_file, params_idx, params)

    z_out_filename = 'outputs'

    for run_number in range(total_runs):

        run_name = f'{base_run_name}_{run_number}'
        camb_name = f'pk_camb_z0_{run_name}.dat'
        
        run_dir = f'{dir_path}/{run_name}'
        os.mkdir(run_dir)
        printlog(f'Created directory {run_dir}')

        with open(f'{run_dir}/{z_out_filename}', 'w') as f:
            for z in sorted(z_out, reverse = True):
                f.write(f'{z}\n')
        
        P = params_file(
            RunFlag = run_name,
            # Same scheme for each cosmo parameter that needs to be read and written to the parameter file
            Omega0 = cosmo_params[run_number, params.index('Omega_m')] if 'Omega_m' in params else 'DEFAULT',
            OmegaLambda = 1. - cosmo_params[run_number, params.index('Omega_m')] if 'Omega_m' in params else 'DEFAULT',
            #Sigma8 = cosmo_params[run_number, params.index('sigma_8')] if 'sigma_8' in params else 'DEFAULT',
            Sigma8 = 0., # Sigma8 in Pinocchio is computed from the given P(k) (that uses the one provided) if this is zero
            Hubble100 = cosmo_params[run_number, params.index('h')] if 'h' in params else 'DEFAULT',
            DEw0 = cosmo_params[run_number, params.index('w0')] if 'w0' in params else 'DEFAULT',
            DEwa = cosmo_params[run_number. params.index('wa')] if 'wa' in params else 'DEFAULT',
            FileWithInputSpectrum = camb_name,
            OutputList = z_out_filename,
            RandomSeed = seed * (run_number + 1),
            **kwargs
        )

        P.write(f'{run_dir}/parameter_file_{run_name}')

        kh, Pk = camb_linear_powerspectrum(
                P.cosmo['Omega0'],
                P.cosmo['Sigma8'],
                P.cosmo['Hubble100'],
                P.cosmo['OmegaBaryon'],
                P.cosmo['PrimordialIndex'],
                minkh=1.1e-5,
                maxkh=50.375,
                kmax=1e2,
                npoints=1125
                )

        with open(f'{run_dir}/{camb_name}', 'w') as f:
            for kh_el, Pk_el in zip(kh, Pk):
                f.write(f'{kh_el:<15.10f}{Pk_el:.10f}\n')

        printlog(f'{run_name} setup done.')

    printlog('Setup done.')

def launch_pinocchio_runs(exec: str, dir_path: str, base_run_name: str, total_runs: int, mpi_procs: int = 5):
    """
    A simple way to lunch multiple Pinocchio runs.
    Use this only for small simulations.

    Args:
        exec (str): absolute path to the Pinocchio executable
        dir_path (str): working directory containing all runs
        base_run_name (str): run template name. run number added at the end (e.g "base_run_name=example"-> "example_1")
        total_runs (int): total number of Pinocchio runs to execute
        mpi_procs (int, default=5): number of MPI processors for each run
    """
    start_time = datetime.now()
    printlog(f'Starting Pinocchio runs. TOTAL = {total_runs}.')

    for run_number in range(total_runs):
        
        thisrun_start_time = datetime.now()
        printlog(f'Starting run {run_number+1}/{total_runs}.')

        run_name = f'{base_run_name}_{run_number}'
        parameter_file = f'parameter_file_{run_name}'
        
        run_dir = f'{dir_path}/{run_name}'
        os.chdir(run_dir)

        # Run CAMB here to generate the Pk so that Pinocchio can start from it

        # os.system(f'mpirun -np {mpi_procs} {exec} {parameter_file} > pinocchio_{run_name}.log')
        os.system(f"echo 'Dummy: mpirun -np {mpi_procs} {exec} {parameter_file}' > pinnocchio_{run_name}.log")
        time.sleep(np.random.uniform(2.,8.))

        thisrun_elapsed = format_timedelta_dhms(datetime.now() - thisrun_start_time)
        printlog(f'Finished run {run_number+1}/{total_runs}. Elapsed time: {thisrun_elapsed}')
    
    tot_elapsed = format_timedelta_dhms(datetime.now() - start_time)
    printlog(f'Finished all computations. Total elapsed time: {tot_elapsed}.')

def cast_type(val):
    """
    Takes a string value and returns the correct type for that value
    """
    for typ in (int, float):

        try:
            return typ(val)
        except ValueError:
            pass

    return val

def parse_split_and_convert(parsed_list):
    """
    Splits a list of 'key=val' into a dict {key:val} with
    the correct type
    """
    res = {}

    for elem in parsed_list:
        key, val = elem.split('=', 1)
        res[key] = cast_type(val)

    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Setup multiple Pinocchio runs from the Sobol sequence')

    parser.add_argument('--main-dir', type=str, required=True)
    parser.add_argument('--base-name', type=str, default='model')
    parser.add_argument('--cosmo-file', type=str, required=True)
    parser.add_argument('--total-runs', type=int, required=True)
    parser.add_argument('--params',
                        nargs='+',
                        help='list of cosmological parameters to use from the Sobol sequence',
                        default=['Omega_m', 'sigma_8', 'h']
                        )
    parser.add_argument('--z-out',
                        nargs='*',
                        help='redshift for the box snapshots',
                        type=float,
                        default=[0.0] # safe default
                        )
    parser.add_argument('--setup-args',
                        nargs='*',
                        help='key=value run properties'
                        )

    args = parser.parse_args()
    setup_dict = parse_split_and_convert(args.setup_args)

    # ============= SETUP ============= #
    
    setup_pinocchio_runs(
        dir_path = args.main_dir,
        base_run_name = args.base_name,
        cosmo_params_file = args.cosmo_file,
        params = args.params,
        total_runs = args.total_runs,
        z_out = args.z_out,
        **setup_dict
    )

    # For not long simulations (all running inside one job):
#    launch_pinocchio_runs(
#        exec='/home/dirac/Documenti/DOTTORATO/Code/pinocchio.x',
#        dir_path=dir_path,
#        base_run_name=base_name,
#        total_runs=total_runs,
#        mpi_procs=8
#    )
