import argparse
import numpy as np
import os

from PinocchioParamfile import params_file
from ReadPinocchio5 import plc

import time
from datetime import datetime

"""
Script to read a PLC and save it to disc into multiple redshift shells that match the massmap redshifts.

This is meant to be run before the "painting".
"""

def printlog(text: str):
    now = datetime.now().strftime('%H:%M:%S')
    print(f'\033[92m[ZSHELLS: {now}]>\033[0m ' + text)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Split a PLC into multiple redshift shells, matchi')

    parser.add_argument('--pin-dir', type=str, required=True) 
    parser.add_argument('--param-file', type=str, required=True)
    parser.add_argument('--out-prefix', type=str, required=True) 
    parser.add_argument('--use-truez', action='store_true')
    parser.add_argument('--ext', type=str, default='.npz')

    args = parser.parse_args()

    P = params_file()
    P.load(args.param_file, verb=True)
    run_name = P.setup['RunFlag']
    output_name = P.setup['OutputList']

    output_path = os.path.join(args.pin_dir, output_name)
    plc_path = os.path.join(args.pin_dir, f'pinocchio.{run_name}.plc.out')
    
    if not os.path.isfile(output_path):
        raise FileNotFoundError(f'{output_path} not found')

    if os.path.isfile(plc_path + '.0'):
        raise NotImplementedError('Please provide one-file only PLC')

    if not os.path.isfile(plc_path):
        raise FileNotFoundError(f'pinocchio.{run_name}.plc.out not found in {args.pin_dir}')
 
    Z = np.loadtxt(output_path)
    Z = np.sort(Z) # just to be safe, shouldn't be necessary
    TOT = len(Z) - 1

    printlog(f'Splitting {os.path.basename(plc_path)} in {args.pin_dir} into {TOT} files. Range {Z.min():.4f}-{Z.max():.4f}. Prefix: {args.out_prefix}.')

    data = plc(plc_path).data
    printlog('PLC loaded. Starting to split ...')

    for i in range(TOT):

        z_low = Z[i]
        z_upp = Z[i+1]

        z = data['truez'] if args.use_truez else data['obsz']
        MASK = (z > z_low) * (z < z_upp)
        z = z[MASK]

        theta_deg = 90. - data['theta'][MASK]
        phi_deg = data['phi'][MASK]
        mass = data['Mass'][MASK]

        # convert to radians
        theta = np.deg2rad(theta_deg)
        phi = np.deg2rad(phi_deg)

        # save to file
        save_data = {
                'theta': theta,
                'phi': phi,
                'z': z,
                'Mass': mass,
                }
        save_name = f'{args.out_prefix}_z{Z[i]:.4f}_{Z[i+1]:.4f}{args.ext}'
        np.savez(os.path.join(args.pin_dir, save_name), **save_data)

    printlog('Done.') 
