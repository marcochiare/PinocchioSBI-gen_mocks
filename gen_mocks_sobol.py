import numpy as np
from scipy.stats import qmc

def write_file(params: dict, sample: np.ndarray, save_dir: str, file_name: str):

    with open(f'{save_dir}/{file_name}.txt', 'w') as f:
        col_width = 12

        header = '# '
        for key in params.keys():
            header += f'{key:<{col_width}}'
        header += '\n'
        f.write(header)

        for line in sample:
            row = 2*' '
            for elem in line:
                row += (f'{elem:<{col_width-1}.6f} ')
            row += '\n'
            f.write(row)

    print(f'Sobol sequence saved to file: {save_dir}/{file_name}.txt')

def write_bounds(params: dict, save_dir: str, file_name: str):

    with open(f'{save_dir}/{file_name}.txt', 'w') as f:
        
        f.write("# PARAMETER           MIN            MAX\n")

        for key, (vmin, vmax) in params.items():
            f.write(f"  {key:<18} {vmin:>12.6g} {vmax:>12.6g}\n")

    print(f'Boundaries saved to file: {save_dir}/{file_name}.txt')

if __name__ == "__main__":
    
    save_dir = 'SobolSeq'
    file_name = 'models_parameters_3dim'

    params = {
            'Omega_m': [0.1, 0.5],
            'sigma_8': [0.6, 1.2],
            'h': [0.6, 0.8],
            # generate some extra parameters in the sequence
            # to be later rescaled to the desired cosmo param.
            #'empty1': [0., 1.], # e.g. w0
            #'empty2': [0., 1.], # e.g. wa
            }

    u_bounds = [val[1] for val in params.values()]
    l_bounds = [val[0] for val in params.values()]

    ndim = len(params.keys())
    
    print('Generating sequence for the following parameters and boundaries:')
    for key, val in params.items():
        print(f'{key}: {val}')

    continue_seq = input('Continue an existing sequence? [y/n]: ').strip().lower() == 'y'

    if continue_seq:

        existing_file = input('Path to existing sequence (_unscaled.txt): ').strip()
        assert existing_file.endswith('_unscaled.txt'), 'File does not end with `_unscaled.txt`'

        existing = np.loadtxt(existing_file)
        n_existing = len(existing)
        print(f'Loaded {n_existing} values from file')

    else:
        print('Generating a new sequence')
        n_existing = 0

    # N = 2^m samples
    x = int(input('How many total samples (existing, if any, + new)? (approximated to the closest power of 2): '))
    m = int(np.rint(np.log2(x)))
    total = 2**m

    if total < n_existing:
        raise ValueError(f'Closest power of 2 below {x} is {total}, not larger than {n_existing}. Choose a larger value.')

    sampler = qmc.Sobol(d=ndim, scramble=True)
    if continue_seq:
        n_new = total - n_existing
        sampler.fast_forward(n_existing)
        
        print(f'Adding {n_new} new samples to the sequence') 
        sample = sampler.random(n=n_new)
    else:
        sample = sampler.random_base2(m=m)
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

    print(f'Sobol sequence generated ({2**m} samples)\nDiscrepancy: {qmc.discrepancy(sample):.3e}')

    # Add the new sequence to the existing one
    if continue_seq:
        sample = np.vstack([existing, sample])
        scaled_existing = np.loadtxt(existing_file.replace('_unscaled',''))
        scaled_sample = np.vstack([scaled_existing, scaled_sample])

        print(f'Total number of samples after continuation: {len(sample)}')

    write_bounds(params, save_dir, file_name + '_boundaries')
    write_file(params, sample, save_dir, file_name + '_unscaled')
    write_file(params, scaled_sample, save_dir, file_name)
