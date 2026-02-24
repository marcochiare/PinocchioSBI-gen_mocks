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
    file_name = 'models_parameters_5dim_mixed'

    # N = 2^m samples
    x = int(input('How many samples? (approximated to the closest power of 2): '))
    m = int(np.rint(np.log2(x)))

    params = {
            'Omega_m': [0.1, 0.5],
            'sigma_8': [0.6, 1.2],
            'h': [0.6, 0.8],
            # generate some extra parameters in the sequence
            # to be later rescaled to the desired cosmo param.
            'empty1': [0., 1.], # e.g. w0
            'empty2': [0., 1.], # e.g. wa
            }

    u_bounds = [val[1] for val in params.values()]
    l_bounds = [val[0] for val in params.values()]

    ndim = len(params.keys())
    
    print('Generating sequence for the following parameters and boundaries:')
    for key, val in params.items():
        print(f'{key}: {val}')

    sampler = qmc.Sobol(d=ndim, scramble=True)
    sample = sampler.random_base2(m=m)
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

    print(f'Sobol sequence generated ({2**m} samples)\nDiscrepancy: {qmc.discrepancy(sample):.3e}')

    write_bounds(params, save_dir, file_name + '_boundaries')
    write_file(params, sample, save_dir, file_name + '_unscaled')
    write_file(params, scaled_sample, save_dir, file_name)
