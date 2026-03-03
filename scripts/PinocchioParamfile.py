
import re
from collections import OrderedDict

"""
This script provides a class that can be used to:
    1. Generate a default Pinocchio parameter file and edit it in python
    2. Save the parameter file to disc with proper formatting
    3. Read a Pinocchio parameter file (full or partial) from disc into the class
"""

class params_file(object):
    """
    A class to generate and edit a Pinocchio parameter file
    """
    n_groups = 9

    @staticmethod
    def _build_defaults():
        # This function is largely inspired from Florent Leclercq's params_file in the Simbelmyne code

        default_params = OrderedDict()

        default_params[0] = '# Run Properties'
        default_params['group0'] = {'PARAM': OrderedDict(), 'COMM': {}}
        default_params['group0']['PARAM']['RunFlag'] = 'example'
        default_params['group0']['PARAM']['OutputList'] = 'outputs'
        default_params['group0']['PARAM']['BoxSize'] = 500.
        default_params['group0']['PARAM']['BoxInH100'] = ''
        default_params['group0']['PARAM']['GridSize'] = 128
        default_params['group0']['PARAM']['RandomSeed'] = 123456
        default_params['group0']['PARAM']['FixedIC'] = ['DISABLE', None]
        default_params['group0']['PARAM']['PairedIC'] = ['DISABLE', None]

        default_params['group0']['COMM']['RunFlag'] = 'name of the run'
        default_params['group0']['COMM']['OutputList'] = 'output list'
        default_params['group0']['COMM']['BoxSize'] = 'physical size of the box in Mpc'
        default_params['group0']['COMM']['BoxInH100'] = 'specify that the box is in Mpc/h'
        default_params['group0']['COMM']['GridSize'] = 'number of grid points per side'
        default_params['group0']['COMM']['RandomSeed'] = 'random seed for initial conditions'
        default_params['group0']['COMM']['FixedIC'] = 'if present, the modulus in ICs is fixed to the average'
        default_params['group0']['COMM']['PairedIC'] = 'if present, the phase in ICs is shifted by PI'

        default_params[1] = '# Cosmology'
        default_params['group1'] = {'PARAM': OrderedDict(), 'COMM': {}}
        # PLANCK-18 TT,TE,EE+lowE+lensing+BAO
        default_params['group1']['PARAM']['Omega0'] = 0.3111
        default_params['group1']['PARAM']['OmegaLambda'] = 0.6889
        default_params['group1']['PARAM']['OmegaBaryon'] = 0.0489
        default_params['group1']['PARAM']['Hubble100'] = 0.6766
        default_params['group1']['PARAM']['Sigma8'] = 0.8102
        default_params['group1']['PARAM']['PrimordialIndex'] = 0.9665
        default_params['group1']['PARAM']['DEw0'] = -1.0
        default_params['group1']['PARAM']['DEwa'] = 0.0
        default_params['group1']['PARAM']['TabulatedEoSfile'] = 'no'
        default_params['group1']['PARAM']['FileWithInputSpectrum'] = 'no'

        default_params['group1']['COMM']['Omega0'] = 'Omega_0 (CDM + Baryons)'
        default_params['group1']['COMM']['OmegaLambda'] = 'OmegaLambda'
        default_params['group1']['COMM']['OmegaBaryon'] = 'Omega_b (baryonic matter)'
        default_params['group1']['COMM']['Hubble100'] = 'little h'
        default_params['group1']['COMM']['Sigma8'] = 'sigma8; if 0, it is computed from the provided P(k)'
        default_params['group1']['COMM']['PrimordialIndex'] = 'n_s'
        default_params['group1']['COMM']['DEw0'] = 'w0 of parametric dark energy equation of state'
        default_params['group1']['COMM']['DEwa'] = 'wa of parametric dark energy equation of state'
        default_params['group1']['COMM']['TabulatedEoSfile'] = 'equation of state of dark energy tabulated in a file'
        default_params['group1']['COMM']['FileWithInputSpectrum'] = 'P(k) tabulated in a file or set to no if Sigma8 != 0'

        default_params[2] = '# From N-GenIC'
        default_params['group2'] = {'PARAM': OrderedDict(), 'COMM': {}}
        default_params['group2']['PARAM']['InputSpectrum_UnitLength_in_cm'] = 0
        default_params['group2']['PARAM']['WDM_PartMass_in_kev'] = 0.0

        default_params['group2']['COMM']['InputSpectrum_UnitLength_in_cm'] = 'units of tabulated P(k), or 0 if it is in h/Mpc'
        default_params['group2']['COMM']['WDM_PartMass_in_kev'] = 'WDM cut following Bode, Ostriker & Turok (2001)'

        default_params[3] = '# Control of Memory Requirements'
        default_params['group3'] = {'PARAM': OrderedDict(), 'COMM': {}}
        default_params['group3']['PARAM']['BoundaryLayerFactor'] = 3.0
        default_params['group3']['PARAM']['MaxMem'] = 3600
        default_params['group3']['PARAM']['MaxMemPerParticle'] = 300
        default_params['group3']['PARAM']['PredPeakFactor'] = 0.8
        
        default_params['group3']['COMM']['BoundaryLayerFactor'] = 'width of the boundary layer for fragmentation'
        default_params['group3']['COMM']['MaxMem'] = 'max available memory to an MPI task in Mbyte'
        default_params['group3']['COMM']['MaxMemPerParticle'] = 'max available memory in bytes per particle'
        default_params['group3']['COMM']['PredPeakFactor'] = 'guess for the number of peaks in the subvolume'

        default_params[4] = '# Output'
        default_params['group4'] = {'PARAM': OrderedDict(), 'COMM': {}}
        default_params['group4']['PARAM']['CatalogInAscii'] = ''
        default_params['group4']['PARAM']['OutputInH100'] = ''
        default_params['group4']['PARAM']['NumFiles'] = 1
        default_params['group4']['PARAM']['MinHaloMass'] = 10 
        default_params['group4']['PARAM']['AnalyticMassFunction'] = 9

        default_params['group4']['COMM']['CatalogInAscii'] = 'catalogs are written in ascii and not in binary format'
        default_params['group4']['COMM']['OutputInH100'] = 'units are in H=100 instead of the true H value'
        default_params['group4']['COMM']['NumFiles'] = 'number of files in which each catalog is written'
        default_params['group4']['COMM']['MinHaloMass'] = 'smallest halo that is given in output'
        default_params['group4']['COMM']['AnalyticMassFunction'] = 'form of analytic mass function given in the .mf.out files'

        default_params[5] = '# Output options'
        default_params['group5'] = {'PARAM': OrderedDict(), 'COMM': {}}
        default_params['group5']['PARAM']['WriteTimelessSnapshot'] = ['DISABLE', None]
        default_params['group5']['PARAM']['DoNotWriteCatalogs'] = ['DISABLE', None]
        default_params['group5']['PARAM']['DoNotWriteHistories'] = ['DISABLE', None]

        default_params['group5']['COMM']['WriteTimelessSnapshot'] = 'writes a Gadget2 snapshot as an output'
        default_params['group5']['COMM']['DoNotWriteCatalogs'] = 'skips the writing of full catalogs (including PLC)'
        default_params['group5']['COMM']['DoNotWriteHistories'] = 'skips the writing of merger histories'

        default_params[6] = '# Past Light Cone'
        default_params['group6'] = {'PARAM': OrderedDict(), 'COMM': {}}
        default_params['group6']['PARAM']['StartingzForPLC'] = 0.3
        default_params['group6']['PARAM']['LastzForPLC'] = 0.0
        default_params['group6']['PARAM']['PLCAperture'] = 30
        default_params['group6']['PARAM']['PLCProvideConeData'] = ''
        default_params['group6']['PARAM']['PLCCenter'] = '0. 0. 0.'
        default_params['group6']['PARAM']['PLCAxis'] = '1. 1. 0.'
        default_params['group6']['PARAM']['MassMapNSIDE'] = 256

        default_params['group6']['COMM']['StartingzForPLC'] = 'starting (highest) redshift for the past light cone'
        default_params['group6']['COMM']['LastzForPLC'] = 'final (lowest) redshift for the past light cone'
        default_params['group6']['COMM']['PLCAperture'] = 'cone aperture for the past light cone'
        default_params['group6']['COMM']['PLCProvideConeData'] = 'read vertex and direction of cone from paramter file'
        default_params['group6']['COMM']['PLCCenter'] = 'cone vertex in the same coordinates as the BoxSize'
        default_params['group6']['COMM']['PLCAxis'] = 'un-normalized direction of the cone axis'
        default_params['group6']['COMM']['MassMapNSIDE'] = 'NSIDE for healpix mass maps'

        default_params[7] = '# Table of collapseTime file, needed if the code is compiled with TABULATED_CT'
        default_params['group7'] = {'PARAM': OrderedDict(), 'COMM': {}}
        default_params['group7']['PARAM']['CTtableFile'] = ['DISABLE', None]

        default_params[8] = '# CAMB Pk files, needed if the code is compiled with READ_PK_TABLE'
        default_params['group8'] = {'PARAM': OrderedDict(), 'COMM': {}}
        default_params['group8']['PARAM']['CAMBMatterFile'] = ['DISABLE', 'matterpower']
        default_params['group8']['PARAM']['CAMBTransferFileTag'] = ['DISABLE', 'transfer_out']
        default_params['group8']['PARAM']['CAMBRunName'] = ['DISABLE', 'nilcdm_0.3eV']
        default_params['group8']['PARAM']['CAMBRedshiftsFile'] = ['DISABLE', 'inputredshift']
        default_params['group8']['PARAM']['CAMBHubbleTableFile'] = ['DISABLE', 'hubble.dat']

        default_params['group8']['COMM']['CAMBMatterFile'] = 'label for matter power spectrum files (CDM+Baryons)'
        default_params['group8']['COMM']['CAMBRunName'] = 'name of the CAMB run'
        default_params['group8']['COMM']['CAMBRedshiftsFile'] = 'list of redshifts for the Pk table'
        default_params['group8']['COMM']['CAMBHubbleTableFile'] = 'Hubble table file'

        return default_params
 
    def __init__(self, no_header: bool = False, **kwargs):
        """
        Initialise a Pinocchio parameter file to its default values.

        Custom values can be passed via keyword arguments where the key is exactly
        the Pinocchio entry that you wish to change.

        e.g.:
            P = params_file(RunFlag='MyRunName') # to change a value
            P = params_file(DoNotWriteHistories='') # to enable one flag, off by default

        Special values:
            - "DEFAULT" or "IGNORE" keep the default value assigned to that key
              e.g. Omega0 = 0.4 if randint(10) % 2 == 0 else 'DEFAULT'

            - "DISABLE" disables that key by commenting the line out

        The class provides the following dictionaries as attributes:
            - setup
            - cosmo
        containing the specifications of their respective group of parameters

        Args:
            no_header (bool, default=False): suppress the fancy header in the parameter file.
            **kwargs: Pinocchio parameter file keyword argument to change.
        """

        self.default_params = self._build_defaults()

        self.no_header = no_header
        
        for group in range(self.n_groups):

            param_dict = self.default_params[f'group{group}']['PARAM']

            for key, val in kwargs.items():

                if val in ['DEFAULT', 'IGNORE']:
                    continue

                if key in param_dict.keys():
                    param_dict[key] = val

    @property
    def setup(self):
        return self.default_params['group0']['PARAM']

    @property
    def cosmo(self):
        return self.default_params['group1']['PARAM']

    def __repr__(self):

        KEY_WIDTH = 35
        COMM_WIDTH = 50

        if self.no_header:

            text = "# PINOCCHIO parameter file\n"
            text += "#========================#\n\n"
    
        else:

            text = r"""
#===============================================#
#         _                      _     _        #
#   _ __ (_)_ __   ___   ___ ___| |__ (_) ___   #
#  | '_ \| | '_ \ / _ \ / __/ __| '_ \| |/ _ \  #
#  | |_) | | | | | (_) | (_| (__| | | | | (_) | #
#  | .__/|_|_| |_|\___/ \___\___|_| |_|_|\___/  #
#  |_|                                          #
#===============================================#"""
            text += '\n\n'

        for group in range(self.n_groups):

            group_dict = self.default_params[f'group{group}']
            param_dict = group_dict['PARAM']
            comms_dict = group_dict['COMM']

            text += str(self.default_params[group]) + '\n'

            for key, val in param_dict.items():

                if 'DISABLE' in str(val):
                    if isinstance(val, (list, tuple)):
                        base = f'%{key:<{KEY_WIDTH - 1}}{val[1]}'

                    else:
                        base = f'%{key:<{KEY_WIDTH - 1}}'

                else:
                    base = f'{key:<{KEY_WIDTH}}{val}'
                
                comm = comms_dict.get(key)
                
                if comm:
                    base = base.ljust(COMM_WIDTH) + f' % {comm}'
            
                text += base + '\n'

            text += '\n'

        return text

    def write(self, save_name: str = './parameter_file', verb: bool = False):
        """
        Write to disc the parameter_file

        Args:
            save_name (str, default='./parameter_file'): name of the file saved on disc. Can include a full path.
            verb (bool, default=False): verbosity
        """            

        with open(save_name, 'w') as param_file:
            param_file.write(repr(self))

        if verb:
            print(f'Parameter file saved to disc: {save_name}')

    def load(self, file_path: str, from_default: bool = True, verb: bool = False):
        """
        Loads the values from a parameter file saved on disc into the class.
        This works also with incomplete parameter files (i.e. containing only some values to change)

        Args:
            file_path (str): path to parameter file
            from_default (bool, default=True): loads the values in the file over a default class, overwriting any already-initialised
                                               value that may or not be included in the given parameter file
            verb (bool, default=False): verbosity

        """

        if from_default:
            self.default_params = self._build_defaults()

        all_keys = {}
        for group in range(self.n_groups):

            param_dict = self.default_params[f'group{group}']['PARAM']

            for key in param_dict:
                all_keys[key] = param_dict

        with open(file_path, 'r') as file:

            for raw_line in file:

                line = raw_line.strip()

                if not line:
                    continue

                disabled = False

                if line.startswith('#') or line.startswith('%'):
                    disabled = True
                    line = line[1:].strip()

                # Remove inline comment and strip into [key, value]
                line = re.split(r'[#%]', line, maxsplit=1)[0].strip()
                
                if not line:
                    continue
                
                # Split key/val in max 2 elements (spec. for PLCCenter and PLCAxis)
                tokens = line.split(maxsplit=1)
                key = tokens[0]

                if key not in all_keys:
                    continue

                param_dict = all_keys[key]

                # Flag only parameters
                if len(tokens) == 1:
                    value = ''

                else:
                    value = tokens[1]

                # Convert type; all values are strings initially
                if value is not None:

                    default_value = param_dict[key]

                    try:
                        if isinstance(default_value, int):
                            value = int(value)
                        elif isinstance(default_value, float):
                            value = float(value)
                        else:
                            pass

                    except ValueError:
                        raise ValueError(f'I am reading a value ({value}) that does not match with the default type for {key}. Expected {type(default_value)}.')

                # Handle DISABLED
                if disabled:
                    param_dict[key] = ['DISABLE', value]

                else:
                    param_dict[key] = value

        if verb:
            print(f'Parameter file loaded from disc: {file_path}')


if __name__ == "__main__":

    from numpy.random import randint

    # Example

    P = params_file(
        RunFlag = 'test',
        GridSize = 256,
        Omega0 = 0.4 if randint(10) % 2 == 0 else 'DEFAULT', # use case for 'DEFAULT' or 'IGNORE'
        BoxInH100 = 'DISABLE', # put string DISABLE if you want to disable a line that is not by default
        DoNotWriteHistories = '' # add a value that is disabled by default to include it (in this case, empty)
    )

    P.write(save_name = 'parameter_file', verb = True)

    print('Example program finished.')
