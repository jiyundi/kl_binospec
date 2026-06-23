import numpy as np
from collections import Counter
import astropy.units as u

# From: https://sdss-mangadap.readthedocs.io/en/latest/emissionlines.html
line_list ={'O2':  np.array([3727.092, 3729.875])* u.Angstrom,
            'Ha':  np.array([6564.608])* u.Angstrom,
            'Hb':  np.array([4862.683])* u.Angstrom, 
            'Hg':  np.array([4341.684])* u.Angstrom, 
            'O3a': np.array([4960.295])* u.Angstrom, 
            'O3b': np.array([5008.240])* u.Angstrom, 
            'N2a': np.array([6549.86 ])* u.Angstrom, 
            'N2b': np.array([6585.27 ])* u.Angstrom,
            }


class Parameters():
    '''
    Class for kl_measurement pars
    '''
    def __init__(self, input_params=None, line_species=['Ha'], verbose=True):
        """_summary_

        Parameters
        ----------
        input_params : _type_, optional
            _description_, by default None
        line_species : list, optional
            The names are inconsequential but required to index the dictionary
            of emission line specific parameters, by default ['Ha']
        """
        self.params = self._get_default_params(line_species)
        self._update_params(input_params, line_species)
        self.print_updates = verbose
        # self.params['central_wavelength'] = self._get_species(line_species, self.params['redshift'])
        self.names = list(self._flatten(self.params).keys())

    @staticmethod
    def _get_default_params(line_species):
        '''
        Default pars dictionary
        '''
        params = {}
        
        ## Shared parameters
        shared_params = {
            'spec_snr':  20.0,
            'image_snr': 80.0,
            
            # 'gamma_t':    0.5,
            'g1' :        0.0,
            'g2' :        0.0,
            'cosi':       0.2, # Inclination angle
            'theta_int':  0.0, # Galaxy intrinsic PA: radians
            'vscale':     0.5, # Velocity scale radius: arcsec
            'r_hl_disk':  1.0, # disk Half_light_radius: arcsec
            'r_hl_bulge': 0.8, # Bulge Half_light_radius: arcsec
                         
            'vcirc':    500.0, # Maximum circular velcoity: km/s
            'v_0':        0.0, # Systematic velocity: km/s
            'v_outer':    0.0, # Velocity scale radius: km/s/arcsec
            'dx_disk':    0.0,
            'dy_disk':    0.0,
            'dx_bulge':   0.0,
            'dy_bulge':   0.0,
            'flux':       10.0, # Total image flux
            'flux_bulge':  0.0, # Total image flux
                         
            'aspect': 0.2,       # Aspect ratio
            'sersic_image': 1.0, # Sersic-index for disk profile
            'beta': 0.0,         # PA of the galaxy w.r.t the cluster (from east)
            # 'img_var_scale': 0,  # log scale
            }

        params['shared_params'] = shared_params
        ## Emission Line specific parameters
        if type(line_species)!=list:
            line_species = [line_species]
        
        num_spectra = max(Counter(line_species).values())
        
        doublet_lines = ['O2']
        
        for line in list(dict.fromkeys(line_species)):
            params[f'{line}_params'] = {
                'v_0'   : 0.0,    # Systematic galaxy velocity  : km/s
                'dx_vel'   : 0.0,
                'dy_vel'   : 0.0,
                'dx_spec'  : 0.0,  # deprecated
                'dy_spec'  : 0.0,  # deprecated
                'r_hl_spec': 1.0,  # Spec Half_light_radius: arcsec
                'f1_0': 1,         # Rescale factor of line profile (sigma_x)
                'f1_1': 1,         # Left wing of line profile, line i. See 'f{i+1}_1' in spec_model.py
                'f1_2': 1,         # Right wing of line profile, line i.
                'I01' : 100.0,     # Central Intensity line 1: arbitrary units
                'bkg_level' : 0.0, # Background level: arbitrary units
                'sersic_spec':  1, # Sersic-index for spectrum brightness profile
                }
            if line in doublet_lines:
                params[f'{line}_params']['v_0_2']    = 0
                params[f'{line}_params']['dx_vel_2'] = 0
                params[f'{line}_params']['dy_vel_2'] = 0
                params[f'{line}_params']['I02']      = 0
                params[f'{line}_params']['f2_0']     = 1
                params[f'{line}_params']['f2_1']     = 1
                params[f'{line}_params']['f2_2']     = 1
                
            for i in range(1, num_spectra+1):
                suffix  = f'_spec{i}'
                prm_add = {'I01':    None, 'bkg_level': None, 
                           'dx_vel': None, 'dy_vel':    None,
                           'f1_0':   None}
                if line in doublet_lines:
                    prm_add['I02']  = None
                    prm_add['f2_0'] = None
                for key, _ in prm_add.copy().items():
                    if '_spec' not in key:
                        params[f'{line}_params'][key + suffix] = params[f'{line}_params'][key] + i/10
                    elif key=='dx_vel' or key=='dy_vel':
                        params[f'{line}_params'][key + f'{i}'] = params[f'{line}_params'][key] + i/10
        
        return params

    def _update_params(self, input_params, 
                       line_species=['O2', 'O2', 'O2']):
        # Get emission lines (non-duplicated)
        lines_in_fit = []
        for key in list(dict.fromkeys(line_species)):
            if key.split('_')[0] != 'shared':
                lines_in_fit.append(key.split('_')[0])
        
        if input_params is not None: 
            # Iterate over shared/line parameter dicts
            for key in input_params.keys():
                # input_params keys: shared_params, O2_params, Ha_params...
                if key in self.params.keys():
                    new_params = input_params[key]
                    for name, value in new_params.items():
                        if self.print_updates: 
                            print(f'Updating {name:10} in {key}: {name:10} = {value}')
                        self.params[key][name] = value
                
                # Now input_params keys: shared/line_params 
                elif key=='line_params':
                    lines = line_list.keys()
                    for line in lines:
                        if line in lines_in_fit:
                            self_params_key = f'{line}_params'
                            new_params = input_params['line_params']
                            for name, value in new_params.items():
                                if self.print_updates: print(f'Updating {name:15} in {self_params_key}: {name:15} = {value}')
                                self.params[self_params_key][name] = value
                
                else:
                    raise KeyError(f'Unkown parameter set {key}, only {list(self.params.keys())} exist!')

            if 'gamma_t' in self.params['shared_params'].keys():
                self._updated_derived_param(self.params)

    @staticmethod
    def _get_species(line_species, z, line_list=line_list):
        '''
        Central wavelength of passed line species
        '''

        if isinstance(line_species, list):
            if len(line_species)==1 and isinstance(line_species[0], str):
                line = line_species[0]
            else:
                raise TypeError('Not sure what type of line_species. Try a string instead.')
        else: 
            line = line_species

        return line, line_list[line] * (1 + z)

    def gen_param_dict(self, par_names, par_values):
        """Returns a copy of the default dict with updated parameter values.
        Intended to be used during inference and thus requires a flat list of parameters.

        Parameters
        ----------
        par_names : list
            List of parameter names. Should match the flattened parameter dictionary
        par_values : list
            List of values to be updated

        Returns
        -------
        Dict
            Unflattened dictionary with updated parameter values
        """
        new_dict = self.params.copy()

        flat_dict = self._flatten(new_dict, level=1)

        # Update parameters
        for key, value in zip(par_names, par_values):
            if key in flat_dict:
                flat_dict[key] = value

            else:
                raise KeyError(f'Keyname "{key}" in your config .yaml file does not match "flat_dict" keys. Here you have {[k for k in flat_dict.keys()]})!')

        # Unflatten
        new_dict = self._unflatten(flat_dict)

        if 'shared_params-gamma_t' in par_names:
            new_dict = self._updated_derived_param(new_dict)

        return new_dict

    @staticmethod
    def _flatten(d, parent_key='', sep='-', level=0):
        """Recursive method for flattening a nested dict,
        Note: Do not change sep='-'
        """
        flat_dict = {}
        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                if level is None or level > 0:
                    flat_dict.update(Parameters._flatten(value, new_key, sep=sep, level=None if level is None else level - 1))
                else:
                    flat_dict[new_key] = value
            else:
                flat_dict[new_key] = value

        return flat_dict


    @staticmethod
    def _unflatten(d, sep='-'):
        """Reconstruct nested dict from flattened version
        Note: Do not change sep='-'
        """
        nested_dict = {}
        for key, value in d.items():
            keys = key.split(sep)
            current = nested_dict
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value

        return nested_dict


    @staticmethod
    def _updated_derived_param(this_dict):
        # Update derived parameters
        shared_dict = this_dict['shared_params'].copy()
        this_dict['shared_params']['g1'] = -shared_dict['gamma_t']*np.cos(2*shared_dict['beta']).value
        this_dict['shared_params']['g2'] = -shared_dict['gamma_t']*np.sin(2*shared_dict['beta']).value

        return this_dict

class FitParameters():
    '''
    Subclass for fit pars
    '''
    def __init__(self, fit_param_names, line_species):
        self._get_fit_pars(fit_param_names, line_species)
        self.names = list(Parameters._flatten(self.params).keys())

    def _get_fit_pars(self, fit_param_names, line_species):
        '''
        Generates attributes (dicts of standard deviations, bounds, latex names) for fit parameters

        Args:
            fit_param_names (list): List of fit parameters
        '''
        default_params = Parameters._get_default_params(line_species=line_species)      # (nested) Default parameters
        flat_default_params = Parameters._flatten(default_params, level=1)
        flat_default_params_names = list(flat_default_params.keys())

        default_param_std = self._get_param_std()        # Default std. deviation
        default_param_limits = self._get_param_limits()    # Default limits
        default_latex_names = self._get_latex_names()  # Default latex names

        params, self.param_std = {}, {}
        self.param_limits, self.latex_names = {}, {}

        # First create shared parameters
        for key in fit_param_names['shared_params']:
            assert f'shared_params-{key}' in flat_default_params_names, f'Undefined fit parameter: {key} in shared_params!'
            params[f'shared_params-{key}'] = flat_default_params[f'shared_params-{key}']

            self.param_std[f'shared_params-{key}'] = default_param_std[key]
            self.param_limits[f'shared_params-{key}'] = default_param_limits[key]
            self.latex_names[f'shared_params-{key}'] = default_latex_names[key]

        # Now create line params
        for key in fit_param_names['line_params']:
            for line in line_species:
                assert f'{line}_params-{key}' in flat_default_params_names, f'Undefined fit parameter: {key} in {line}_params!'

                # If the line is not a doublet don't need to fit for I0_2
                if key in ['v_0_2', 'dx_vel_2', 'dy_vel_2', 'I02', 'f2_1', 'f2_2'] and line not in ['O2', 'O2a', 'O2b']:
                    print(f'Removed {key} from {line} fit parameters...')
                    continue

                params[f'{line}_params-{key}'] = flat_default_params[f'{line}_params-{key}']

                self.param_std[f'{line}_params-{key}'] = default_param_std[key]
                self.param_limits[f'{line}_params-{key}'] = default_param_limits[key]
                self.latex_names[f'{line}_params-{key}'] = default_latex_names[key] + ' ' + line

        self.params = params


    @staticmethod
    def _get_param_std():
        '''
        Generates dictionary of fit parameter standard deviations

        Returns:
            dict: Dictionary of parameter std. deviations
        '''
        param_std = {'gamma_t' : 0.03,
                    'g1': 0.03,
                    'g2': 0.03,
                    'v_0' : 1.,
                    'vcirc' : 30,
                    'cosi' :  0.3,
                    'theta_int' : 0.3,
                    'vscale' : 0.1,
                    'dx_vel' : 0.3,
                    'dy_vel' : 0.3,
                    'dx_vel_2' : 0.3,
                    'dy_vel_2' : 0.3,
                    'r_hl_spec' : 0.1,
                    'I01' : 10.0,
                    'I02' : 10.0,
                    'bkg_level' : 1.,
                    'sersic_spec' : 0.2,
                    'f1_1': 0.2,
                    'f1_2': 0.2,
                    'f2_1': 0.2,
                    'f2_2': 0.2,
                    'sersic_image' : 0.2,
                    'r_hl_disk' : 0.1,
                    'flux' : 2,
                    'dx_disk': 0.2,
                    'dy_disk':0.2,
                    'r_hl_bulge' : 0.1,
                    'flux_bulge' : 2,
                    'dx_bulge': 0.2,
                    'dy_bulge':0.2}

        return param_std

    @staticmethod
    def _get_param_limits():
        '''
        Generates dictionary of upper and lower bounds for fit parameters

        Returns:
            dict: Dictionary of parameter bounds
        '''
        param_limits = {'gamma_t': [-0.3, 0.3],
                    'g1': [-0.5, 0.5],
                    'g2': [-0.5, 0.5],
                    'v_0': [-500., 500.],            # km/s
                    'vcirc': [0, 1e3],                 # km/s
                    'cosi': [0, 1],
                    'theta_int': [0, 2*np.pi],         # radians
                    'vscale': [0.05, 3],                # arcseconds
                    'dx_vel' : [-.3, .3],
                    'dy_vel' : [-.3, .3],
                    'dx_vel_2' : [-.3, .3],
                    'dy_vel_2' : [-1., .3],
                    'r_hl_spec': [0.15, 2],            # arcseconds
                    'I01': [0, 5e3],
                    'I02': [0, 5e3],
                    'bkg_level': [-10, 10],            # arbitrary units
                    'sersic_spec': [0.3, 2],
                    'f1_1': [0., 1.],
                    'f1_2': [0., 1.],
                    'f2_1': [0., 1.],
                    'f2_2': [0., 1.],
                    'sersic_image': [0.3, 2],
                    'r_hl_disk': [0.15, 2],           # arcseconds
                    'flux': [0, 1e3],
                    'dx_disk': [-0.1, 0.1],
                    'dy_disk': [-0.1, 0.1],
                    'r_hl_bulge': [0.15, 3],           # arcseconds
                    'flux_bulge': [0, 1e3],
                    'dx_bulge': [-.1, .1],
                    'dy_bulge': [-.1, .1]}

        return param_limits

    @staticmethod
    def _get_latex_names():
        '''
        Generates dictionary of fit parameter latex names

        Returns:
            dict: Dictionary of parameter latex names
        '''
        latex_names = {'gamma_t': '\\gamma_t',
                        'g1': '\\gamma_1',
                        'g2': '\\gamma_2',
                        'v_0': 'v_0',
                        'vcirc': 'v_{\mathrm{circ}}',
                        'cosi': '\\cos i',
                        'theta_int': '\\theta_{\mathrm{int}}',
                        'dx_vel' : '\delta_x^{\mathrm{vel}}',
                        'dy_vel' : '\delta_y^{\mathrm{vel}}',
                        'dx_vel_2' : '\delta_{x,2}^{\mathrm{vel}}',
                        'dy_vel_2' : '\delta_{y,2}^{\mathrm{vel}}',
                        'vscale': 'r_{\mathrm{vscale}}',
                        'r_hl_spec': 'r_{\mathrm{hl}}^{\mathrm{spec}}',
                        'sersic_spec': '\mathrm{Sersic} n^\mathrm{spec}',
                        'f1_1': 'f_{1,1}',
                        'f1_2': 'f_{1,2}',
                        'f2_1': 'f_{2,1}',
                        'f2_2': 'f_{2,2}',
                        'I01': 'I_0^1',
                        'I02': 'I_0^2',
                        'bkg_level': 'bkg',
                        'sersic_image': '\mathrm{Sersic} n^\mathrm{image}',
                        'r_hl_disk': 'r_{\mathrm{hl}}^{\mathrm{disk}}',
                        'flux': 'Flux',
                        'dx_disk': '\delta_x^{\mathrm{disk}}',
                        'dy_disk': '\delta_y^{\mathrm{disk}}',
                        'r_hl_bulge': 'r_{\mathrm{hl}}^{\mathrm{bulge}}',
                        'flux_bulge': 'Flux-bulge',
                        'dx_bulge': '\delta_x^{\mathrm{bulge}}',
                        'dy_bulge': '\delta_y^{\mathrm{bulge}}'}

        return latex_names

    @staticmethod
    def _get_line_latex(line):
        if line == 'Ha':
            return '\(H\\alpha\)'

        elif line == 'Hb':
            return '\(H\\beta\)'

        else:
            return f'\({line}\)'








