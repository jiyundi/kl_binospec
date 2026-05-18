from functools import reduce
# import numpy as np
import scipy.stats
from collections import Counter

from klm.parameters import Parameters


class Config():
    def __init__(self, config_dict=None):
        if config_dict is None or config_dict is {}:
            print('No config file provided. Using default values.')
            config_dict = {}
        self._set_config(config_dict)

    def _set_config(self, config_dict):
        self.init_galaxy_params(config_dict)
        self.init_TFprior(config_dict)
        self.init_params(config_dict)
        self.init_likelihood(config_dict)
        self.init_truevalues(config_dict)

        # Misc
        self.verbose = config_dict.get('verbose', False)

    def init_likelihood(self, config_dict):
        likelihood_params = config_dict['likelihood']
        self.likelihood = Container()
        # General options
        self.likelihood.isFitImage = likelihood_params.get('fit_image', True)
        self.likelihood.isFitSpec  = likelihood_params.get('fit_spec', True)
        self.likelihood.fid_params = likelihood_params.get('fid_params', None)
        self.likelihood.set_non_analytic_prior = likelihood_params.get('set_non_analytic_prior', None)  # should be a dict of prior for each parameter

        if all(f'shared_params-{p}' in self.params.names for p in ['r_hl_disk', 'r_hl_bulge']):
            self.likelihood.apply_rhl_constraint = True
        else:
            self.likelihood.apply_rhl_constraint = False
            print('Note config.py: Not applying r_hl_bulge < r_hl_disk.')
            
        line_species = config_dict['galaxy_params']['line_species']
        num_spectra  = max(Counter(line_species).values())
        line         = line_species[0]
        for i in range(1, num_spectra+1):
            if all(f'{line}_params-{p}_spec{i}' in self.params.names for p in ['I01', 'bkg_level']):
                self.likelihood.apply_line_flux_constraint = True
                self.likelihood.apply_which_line_flux_constraint = line_species
                self.likelihood.num_spectra = num_spectra
            else:
                self.likelihood.apply_line_flux_constraint = False
                print('Note config.py: Not applying flux > bkg constraint.')
        

    def init_galaxy_params(self, config_dict):
        '''
        Initializes galaxy parameters based on the observation type.
        '''
        self.galaxy_params = Container()
        galaxy_params = config_dict.get('galaxy_params', {})

        self.galaxy_params.obs_type = galaxy_params.get('obs_type', 'slit')
        self.galaxy_params.rc_type = galaxy_params.get('rc_type', 'arctan')

        if self.galaxy_params.obs_type == 'slit':
            line_species = galaxy_params.get('line_species', 'Ha')
            if isinstance(line_species, str):
                self.galaxy_params.line_species = [line_species]

            elif isinstance(line_species, list):
                # if len(line_species)>1:
                #     assert all(x==line_species[0] for x in line_species), 'Can only fit multi-slit obs. of same emission line'
                self.galaxy_params.line_species = line_species

            self.galaxy_params.line_profile_path = galaxy_params.get('line_profile_path', None)
            assert config_dict.get('vmap_type', None) is None, 'vmap_type should not be set for slit data'
            self.vmap_type = None

        elif self.galaxy_params.obs_type == 'IFU':
            self.galaxy_params.Rmax_G = float(galaxy_params.get('Rmax_G', None))
            self.galaxy_params.Rmax_ST = float(galaxy_params.get('Rmax_ST', None))

            self.vmap_type = galaxy_params.get('vmap_type', 'gas')
            assert config_dict.get('line_species', None) is None, 'line_species should not be set for IFU data'
            self.galaxy_params.line_species = [self.vmap_type]

        self.galaxy_params.log10_Mstar = float(galaxy_params.get('log10_Mstar', None))
        self.galaxy_params.log10_Mstar_err = float(galaxy_params.get('log10_Mstar_err', 0.))

        #If any attribute is None, raise an error
        for attr in self.galaxy_params.__dict__:
            if getattr(self.galaxy_params, attr) is None:
                if attr == 'line_profile_path':
                    print('Note config.py: line_profile_path is set to None')
                    continue

                raise ValueError(f'{attr} is not set in galaxy_params')


    def init_TFprior(self, config_dict):
        '''
        Initializes the Tully-Fisher prior based on the observation type.
        '''
        self.TFprior = Container()
        TFprior = config_dict.get('TFprior', {})
        self.TFprior.use_TFprior = TFprior.get('use_TFprior', True)

        if self.TFprior.use_TFprior is False:
            return

        # First check if log10vTF is set
        if TFprior.get('log10_vTF', None) is not None:
            self.TFprior.log10_vTF = TFprior['log10_vTF']
            self.set_sigmaTF(TFprior)
            self.TFprior.a = None
            self.TFprior.b = None


        # Next we check if the relation is supplied
        elif TFprior.get('relation', None) is not None:
            a = TFprior.get('a', None)
            b = TFprior.get('b', None)
            self.set_sigmaTF(TFprior)

            # log10_Mstar     = self.galaxy_params.log10_Mstar
            # log10_Mstar_err = self.galaxy_params.log10_Mstar_err

            self.TFprior.log10_vTF = eval(TFprior['relation'])
            self.TFprior.a = a
            self.TFprior.b = b

        else:
            print('Note config.py: TF prior not specified. Using default values.')
            self._init_TF_relation()
    
    def init_truevalues(self, config_dict):
        '''
        Pass true values used in building mock data to draw lines in corner plots.

        Args:
            None.

        Returns:
            None
        '''
        if config_dict.get('truevalues') is not None:
            self.truevalues = config_dict['truevalues']
        else:
            self.truevalues = None

    def init_params(self, config_dict):
        self.params = Container()
        params = config_dict.get('params', {})

        shared_params_dict = params.get('shared_params', {})
        shared_params_dict = {} if shared_params_dict is None else shared_params_dict
        shared_params      = Parameters._flatten(shared_params_dict, level=0)
        
        # Default line config (key: line_params)
        line_paramss = {}
        if params.get('line_params', None) is not None:
            line_params_dict = params.get('line_params', {})
            line_params_dict = {} if line_params_dict is None else line_params_dict
            line_params      = Parameters._flatten(line_params_dict, level=0)
            line_paramss['line_params'] = line_params

        # Specified line configs (Hb_params, O3a_params, or more)
        else:
            for key, line_params_dict in params.items():
                line_params  = Parameters._flatten(line_params_dict, level=0)
                line_paramss[key] = line_params
        
        self.params.names = []
        self.params.latex_names = {}
        self.params.prior = {}

        # Iterate over shared params
        for p in shared_params.keys():
            this_prior = shared_params[p].get('prior', None)
            latex_name = shared_params[p].get('latex_name', p)
            if this_prior is None:
                raise ValueError(f'Prior not set for {p}')

            p = 'shared_params-'+p
            self._init_prior(p, this_prior)

            self.params.names.append(p)
            self.params.latex_names[p] = latex_name

        # If default line_params: iterate over all lines
        if line_paramss.get('line_params', None) is not None:
            line_params = line_paramss['line_params']
            for line in list(dict.fromkeys(self.galaxy_params.line_species)):
                self._complete_line_config(line_params, line, 
                                           specified_lines=False)
        # If specified lines...
        else:
            for linekey, line_params in line_paramss.items():
                if linekey != 'shared_params':
                    self._complete_line_config(line_params, linekey.split('_')[0], 
                                               specified_lines=True)
    
    def _complete_line_config(self, line_params, line, specified_lines=False):
        doub_pars      = ['v_0_2','dx_vel_2','dy_vel_2','I02','f2_0','f2_1','f2_2']
        doub_pars_twin = ['v_0',  'dx_vel',  'dy_vel',  'I01','f1_0','f1_1','f1_2']
        doublet_lines  = ['O2', 'O2a', 'O2b']
        for p in line_params.keys(): 
            this_prior = line_params[p].get('prior', None)
            latex_name = line_params[p].get('latex_name', p)

            # Remove doublet params for singlet lines
            if any([name in p for name in doub_pars]) and \
            (line not in doublet_lines):
                print(f'Removed {p} from {line} fit parameters...')
                continue

            # Now add line name to the parameter name
            line_p = f'{line}_params-{p}'
            
            self._init_prior(line_p, this_prior)
            self.params.names.append(line_p)
            self.params.latex_names[line_p] = latex_name
            
            if line in doublet_lines:
                if p in doub_pars_twin:
                    i_q    = doub_pars_twin.index(p)
                    q      = doub_pars[i_q]
                    line_q = f'{line}_params-{q}'
                    
                elif p.split('_spec')[0] in doub_pars_twin:
                    i_q    = doub_pars_twin.index(p.split('_spec')[0])
                    q      = doub_pars[i_q] + '_spec' + p.split('_spec')[1]
                    line_q = f'{line}_params-{q}'
                
                self._init_prior(line_q, this_prior)
                self.params.names.append(line_q)
                latex_n = latex_name[:-1] + "\,_{(2)}$"
                self.params.latex_names[line_q] = latex_n
        
    def _init_prior(self, param_name, prior_dict):
        '''
        Initialize the prior distribution for a given parameter.

        Args:
            param_name (str): The name of the parameter.
            prior_dict (dict): A dictionary containing the prior information.

        Returns:
            None
        '''
        if 'min' in prior_dict.keys():
            min_val, max_val = prior_dict.get('min', None), prior_dict.get('max', None)
            if type(min_val) is str:
                min_val = eval(min_val)
            if type(max_val) is str:
                max_val = eval(max_val)
            self.params.prior[param_name] = [min_val, max_val]

        if 'norm' in prior_dict.keys():
            if (type(prior_dict['norm']['loc']) is float) or (type(prior_dict['norm']['loc']) is int):
                loc   = prior_dict['norm']['loc']
                scale = prior_dict['norm']['scale']
            else:
                loc   = reduce(getattr, [self]+prior_dict['norm']['loc'].split('.'))  # Hack to get attribute of attribute
                scale = reduce(getattr, [self]+prior_dict['norm']['scale'].split('.')) # From https://stackoverflow.com/questions/4247036/python-recursively-getattribute
            self.params.prior[param_name] = scipy.stats.norm(loc=loc, scale=scale)

    def _init_TF_relation(self):
        '''
        Initializes the Tully-Fisher relation based on the observation type and velocity map type.
        For slit data uses relation from Miller et al. 2011: https://arxiv.org/pdf/1102.3911.pdf
        For IFU data uses relation from Ristea et al. 2023: https://arxiv.org/pdf/2311.13251.pdf
        '''
        log10_Mstar     = float(self.galaxy_params.log10_Mstar)
        log10_Mstar_err = float(self.galaxy_params.log10_Mstar_err)

        if self.galaxy_params.obs_type == 'IFU':
            if self.vmap_type == 'stellar':

                if self.galaxy_params.Rmax_ST == 1:
                    a = 0.282
                    b = -0.78
                    sigmaTF_intr = 0.07

                if self.galaxy_params.Rmax_ST == 1.3:
                    a = 0.279
                    b = -0.73
                    sigmaTF_intr = 0.06

                if self.galaxy_params.Rmax_ST == 2:
                    a = 0.27
                    b = -0.60
                    sigmaTF_intr = 0.05

                print(f'Using TFR for Rmax_ST = {self.galaxy_params.Rmax_ST}')

            elif self.vmap_type == 'gas':

                if self.galaxy_params.Rmax_G == 1:
                    a = 0.282
                    b = -0.75
                    sigmaTF_intr = 0.06

                if self.galaxy_params.Rmax_G == 1.3:
                    a = 0.275
                    b = -0.65
                    sigmaTF_intr = 0.06

                if self.galaxy_params.Rmax_G == 2:
                    a = 0.26
                    b = -0.48
                    sigmaTF_intr = 0.04

                print(f'Using TFR for Rmax_G = {self.galaxy_params.Rmax_G}')

            TF_relation_str = 'log10_Mstar * a + b'
            log10_vTF = eval(TF_relation_str)
            self.TFprior.sigmaTF = sigmaTF_intr

        elif self.galaxy_params.obs_type == 'slit':
            # Stellar mass - TF relation
            a = 1.718
            b = 3.869
            sigmaTF_intr = 0.058
            log10_Mstar  = log10_Mstar
            TF_relation_str = '(log10_Mstar - a) / b'
            log10_vTF = eval(TF_relation_str)
            self.TFprior.sigmaTF = (sigmaTF_intr**2 + (log10_Mstar_err/b)**2)**0.5

        self.TFprior.a = a
        self.TFprior.b = b
        self.TFprior.sigmaTF_intr = sigmaTF_intr
        self.TFprior.log10_vTF = log10_vTF
        self.TFprior.TF_relation_str = TF_relation_str

        # if self.config.verbose:
        #     print('\n')
        #     print('Initializing TFR...')
        #     print(f'Stellar Mass is 10^{log10_Mstar:.2f}')
        #     print(f'log10_v prior at {self.TFprior.log10_vTF:0.2f} dex')

    def __repr__(self):
        config_str = repr(self)
        # return the string
        return config_str

    def set_sigmaTF(self, TFprior):
        # Check if sigmaTF is set
        if TFprior.get('sigmaTF', None) is None:
            if  TFprior.get('sigmaTF_intr', None) is None:
                raise ValueError('sigmaTF or sigmaTF_intr not set in TFprior')
            else:
                self.TFprior.sigmaTF = TFprior['sigmaTF_intr']
                print('Using intrinsic scatter as sigmaTF')

        else:
            self.TFprior.sigmaTF = TFprior['sigmaTF']


class Container():
    def __init__(self):
        pass

    def __repr__(self):
        config_str = repr(self)
        # return the string
        return config_str

def repr(self):
    # Get all the attributes of the class
    attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

    # Initialize an empty string
    config_str = '\n'

    # Loop over all the attributes
    for attr in attributes:
        # Get the value of the attribute
        value = getattr(self, attr)
        print(attr)
        # Add the attribute name and value to the string
        if isinstance(value, dict):
            config_str += f'{attr}:\n'
            for k, v in value.items():
                v = isclassinstance(v)
                config_str += f'    {k}: {v}\n'

        elif isinstance(value, list):
            config_str += f'    {attr}:\n'
            for v in value:
                v = isclassinstance(v)
                config_str += f'    {v}\n'

        # Determine if value object is any class

        else:
            config_str += f'{attr}: {value}\n'

    return config_str

def isclassinstance(obj):
    if hasattr(obj, '__dict__'):
        return obj.__class__.__name__
    else:
        return obj
