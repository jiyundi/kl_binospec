import numpy as np
from scipy.ndimage import binary_erosion

from klm.parameters import Parameters
from klm.spec_model import SlitModel #, IFUModel
from klm.image_model import ImageModel
from klm.config import Config
from klm.line_width_profile import find_line_sigma, exam_line_profile


class KLInference():
    '''
    Base class for KL inference
    '''

    def __init__(self, data_info, config, verbose=True):
        '''
        Initializes the KLInference object.

        Args:
            data_info (str): Information about the data.
            config (dict): Configuration parameters.

        '''
        self.config = Config(config)

        self._init_data(data_info, verbose)

        if self.config.likelihood.fid_params is not None:
            fid_params = self.config.likelihood.fid_params.copy()
            
            update_dict = {}
            if 'shared_params-gamma_t' in self.config.params.names:
                if not np.any([('shared_params' in parname) for parname in self.config.params.names]):
                    fid_params['shared_params'] = {}
                fid_params['shared_params']['beta'] = data_info['galaxy']['beta'].value
                print(f'kl_inference.py: Updated beta = {fid_params["shared_params"]["beta"]:.3f} since you included shared_params-gamma_t in config.')

                # Since we added beta in fid, update_dict must contain shared_params
                update_dict['shared_params'] = {} # We will update this in the next block
                
            # update only when 'shared_params' key exists
            if 'shared_params' in fid_params.keys():
                if 'shared_params' not in update_dict.keys():
                    update_dict['shared_params'] = {}
                update_dict['shared_params'] = {
                    **fid_params['shared_params']
                    }
            
            # update only when '{}_params'.format(line) key exists
            if 'line_params' in update_dict.keys():
                update_dict['line_params'] = {
                    **fid_params['line_params']
                    }
            else:
                for key, _ in fid_params.items():
                    if key != 'shared_params':
                        update_dict[key] = {
                            **fid_params[key]
                            }
            
            # update only when 'update_dict' not empty
            if update_dict:
                spec = data_info['spec']
                self.params._update_params(
                    update_dict,
                    [spec[i]['meta']['line_species'] for i in range(len(spec))]
                    )
                    
    def _init_data(self, data_info, verbose=True):
        self.meta_gal = data_info['galaxy']
        if self.config.likelihood.isFitImage:
            self.meta_image = data_info['image']['meta']
            self.data_image = data_info['image']['data']
            self.mask_image = data_info['image']['mask']
            self.var_image  = data_info['image']['var']
            self.image_model = ImageModel(self.meta_image)


        if self.config.galaxy_params.obs_type == 'slit':
            self.params = Parameters(
                line_species=self.config.galaxy_params.line_species,
                verbose=verbose
                )
            self._init_slit_data(data_info, verbose)


    def _init_slit_data(self, data_info, verbose=True):
        '''
        Initializes slit data.

        Parameters:
        - data_info (dict): A dictionary containing information about the spectral data.

        '''
        self.meta_spec = []
        self.data_spec = []
        self.mask_spec = []
        self.var_spec  = []
        self.spec_model = []
        if self.config.likelihood.isFitSpec == 'rotation_curve':
            self.data_k     = []
            self.data_k_err = []
            self.mask_k     = []

        for i in range(len(data_info['spec'])):
            this_spec = data_info['spec'][i]

            if this_spec['meta']['line_species'] not in self.config.galaxy_params.line_species:
                continue

            self.meta_spec.append(this_spec['meta'])
            self.data_spec.append(this_spec['data'])
            self.mask_spec.append(this_spec['mask'])
            self.var_spec.append( this_spec['var' ])

            if self.config.likelihood.isFitSpec == 'rotation_curve':
                # Pass the line profiles available from data 
                if 'data_k' in this_spec.keys():
                    self.data_k.append(    this_spec['data_k'    ])
                    self.data_k_err.append(this_spec['data_k_err'])
                    self.mask_k.append(    this_spec['mask_k'    ])
                
                # If none are available, generate the LPF now in here
                else:
                    print('[INFO] kl_inference.py: Fitting mode = rotation_curve.', 
                          'Please check if line centers are well-extracted in', 
                          'test_converted_spec_data_to_LPF.jpg')
                    # from scipy.ndimage import median_filter
                    LPFs = find_line_sigma(
                        # median_filter(
                            this_spec['data'], 
                            # size=3), 
                        line = this_spec['meta']['line_species'],
                        lambda_scale = this_spec['meta']['dispersion'] # A/px
                        )
                    exam_line_profile(
                        LPFs, 
                        this_spec['data'], this_spec['mask'], 
                        this_spec['meta']['line_species'], 
                        lambda_scale = this_spec['meta']['dispersion'], # A/px
                        filename='test_converted_spec_data_to_LPF.jpg'
                        )
                    if this_spec['meta']['line_species'] not in ('O2'):
                        self.data_k.append(    LPFs[0][1:, 0].astype(float))
                        self.data_k_err.append(LPFs[0][1:, 1].astype(float))

                        # Since the mask doesn't consider the 3-row binning,
                        # we need to exclude the upper & lower rows whose bin 
                        # crossed the valid and invalid emission line regions.
                        k_mask = (LPFs[0][1:, 4] == 'True') # 'True' -> boolean array
                        fit_mask = binary_erosion(
                            k_mask,
                            structure=np.ones(3, dtype=bool),
                            iterations=1,
                            )
                        self.mask_k.append(fit_mask)

            S = SlitModel(
                obj_param=self.meta_gal, 
                meta_param=this_spec['meta'], 
                line_profile=self.config.galaxy_params.line_profile_path,
                redo_data=this_spec['data'],
                rc_type=self.config.galaxy_params.rc_type,
                )
            self.spec_model.append(S)
                
            if self.config.verbose:
                print(f'[INFO]: Finished initializing model for {self.config.line_species[i]}')

        if len(self.config.galaxy_params.line_species) > len(self.spec_model):
            print(f'[WARNING]: Specified {self.config.galaxy_params.line_species} line specie(s) but only {len(self.spec_model)} emission lines in data.')
        
        if len(self.spec_model) > len(self.config.galaxy_params.line_species):
            raise Exception(f'Data has {len(self.spec_model)} emission lines but only {self.config.galaxy_params.line_species} line species are specified in config file.')
    
    
    # def _init_ifu_data(self, data_info):
    #     '''
    #     Initializes IFU data.

    #     Parameters:
    #     - data_info (dict): A dictionary containing information about the data.

    #     '''
    #     this_vmap = data_info['vmap'][self.config.vmap_type]
    #     self.meta_2Dmap = this_vmap['par_meta']
    #     self.data_2Dmap = this_vmap['data']
    #     self.var_2Dmap = this_vmap['var']
    #     self.mask_2Dmap = this_vmap['mask']

    #     self.IFU_model = IFUModel(this_vmap['par_meta'], self.config.galaxy_params.rc_type)

    #     if self.config.verbose:
    #         print('Initializing IFU data...')
    #         print(f'Finished initializing model for {self.config.vmap_type} vmap')


    def _print_TF_relation_table(self, a, b, sigmaTF_intr):
        '''
        Prints a table of the Tully-Fisher relation parameters.

        Parameters:
        - a (float): The slope of the Tully-Fisher relation.
        - b (float): The intercept of the Tully-Fisher relation.
        - sigmaTF_intr (float): The intrinsic scatter of the Tully-Fisher relation.
        '''
        print('Tully-Fisher Relation:')
        print('----------------------')
        print(f'Slope (a): {a}')
        print(f'Intercept (b): {b}')
        print(f'Intrinsic Scatter: {sigmaTF_intr}')
        print('\n')


    def calc_spectrum_loglike(self, pars):
        '''
        Computes log likelihood for the spectrum fit

        Args:
            pars (dict): Dictionary of model parameters

        Attributes:
            meta_pars (float): Slit angle
            data_info (2d array): Spectrum and spectrum variance

        Returns:
                float: log likelihood spectrum
        '''
        return self._loglike_slit(pars)


    def _loglike_slit(self, pars):
        '''
        Calculate the log-likelihood of the slit data.

        Parameters:
        - pars (dict): A dictionary containing the parameter values for the emission lines.

        Returns:
        - chi2 (float): The calculated chi-squared of the slit data.
        '''
        chi2     = 0
        lines    = list(dict.fromkeys(self.config.galaxy_params.line_species))
        n_lines  = len(lines)
        n_sets   = len(self.spec_model) // n_lines
        
        for j_line in range(n_lines):
            line = lines[j_line]
            for i in range(n_sets):
                
                # Multi spec may have various I0 due to exposure
                key_correct   = ['I01', 'bkg_level', 
                                #  'f1_0', 
                                 'f1_1', 'f1_2']
                doublet_lines = ['O2']
                if line in doublet_lines:
                    key_correct.append('I02')
                    # key_correct.append('f2_0')
                    key_correct.append('f2_1')
                    key_correct.append('f2_2')
                    
                suffix = f'_spec{i+1}'
                for key in key_correct:
                    full_key_name = key + suffix
                    pars[f'{line}_params'][key] = pars[f'{line}_params'][full_key_name]
                
                this_line_dict = {**pars['shared_params'], **pars[f'{line}_params']}
                
                spec_i = j_line * n_sets + i

                fit_type = self.config.likelihood.isFitSpec
                if fit_type is True: fit_type = 'spec' # default value
                
                if fit_type not in ('rotation_curve', 'spec'): # , 'spec_and_rotation_curve'
                    raise ValueError(f'Invalid isFitSpec: {fit_type}')

                chi2_one_spec = 0.0

                if 'rotation_curve' in fit_type:
                    data_k     = self.data_k[spec_i].astype(float)
                    data_k_err = self.data_k_err[spec_i].astype(float)
                    mask_k     = self.mask_k[spec_i].astype(bool)

                    model_k = self.spec_model[spec_i].get_observable(
                        this_line_dict, return_type='rotation_curve'
                    )

                    chi2_one_spec += self._loglike_one_rotation_curve(
                        data_k, mask_k, data_k_err, model_k
                    )

                if 'spec' in fit_type:
                    data_spec = self.data_spec[spec_i]
                    mask_spec = self.mask_spec[spec_i]
                    var_spec  = self.var_spec[spec_i]

                    model_spec = self.spec_model[spec_i].get_observable(this_line_dict)

                    chi2_one_spec += self._loglike_one_slit(
                        data_spec, mask_spec, var_spec, model_spec
                    )
                
                chi2 += chi2_one_spec
                
        return chi2

    
    def _loglike_one_rotation_curve(self, 
                                    data_k, mask_k, data_k_err, model_k):
        # Some extreme v0 or other params may cause model_k to be NaN
        # This model_k cannot be used to calculate chi2, so we return a panelty chi2 value
        if np.sum(np.isnan(model_k[mask_k])) > 0:
            chi2 = 1e10 * np.sum(np.isnan(model_k[mask_k]))
            return chi2

        resi2     = (data_k[mask_k] - model_k[mask_k])**2
        var_total =  data_k_err[mask_k]**2

        chi2 = np.sum(resi2 / var_total)

        assert chi2 >= 0, f'Found negative var_total in spec chi2 ({np.round(chi2, 2)})'
        
        return chi2
    
    
    def _loglike_one_slit(self, 
                          data_spec, mask_spec, var_spec, model_spec, 
                          reduce=False):
        resi2     = (data_spec[mask_spec] - model_spec[mask_spec])**2
        var_total =  data_spec[mask_spec] + var_spec[mask_spec]
        
        if reduce:
            chi2 = np.sum( resi2 / var_total )
        else:
            chi2 = np.sum((resi2 / var_total) + np.log(2.0 * np.pi * var_total))
        
        assert chi2 >= 0, f'Found negative var_total in spec chi2 ({np.round(chi2, 2)})'
        
        return chi2


    def calc_image_loglike(self, pars, reduce=False):
        '''
        Computes log likelihood for the image fit

        Args:
            pars (dict): Dictionary of model parameters

        Attributes:
            data_info (2d array): Image and image variance

        Returns:
            float: chi-squared of image
        '''
        model_image = self.image_model.get_image(pars['shared_params'])
        
        # (1) Normal calculation: Do not count negative-flux data's pixels in
        data_image = self.data_image
        var_image  = self.var_image
        mask_image = self.mask_image
        
        resi2     = (data_image[mask_image] - model_image[mask_image])**2
        var_total =  data_image[mask_image] + var_image[mask_image]

        if reduce:
            chi2 = np.sum( resi2 / var_total )
        else:
            chi2 = np.sum((resi2 / var_total) + np.log(2.0 * np.pi * var_total))
        
        assert chi2 >= 0, 'Found negative var_total in image chi2'
        
        return chi2
