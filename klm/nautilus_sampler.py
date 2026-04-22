import numpy as np
import scipy
from scipy.stats import lognorm
from nautilus import Sampler, Prior
from klm.kl_inference import KLInference

class NautilusSampler(KLInference):
    '''
    Sub-class for parameter inference using nested sampling
    '''

    def __init__(self, data_info=None, config=None):
        KLInference.__init__(self, data_info, config)
        self.n_params = len(self.config.params.names)

        # For deltax_vel/deltay_vel
        a, b = (-1 - 0.)/0.5, (1 - 0.)/0.5
        self.truncnormprior = scipy.stats.truncnorm(a, b)
        if not self.config.TFprior.use_TFprior:
            print('Warning: Not using using TF prior')
        self._init_prior(self.config.likelihood.set_non_analytic_prior)


    def _get_wrapped_params(self):
        '''Returns a list of bools, True for wrapped (circular) params
        otherwise False
        '''
        master_wrapped_params = ['theta_int', 'phi0']
        wrapped_params = [False]*len(self.config.params.names)

        for item in master_wrapped_params:
            for i, key in enumerate(self.config.params.names):
                if item in key:
                    wrapped_params[i] = True

        return wrapped_params
    
    
    def _init_prior(self, set_prior):
        self.prior_ppf = {}
        if set_prior is not None:
            for par in set_prior:
                prior_samples = set_prior[par]
                hist, bin_edges = np.histogram(prior_samples, bins=100)
                hist_cumulative = np.cumsum(hist / hist.sum())
                bin_middle = (bin_edges[:-1] + bin_edges[1:]) / 2

                self.prior_ppf[par] = scipy.interpolate.interp1d(hist_cumulative, bin_middle,
                             bounds_error=False, fill_value=(bin_middle[0], bin_middle[-1]))
        
        return
    
    
    def calc_joint_loglike(self, cube):
        '''
        Computes the joint likelihood of image and spectra

        Args:
            fit_par_values (list): list of fit parameter values from sampler
            ndim (int): Number of dimensions
            nparams (int): Number of fit parameters

        Returns:
            float: log likelihood
        '''
        # Get a dictionary of updated fit parameter values
        pars = self.params.gen_param_dict(self.config.params.names, cube)

        image_loglike, spec_loglike = 0., 0.
        
        # Disk > bulge constraint
        if self.config.likelihood.apply_rhl_constraint:
            constraint  = pars['shared_params']['r_hl_disk'] - pars['shared_params']['r_hl_bulge']
            constraint2 = pars['shared_params']['flux']      - pars['shared_params']['flux_bulge']

            if constraint < 0.:
                return -1e100 * (1 - constraint)

            if constraint2 < 0.:
                return -1e100 * (1 - constraint2)
        
        # vcirc constraint
        constraint_vcirc = 500 - pars['shared_params']['vcirc']
        if constraint_vcirc < 0:
            print('VCIRC ILLEGAL..................')
            return -1e100 * (1 - constraint_vcirc)
        
        # Background < flux constraint
        if self.config.likelihood.apply_line_flux_constraint == True:
            n_spec = self.config.likelihood.num_spectra
            lines  = self.config.likelihood.apply_which_line_flux_constraint
            for i in range(1, n_spec+1):
                this_I01 = pars[f'{lines[0]}_params'][f'I01_spec{i}']
                this_bkg = pars[f'{lines[0]}_params'][f'bkg_level_spec{i}']
                if this_I01 < this_bkg:
                    return -1e100 * (1 - (this_I01 - this_bkg))

        # Get image log likelihood
        if self.config.likelihood.isFitImage is True:
            image_loglike = self.calc_image_loglike(pars)

        # Get spectrum log likelihood
        if self.config.likelihood.isFitSpec is True:
            spec_loglike = self.calc_spectrum_loglike(pars)

        # Compute joint likelihood
        joint_loglike = -0.5 * (spec_loglike + image_loglike)
        
        return joint_loglike
    
    
    def run(self, output_dir='./nautilus_output/', test_run=False, **kwargs):
    
        assert self.n_params != 0, 'No fit parameters entered'
    
        # -------------------------
        # Build prior
        # -------------------------
        prior = Prior()
    
        for i, key in enumerate(self.config.params.names):
            # TF prior
            if key == 'shared_params-vcirc' and self.config.TFprior.use_TFprior:
                
                # log10(vcirc) ~ Gaussian(mu, sigma)
                mu    = self.config.params.prior[key].mean()
                sigma = self.config.params.prior[key].std()
                
                # ln(vcirc) = log10(vcirc) * ln10
                # mu --> mu * ln10, sigma --> sigma * ln10
                mu_ln    = mu    * np.log(10)
                sigma_ln = sigma * np.log(10)
                
                # vcirc ~ Log10Normal(mu,        sigma)
                # scipy.stats.lognorm(exp(μ_ln), σ_ln)
                logN = lognorm(scale = np.exp(mu_ln), 
                               s = sigma_ln)
                
                prior.add_parameter(key, logN)
                
            else:
                low_lim, up_lim = self.config.params.prior[key]
                
                prior.add_parameter(key, (low_lim, up_lim))
    
        # -------------------------
        # Likelihood wrapper
        # -------------------------
        def loglike(theta):
            # theta is a dict {param_name:value}
            # theta dict --> data cube (value only)
            cube = np.array([theta[name] for name in self.config.params.names])
    
            return self.calc_joint_loglike(cube)
    
        # -------------------------
        # Sampler
        # -------------------------
        if test_run:
            print("Testing likelihood...")
            sampler = Sampler(
                prior, loglike, n_live=400, 
                filepath=output_dir + "chain.hdf5"
            )
            sampler.run(n_like_max=1000, verbose=True)
            print("\033[42m" + 'WARNING:' + "\033[0m " + 
                  'Testing done. OK ✅\n')
    
        else:
            sampler = Sampler(
                prior, loglike, n_live=400, 
                pool=8, # pool = CPU cores 
                # HPC ONLY: pool -> n_threads and set 
                # SLURM with --cpus-per-task=n_threads
                filepath=output_dir + "chain.hdf5"
            )
            sampler.run(verbose=True)
    
        return sampler
