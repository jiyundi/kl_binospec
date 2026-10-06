import numpy as np
import scipy
from scipy.stats import lognorm
from nautilus import Sampler, Prior
from klm.kl_inference import KLInference
import os
import multiprocessing as mp


# Module level (Do not nested within the run() function)
_worker_self = None

def _loglike_worker(theta):
    # Previous transformation logic in the loglike section
    cube = np.array([theta[name] for name in _worker_self.config.params.names])  
    return _worker_self.calc_joint_loglike(cube)

def _init_worker(self_obj):
    global _worker_self
    _worker_self = self_obj

    # print(
    #     f"Likelihood worker started: "
    #     f"pid={os.getpid()}, "
    #     f"affinity={sorted(os.sched_getaffinity(0))}",
    #     flush=True,
    # )


class NautilusSampler(KLInference):
    '''
    Sub-class for parameter inference using nested sampling
    '''

    def __init__(self, data_info=None, config=None, verbose=True):
        KLInference.__init__(self, data_info, config, verbose)
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
        constraint_vcirc = 1000 - pars['shared_params']['vcirc']
        if constraint_vcirc < 0:
            print('VCIRC ILLEGAL (> 1000 km/s) ..................')
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

        # Get spectrum log likelihood, data array vs. model array
        if self.config.likelihood.isFitSpec in (True, 'rotation_curve'):
            spec_loglike = self.calc_spectrum_loglike(pars)

        # Compute joint likelihood
        joint_loglike = -0.5 * (spec_loglike + image_loglike)
        
        return joint_loglike
    
    def run(self, output_dir='./nautilus_output/', test_run=False, parallel=True, **kwargs):
    
        assert self.n_params != 0, 'No fit parameters entered'

        # Check whether previous sampling results exist
        saved_chain = os.path.join(output_dir, "chain.hdf5")

        if os.path.isfile(saved_chain):
            parallel = False
            print(f"[INFO] Found existing chain: {saved_chain}")
            print("[INFO] Switching to serial mode.")

        # -------------------------
        # Nautilus 运行过程中反复向 /xdisk 写 HDF5 checkpoint 会导致出现 Errno 5 类报错。
        # 最实用的解决办法是：运行过程中把 chain.hdf5 写到计算节点本地 /tmp，拟合完成后再一次性复制到 /xdisk。
        # 一次性复制 135 MB，比持续进行 HDF5 checkpoint 写入可靠得多。
        # -------------------------
        if parallel == True:
            os.makedirs(output_dir, exist_ok=True)

            final_chain = os.path.join(output_dir, "chain.hdf5")

            scratch_root = os.environ.get(
                "SLURM_TMPDIR",
                f"/tmp/{os.environ.get('USER', 'user')}"
            )

            job_id = os.environ.get("SLURM_JOB_ID", "interactive")
            task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "single")

            scratch_dir = os.path.join(
                scratch_root,
                f"nautilus_{job_id}_{task_id}_{os.getpid()}"
            )
            os.makedirs(scratch_dir, exist_ok=True)

            local_chain = os.path.join(scratch_dir, "chain.hdf5")

            print(f"Nautilus checkpoint: {local_chain}")
            print(f"Final chain:         {final_chain}")

        # -------------------------
        # Build prior
        # -------------------------
        prior = Prior()
    
        for _, key in enumerate(self.config.params.names):
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
        # Sampler
        # -------------------------
        def loglike(theta):
            # theta is a dict {param_name:value}
            # theta dict --> data cube (value only)
            cube = np.array([theta[name] for name in self.config.params.names])

            return self.calc_joint_loglike(cube)

        if parallel == False:
            sampler = Sampler(
                prior, loglike, n_live=400, 
                filepath=saved_chain if os.path.isfile(saved_chain)
                         else os.path.join(output_dir, "chain.hdf5"), # None
                )
            if os.path.isfile(saved_chain):
                print("[INFO] Loading previous sampling results.")
            elif test_run:
                print("Testing likelihood...")
                sampler.run(n_like_max=1000, verbose=True)
                print("\033[42m" + 'INFO:' + "\033[0m " + 'Testing done. OK ✅\n')
            else:
                sampler.run(verbose=True)

        else: # parallel == True
            n_cores = self.get_n_cores()
            print('[INFO] You are using the parallel mode:', 
                  f"CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')}", 
                  f"affinity={len(os.sched_getaffinity(0))}", 
                  f"workers={n_cores}")

            with mp.Pool(n_cores, initializer=_init_worker, initargs=(self,)) as pool:
                sampler = Sampler(
                    prior, _loglike_worker, n_live=400, 
                    pool=pool,
                    filepath=local_chain, # output_dir + "chain.hdf5"
                )
                if test_run:
                    print("Testing likelihood...")
                    sampler.run(n_like_max=2000, verbose=True)
                    print("\033[42m" + 'INFO:' + "\033[0m " + 'Testing done. OK ✅\n')
                else:
                    sampler.run(verbose=True)

            # Pool 已关闭，拟合也已完成，现在一次性复制到 /xdisk
            import shutil
            temporary_destination = final_chain + ".copying"

            try:
                shutil.copy2(local_chain, temporary_destination)
                os.replace(temporary_destination, final_chain)

                print(f"chain.hdf5 successfully saved to {final_chain}")

            except Exception:
                print(
                    f"ERROR: Failed to copy chain to /xdisk.\n"
                    f"The local chain is still available at:\n{local_chain}"
                )
                raise
    
        return sampler


    @staticmethod
    def get_n_cores():
        affinity_cores = len(os.sched_getaffinity(0))
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

        if slurm_cpus is None:
            return affinity_cores

        return min(int(slurm_cpus), affinity_cores)


    def check_parallel_loglike(
        self,
        cube,
        n_workers=None,
        n_repeat=16,
        ):
        """
        检查同一个参数点在串行和 multiprocessing 下
        是否返回相同的 likelihood。
        """
        cube = np.asarray(cube, dtype=float)

        if cube.shape != (self.n_params,):
            raise ValueError(
                f"cube shape should be ({self.n_params},), "
                f"but got {cube.shape}"
            )

        if n_workers is None:
            affinity_cores = len(os.sched_getaffinity(0))
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")

            if slurm_cpus is None:
                n_workers = affinity_cores
            else:
                n_workers = min(
                    int(slurm_cpus),
                    affinity_cores,
                )

        # _loglike_worker 需要的输入是 theta 字典
        theta = dict(zip(self.config.params.names, cube))

        # 串行：直接调用真正的 likelihood
        serial_values = np.asarray(
            [
                self.calc_joint_loglike(cube.copy())
                for _ in range(n_repeat)
            ],
            dtype=float,
        )

        # 并行：使用正式拟合所使用的 worker
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(self,),
        ) as pool:
            parallel_values = np.asarray(
                pool.map(
                    _loglike_worker,
                    [theta.copy() for _ in range(n_repeat)],
                ),
                dtype=float,
            )

        serial_consistent = np.allclose(
            serial_values,
            serial_values[0],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

        parallel_consistent = np.allclose(
            parallel_values,
            parallel_values[0],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

        serial_parallel_equal = np.allclose(
            serial_values,
            parallel_values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

        print("\n========== Likelihood consistency check ==========")
        print(f"Workers: {n_workers}")
        print(f"Serial values:   {serial_values}")
        print(f"Parallel values: {parallel_values}")
        print(f"Serial internally consistent:   {serial_consistent}")
        print(f"Parallel internally consistent: {parallel_consistent}")
        print(f"Serial vs parallel equal:       {serial_parallel_equal}")
        print("==================================================\n")

        if not serial_consistent:
            raise RuntimeError(
                "Serial likelihood is not deterministic."
            )

        if not parallel_consistent:
            raise RuntimeError(
                "Parallel workers return inconsistent likelihoods."
            )

        if not serial_parallel_equal:
            raise RuntimeError(
                "Serial and parallel likelihoods are different."
            )

        return {
            "serial": serial_values,
            "parallel": parallel_values,
            "serial_consistent": serial_consistent,
            "parallel_consistent": parallel_consistent,
            "serial_parallel_equal": serial_parallel_equal,
        }