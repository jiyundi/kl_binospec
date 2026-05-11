# No dependencies required in this script

def make_config_dic(linespecies, fitting_params, fid_params, 
                    log10_Mstar=9.30, log10_Mstar_err=0.05, 
                    use_line_profile=None):
    config_dic = {
        'galaxy_params': {
            'obs_type':     'slit', 
            'line_species':    linespecies, 
            'log10_Mstar':     log10_Mstar, 
            'log10_Mstar_err': log10_Mstar_err, 
            'line_profile_path': use_line_profile,
            }, 
        'likelihood': {
            'fit_image':  True, 
            'fit_spec':   True, 
            'set_non_analytic_prior': None,
            'fid_params': fid_params
            }, 
        'TFprior': {
            'use_TFprior': True, 
            'log10_vTF':   None,
            'sigmaTF':     None, 
            'a':            None, 
            'b':            None, 
            'sigmaTF_intr': None, 
            'relation':     None
            }, 
        'params': fitting_params,
        'truevalues': None
        }
    return config_dic

