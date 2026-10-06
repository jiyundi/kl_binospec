{
'O2_params': {
    'v_0': {'latex_name': '$v_0$', 'prior': {'min': -200, 'max': 200}}, 
    'v_0_2': {'latex_name': '$v_0$', 'prior': {'min': -200, 'max': 200}}, 
    'I01_spec1': {'latex_name': '$I_0^{\\rm Spec 1}$', 'prior': {'min': 0, 'max': 1000}}, 
    'I02_spec1': {'latex_name': '$I_0^{\\rm Spec 1}$', 'prior': {'min': 0, 'max': 1000}}, 
    'I01_spec2': {'latex_name': '$I_0^{\\rm Spec 2}$', 'prior': {'min': 0, 'max': 1000}}, 
    'I02_spec2': {'latex_name': '$I_0^{\\rm Spec 2}$', 'prior': {'min': 0, 'max': 1000}}, 
    'I01_spec3': {'latex_name': '$I_0^{\\rm Spec 3}$', 'prior': {'min': 0, 'max': 1000}}, 
    'I02_spec3': {'latex_name': '$I_0^{\\rm Spec 3}$', 'prior': {'min': 0, 'max': 1000}}
    }, 
'Hg_params': {
    'v_0': {'latex_name': '$v_0$', 'prior': {'min': -200, 'max': 200}}, 
    'I01_spec1': {'latex_name': '$I_0^{\\rm Spec 1}$', 'prior': {'min': 0, 'max': 1000}}, 
    'I01_spec2': {'latex_name': '$I_0^{\\rm Spec 2}$', 'prior': {'min': 0, 'max': 1000}}, 
    'I01_spec3': {'latex_name': '$I_0^{\\rm Spec 3}$', 'prior': {'min': 0, 'max': 1000}}
    }, 
'Hb_params': {
    'v_0': {'latex_name': '$v_0$', 'prior': {'min': -200, 'max': 200}}, 
    'I01_spec1': {'latex_name': '$I_0^{\\rm Spec 1}$', 'prior': {'min': 0, 'max': 1000}}, 
    'I01_spec2': {'latex_name': '$I_0^{\\rm Spec 2}$', 'prior': {'min': 0, 'max': 1000}}, 
    'I01_spec3': {'latex_name': '$I_0^{\\rm Spec 3}$', 'prior': {'min': 0, 'max': 1000}}
    }, 
'shared_params': {
    'g1': {'latex_name': '$g_1$', 'prior': {'max': 0.15, 'min': -0.15}}, 
    'g2': {'latex_name': '$g_2$', 'prior': {'max': 0.15, 'min': -0.15}}, 
    'theta_int': {'latex_name': '$\\theta_{\\mathrm{int}}$', 'prior': {'max': 3.141592653589793, 'min': -3.141592653589793}}, 
    'vcirc': {'latex_name': '$v_{\\rm circ}$', 'prior': {'norm': {'loc': 'TFprior.log10_vTF', 'scale': 'TFprior.sigmaTF'}}}, 
    'vscale': {'latex_name': '$r_{\\mathrm{vscale}}$', 'prior': {'max': 2, 'min': 0.01}}, 
    'cosi': {'latex_name': '$\\cos{(i)}$', 'prior': {'max': 1, 'min': 0}}, 
    'r_hl_disk': {'latex_name': '$r_{\\mathrm{hl, disk}}$', 'prior': {'min': 0.2, 'max': 2.0}}, 
    'flux': {'latex_name': '$\\log{(F_{\\rm disk})}$', 'prior': {'max': 2, 'min': 0}}
    }
}