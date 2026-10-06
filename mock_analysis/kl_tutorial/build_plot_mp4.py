import yaml
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from build_multi_mock import make_mock_data

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Inter",
    "font.serif": "Inter",
})


def init_mock(mock_params):
    spec_this_setA, image_A = make_mock_data(
        lines, meta_gal, mock_params, Set='A', if_add_noise=False
        )
    spec_this_setC, image_C = make_mock_data(
        lines, meta_gal, mock_params, Set='C', if_add_noise=False
        )
    spec_this_setB, image_B = make_mock_data(
        lines, meta_gal, mock_params, Set='B', if_add_noise=False
        )
    
    all_image = image_C
    all_specs = []
    for line in lines:
        all_specs.append(spec_this_setA[line])
        all_specs.append(spec_this_setC[line])
        all_specs.append(spec_this_setB[line])
        
    specs_data_info = []
    for spec_this_line_this_set in all_specs:
        specs_data_info.append(
            {'data':       spec_this_line_this_set['spec_data'],
             'var':        spec_this_line_this_set['spec_var'],
             'cont_model': spec_this_line_this_set['cont_model_spec'],
             'par_meta':   spec_this_line_this_set['meta_spec'] 
             }
            )
    
    image_data_info = {
        'data':     all_image['image_data'],
        'var':      all_image['image_var'],
        'par_meta': all_image['meta_image']
        }
    
    mock_data_info = {
        'spec':    specs_data_info,
        'image':   image_data_info,
        'galaxy':  meta_gal
        }
    
    if mock_params is not None:
        mock_data_info['par_fit'] = mock_params
    
    # Unfortunately, my galsim.wcs objects cannot be packed in PKL files.
    # To pack galsim.wcs, DELETE it before packing in PKL.
    # To read, always regenerate by using ap_wcs. (by JD)
    mock_data_info['image']['par_meta'].pop('wcs') # delete it!
    
    return mock_data_info


if __name__ == '__main__':
    """
        Other external Python scripts may load functions above
        If so, "iter_num" or "run" variables defined wherelse may 
        be contaminated/destroyed!!!
        To protect them, only when directly execute this script,
        the variables below will then be executed.
    """
    mock_folder = 'kl_tutorial_pkl/'
    slitname = 111
    lines = ['Hb']
    cosis = np.cos(np.linspace(0, 60,  181)/57.29583)
    g1s   =        np.linspace(0, 0.3, 181)
            
    # Now get object data from object catalog
    slit_name = '{:03d}'.format(slitname)
    RA_obj     = (10/3600)*u.deg
    Dec_obj    = (10/3600)*u.deg
    redshift   = 1
    meta_gal = {
        'redshift': redshift,
        'RA':       RA_obj,
        'Dec':      Dec_obj,
        'beta':     0*u.deg,
        'log10_Mstar':     10.00, 
        'log10_Mstar_err': 0.100,
        }
    
    with open("./mock_params.yaml", "r", encoding="utf-8") as file:
        mock_params = yaml.safe_load(file)
    
    mock_data_info = init_mock(mock_params)

    fig = plt.figure(figsize=(8, 8))  # (length, height)
    gs = fig.add_gridspec(nrows=1, ncols=3, height_ratios=[1], width_ratios=[1,1,1])
    plt.subplots_adjust(hspace=0.35, wspace=0.18) # h=height
    ax1 = fig.add_subplot(gs[0, 0])
    
    # image observed
    image_obs = np.flip(mock_data_info['image']['data'], axis=1)
    noise     = np.std(image_obs)
    imshow1 = ax1.imshow(image_obs, origin='lower', 
                         cmap='gray', aspect='equal', 
                         vmin=0, vmax=0 + 5*noise)
    
    # -----------------------------------------------------------------------
    
    # i = np.arccos(mock_params["shared_params-cosi"]) * 57.29583
    # ax1.set_title(r'$i$ = '+f'{i:.0f}'+r'$^\circ$', fontsize=16)
    
    g1 = mock_params["shared_params-g1"]
    ax1.set_title(r'$g_1$ = '+f'{g1:.2f}', fontsize=16)
    
    # -----------------------------------------------------------------------
    
    plt.axis('off')
    
    def update(i):
        # cosi = cosis[i]
        # mock_params['shared_params-cosi'] = cosi
        # ax1.set_title(r'$i$ = '+f'{np.arccos(cosi) * 57.29583:.0f}'+r'$^\circ$', fontsize=16)
        
        g1 = g1s[i]
        mock_params['shared_params-g1'] = g1
        ax1.set_title(r'$g_1$ = '+f'{g1:.2f}', fontsize=16)
        
        image_data = init_mock(mock_params)['image']['data']
        imshow1.set_data(np.flip(image_data, axis=1))
        
    ani = FuncAnimation(fig, update, frames=list(range(len(cosis)))+[len(cosis)-1]*180)
    
    ani.save("mock_g1s.mp4", fps=60)