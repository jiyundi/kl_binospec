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


def rot_rectangle(ax, x0, y0, dx, dy, rotation, color, ls):
    xUL = x0 + (-dx/2)*np.cos(rotation) - (+dy/2)*np.sin(rotation)
    yUL = y0 + (-dx/2)*np.sin(rotation) + (+dy/2)*np.cos(rotation)
    xUR = x0 + (+dx/2)*np.cos(rotation) - (+dy/2)*np.sin(rotation)
    yUR = y0 + (+dx/2)*np.sin(rotation) + (+dy/2)*np.cos(rotation)
    xLL = x0 + (-dx/2)*np.cos(rotation) - (-dy/2)*np.sin(rotation)
    yLL = y0 + (-dx/2)*np.sin(rotation) + (-dy/2)*np.cos(rotation)
    xLR = x0 + (+dx/2)*np.cos(rotation) - (-dy/2)*np.sin(rotation)
    yLR = y0 + (+dx/2)*np.sin(rotation) + (-dy/2)*np.cos(rotation)
    ax.plot([xUL, xUR, xLR, xLL, xUL], 
            [yUL, yUR, yLR, yLL, yUL], color=color, linestyle=ls)
    return 


def init_mock(mock_params):
    # spec_this_setA, image_A = make_mock_data(
    #     lines, meta_gal, mock_params, Set='A', if_add_noise=False
    #     )
    spec_this_setC, image_C = make_mock_data(
        lines, meta_gal, mock_params, Set='C', if_add_noise=False
        )
    # spec_this_setB, image_B = make_mock_data(
    #     lines, meta_gal, mock_params, Set='B', if_add_noise=False
    #     )
    spec_this_setP, image_P = make_mock_data(
        lines, meta_gal, mock_params, Set='P', if_add_noise=False
        )
    
    all_image = image_C
    all_specs = []
    for line in lines:
        # all_specs.append(spec_this_setA[line])
        all_specs.append(spec_this_setC[line])
        # all_specs.append(spec_this_setB[line])
        all_specs.append(spec_this_setP[line])
        
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

    fig = plt.figure(figsize=(3+3*len(mock_data_info['spec']), 3))  # (length, height)
    gs = fig.add_gridspec(nrows=1, ncols=1+len(mock_data_info['spec']), 
                          height_ratios=[1], width_ratios=[1]+[1]*len(mock_data_info['spec']))
    plt.subplots_adjust(hspace=0.35, wspace=0.05) # h=height
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_axis_off()
    colors = ['orangered', 'cyan', 'gold']
    
    # image observed
    image_obs = np.flip(mock_data_info['image']['data'], axis=1)
    noise     = np.std(image_obs)
    imshow1 = ax1.imshow(image_obs, origin='lower', 
                         cmap='gray', aspect='equal', 
                         vmin=0, vmax=0 + 5*noise)
    
    # spec observed
    imshows = [None] * len(mock_data_info['spec'])
    axs     = [None] * len(mock_data_info['spec'])
    for i in range(len(mock_data_info['spec'])):
        ax2 = fig.add_subplot(gs[0, 1+i])
        ax2.set_axis_off()
        spec_data = mock_data_info['spec'][i]['data']
        spec_cont = mock_data_info['spec'][i]['cont_model']
        noise     = np.std(spec_data)
        imshows[i]= ax2.imshow(spec_data+spec_cont, origin='lower', 
                               cmap='pink', aspect='auto', 
                               vmin=0, vmax=0 + 5*noise)
        rot_rectangle(ax1, 
                      mock_data_info['image']['par_meta']['ap_wcs'].wcs_world2pix([[mock_data_info['spec'][i]['par_meta']['slitRA'].value, mock_data_info['spec'][i]['par_meta']['slitDec'].value]], 0)[0][0], 
                      mock_data_info['image']['par_meta']['ap_wcs'].wcs_world2pix([[mock_data_info['spec'][i]['par_meta']['slitRA'].value, mock_data_info['spec'][i]['par_meta']['slitDec'].value]], 0)[0][1], 
                      mock_data_info['spec'][i]['par_meta']['slitWidth']/mock_data_info['image']['par_meta']['pixScale'], 
                      mock_data_info['spec'][i]['par_meta']['slitLen']/mock_data_info['image']['par_meta']['pixScale'], 
                      (90-mock_data_info['spec'][i]['par_meta']['slitLPA'].value)/57.3, 
                      colors[i], '-')
        ax2.text(0.1, 0.9, f'{i+1}', fontsize=16, color=colors[i], ha='left', va='top', 
                 transform=ax2.transAxes)
        ax2.set_title( 'Spectrum', fontsize=16)
        axs[i] = ax2
    
    # -----------------------------------------------------------------------
    
    i = np.arccos(mock_params["shared_params-cosi"]) * 57.29583
    ax1.set_title(r'$i$ = '+f'{i:.0f}'+r'$^\circ$ Image', fontsize=16)
    
    # g1 = mock_params["shared_params-g1"]
    # ax1.set_title(r'$g_1$ = '+f'{g1:.2f} Image', fontsize=16)
    
    # -----------------------------------------------------------------------
    
    def update(i):
        # -------------------------------------------------------------------
        cosi = cosis[i]
        mock_params['shared_params-cosi'] = cosi
        ax1.set_title(r'$i$ = '+f'{np.arccos(cosi) * 57.29583:.0f}'+r'$^\circ$ Image', fontsize=16)
        
        # g1 = g1s[i]
        # mock_params['shared_params-g1'] = g1
        # ax1.set_title(r'$g_1$ = '+f'{g1:.2f} Image', fontsize=16)
        # -------------------------------------------------------------------
        
        image_data = init_mock(mock_params)['image']['data']
        
        imshow1.set_data(np.flip(image_data, axis=1))
        ax1.set_axis_off()
        for i_spec in range(len(mock_data_info['spec'])):
            spec_data  = init_mock(mock_params)['spec'][i_spec]['data']
            spec_cont  = init_mock(mock_params)['spec'][i_spec]['cont_model']
            imshows[i_spec].set_data(spec_data+spec_cont)
            ax2 = axs[i_spec]
            ax2.set_axis_off()
        
    ani = FuncAnimation(fig, update, frames=list(range(len(cosis)))+[len(cosis)-1]*180)
    
    ani.save("mock_cosis_two_spec_.mp4", fps=60)