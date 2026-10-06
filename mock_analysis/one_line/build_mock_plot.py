import matplotlib.pyplot as plt
import numpy as np

def make_exam_plots(mock_data_info, slit_name, iter_num, mock_folder):
    n_sets = len(mock_data_info['spec'])
    line   = mock_data_info['spec'][0]['par_meta']['line_species']
    
    fig = plt.figure(figsize=(11, 5*n_sets))  # (length, height)
    plt.subplots_adjust(hspace=0.4, wspace=0.0) # h=height
    gs = fig.add_gridspec(nrows=1*n_sets, ncols=2, 
                          height_ratios=[1]*n_sets, 
                          width_ratios=[1, 2])
    
    for i in range(n_sets):
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1], 
                              projection=mock_data_info['image']['par_meta']['ap_wcs'])
        
        spec_data = mock_data_info['spec'][i]['data']
        noise = np.std(spec_data)
        im_spec = ax1.imshow(spec_data, origin='lower', 
                             extent=[mock_data_info['spec'][i]['par_meta']['lambda_grid'][0, 0].value, # x_left
                                     mock_data_info['spec'][i]['par_meta']['lambda_grid'][0,-1].value, # x_right
                                     -mock_data_info['spec'][i]['par_meta']['pixScale']*len(spec_data)/2,  # y_bottom
                                     +mock_data_info['spec'][i]['par_meta']['pixScale']*len(spec_data)/2], # y_top
                             cmap='viridis', aspect='auto', 
                             vmin=0-noise, vmax=0 + 5*noise)
        fig.colorbar(im_spec, ax=ax1)
        ax1.set_ylim(-mock_data_info['spec'][i]['par_meta']['pixScale']*len(spec_data)/2,
                     +mock_data_info['spec'][i]['par_meta']['pixScale']*len(spec_data)/2)
        ax1.xaxis.get_major_formatter().set_useOffset(False)
        ax1.set_xlabel(r'Observed Wavelength $\lambda$ ($\AA$ or nm)')
        ax1.set_ylabel(r'Spatial Position (arcsec)')
        ax1.grid(linestyle=':', color='white', alpha=0.5)
        ax1.set_title('Mock MMT/Binospec 2D Spec')
        
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
        
        image_data = mock_data_info['image']['data']
        objRA      = mock_data_info['image']['par_meta']['RA']
        objDec     = mock_data_info['image']['par_meta']['Dec']
        ap_wcs     = mock_data_info['image']['par_meta']['ap_wcs']
        pixScale   = mock_data_info['image']['par_meta']['pixScale']
        slitRA     = mock_data_info['spec'][i]['par_meta']['slitRA'].value
        slitDec    = mock_data_info['spec'][i]['par_meta']['slitDec'].value
        slitLen    = mock_data_info['spec'][i]['par_meta']['slitLen']
        slitWidth  = mock_data_info['spec'][i]['par_meta']['slitWidth']
        slit_LPA   = mock_data_info['spec'][i]['par_meta']['slitLPA'].value
        noise = np.std(image_data)
        im_imag = ax2.imshow(image_data.T,  
                             cmap='viridis', aspect='equal', 
                             vmin=0-noise, vmax=0 + 5*noise)
        fig.colorbar(im_imag, ax=ax2)
        x0obj,  y0obj  = ap_wcs.wcs_world2pix([[objRA,  objDec ]], 0)[0]  
        x0slit, y0slit = ap_wcs.wcs_world2pix([[slitRA, slitDec]], 0)[0]  
        ax2.scatter(x0obj, y0obj, marker='x', s=360, color='black', zorder=1)
        rot_rectangle(ax2, x0slit, y0slit, slitWidth/pixScale, slitLen/pixScale, 
                      (slit_LPA)/57.3, 'white', '-')
        # ax2.set_xlim(left=0, right=image_data.shape[1]-1)
        # ax2.set_ylim(bottom=0, top=image_data.shape[0]-1)
        ax2.coords['ra' ].set_major_formatter('dd:mm:ss')
        ax2.coords['dec'].set_major_formatter('dd:mm:ss')
        ax2.set_xlabel('RA')
        ax2.set_ylabel('Dec', labelpad=-1)
        ax2.grid(linestyle=':', color='white', alpha=0.5)
        ax2.set_title('Mock Subaru Imaging')
        
        ax1txts, ax2txts = '', ''
        for key, arr in mock_data_info['par_fit'].items():
            if key[:14] == 'shared_params-':
                ax2txts += (key[14:] + ' = ' + '{:.3g}'.format(arr) + '\n')
            elif key.split('-')[0] == f'{line}_params':
                ax1txts += (key.split('-')[1] + ' = ' + '{:.0f}'.format(arr) + '\n')
        ax2.text(1, 1, ax2txts, fontsize=10, color='white', ha='right', va='top', 
                 transform=ax2.transAxes)
        ax1.text(1, 1, ax1txts, fontsize=10, color='white', ha='right', va='top', 
                 transform=ax1.transAxes)
    
    plt.savefig(f'{mock_folder}mock_{slit_name}_{iter_num}.png', 
                dpi=150, bbox_inches='tight')
    return

