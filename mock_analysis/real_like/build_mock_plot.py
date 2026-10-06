import matplotlib.pyplot as plt
import numpy as np

def make_exam_plots(mock_data_info, slit_name, iter_num, mock_folder):
    setnames = ['A', 'C', 'B']
    n_sets   = len(setnames)
    n_specs  = len(mock_data_info['spec'])
    n_lines  = int(n_specs / n_sets)
    
    fig = plt.figure(figsize=(5*n_lines+3, 4*n_sets))  # (length, height)
    plt.subplots_adjust(hspace=0.35, wspace=0.25) # h=height
    gs = fig.add_gridspec(nrows=n_sets, ncols=(n_lines+1), 
                          height_ratios=[1]*n_sets, 
                          width_ratios=[0.7]*n_lines + [1])
    
    colors = ['orangered', 'cyan', 'gold']
    
    # Spectra
    for j in range(n_lines): # start with a column first
        for i in range(n_sets):
            ax1 = fig.add_subplot(gs[i, j])
        
            spec_data = mock_data_info['spec'][j*n_sets+i]['data']
            noise     = np.std(spec_data)
            line      = mock_data_info['spec'][j*n_sets+i]['par_meta']['line_species']
            im_spec = ax1.imshow(spec_data, # origin='lower', 
                                 extent=[mock_data_info['spec'][j*n_sets+i]['par_meta']['lambda_grid'][0, 0].value, # x_left
                                         mock_data_info['spec'][j*n_sets+i]['par_meta']['lambda_grid'][0,-1].value, # x_right
                                         -mock_data_info['spec'][j*n_sets+i]['par_meta']['pixScale']*len(spec_data)/2,  # y_bottom
                                         +mock_data_info['spec'][j*n_sets+i]['par_meta']['pixScale']*len(spec_data)/2], # y_top
                                 cmap='viridis', aspect='auto', 
                                 vmin=0 - 1*noise, vmax=0 + 5*noise)
            fig.colorbar(im_spec, ax=ax1)
            ax1.set_ylim(-mock_data_info['spec'][j*n_sets+i]['par_meta']['pixScale']*len(spec_data)/2,
                         +mock_data_info['spec'][j*n_sets+i]['par_meta']['pixScale']*len(spec_data)/2)
            ax1.xaxis.get_major_formatter().set_useOffset(False)
            ax1.set_xlabel(r'Observed Wavelength $\lambda$ ($\AA$ or nm)')
            ax1.set_ylabel(r'Spatial Position (arcsec)')
            ax1.grid(linestyle=':', color='white', alpha=0.5)
            ax1.set_title(f'Set {setnames[(j*n_sets+i)%3]}, Line: {line}')
    
    # Imaging
    for i in range(n_sets):
        ax2 = fig.add_subplot(gs[i, -1], 
                              projection=mock_data_info['image']['par_meta']['ap_wcs'])
        
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
        im_imag = ax2.imshow(image_data,  
                             cmap='viridis', aspect='equal', 
                             vmin=0-noise, vmax=0 + 5*noise)
        fig.colorbar(im_imag, ax=ax2)
        x0obj,  y0obj  = ap_wcs.wcs_world2pix([[objRA,  objDec ]], 0)[0]  
        x0slit, y0slit = ap_wcs.wcs_world2pix([[slitRA, slitDec]], 0)[0]  
        ax2.scatter(x0obj,  y0obj,  marker='x', s=360, color='black',   zorder=1)
        ax2.scatter(x0slit, y0slit, marker='o', s=30,  color=colors[i], zorder=2)
        rot_rectangle(ax2, x0slit, y0slit, slitWidth/pixScale, slitLen/pixScale, 
                      (90-slit_LPA)/57.3, colors[i], '-')
        # ax2.set_xlim(left=0, right=image_data.shape[1]-1)
        # ax2.set_ylim(bottom=0, top=image_data.shape[0]-1)
        ax2.coords['ra' ].set_major_formatter('dd:mm:ss')
        ax2.coords['dec'].set_major_formatter('dd:mm:ss')
        ax2.set_xlabel('RA')
        ax2.set_ylabel('Dec', labelpad=-1)
        ax2.grid(linestyle=':', color='white', alpha=0.5)
        ax2.set_title(f'Set {setnames[i]} imaging')
        
        ax1txts, ax2txts = '', ''
        for key, arr in mock_data_info['par_fit'].items():
            if key[:14] == 'shared_params-':
                ax2txts += (key[14:] + ' = ' + '{:.3g}'.format(arr) + '\n')
            elif key.split('-')[0] == 'line_params':
                ax1txts += (key.split('-')[1] + ' = ' + '{:.0f}'.format(arr) + '\n')
        ax2.text(1, 1, ax2txts, fontsize=9, color='orangered', ha='right', va='top', 
                 transform=ax2.transAxes)
        ax1.text(1, 1, ax1txts, fontsize=9, color='orangered', ha='right', va='top', 
                 transform=ax1.transAxes)
    
    plt.savefig(f'{mock_folder}mock_{slit_name}_{iter_num:03d}.png', 
                dpi=150, bbox_inches='tight')
    return

