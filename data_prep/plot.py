import numpy as np
import matplotlib.pyplot as plt


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
    

def solve_snr(data, var, data_type):
    if data_type == 'image':
        return np.sum(data) / np.sqrt(np.sum(var))

    elif data_type == 'spec':
        S = np.sum(data)
        N = np.sqrt(np.sum(data + var))
        
        if S/N < 0:
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                  f'Negative SNR found: {S/N:.0f}')
        
        return S/N


def make_exam_plots(data_info, slit_name, how_cut=None, 
                    pkl_folder='./', savefig=True):
    setnames = ['A', 'C', 'B']
    n_sets   = len(setnames)
    n_specs  = len(data_info.spec)
    n_lines  = int(n_specs / n_sets)
    
    fig = plt.figure(figsize=(5*n_lines+2, 4*n_sets))  # (length, height)
    plt.subplots_adjust(hspace=0.35, wspace=0.35) # h=height
    gs = fig.add_gridspec(nrows=n_sets, ncols=(n_lines+1), 
                          height_ratios=[1]*n_sets, 
                          width_ratios=[1]*n_lines + [1.2])
    
    colors = ['orangered', 'cyan', 'gold']
    sets_slitRA  = []
    sets_slitDec = []
    
    # Spectra
    for j in range(n_lines): # start with a column first
        for i in range(n_sets):
            linename   = data_info.spec[j*n_sets+i].meta['line_species']
            slitRA     = data_info.spec[j*n_sets+i].meta['slitRA'].value
            slitDec    = data_info.spec[j*n_sets+i].meta['slitDec'].value
            slitLen    = data_info.spec[j*n_sets+i].meta['slitLen']
            slitWidth  = data_info.spec[j*n_sets+i].meta['slitWidth']
            slit_LPA   = data_info.spec[j*n_sets+i].meta['slitLPA'].value
            
            ax1 = fig.add_subplot(gs[i, j])
            
            spec_data = data_info.spec[j*n_sets+i].data
            spec_mask = data_info.spec[j*n_sets+i].mask
            
            noise = np.nanstd(spec_data[spec_mask])
            ny, nx = spec_data.shape
            xmin =  data_info.spec[j*n_sets+i].meta['lambda_grid'][0, 0].value # x_left
            xmax =  data_info.spec[j*n_sets+i].meta['lambda_grid'][0,-1].value # x_right
            ymin = -data_info.spec[j*n_sets+i].meta['pixScale']*len(spec_data)/2 # y_bottom
            ymax = +data_info.spec[j*n_sets+i].meta['pixScale']*len(spec_data)/2 # y_top
            im_spec = ax1.imshow(np.where(spec_mask, spec_data, np.nan), 
                                 extent=[xmin, xmax, ymin, ymax],
                                 cmap='viridis', aspect='auto', origin='lower', 
                                 vmin=0-noise, vmax=0 + 5*noise, 
                                 zorder=1
                                 )
            fig.colorbar(im_spec, ax=ax1)
            ax1.xaxis.get_major_formatter().set_useOffset(False)
            ax1.set_xlabel(r'Observed Wavelength $\lambda$ ($\AA$)')
            ax1.set_ylabel(r'Spatial Position (arcsec)')
            ax1.grid(linestyle=':', color='orangered', alpha=0.5)
            ax1.set_title(f'#{slit_name} Set {setnames[(j*n_sets+i)%3]}, Line: {linename}', 
                          size=15)
            
            if how_cut is not None:
                UP = how_cut[linename][f'Set{i}']['UP']
                DN = how_cut[linename][f'Set{i}']['DN']
                
                ax1.text(0.98, 0.02, 
                         f'Size {spec_data.shape}\n'+
                         f'Row of center in raw: {(UP+DN)/2:.1f}', 
                         fontsize=10, color='orangered', ha='right', va='bottom', 
                         transform=ax1.transAxes)
                
            else:
                ax1.text(0.98, 0.02, 
                         f'Size {spec_data.shape}', 
                         fontsize=10, color='orangered', ha='right', va='bottom', 
                         transform=ax1.transAxes)
            
            spec_var1 = spec_data # as a Poisson noise
            spec_var2 = data_info.spec[j*n_sets+i].var
            specSNR   = solve_snr(spec_data[spec_mask], 
                                  spec_var1[spec_mask] + spec_var2[spec_mask], 
                                  "spec")
            ax1.text(0.98, 0.98, 
                     f'Min: {np.min(spec_data[spec_mask]):.1f}'+'\n'+
                     f'Max: {np.max(spec_data[spec_mask]):.1f}'+'\n'+ 
                     f'SNR: {specSNR:.0f}',
                     fontsize=10, color='orangered', ha='right', va='top', 
                     transform=ax1.transAxes)
            sets_slitRA.append(slitRA)
            sets_slitDec.append(slitDec)
    sets_slitRA  = sets_slitRA[ : n_sets] # first n_sets entries
    sets_slitDec = sets_slitDec[: n_sets] # are enough
    
    # Imaging
    for i in range(n_sets):
        objRA      = data_info.image.meta['RA']
        objDec     = data_info.image.meta['Dec']
        ap_wcs     = data_info.image.meta['ap_wcs']
        pixScale   = data_info.image.meta['pixScale']
        
        ax2 = fig.add_subplot(gs[i, -1], 
                              projection=ap_wcs)
        
        image_mask = data_info.image.mask
        image_data = data_info.image.data
        # image_data = np.where(image_data>-1, 
        #                       image_data, 0)
        noise = np.nanstd(np.where(image_mask, image_data, np.nan))
        im_imag = ax2.imshow(np.where(image_mask, image_data, np.nan),  
                             cmap='viridis', aspect='equal', 
                             vmin=0-noise, vmax=0 + 5*noise)
        fig.colorbar(im_imag, ax=ax2)
        x0obj,  y0obj  = ap_wcs.wcs_world2pix([[objRA,  objDec ]], 0)[0]  
        x0slit, y0slit = ap_wcs.wcs_world2pix([[sets_slitRA[i], sets_slitDec[i]]], 0)[0]  
        ax2.scatter(x0obj,  y0obj,  marker='x', s=360, color='black',   zorder=1)
        ax2.scatter(x0slit, y0slit, marker='o', s=30,  color=colors[i], zorder=2)
        rot_rectangle(ax2, x0slit, y0slit, slitWidth/pixScale, slitLen/pixScale, 
                      (90-slit_LPA)/57.3, colors[i], '-')
        
        image_var1 = image_data # as a Poisson noise
        image_var2 = data_info.image.var
        imageSNR   = solve_snr(image_data[image_mask], 
                               image_var1[image_mask] + image_var2[image_mask], 
                               "image")
        ax2.text(0.98, 0.98, 
                 f'Min = {np.nanmin(image_data[image_mask]):.1f}'+'\n'+
                 f'Max = {np.nanmax(image_data[image_mask]):.1f}'+'\n'+
                 f'SNR = {imageSNR:.0f}',
                 fontsize=10, color='orangered', ha='right', va='top', 
                 transform=ax2.transAxes)
        ax2.text(0.98, 0.02, 
                 f'Size {image_data.shape}', 
                 fontsize=10, color='orangered', ha='right', va='bottom', 
                 transform=ax2.transAxes)
        ax2.coords['ra' ].set_major_formatter('dd:mm:ss')
        ax2.coords['dec'].set_major_formatter('dd:mm:ss')
        ax2.set_xlabel('RA')
        ax2.set_ylabel('Dec', labelpad=-2)
        ax2.grid(linestyle=':', color='orangered', alpha=0.5)
        ax2.set_title(f'Set {setnames[i]} imaging', size=15)
    
    if savefig:
        plt.savefig(f'{pkl_folder}slit_{slit_name}.png', 
                    dpi=150, bbox_inches='tight')
    else:
        plt.show()
    return


