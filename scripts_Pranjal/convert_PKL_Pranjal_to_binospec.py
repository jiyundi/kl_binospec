import joblib
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

_mapping = {
    'OII': 'O2',
    'OIIa': 'O2',
    'OIIb': 'O2',
    'OIIIa': '',
    'OIIIb': 'O3a',
    'OIIIc': 'O3b',
    'Halpha': 'Ha',
    'Hb': 'Hb'
    }

def another_load_mock(pkl_folder='mock/', slit_num=95):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}_raw.pkl', "rb") as f:
        data_info = joblib.load(f)
    
    # assert data_info['galaxy']['log10_Mstar'] != None, \
    #     "Cannot find corresponing stallar mass M*"  # if not, error
    
    # Recover wcs(galsim.wcs) from ap_wcs
    # ap_wcs  = data_info['image']['par_meta']['ap_wcs']
    # data_info['image']['par_meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)
    return data_info


def rot_rectangle(ax, x0, y0, dx, dy, rotation=0, color='white', ls='-'):
    """
    Warning: 
        1. rotation angle is defined counter-clockwise on xy-plane, 
           not East-North plane.
        2. dx is defined differently by how you choose the slit will 
           start rotating. 
           Here, we define the slit has horizontal length > vertical width.
           If define the horizontal as the starting position, 
           dx = slit length/height. 
           Think this twice as it is annoying but important!
           E.g. = 8 arcsecs
        3. dy: e.g. dy = 1 arcsec
    """
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
    ax.scatter((xUR + xLR)/2, 
               (yUR + yLR)/2, marker='o', s=30, color='white')
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
                    pkl_folder='./', special_idx=None, savefig=True):
    n_specs  = len(data_info['spec'])
    n_sets   = n_specs
    n_lines  = 1
    specnums = np.arange(1, n_specs+1)
    
    fig = plt.figure(figsize=(5*n_lines+2, 4*n_sets))  # (length, height)
    plt.subplots_adjust(hspace=0.35, wspace=0.35) # h=height
    gs = fig.add_gridspec(nrows=n_sets, ncols=(n_lines+1), 
                          height_ratios=[1]*n_sets, 
                          width_ratios=[1]*n_lines + [1.2])
    
    colors = ['orangered', 'cyan', 'gold', 'green', 'red']
    sets_slitRA  = []
    sets_slitDec = []
    
    # Spectra
    for j in range(n_lines): # start with a column first
        for i in range(n_sets):
            linename   = data_info['spec'][j*n_sets+i]['meta']['line_species']
            slitRA     = data_info['spec'][j*n_sets+i]['meta']['slitRA'].value
            slitDec    = data_info['spec'][j*n_sets+i]['meta']['slitDec'].value
            slitLen    = data_info['spec'][j*n_sets+i]['meta']['slitLen']
            slitWidth  = data_info['spec'][j*n_sets+i]['meta']['slitWidth']
            slit_LPA   = data_info['spec'][j*n_sets+i]['meta']['slitLPA'].value
            
            ax1 = fig.add_subplot(gs[i, j])
            
            spec_data = data_info['spec'][j*n_sets+i]['data']
            spec_mask = data_info['spec'][j*n_sets+i]['mask']
            
            noise = np.nanstd(spec_data[spec_mask])
            ny, nx = spec_data.shape
            xmin =  data_info['spec'][j*n_sets+i]['meta']['lambda_grid'][0, 0].value # x_left
            xmax =  data_info['spec'][j*n_sets+i]['meta']['lambda_grid'][0,-1].value # x_right
            ymin = -data_info['spec'][j*n_sets+i]['meta']['pixScale']*len(spec_data)/2 # y_bottom
            ymax = +data_info['spec'][j*n_sets+i]['meta']['pixScale']*len(spec_data)/2 # y_top
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
            ax1.set_title(f'#{slit_name} Spec {specnums[(j*n_sets+i)]}, Line: {linename}', 
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
            spec_var2 = data_info['spec'][j*n_sets+i]['var']
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
        objRA      = data_info['image']['meta']['RA']
        objDec     = data_info['image']['meta']['Dec']
        ap_wcs     = data_info['image']['meta']['ap_wcs']
        pixScale   = data_info['image']['meta']['pixScale']
        
        ax2 = fig.add_subplot(gs[i, -1], 
                              projection=ap_wcs)
        
        image_mask = data_info['image']['mask']
        image_data = data_info['image']['data']
        # image_data = np.where(image_data>-1, 
        #                       image_data, 0)
        noise = np.nanstd(np.where(image_mask, image_data, np.nan))
        im_imag = ax2.imshow(np.where(image_mask, image_data, np.nan),  
                             cmap='viridis', aspect='equal', 
                             vmin=0-noise, vmax=0 + 5*noise)
        fig.colorbar(im_imag, ax=ax2)
        
        # Since Pranjal did not provide the correct ap_wcs after cutout,
        # here, we calculate the estimated offset.
        # 1. 用大图的 WCS,算出目标在大图里的像素坐标
        x_big, y_big = ap_wcs.wcs_world2pix([[objRA, objDec]], 0)[0]
        
        # 2. 假设这个坐标就是小图的中心像素
        ny, nx = image_data.shape  # (63, 61)
        x_start = x_big - nx / 2.0
        y_start = y_big - ny / 2.0
        
        # 3. 修正 WCS:让 CRPIX 平移到小图自己的坐标系里
        local_wcs = ap_wcs.deepcopy()
        local_wcs.wcs.crpix = ap_wcs.wcs.crpix - np.array([x_start, y_start])
        
        # 4. 之后用 local_wcs 代替 ap_wcs 去做 wcs_world2pix
        x0obj,  y0obj  = local_wcs.wcs_world2pix([[objRA,  objDec ]], 0)[0]  
        x0slit, y0slit = local_wcs.wcs_world2pix([[sets_slitRA[i], sets_slitDec[i]]], 0)[0]  
        ax2.scatter(x0obj,  y0obj,  marker='x', s=360, color='black',   zorder=1)
        ax2.scatter(x0slit, y0slit, marker='o', s=30,  color=colors[i], zorder=2)
        rot_rectangle(ax2, x0slit, y0slit, slitWidth/pixScale, slitLen/pixScale, 
                      (90-slit_LPA)/57.3, colors[i], '-')
        
        image_var1 = image_data # as a Poisson noise
        image_var2 = data_info['image']['var']
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
        ax2.set_title(f'Imaging and slit for spec {specnums[(j*n_sets+i)]}', size=15)
    
    if savefig:
        plt.savefig(f'{pkl_folder}slit_{slit_name}_{special_idx}.png', 
                    dpi=150, bbox_inches='tight')
    else:
        plt.show()
    return


def add_colorbar_by_alpha(fig, ax, color, label='label', bar_low=0, bar_high=1):
    # 创建一个只改变 alpha 的自定义 cmap，转换成 RGB，然后定义渐变
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    cdict = {
        'red':   [(0.0, rgb[0], rgb[0]), (1.0, rgb[0], rgb[0])],
        'green': [(0.0, rgb[1], rgb[1]), (1.0, rgb[1], rgb[1])],
        'blue':  [(0.0, rgb[2], rgb[2]), (1.0, rgb[2], rgb[2])],
        'alpha': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] # 这里控制透明度渐变
    }
    alpha_cmap = mcolors.LinearSegmentedColormap('AlphaMap', cdict)

    # 添加色条，创建归一化映射
    norm = mcolors.Normalize(vmin=bar_low, vmax=bar_high)
    
    # 创建 ScalarMappable 并传给 colorbar
    sm = plt.cm.ScalarMappable(cmap=alpha_cmap, norm=norm)
    sm.set_array([]) # 必须设置一个空数组

    # 在指定的子图 ax 旁边添加色条
    cbar = fig.colorbar(sm, ax=ax, label=label)
    
    return cbar


def make_line_profile_and_exam(data_info, spec_idx, lambda_scale):
    spec_data    = data_info['spec'][spec_idx]['data']
    spec_mask    = data_info['spec'][spec_idx]['mask']
    line_species = data_info['spec'][spec_idx]['meta']['line_species']
    patch_size = 3
    from scipy.ndimage import median_filter
    arr = np.where(spec_mask, spec_data, np.nan)
    smoothed = median_filter(arr, size=patch_size)
    
    from line_width_profile import find_line_sigma, _gaussian_nonzero, _double_gaussian_nonzero
    x0_sigma_amp_1, x0_sigma_amp_2 = find_line_sigma( # note: x0_sigma_amp sigmas are now in Angstrom
        smoothed, line_species, lambda_scale, 
        )
    x0_sigma_amp_1_to_return = x0_sigma_amp_1.copy()
    x0_sigma_amp_2_to_return = x0_sigma_amp_2.copy()
    
    # sigma Angstrom --> pixels for line restoration only
    xsa1_in_pixels = x0_sigma_amp_1.copy() # COPY: OTHERWISE YOU WILL 
    xsa2_in_pixels = x0_sigma_amp_2.copy() #       DESTROY x0_sigma_amp_1s
    xsa1_in_pixels[1:, 1] = xsa1_in_pixels[1:, 1].astype(float) / lambda_scale
    xsa2_in_pixels[1:, 1] = xsa2_in_pixels[1:, 1].astype(float) / lambda_scale
    
    ny, nx = arr.shape
    lw_restored = np.tile([np.arange(nx)], (1, ny)).reshape(ny, nx)
    
    fit_func = _double_gaussian_nonzero if line_species == "O2" else _gaussian_nonzero
    
    for y in range(ny):
        params = xsa1_in_pixels[y+1].astype(float)
        if line_species == "O2":
            params = np.append(params[:-1],  # keep one shared_bkg
                               xsa2_in_pixels[y+1]).astype(float)
            idx_mean1 = np.where(xsa1_in_pixels[0] == 'mean1')[0][0]
            idx_mean2 = np.where(xsa2_in_pixels[0] == 'mean2')[0][0] + xsa1_in_pixels.shape[1]-1
            params[idx_mean2] -= params[idx_mean1] # mean2 -> dmean
        lw_restored[y] = fit_func(lw_restored[y], *params)
    
    # ---------------------------------------------------
    # Plotting
    # ---------------------------------------------------
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,6))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    ax[0,2].remove()
    
    spec_data  = data_info['spec'][spec_idx]['data']
    spec_mask  = data_info['spec'][spec_idx]['mask']
    
    # Note: spec_data[0, 0] was defined at spec's lower-left corner
    im0 = ax[0,0].imshow(
        np.where(spec_mask, spec_data, np.nan), 
        aspect='auto', cmap='viridis', 
        # origin='lower'
        )
    
    vmin = np.nanmin(smoothed)
    vmax = np.nanmax(smoothed)
    im1 = ax[1,0].imshow(
        np.where(spec_mask, smoothed, np.nan), 
        aspect='auto', cmap='viridis', 
        # origin='lower'
        )
    im2 = ax[1,1].imshow(
        lw_restored, 
        aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, 
        # origin='lower'
        )
    im3 = ax[1,2].imshow(
        np.where(spec_mask, smoothed - lw_restored, np.nan), 
        aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax, 
        # origin='lower'
        )
    plt.colorbar(im0, ax=ax[0,0])
    plt.colorbar(im1, ax=ax[1,0])
    plt.colorbar(im2, ax=ax[1,1])
    plt.colorbar(im3, ax=ax[1,2])
    
    x0_sigma_amp_1 = x0_sigma_amp_1[1:].astype(float).copy()
    x0_sigma_amp_2 = x0_sigma_amp_2[1:].astype(float).copy()
    
    # sigma Angstrom --> pixels
    x0_sigma_amp_1[1:, 1] = x0_sigma_amp_1[1:, 1].astype(float) / lambda_scale
    x0_sigma_amp_2[1:, 1] = x0_sigma_amp_2[1:, 1].astype(float) / lambda_scale
    
    max_alpha_1 = np.max(x0_sigma_amp_1)
    max_alpha_2 = np.max(x0_sigma_amp_2)
    max_alpha   = np.max([max_alpha_1, max_alpha_2])
    alphas_1 = x0_sigma_amp_1[:, 2] / max_alpha
    if x0_sigma_amp_2[0,1] != 0:
        alphas_2 = x0_sigma_amp_2[:, 2] / max_alpha
    
    for y in range(len(x0_sigma_amp_1)):
        ax[0,1].errorbar(
            x0_sigma_amp_1[y, 0], 
            y, 
            xerr=x0_sigma_amp_1[y, 1], fmt='o', capsize=5, 
            label='Line 1', color='blue', alpha=alphas_1[y]
            )
        
        if x0_sigma_amp_2[0,1] != 0:
            ax[0,1].errorbar(
                x0_sigma_amp_2[y, 0], 
                y, 
                xerr=x0_sigma_amp_2[y, 1], fmt='x', capsize=5, 
                label='Line 2', color='blue', alpha=alphas_2[y]
                )
    
    add_colorbar_by_alpha(fig, ax[0,1], 'blue', label='amp', 
                          bar_low=0, bar_high=max_alpha)
    
    ax[0,1].invert_yaxis()
    ax[0,1].set_xlim(0, arr.shape[1])
    ax[0,1].set_title('line width fit')
    
    ax[0,0].set_title(f'#{slit_num} spec_idx {spec_idx}: Data')
    ax[1,0].set_title(f'median filtered ({patch_size}x{patch_size})')
    ax[1,1].set_title('line width restored')
    ax[1,2].set_title('Filtered - restored')
    
    return x0_sigma_amp_1_to_return, x0_sigma_amp_2_to_return









if __name__ == '__main__':
    # slit_nums = [2] # 
    slit_num  = 2
    special_idxs = [
        '0.0', 
        '0.3', 
        '0.7', 
        '1.0', 
        '1.4', '1.6', '1.7', 
        '2.1', '2.4', '2.8',
        '3.1' 
        ]
    mock_folder = '008b_vary_thetaint_slitLPA_major/'
    DATA_BASKET = []
    for special_idx in special_idxs:
        # Load
        # data_info = another_load_mock(
        #     pkl_folder='../scripts_Pranjal/', 
        #     slit_num=slit_num)
        filename = f'{mock_folder}/thetaint_slitLPA_{special_idx}.pkl'
        # filename = f'{mock_folder}/slit_002_{special_idx}.pkl'
        with open(filename, "rb") as f:
            data_info = joblib.load(f)
        print(filename)

        DATA_BASKET.append(data_info)
        
        # add key names
        imageshape = data_info['image']['data'].shape

        # calculate beta given the coordinates of the slit w.r.t the galaxy cluster
        # A2261_ctr_RA  = 260.612917 # ( (17)+(22)/60+(26.986)/3600 ) * 15 
        # A2261_ctr_DEC =  32.133889 # ( (32)+( 7)/60+(57.89 )/3600 )
        # theta_cl_rad  = np.arctan2(data_info['galaxy']['Dec'].value - A2261_ctr_DEC, 
        #                            data_info['galaxy']['RA'].value  - A2261_ctr_RA)
        # data_info['galaxy']['beta'] = theta_cl_rad * u.radian

        # Although the slit has RA/DEC, 
        # Pranjal forced the beta to be 0, so we follow his convention here.
        data_info['galaxy']['beta'] = 0 * u.radian

        # Add masks by manual inspection
        data_info['image']['mask'] = np.ones(imageshape, dtype=bool)
        
        # rename key names
        if 'sky_var' in data_info['image'].keys():
            data_info['image']['var' ] = data_info['image']['sky_var']
            data_info['image'].pop('sky_var')
        if 'par_meta' in data_info['image'].keys():
            data_info['image']['meta'] = data_info['image']['par_meta']
            data_info['image'].pop('par_meta')
        
        for spec_idx in range(len(data_info['spec'])):
            # add key names
            specshape = data_info['spec'][spec_idx]['data'].shape
            data_info['spec'][spec_idx]['mask'] = np.ones(specshape, dtype=bool)
            
            # rename key names
            data_info['spec'][spec_idx]['cont'] = data_info['spec'][spec_idx]['cont_model']
            data_info['spec'][spec_idx]['meta'] = data_info['spec'][spec_idx]['par_meta']
            data_info['spec'][spec_idx].pop('cont_model')
            data_info['spec'][spec_idx].pop('par_meta')
            
            # Edit some params in meta_spec
            import astropy.units as u
            lambda_scale = 0.33 * u.Angstrom # in Angstrom/px
            data_info['spec'][spec_idx]['meta'].pop('FWHMresolution')
            data_info['spec'][spec_idx]['meta'].pop('psfFWHM')
            data_info['spec'][spec_idx]['meta']['rhl'] = None
            data_info['spec'][spec_idx]['meta']['dispersion'] = lambda_scale.value
            
            # Pranjal used different definition of slitLPA:
            # In his/my KL paper, slitLPA starts from -x (East) on RA/DEC-plane, counter-clockwise
            # But his fid_param,  slitLPA starts from +x (West) on RA/DEC-plane, clockwise
            # slitPA = data_info['spec'][spec_idx]['meta']['slitLPA']
            # slitPA = 120 * u.deg
            # data_info['spec'][spec_idx]['meta']['slitLPA'] = slitPA
            # data_info['spec'][spec_idx]['meta']['slitWPA'] = 90*u.deg + slitPA
            
            # Edit line profile
            Pname = data_info['spec'][spec_idx]['meta']['line_species']
            if Pname in _mapping.keys():
                Dname = _mapping[Pname]
                data_info['spec'][spec_idx]['meta']['line_species'] = Dname
            x0_sigma_amp_1, x0_sigma_amp_2 = make_line_profile_and_exam(
                data_info, spec_idx, lambda_scale
                )
            data_info['spec'][spec_idx]['meta']['line_profile'] = (
                x0_sigma_amp_1, x0_sigma_amp_2
                )
        
        with open(f'{mock_folder}slit_{slit_num:03d}_{special_idx}.pkl', "wb") as f:
            joblib.dump(data_info, f)
        
        # Plot
        # from plot_temp import make_exam_plots
        make_exam_plots(data_info, f'{slit_num:03d}', 
                        pkl_folder=f'{mock_folder}', 
                        special_idx=special_idx)

    
    fig, axs = plt.subplots(nrows=2, ncols=len(DATA_BASKET), figsize=(2.5*len(DATA_BASKET),6), gridspec_kw={'height_ratios': [1, 2]})
    for i in range(len(DATA_BASKET)):
        data_info = DATA_BASKET[i]
        
        # Special treatment regarding WCS
        # If read ap_wcs:
        #     PC1_1 PC1_2  : -5.555556e-05 0.0 
        #     PC2_1 PC2_2  : 0.0 5.555556e-05
        # PC1_1 < 0: RA decreases as j increases in image_data[i, j].
        #            Thus, image_data: left -> right, vector: -Dec (E->W).
        #            left = East, right = West. [0,0] = SE.
        im_img  = axs[0,i].imshow(
            np.flip(data_info['image']['data'], axis=1), # because PC1_1 < 0 
            aspect='equal', cmap='viridis', 
            origin='lower', # because PC2_2 > 0
            vmax=50
            )
        
        im_spec = axs[1,i].imshow(
            data_info['spec'][0]['data'], 
            aspect='auto',  cmap='viridis', origin='lower'
            )
        plt.colorbar(im_img,  ax=axs[0,i])
        plt.colorbar(im_spec, ax=axs[1,i])
    
        if 'par_meta' in data_info['image'].keys():
            objRA      = data_info['image']['par_meta']['RA']
            objDec     = data_info['image']['par_meta']['Dec']
            ap_wcs     = data_info['image']['par_meta']['ap_wcs']
            pixScale   = data_info['image']['par_meta']['pixScale']
        else:
            objRA      = data_info['image']['meta']['RA']
            objDec     = data_info['image']['meta']['Dec']
            ap_wcs     = data_info['image']['meta']['ap_wcs']
            pixScale   = data_info['image']['meta']['pixScale']
        
        # Since Pranjal did not provide the correct ap_wcs after cutout,
        # here, we calculate the estimated offset.
        # 1. 用大图的 WCS,算出目标在大图里的像素坐标
        x_big, y_big = ap_wcs.wcs_world2pix([[objRA, objDec]], 0)[0]
        
        # 2. 假设这个坐标就是小图的中心像素
        ny, nx = data_info['image']['data'].shape  # (63, 61)
        x_start = x_big - nx / 2.0
        y_start = y_big - ny / 2.0
        
        # 3. 修正 WCS:让 CRPIX 平移到小图自己的坐标系里
        local_wcs = ap_wcs.deepcopy()
        local_wcs.wcs.crpix = ap_wcs.wcs.crpix - np.array([x_start, y_start])
        
        # 4. 之后用 local_wcs 代替 ap_wcs 去做 wcs_world2pix
        if 'par_meta' in data_info['spec'][0].keys():
            slitRA     = data_info['spec'][0]['par_meta']['slitRA'].value
            slitDec    = data_info['spec'][0]['par_meta']['slitDec'].value
            slitLen    = data_info['spec'][0]['par_meta']['slitLen']
            slitWidth  = data_info['spec'][0]['par_meta']['slitWidth']
            slit_LPA   = data_info['spec'][0]['par_meta']['slitLPA'].value
        else:
            slitRA     = data_info['spec'][0]['meta']['slitRA'].value
            slitDec    = data_info['spec'][0]['meta']['slitDec'].value
            slitLen    = data_info['spec'][0]['meta']['slitLen']
            slitWidth  = data_info['spec'][0]['meta']['slitWidth']
            slit_LPA   = data_info['spec'][0]['meta']['slitLPA'].value
        x0obj,  y0obj  = local_wcs.wcs_world2pix([[objRA,  objDec ]], 0)[0]  
        x0slit, y0slit = local_wcs.wcs_world2pix([[slitRA, slitDec]], 0)[0]  
        axs[0,i].scatter(x0obj,  y0obj,  marker='x', s=360, color='black', zorder=1)
        axs[0,i].scatter(x0slit, y0slit, marker='o', s=30,  color='cyan',  zorder=2)
        print('slitLen', slitLen, 'rotation', slit_LPA/57.3)
        rot_rectangle(axs[0,i], x0slit, y0slit, 
                      dx=slitLen/pixScale, 
                      dy=slitWidth/pixScale, 
                      rotation=slit_LPA/57.3, 
                      color='cyan', 
                      ls='-')

        axs[0,i].set_title(r'$\theta_{\rm int}$'+'_slitLPA_'+f'{special_idxs[i]}.pkl')
    
    plt.tight_layout()
    plt.savefig('test.jpg')