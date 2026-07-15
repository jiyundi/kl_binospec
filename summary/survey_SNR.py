import joblib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})
# plt.style.use('dark_background')
plt.style.use('seaborn-whitegrid')

def get_snr(flux, mask, var):
    S = np.sum(flux[mask])
    N = np.sqrt(np.sum(flux[mask] + var[mask]))
    return S/N


def get_Fabian_snr(flux, mask, var=None, verbose=False):
    left, right = flux.shape[1]//4, flux.shape[1] - flux.shape[1]//4
    up,   down  = flux.shape[0]//4, flux.shape[0] - flux.shape[0]//4
    
    mask_var = np.ones(flux.shape, dtype=bool)
    mask_var[up:down, left:right] = False
    
    if np.sum(mask_var & mask) <= 4:
        print( "\033[43m" + 'WARNING:' + "\033[0m " + 'SNR & sky mask skipped all pixels. Gauss background noise may be not robust.')
        gaussian_background_noise = np.nanstd(flux[mask_var       ])
    else:
        gaussian_background_noise = np.nanstd(flux[mask_var & mask])
    
    del var
    var = np.ones(flux.shape) * gaussian_background_noise**2
    
    S = np.sum(flux[mask])
    N = np.sqrt(np.sum(flux[mask] + var[mask]))
    
    if S/N < 0:
        print( "\033[43m" + 'WARNING:' + "\033[0m " + 
              f'Negative SNR found: {S/N:.0f}')
    
    if verbose: 
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6,3))
        im1 = ax1.imshow(np.where(mask_var & mask, flux, np.nan), 
                         aspect='auto', cmap='viridis')
        im2 = ax2.imshow(np.where(mask_var & mask, var, np.nan), 
                         aspect='auto', cmap='viridis')
        plt.colorbar(im1, ax=ax1)
        plt.colorbar(im2, ax=ax2)
        plt.show()
        print(f'SNR = {S/N:.1f}')
    
    return S/N


def another_load_mock(pkl_folder='mock/', slit_num=95):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as f:
        data_info = joblib.load(f)
    return data_info


if __name__ == '__main__':
    slitnums = []
    spec_idxs= []
    specs    = []
    variances= []
    masks    = []
    SNRs     = []
    lines    = []
    
    # func_SNR = get_snr
    func_SNR = get_Fabian_snr
    print('Using Fabian\'s method to get SNRs...')
    
    for slit_num in range(1, 143):
        try:
            data_info = another_load_mock(
                pkl_folder='../scripts/binospec_pkl/', 
                slit_num=slit_num)
            
        except FileNotFoundError:
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                  f'Slit {slit_num} skipped because no PKL found.')
            continue
        
        for spec_idx in range(len(data_info['spec'])):
            flux = data_info['spec'][spec_idx]['data']
            var  = data_info['spec'][spec_idx]['var' ]
            mask = data_info['spec'][spec_idx]['mask']
            cont = data_info['spec'][spec_idx]['cont']
            
            SNR  = func_SNR(flux, mask, var+cont)
            
            slitnums.append(slit_num)
            spec_idxs.append(spec_idx)
            specs.append(flux)
            variances.append(var+cont)
            masks.append(mask)
            SNRs.append(SNR)
            lines.append(data_info['spec'][spec_idx]['meta']['line_species'])
    
    # Check if specs are overlapped with any sky line masks
    from survey_not_overlaps import find_not_overlaps
    slitnums_, spec_idxs_, not_overlaps = find_not_overlaps()
    
    # Python's SQL JOINT()
    import pandas as pd
    df1 = pd.DataFrame({'slitnums': slitnums, 'spec_idxs': spec_idxs,
                        'specs': specs, 'variances': variances,
                        'masks': masks, 'SNRs': SNRs, 'lines': lines})
    df2 = pd.DataFrame({'slitnums': slitnums_, 'spec_idxs': spec_idxs_, 
                        'not_overlaps': not_overlaps})
    result = pd.merge(df1, df2, on=['slitnums', 'spec_idxs'], 
                      how='inner')
    slitnums  = list(result['slitnums'])
    spec_idxs = list(result['spec_idxs'])
    specs     = list(result['specs'])
    variances = list(result['variances'])
    masks     = list(result['masks'])
    SNRs      = list(result['SNRs'])
    lines     = list(result['lines'])
    not_overlaps = list(result['not_overlaps'])
    
    # Sort by...
    sorted_SNRs, \
    sorted_slitnums, sorted_spec_idxs, sorted_specs, sorted_variances, \
    sorted_masks, sorted_lines, sorted_not_overlaps = zip(*sorted(
        zip(SNRs, # <-- sort by
            slitnums, spec_idxs, specs, variances, masks, lines, not_overlaps), 
        reverse=True
        ))
    
    from survey_secure_Ms import find_secure_Ms
    s, secures = find_secure_Ms()
    secure_zs  = np.array(s)[secures]
    
    good_list = [117, 39, 135, 42, 132, 96, 118, 141, 
                 95, 36, 29, 88, 91, 70, 43, 50, 84, 75, 79, 49, 7, 8, 
                 # 57, 58
                 ]
    lines_to_delete = []
    # lines_to_delete = [[ 57, 'O3b'], # Slit removed
    #                    [135,  'Hb'], # Slit removed
    #                    [135, 'O3b'], # Slit removed
    #                    [ 42,  'Hb'], # Slit removed
    #                    [ 42, 'O3b'], # Slit removed
    #                    [132, 'O3a'], 
    #                    [ 58, 'O3b'], # Slit removed
    #                    [141,  'Hb'], # Slit removed
    #                    [141, 'O3b'], # Slit removed
    #                    [ 88,  'Hb'], 
    #                    [ 43,  'Hb'], 
    #                    [ 49,  'Hb'], # Slit removed
    #                    [ 79, 'O3b'], # Slit removed
    #                    ]
    
    # =========== 15 x 40 slit numbers plot (with GOOD shears) =========
    n_rows, n_cols = 15, 40
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, 
                           figsize=(n_cols, n_rows), dpi=100)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    for i in range(len(sorted_SNRs)):
        i_row, i_col = i//n_cols, i%n_cols
        
        if int(sorted_slitnums[i]) in good_list:
            if [int(sorted_slitnums[i]), sorted_lines[i]] in lines_to_delete:
                ax[i_row, i_col].text(0.5, 0.5, f'{int(sorted_slitnums[i])}', 
                                      ha='center', va='center', color='red', size=n_rows*2, 
                                      transform=ax[i_row, i_col].transAxes)
            else:
                ax[i_row, i_col].text(0.5, 0.5, f'{int(sorted_slitnums[i])}', 
                                      ha='center', va='center', color='blue', size=n_rows*2, 
                                      transform=ax[i_row, i_col].transAxes)
        elif int(sorted_slitnums[i]) in secure_zs:
            ax[i_row, i_col].imshow(np.zeros((4, 4)),
                                    cmap='gray', aspect='auto', vmin=0, vmax=1)
            ax[i_row, i_col].text(0.5, 0.5, f'{int(sorted_slitnums[i])}', 
                                  ha='center', va='center', color='white', size=n_rows*2, 
                                  transform=ax[i_row, i_col].transAxes)
    
    for i_row in range(n_rows):
        for i_col in range(n_cols):
            ax[i_row, i_col].grid(False)
            ax[i_row, i_col].tick_params(
                axis='both',          # Apply to both x and y axes
                which='both',         # Apply to both major and minor ticks
                bottom=False,         # Hide ticks on the bottom edge
                top=False,            # Hide ticks on the top edge
                left=False,           # Hide ticks on the left edge
                right=False,          # Hide ticks on the right edge
                labelbottom=False,    # Hide text labels on the bottom edge
                labelleft=False       # Hide text labels on the left edge
            )
    
    # ====== 15 x 40 slit numbers plot (without sky masks overlapped) ======
    # n_rows, n_cols = 15, 40
    # fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, 
    #                        figsize=(n_cols, n_rows), dpi=100)
    # plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    # for i in range(len(sorted_SNRs)):
    #     i_row, i_col = i//n_cols, i%n_cols
        
    #     if int(sorted_slitnums[i]) in good_list:
            
    #         if sorted_not_overlaps[i] == False:
    #             ax[i_row, i_col].text(0.5, 0.5, f'{int(sorted_slitnums[i])}', 
    #                                   ha='center', va='center', color='red', size=n_rows*2, 
    #                                   transform=ax[i_row, i_col].transAxes)
    #         else:
    #             ax[i_row, i_col].text(0.5, 0.5, f'{int(sorted_slitnums[i])}', 
    #                                   ha='center', va='center', color='blue', size=n_rows*2, 
    #                                   transform=ax[i_row, i_col].transAxes)
        
    #     elif int(sorted_slitnums[i]) in secure_zs:
    #         ax[i_row, i_col].imshow(np.zeros((4, 4)), # black
    #                                 cmap='gray', aspect='auto', vmin=0, vmax=1)
    #         ax[i_row, i_col].text(0.5, 0.5, f'{int(sorted_slitnums[i])}', 
    #                               ha='center', va='center', color='white', size=n_rows*2, 
    #                               transform=ax[i_row, i_col].transAxes)
    
    # for i_row in range(n_rows):
    #     for i_col in range(n_cols):
    #         ax[i_row, i_col].grid(False)
    #         ax[i_row, i_col].tick_params(
    #             axis='both',          # Apply to both x and y axes
    #             which='both',         # Apply to both major and minor ticks
    #             bottom=False,         # Hide ticks on the bottom edge
    #             top=False,            # Hide ticks on the top edge
    #             left=False,           # Hide ticks on the left edge
    #             right=False,          # Hide ticks on the right edge
    #             labelbottom=False,    # Hide text labels on the bottom edge
    #             labelleft=False       # Hide text labels on the left edge
    #         )
    
    # for slitnum in good_list:
    #     this_slit = result.loc[result['slitnums'] == slitnum]
    #     goods_or_bads = np.array(this_slit['not_overlaps'])
        
    #     # If all are bads...
    #     all_specs_rejected   = ~np.any(goods_or_bads) 
        
    #     # If >= 50% are good, < 50% specs are bad...
    #     major_specs_accepted = (np.sum(goods_or_bads) >= 0.5 * len(goods_or_bads))
        
    #     if all_specs_rejected:
    #         print(f'Slit {slitnum:03d}:  NO  spec   is not overlapped by sky line masks.')
    #     elif not major_specs_accepted:
    #         print(f'Slit {slitnum:03d}: <50% specs are not overlapped by sky line masks.')
        
    # =============== 15 x 40 slits specs ==================
    n_rows, n_cols = 15, 40
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, 
                           figsize=(n_cols, n_rows), dpi=100)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    for i in range(len(sorted_SNRs)):
        i_row, i_col = i//n_cols, i%n_cols
        noise = np.nanstd(sorted_specs[i][sorted_masks[i]])
        ax[i_row, i_col].imshow(np.where(sorted_masks[i], 
                                         sorted_specs[i], 
                                         np.nan),
                                cmap='viridis', aspect='auto', origin='lower')
    
        ax[i_row, i_col].text(0.9, 0, f'{int(sorted_SNRs[i])}', 
                              ha='right', color='white', size=n_rows*1.5, 
                              transform=ax[i_row, i_col].transAxes)
    
    for i_row in range(n_rows):
        for i_col in range(n_cols):
            ax[i_row, i_col].grid(False)
            ax[i_row, i_col].tick_params(
                axis='both',          # Apply to both x and y axes
                which='both',         # Apply to both major and minor ticks
                bottom=False,         # Hide ticks on the bottom edge
                top=False,            # Hide ticks on the top edge
                left=False,           # Hide ticks on the left edge
                right=False,          # Hide ticks on the right edge
                labelbottom=False,    # Hide text labels on the bottom edge
                labelleft=False       # Hide text labels on the left edge
            )

        