import joblib
import numpy as np
# from astropy.io import fits
# from astropy.wcs import WCS
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from tqdm import tqdm
plt.style.use('classic')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})
# darkred = '#C82423'
# oranred = '#F54C22'
# lighred = '#FF6B6B'
# pinkred = '#FF8884'
# ylwgold = '#f5cf8b'
# greelig = '#E6F4EA'
# greemid = '#CCE3D0'
# greehgh = '#b0c64d'
# greeblu = '#5ea79c'
# greehvy = '#B4D9C3'
# greedrk = '#8fc4a5'
# blueblk = '#1D3557'
# blueash = '#457B9D'
# blueook = '#74A9CF'
# bluegre = '#A8DADC'
# bluelim = '#BAEFFF'
# colors = {'B' : blueblk, 'V' : blueook, 
#           'Rc': greemid, 'Ip': greedrk, 
#           'Ic': oranred, 'z' : darkred}


def get_snr(flux, mask, var):
    S = np.sum(flux[mask])
    N = np.sqrt(np.sum(flux[mask] + var[mask]))
    return S/N


def another_load_mock(pkl_folder='mock/', slit_num=95):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as f:
        data_info = joblib.load(f)
    
    # assert data_info['galaxy']['log10_Mstar'] != None, \
    #     "Cannot find corresponing stallar mass M*"  # if not, error
    
    # Recover wcs(galsim.wcs) from ap_wcs
    # ap_wcs  = data_info['image']['par_meta']['ap_wcs']
    # data_info['image']['par_meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)
    return data_info


# def plot_for_each_set(obs_session, arr_slit_list, if_add_set,
#                       ax1, ax2, dic_spec_snr_of_slits):
#     if if_add_set:
#         for key, arr in dic_spec_snr_of_slits.items():
#             if len(arr) != 0:
#                 Mean, Std      = np.mean(arr[arr>0]), np.std( arr[arr>0])
#                 MinSNR, MaxSNR = np.max([1e-3, Mean-3*Std]), np.max([np.max(arr), Mean+3*Std])
#                 bin_edges      = np.linspace(MinSNR, MaxSNR,
#                                              num=41, endpoint=True)
#                 hist, bin_edges = np.histogram(arr[(arr>=MinSNR) & (arr<=MaxSNR)], 
#                                                bins=bin_edges)
#                 ax1.plot(arr_slit_list, arr, drawstyle='steps-mid',
#                          color='black', linewidth=1, label='Counts')
#                 ax1.text(0.5, 0.95, 
#                          r'Mean SNR = {0:.0f} $\pm$ {1:.0f}'.format(Mean, Std), 
#                          fontsize=9, transform=ax1.transAxes, ha='center', va='top')
#                 ax2.fill_betweenx(y=bin_edges[:-1],
#                                   x1=hist, x2=0, step='pre',
#                                   edgecolor='black', facecolor='black', alpha=0.6, linewidth=1)
#         ax1.minorticks_on()
#         ax1.set_xlim(0, 143)
#         ax1.set_ylabel('Spec SNR')
#         ax1.set_title('Set '+obs_session, loc='left', fontsize=14)
#         ax1.grid(linestyle=':', color='black', alpha=0.5, zorder=-1)
#         ax1.legend(prop={'size': 9})
#         ax2.minorticks_on()
#         ax2.set_ylim(bottom=0, top=ax1.get_ylim()[1])
#         ax2.set_ylabel('Spec SNR')
#         ax2.set_xlabel('# of Slits', labelpad=1)
#         ax2.grid(linestyle=':', color='black', alpha=0.5, zorder=-1)
#     return


if __name__ == '__main__':
    fig, ax = plt.subplots(nrows=2, ncols=2, 
                           gridspec_kw={'height_ratios': [2, 1],
                                        'width_ratios':  [4, 1]},
                           figsize=(18, 4), dpi=100) # (width, height)
    plt.subplots_adjust(hspace=0.16, wspace=0.2)
    
    specSNRs_all_slits = np.zeros((1,2)) # slit, SNR
    for slit_num in range(1, 142):
        try:
            data_info = another_load_mock(
                pkl_folder='../scripts/binospec_pkl/', 
                slit_num=slit_num)
            
        except FileNotFoundError:
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                  f'Slit {slit_num} skipped because no PKL found.\n')
            continue
        
        specSNRs = np.array([])
        for spec_idx in range(len(data_info['spec'])):
            flux = data_info['spec'][spec_idx]['data']
            var  = data_info['spec'][spec_idx]['var' ]
            mask = data_info['spec'][spec_idx]['mask']
            SNR  = get_snr(flux, mask, var)
            specSNRs = np.append(specSNRs, SNR)
            specSNRs_all_slits = np.append(specSNRs_all_slits, [[slit_num, SNR]], axis=0)
        
        positive_SNRs = specSNRs[specSNRs >= 0]
        negative_SNRs = specSNRs[specSNRs <  0]
        ax[0,0].scatter(slit_num * np.ones(len(positive_SNRs)), 
                        positive_SNRs, linestyle='-', marker='s', 
                        facecolor='cyan', edgecolor='black', zorder=2)
        ax[1,0].scatter(slit_num * np.ones(len(negative_SNRs)), 
                        negative_SNRs, linestyle='-', marker='s', 
                        facecolor='magenta', edgecolor='black', zorder=2)
        if len(negative_SNRs) != 0:
            ax[0,0].plot(slit_num * np.ones(len(positive_SNRs)+1), 
                         list(positive_SNRs)+[0], linestyle='-', color='cyan', zorder=1)
            ax[1,0].plot(slit_num * np.ones(len(negative_SNRs)+1), 
                         list(negative_SNRs)+[0], linestyle='-', color='magenta', zorder=1)
        else:
            ax[0,0].plot(slit_num * np.ones(len(positive_SNRs)), 
                         list(positive_SNRs), linestyle='-', color='cyan', zorder=1)
    
    # Plot the Position Cost Distribution
    specSNRs_all_slits = np.delete(specSNRs_all_slits, (0), axis=0)
    specSNR_max     = ax[0,0].get_ylim()[1]
    specSNR_min     = ax[1,0].get_ylim()[0]
    slit_num_edges1 = np.linspace(0, specSNR_max, num=int( specSNR_max)+1, endpoint=True)
    slit_num_edges2 = np.linspace(specSNR_min, 0, num=int(-specSNR_min)+1, endpoint=True)
    hist1, bin_edge = np.histogram(specSNRs_all_slits[:,1], bins=slit_num_edges1)
    hist2, bin_edge = np.histogram(specSNRs_all_slits[:,1], bins=slit_num_edges2)
    ax[0,1].fill_betweenx(y=slit_num_edges1[:-1],
                          x1=0, x2=hist1, step='pre',
                          edgecolor='black', facecolor='cyan', linewidth=1, zorder=2)
    ax[1,1].fill_betweenx(y=slit_num_edges2[:-1],
                          x1=0, x2=hist2, step='pre',
                          edgecolor='black', facecolor='magenta', linewidth=1, zorder=2)
    ax[0,1].grid(which='major', linestyle=':', color='black', alpha=0.5, zorder=0)
    ax[1,1].grid(which='major', linestyle=':', color='black', alpha=0.5, zorder=0)
    ax[0,1].set_ylabel('Spec SNR')
    ax[1,1].set_ylabel('Negative\nspec SNR')
    ax[1,1].set_xlabel('# of spectra', labelpad=1)
    
    ax[0,0].minorticks_on()
    ax[1,0].minorticks_on()
    ax[0,0].tick_params(axis='y', right=True, labelright=True)
    ax[1,0].tick_params(axis='y', right=True, labelright=True)
    ax[0,0].set_xlim(left=0, right=slit_num+1)
    ax[1,0].set_xlim(left=0, right=slit_num+1)
    ax[0,0].set_ylim(bottom=0, top=None)
    ax[1,0].set_ylim(top=0, bottom=None)
    ax[0,0].set_ylabel('Spec SNR')
    ax[1,0].set_ylabel('Negative\nspec SNR')
    ax[1,0].set_xlabel('Slit number', labelpad=1)
    ax[0,0].grid(which='major', linestyle=':', color='black', alpha=0.5, zorder=-1)
    ax[1,0].grid(which='major', linestyle=':', color='black', alpha=0.5, zorder=-1)
    
    plt.savefig('specSNR_distri.jpg', dpi=100, bbox_inches='tight')
        