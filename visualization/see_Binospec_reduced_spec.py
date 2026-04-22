import numpy as np
from astropy.io  import fits
# import matplotlib.patches as patches
# import matplotlib.lines   as mlines
import matplotlib.pyplot  as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Inter",
    "font.serif": "Inter",
})


if __name__ == '__main__':
    spec_CCD_root_dir = '../../../RSCH3/UAO-S156-23B-A383/'
    set_A_dir = 'psf_m1/231023/'
    set_B_dir = 'psf_p1/231019/'
    set_B_dir = 'psf/231019/'
    fits_name_all_2d = 'obj_abs_slits_lin.fits'
    fits_name_all_2d_err = 'obj_counts_err_slits_lin.fits'
    
    set_dir   = set_B_dir
    # fits_name = fits_name_all_2d_err
    
    science_hdul = fits.open(spec_CCD_root_dir + set_dir + fits_name_all_2d)
    science_err_hdul = fits.open(spec_CCD_root_dir + set_dir + fits_name_all_2d_err)
    
    i = 5
    
    flux_ar = science_hdul[i].data
    erro_ar = science_err_hdul[i].data
    # science_h1 = science_hdul[i].header
    # science_hdr1 = [{'key':     card.keyword,
    #                  'value':   card.value, 
    #                  'comment': card.comment
    #                  }
    #                 for card in science_h1.cards ]
    
    
    # # CCD PLotting
    # arr = science_ar
    
    # fig = plt.figure(figsize=(arr.shape[1]/500, 
    #                           arr.shape[0]/ 80), 
    #                  dpi=200)
    # gs = fig.add_gridspec(nrows=1, ncols=1)
    # ax1 = fig.add_subplot(gs[0, 0])
    # noise = np.nanstd(arr)
    # imshow1 = ax1.imshow(arr, cmap='bone', aspect='auto',
    #                      vmin=-noise, vmax=3*noise)
    # cbar = fig.colorbar(imshow1, ax=ax1)
    # ax1.minorticks_on() # enable minor ticks
    # ax1.tick_params(right=True, labelright=True)
    # # ax1.tick_params(axis='both', which='major', length=12, width=2.5)
    # # ax1.tick_params(axis='both', which='minor', length=9,  width=1, right=True)
    # ax1.set_title(f'{set_dir+fits_name} (reduced) Slit {i}', loc='left')
    # plt.savefig(f'binospec_reduced_{set_dir.replace("/", "_")}slit_{i:03d}.jpg', bbox_inches='tight')