import joblib
import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})

from read_save_utils import cont, readinfodat
from image_utils     import cutoffimg, Meta_image
from plot import make_exam_plots


# ============================================
#    0 -- Test if changed data_info dict
# ============================================

def check_changed_dict(dic_old, dic_new):
    import pickle
    import hashlib  # Compare by fingerprint
    fp_old = hashlib.md5(pickle.dumps(dic_old)).hexdigest()
    fp_new = hashlib.md5(pickle.dumps(dic_new)).hexdigest()
    if fp_old == fp_new:
        print('Dictionary is not changed.')
        return False
    else:
        print(f'DICTIONARY CHANGED: \nold = {fp_old}, \nnew = {fp_new}.')
        return True


# ==============================
#    1 = ImagingProcessing
# ==============================

class ImagingProcessing:
    def mask_neighbor_star(self, slit_num,
                           mask_x0=0, mask_y0=0, dx=4, dy=3,
                           theta=120/180*np.pi, save=False):
        pkl_folder = '../binospec_pkl/'
        pkl_filenm = f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl'
        with open(pkl_filenm, "rb") as f:
            data_info = joblib.load(f)

        image_data = data_info['image']['data']
        image_mask = data_info['image']['mask']

        ny, nx = image_data.shape

        if not isinstance(mask_x0, list):
            mask_x0 = [mask_x0]
            mask_y0 = [mask_y0]
            dx = [dx]
            dy = [dy]

        # image_masked = image_data
        # image_varmsk = image_var
        masks = np.ones(image_mask.shape, dtype=bool)
        for i in range(len(mask_x0)):
            xs = np.arange(np.min([0,  int(mask_x0[i])]),
                           np.max([nx, int(mask_x0[i]) + nx + 2]))
            ys = np.arange(np.min([0,  int(mask_y0[i])]),
                           np.max([ny, int(mask_y0[i]) + ny + 2]))
            xxs, yys = np.meshgrid(xs, ys)
            mask_pad = np.ones(xxs.shape, dtype=bool)

            r_tranf = ImagingProcessing.Gauss_2d(
                xxs, yys,
                mask_x0[i], mask_y0[i], dx[i], dy[i],
                theta)  # θ parallel dx

            x_origin, y_origin = list(xs).index(0), list(ys).index(0)

            mask_pad[r_tranf > np.exp(-1)] = False

            mask = mask_pad[y_origin: y_origin + ny,
                            x_origin: x_origin + nx]
            masks = masks & mask
            # image_masked = np.where(mask, image_masked, -10)
            # image_varmsk = np.where(mask, image_varmsk, image_varmsk)
        mask = masks
        
        # recover_zeros_from_nan = False
        # if recover_zeros_from_nan:
        #     mask = ~np.isnan(image_masked)
        #     image_masked = np.where(mask, image_masked, -10)
        #     image_varmsk = np.where(mask, image_varmsk, image_varmsk)

        fig = plt.figure(figsize=(6, 6), dpi=100)  # height=6
        gs = fig.add_gridspec(nrows=1, ncols=2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1.imshow(np.where(image_mask, 
                            image_data, np.nan))
        ax2.imshow(np.where((mask & image_mask), 
                            image_data, np.nan))
        ax1.set_title(f'Slit {slit_num}: Before')
        ax2.set_title(f'Slit {slit_num}: After')
        plt.show()

        if save:
            data_info['image']['mask'] = (mask & image_mask)
            # data_info['image']['var'] = image_varmsk

            # Save changed pkl first
            with open(pkl_filenm, "wb") as f:
                joblib.dump(data_info, f)

            # Read to check
            with open(pkl_filenm, "rb") as f:
                real_data_info = joblib.load(f)

            make_exam_plots(real_data_info, f'{slit_num:03d}', pkl_folder=pkl_folder,
                            savefig=True)

        else:
            # Check if changed raw pkl
            with open(pkl_filenm, "rb") as f:
                data_info_raw_0 = joblib.load(f)
            assert check_changed_dict(data_info,
                                      data_info_raw_0) == False, \
                "\033[43m"+'WARNING:'+"\033[0m"+' PKL file changed!'

        return

    @staticmethod
    def Gauss_2d(x, y, x0, y0, dx, dy, theta):
        # x′ =  (x − x0) cosθ + (y − y0) sinθ
        # y′ = −(x − x0) sinθ + (y − y0) cosθ
        # G  = A exp[−​ x′^2 / (2 dx^2) ​− ​y′^2 / (2 dy^2)​]

        a = (np.cos(theta) ** 2) / (2 * dx ** 2) + \
            (np.sin(theta) ** 2) / (2 * dy ** 2)
        b = -(np.sin(2 * theta)) / (4 * dx ** 2) + \
            (np.sin(2 * theta)) / (4 * dy ** 2)
        c = (np.sin(theta) ** 2) / (2 * dx ** 2) + \
            (np.cos(theta) ** 2) / (2 * dy ** 2)
        
        return np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))

        
def reload_raw_imaging(slit_num, image_shape, 
                       arrimg, wcs_loaded, arrwgt, wcs_weight):
    spec1d2dfolder03  = '../../../../RSCH3/UAO-S156-23B-A383/psf_p1/231019/1d2dspecfiles_counts/'
    infdatfilename_temp = 'info.829.{:03d}.{:06d}.dat'
    infdatfilename = infdatfilename_temp.format(slit_num, slit_num+100305)
    dat_dict       = readinfodat(spec1d2dfolder03 + infdatfilename)
    objRA, objDEC  = dat_dict['RA'], dat_dict['DEC']
    
    image_data, RAlim, DEClim = cutoffimg(arrimg, 
                                          wcs_loaded, 
                                          objRA, objDEC, 
                                          image_shape[1], 
                                          image_shape[0])
    image_ivar, RAlim, DEClim = cutoffimg(arrwgt, 
                                          wcs_weight, 
                                          objRA, objDEC, 
                                          image_shape[1], 
                                          image_shape[0])
    
    assert image_data.shape == image_ivar.shape
    
    # It is not needed to "crop" astropy.WCS object.
    return image_data, image_ivar, objRA, objDEC


# ==============================
#    2 = SpecProcessing
# ==============================

class SpecProcessing:
    # def scale_flux(slit_num=109, save=False, uniform_fac=False, 
    #                correct_image_var = False,
    #                arrimg=None, wcs_loaded=None, arrwgt=None, wcs_weight=None):
    #     """
    #     This includes imaging and spec rescaling operations.
    #     """
        
    #     pkl_folder = '../binospec_data_pkl/'
    #     pkl_filenm = f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl'

    #     with open(pkl_filenm, "rb") as f:
    #         data_info_raw = joblib.load(f)

    #     import copy
    #     data_info = copy.deepcopy(data_info_raw)

    #     # Image
    #     # image_data_raw = data_info['image']['data']
    #     # image_var_raw  = data_info['image']['var']
        
    #     # if correct_image_var:
    #     #     image_data, image_ivar, objRA, objDEC = reload_raw_imaging(slit_num)
    #     #     meta_image = Meta_image(image_data, objRA, objDEC)
    #     #     data_info['image']['par_meta'] = meta_image
    #     #     data_info['image']['par_meta'].pop('wcs') # delete it!
    #     #     image_data_raw = image_data
    #     #     image_var_raw = 2 * image_ivar**(-1)
        
    #     # Do:
    #     # flux_high  = np.percentile(image_data_raw, 99)
    #     # scale      = 100 / flux_high
    #     # image_data = image_data_raw * scale
    #     # image_var  = image_var_raw  * scale**2

    #     # Spec
    #     spec_list = data_info['spec']
        
    #     # Before rescaling
    #     # plot_distribution(image_data_raw, spec_list, slit_num)

    #     # Find an uniform rescale factor (spec only)
    #     if uniform_fac:
    #         perc_vals = np.array([
    #             np.percentile(
    #                 spec_list[i]['data'].flatten(), 95
    #             ) for i in range(len(spec_list))
    #         ])
    #         perc_val = np.max(perc_vals)

    #     # Do:
    #     for i in range(len(spec_list)):
    #         spec_obs = spec_list[i]['data']
    #         spec_var = spec_list[i]['var']
    #         spec_con = spec_list[i]['cont_model']

    #         perc_val = np.percentile(spec_obs.flatten(), 95)
    #         spec_obs *= (50 / perc_val)
    #         spec_var *= (50 / perc_val)**2
    #         spec_con *= (50 / perc_val)
    #         spec_list[i]['data'] = spec_obs
    #         spec_list[i]['var'] = spec_var
    #         spec_list[i]['cont_model'] = spec_con

    #     # After rescaling
    #     # plot_distribution(image_data, spec_list, slit_num)

    #     # data_info['image']['data'] = image_data
    #     # data_info['image']['var'] = image_var
    #     data_info['spec'] = spec_list

    #     if save:
    #         # Save pkl
    #         with open(pkl_filenm, "wb") as f:
    #             joblib.dump(data_info, f)

    #         # Read to check
    #         with open(pkl_filenm, "rb") as f:
    #             real_data_info = joblib.load(f)

    #         make_exam_plots(real_data_info, f'{slit_num:03d}', pkl_folder=pkl_folder,
    #                         savefig=True)

    #     else:
    #         make_exam_plots(data_info, f'{slit_num:03d}', pkl_folder=pkl_folder,
    #                         savefig=False)

    #     return data_info

    # def plot_distribution(image_data, spec_data_list, slit_num):
    #     # Plot image
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    #     ax.hist(
    #         image_data.flatten(),
    #         bins=10, color='green', edgecolor='black')

    #     ax.axvline(x=np.std(image_data.flatten()))
    #     ax.text(np.std(image_data.flatten()), 50,
    #             f'Noise = {np.std(image_data.flatten()):.2f}',
    #             )

    #     ax.axvline(x=np.median(image_data.flatten()), color='pink')
    #     ax.text(np.median(image_data.flatten()), 75,
    #             f'Median = {np.median(image_data.flatten()):.2f}',
    #             )

    #     ax.axvline(x=np.percentile(image_data.flatten(), 95), color='purple')
    #     ax.text(np.percentile(image_data.flatten(), 95), 100,
    #             f'95-Percentile = {np.percentile(image_data.flatten(), 95):.2f}',
    #             ha='right')

    #     ax.text(0.5, 0.5, f'{slit_num}', ha='center', va='center',
    #             alpha=0.2, size=60, transform=ax.transAxes)

    #     plt.show()

    #     # Plot spec
    #     n_spec = len(spec_data_list)
    #     fig, axs = plt.subplots(nrows=n_spec//3, ncols=3, figsize=(8, 8))
    #     for i in range(n_spec):
    #         spec_obs = spec_data_list[i]['data']

    #         axs[0, 0].text(0.5, 0.5, f'{slit_num}', ha='center', va='center',
    #                        color='gray', size=60, transform=axs[0, 0].transAxes)

    #         axs[i//3, i % 3].hist(
    #             spec_obs.flatten(),
    #             bins=10, color='skyblue', edgecolor='black')

    #         noise = np.std(spec_obs.flatten())
    #         axs[i//3, i % 3].axvline(x=noise)
    #         axs[i//3, i % 3].text(noise, 50, f'Noise = {noise:.2f}')

    #         median = np.median(spec_obs.flatten())
    #         axs[i//3, i % 3].axvline(x=median, color='green')
    #         axs[i//3, i % 3].text(median, 75, f'Med. = {median:.2f}')

    #         percen = np.percentile(spec_obs.flatten(), 95)
    #         axs[i//3, i % 3].axvline(x=percen, color='purple')
    #         if percen >= 100:
    #             axs[i//3, i % 3].text(percen, 100, f'95-Perc. = {percen:.2f}',
    #                                   ha='right', color='red')
    #         else:
    #             axs[i//3, i % 3].text(percen, 100, f'95-Perc. = {percen:.2f}',
    #                                   ha='right')

    #     plt.show()
    #     return

    def clear_cont(slit_num=7, save=False, fig_verbose=True):
        pkl_folder = '../binospec_data_pkl/'
        pkl_filenm = f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl'

        with open(pkl_filenm, "rb") as f:
            data_info_raw = joblib.load(f)

        data_info = {}
        for key, dic in data_info_raw.items():
            if key == 'spec':
                data_info['spec'] = []
                for i in range(len(data_info_raw['spec'])):
                    spec_info = data_info_raw['spec'][i]
                    spec_cont = cont(spec_info['data'], margin=5)  # px

                    if fig_verbose:
                        fig, axs = plt.subplots(3, 1)  # nrow, ncol
                        ax1, ax2, ax3 = axs

                        vmax = 5 * np.nanstd(spec_info['data'])

                        im1 = ax1.imshow(spec_info['data'],
                                         vmax=vmax)
                        im2 = ax2.imshow(spec_cont,
                                         vmax=vmax)
                        im3 = ax3.imshow(spec_info['data'] - spec_cont,
                                         vmax=vmax)

                        fig.colorbar(im1, ax=ax1)
                        fig.colorbar(im2, ax=ax2)
                        fig.colorbar(im3, ax=ax3)
                        plt.show()

                    spec_data_set0 = {
                        'data':       np.where(
                            (spec_info['data'] - spec_cont > 0),
                            spec_info['data'] - spec_cont, 0),
                        'var':        spec_info['var'],
                        'cont_model': np.zeros(spec_cont.shape),
                        'par_meta':   spec_info['par_meta'],
                    }
                    data_info['spec'].append(spec_data_set0)
            else:
                data_info[key] = dic

        if save:
            # Save changed pkl first
            with open(pkl_filenm, "wb") as f:
                joblib.dump(data_info, f)

            # Read to check
            with open(pkl_filenm, "rb") as f:
                real_data_info = joblib.load(f)

            make_exam_plots(real_data_info, f'{slit_num:03d}', pkl_folder=pkl_folder,
                            savefig=True)

        else:
            # Check if changed raw pkl
            with open(pkl_filenm, "rb") as f:
                data_info_raw_0 = joblib.load(f)
            assert check_changed_dict(data_info_raw,
                                      data_info_raw_0) == False, \
                "\033[43m"+'WARNING:'+"\033[0m"+' PKL file changed!'

            make_exam_plots(data_info, f'{slit_num:03d}', pkl_folder=pkl_folder,
                            savefig=False)

        return data_info

    def add_M_stellar_and_r_hl_disk(slit_num, z_spec, 
                                    correct_Slit_LPA=False, save=False):
        pkl_folder = '../binospec_data_pkl/'
        pkl_filenm = f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl'
        Ms_folder  = '../../../bagpipes-KL/'
        spec1d2dfolder03  = '../../../../RSCH3/UAO-S156-23B-A383/psf_p1/231019/1d2dspecfiles_counts/'
        infdatfilename_temp = 'info.829.{:03d}.{:06d}.dat'
        
        with open(pkl_filenm, "rb") as f:
            data_info_raw = joblib.load(f)

        data_info = {}
        for key, dic in data_info_raw.items():
            if key == 'galaxy':
                infdatfilename = infdatfilename_temp.format(slit_num, slit_num+100305)
                dat_dict       = readinfodat(spec1d2dfolder03 + infdatfilename)
                objRA, objDEC  = dat_dict['RA'], dat_dict['DEC']
                RA_obj     = objRA  * u.deg
                Dec_obj    = objDEC * u.deg
                redshift   = z_spec
                data_info['galaxy'] = {
                    'redshift': redshift,
                    'RA':       RA_obj,
                    'Dec':      Dec_obj,
                    'beta':     0*u.deg, # Just give any number, only for g_t
                    'log10_Mstar':     None, 
                    'log10_Mstar_err': None,
                    }
                
                # Read M_stellar
                dg = pd.read_csv(f"{Ms_folder}Mstellar_table.txt",
                                 sep=r"\s+", header=0,
                                 names=["slit", "median", "std", "err_lo", "mean", "err_hi"])
                dg = dg.drop_duplicates(subset="slit", keep="last")
                Ms = dg.sort_values(by='slit').to_numpy()
                for row in Ms:
                    if row[0] == slit_num:
                        data_info['galaxy']['log10_Mstar'] = row[1]
                        data_info['galaxy']['log10_Mstar_err'] = row[2]
                        print(f'Updated: log10_Mstar = {row[1]}, err = {row[2]}')
                        break
            
            elif key == 'spec':
                data_info['spec'] = data_info_raw['spec'].copy()
                
                # Read r_hl_disk that was finished in pkl
                df = pd.read_csv(f"{pkl_folder}r_hl_table.txt",
                                 sep=r"\s+", header=None, names=["slit", "rhl"])
                df = df.drop_duplicates(subset="slit", keep="last")
                rr = df.sort_values(by='slit').to_numpy()
                for row in rr:
                    if row[0] == slit_num:
                        r_hl_disk = row[1]
                        for i in range(len(data_info_raw['spec'])):
                            data_info_raw['spec'][i]['par_meta']['rhl'] = r_hl_disk
                        print(f'Updated: r_hl_disk = {r_hl_disk}')
                        break
                
                # Correct Slit_LPA from 25 to 65 in meta_utils.py
                if correct_Slit_LPA:
                    for i in range(len(data_info_raw['spec'])):
                        old_L = data_info_raw['spec'][i]['par_meta']['slitLPA']
                        if old_L == 25*u.deg:
                            data_info_raw['spec'][i]['par_meta']['slitLPA'] = 90*u.deg - old_L
                            data_info_raw['spec'][i]['par_meta']['slitWPA'] =180*u.deg - old_L
            
            else:
                data_info[key] = dic

        if save:
            # Save changed pkl first
            with open(pkl_filenm, "wb") as f:
                joblib.dump(data_info, f)

            # Read to check
            with open(pkl_filenm, "rb") as f:
                real_data_info = joblib.load(f)

            make_exam_plots(real_data_info, f'{slit_num:03d}', pkl_folder=pkl_folder,
                            savefig=True)

        else:
            # Check if changed raw pkl
            # with open(pkl_filenm, "rb") as f:
            #     data_info_raw_0 = joblib.load(f)
            # assert check_changed_dict(data_info_raw,
            #                           data_info_raw_0) == False, \
            #     "\033[43m"+'WARNING:'+"\033[0m"+' PKL file changed!'

            make_exam_plots(data_info, f'{slit_num:03d}', pkl_folder=pkl_folder,
                            savefig=False)
            
        return data_info


class RedoImaging:
    @staticmethod
    def redo_img_data():
        from astropy.io  import fits
        from astropy.wcs import WCS
        import pandas as pd
        
        hscimagefolder01 = '../../../../RSCH3/HSC_img_A383/'
        hsc_filename = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz.fits'
        hsc_wght_fnm = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz-weight.fits'
        
        science_hdul = fits.open(hscimagefolder01 + hsc_filename)
        weights_hdul = fits.open(hscimagefolder01 + hsc_wght_fnm)
        
        # Pixel data
        science_raw = science_hdul[0].data
        weights_raw = weights_hdul[0].data
        
        # WCS mapping
        science_wcs = WCS(hscimagefolder01 + hsc_filename)
        weights_wcs = WCS(hscimagefolder01 + hsc_wght_fnm)
        
        science_hdr0_raw = science_hdul[0].header
        weights_hdr0_raw = weights_hdul[0].header
        
        science_hdr0 = [{'key':     card.keyword,
                         'value':   card.value, 
                         'comment': card.comment
                         }
                        for card in science_hdr0_raw.cards ]
        weights_hdr0 = [{'key':     card.keyword,
                         'value':   card.value, 
                         'comment': card.comment
                         }
                        for card in weights_hdr0_raw.cards ]
        
        # Generate PD table (for checking entire FITS header)
        df_science = pd.DataFrame(science_hdr0)
        df_weights = pd.DataFrame(weights_hdr0)
        
        # Ignore row index
        df_science = df_science.set_index('key')
        df_weights = df_weights.set_index('key')
        
        sci_GAIN = df_science.loc['GAIN', 'value']
        wei_GAIN = df_weights.loc['GAIN', 'value']
        
        # Science = science_raw (ADU/s)     * sci_GAIN    (e/ADU)
        # Weights = weights_raw (s^2/ADU^2) * wei_GAIN^-2 (e/ADU)^-2
        science_data = science_raw * sci_GAIN
        weights_data = weights_raw * wei_GAIN**(-2)
        
        return {'science_data': science_data, 
                'weights_data': weights_data, 
                'science_wcs': science_wcs, 
                'weights_wcs': weights_wcs,
                }
    
    @staticmethod
    def run_redo_img_data(slit_num, 
                          science_data, science_wcs, 
                          weights_data, weights_wcs,
                          save=False):
        pkl_folder = '../binospec_data_pkl/'
        pkl_filenm = f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl'
    
        with open(pkl_filenm, "rb") as f:
            data_info_raw = joblib.load(f)
    
        import copy
        data_info = copy.deepcopy(data_info_raw)
    
        # Image
        image_data_shape = data_info['image']['data'].shape
        
        image_data, image_ivar, objRA, objDEC = reload_raw_imaging(
            slit_num, image_data_shape,
            science_data, science_wcs, 
            weights_data, weights_wcs) # cut off
        image_var = image_ivar**(-1)
        
        # Mask flux < 0
        # bad_f_mask = image_data<0
        # image_data = np.where(bad_f_mask, 0, image_data)
        # image_var  = np.where(bad_f_mask, image_var*2, image_var)
        
        # Treat science value as Poisson var (same value as mean) 
        # and Apply to var
        image_var += image_data
        
        meta_image = Meta_image(image_data, objRA, objDEC)
        data_info['image']['par_meta'] = meta_image
        data_info['image']['par_meta'].pop('wcs') # delete it!
        
        data_info['image']['data'] = image_data
        data_info['image']['var']  = image_var
        
        # Verify slit len
        for i in range(len(data_info['spec'])):
            spec_pix_scale = 0.24 # 0.24 arcsec/pix by Binospec
            spec_shape     = data_info['spec'][i]['par_meta']['ngrid']
            slit_len       = spec_shape[0] * spec_pix_scale
            data_info['spec'][i]['par_meta']['slitLen'] = slit_len
        
        if save:
            # Save pkl
            with open(pkl_filenm, "wb") as f:
                joblib.dump(data_info, f)
    
            # Read to check
            with open(pkl_filenm, "rb") as f:
                real_data_info = joblib.load(f)
    
            make_exam_plots(real_data_info, f'{slit_num:03d}', pkl_folder=pkl_folder,
                            savefig=True)
    
        else:
            make_exam_plots(data_info, f'{slit_num:03d}', pkl_folder=pkl_folder,
                            savefig=False)
    
        return data_info
            
            
if __name__ == '__main__':
    Redo_Imaging = RedoImaging()
    imaging      = Redo_Imaging.redo_img_data()
    science_data = imaging['science_data']
    weights_data = imaging['weights_data']
    science_wcs  = imaging['science_wcs' ]
    weights_wcs  = imaging['weights_wcs' ]
    
    
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    # axs[0].hist(
    #     science_data[(science_data < np.percentile(science_data, 99.7)) & 
    #                  (science_data > np.percentile(science_data,  0.3))],
    #     bins=30, 
    #     color='skyblue', edgecolor='black')
    # axs[1].hist(
    #     weights_data[weights_data > np.percentile(weights_data, 4)]**(-0.5),
    #     bins=30, 
    #     color='skyblue', edgecolor='black')
    
    # axs[0].set_yscale('log')
    # axs[1].set_yscale('log')
    
    # =========================================================
    # Do this step individually after iterating all slits
    # =========================================================
    imaging = ImagingProcessing()
    kwds_list = [
        {'slit_num':8, 'mask_x0':11, 'mask_y0':18, 'dx':2, 'dy':2.5},
        {'slit_num':13, 'mask_x0':[3,23], 'mask_y0':[0,23], 'dx':[8,9], 'dy':[5,9]},
        {'slit_num':14, 'mask_x0':0, 'mask_y0':18, 'dx':3, 'dy':3},
        {'slit_num':27, 'mask_x0':22, 'mask_y0':4, 'dx':3, 'dy':3},
        {'slit_num':28, 'mask_x0':0, 'mask_y0':0, 'dx':4, 'dy':4},
        {'slit_num':35, 'mask_x0':8, 'mask_y0':20, 'dx':3, 'dy':1}, 
        {'slit_num':42, 'mask_x0':3, 'mask_y0':16, 'dx':5, 'dy':4},
        {'slit_num':55, 'mask_x0':20, 'mask_y0':23, 'dx':9, 'dy':7},
        {'slit_num':58, 'mask_x0':13, 'mask_y0':4, 'dx':3, 'dy':3},
        {'slit_num':64, 'mask_x0':3, 'mask_y0':3, 'dx':3, 'dy':3}, 
        {'slit_num':67, 'mask_x0':0, 'mask_y0':8, 'dx':3, 'dy':4},
        {'slit_num':69, 'mask_x0':0, 'mask_y0':19, 'dx':3, 'dy':3},
        {'slit_num':70, 'mask_x0':21, 'mask_y0':5, 'dx':3, 'dy':6},
        {'slit_num':75, 'mask_x0':25, 'mask_y0':12, 'dx':8, 'dy':9},
        {'slit_num':91, 'mask_x0':4.5, 'mask_y0':5, 'dx':1.5, 'dy':1.5},
        {'slit_num':99, 'mask_x0':15, 'mask_y0':2, 'dx':3, 'dy':3},
        {'slit_num':128, 'mask_x0':[10,22], 'mask_y0':[0,13], 'dx':[5,3], 'dy':[2,4]},
        {'slit_num':132, 'mask_x0':[4,12,21], 'mask_y0':[0,18,16], 'dx':[3,2,2], 'dy':[3,2,2]},
        {'slit_num':133, 'mask_x0':5, 'mask_y0':20, 'dx':5, 'dy':4},
        {'slit_num':139, 'mask_x0':12, 'mask_y0':23, 'dx':6, 'dy':4},
        ]
    for kwds in kwds_list:
        imaging.mask_neighbor_star(
            kwds['slit_num'], kwds['mask_x0'], kwds['mask_y0'], kwds['dx'], kwds['dy'],
            theta=0/180*np.pi, save=True # False  
        )
    # =========================================================
    
    # for slit_num in [5,23,55,72,95,138]: #range(1,143), 3, 4, 90, 7, 8
    #     print(f'\nWorking on Slit {slit_num}...')
        
    #     z_table_filename = "../redshift_table.xlsx"
    #     df     = pd.read_excel(z_table_filename, header=None, engine='openpyxl')
    #     array  = df.to_numpy()
    #     ztable = array[1:, 0:2] # remove 1st line header, keep 1 remaining
    #     z_spec = ztable[slit_num, 1]
        
    #     if np.isnan(z_spec)==True:
    #         continue
        
    #     elif slit_num in [12, 34, 106, 114, 119]:
    #         continue
        
    #     elif np.isnan(z_spec)==False:
    #         # ==============================
    #         #    1. Imaging Processing
    #         # ==============================
    #         # Redo_Imaging.run_redo_img_data(
    #         #     slit_num, 
    #         #     science_data, science_wcs, 
    #         #     weights_data, weights_wcs,
    #         #     save=True) #  False
            
    #         # ==============================
    #         #    2. Spec Processing
    #         # ==============================
        
    #         # SpecProcessing.clear_cont(
    #         #     slit_num= 7, save=True, fig_verbose=True)
    #         #     slit_num=69, save=False, fig_verbose=True)
    #         #     slit_num=91, save=False, fig_verbose=True)
            
    #         # hscimagefolder01 = '../../../../RSCH3/HSC_img_A383/'
    #         # hsc_filename = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz.fits'
    #         # hsc_wght_fnm = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz-weight.fits'

    #         # from read_save_utils import read_hsc_img_wcs
    #         # arrimg, wcs_loaded = read_hsc_img_wcs(hscimagefolder01 + hsc_filename)
    #         # arrwgt, wcs_weight = read_hsc_img_wcs(hscimagefolder01 + hsc_wght_fnm)
            
    #         # test = SpecProcessing.scale_flux(
    #         #     slit_num, save=True, 
    #         #     correct_image_var = True,
    #         #     arrimg=arrimg, wcs_loaded=wcs_loaded, 
    #         #     arrwgt=arrwgt, wcs_weight=wcs_weight
    #         #     )
    #         # print('slitLPA:', test['spec'][0]['par_meta']['slitLPA'].value)
            
    #         # test = SpecProcessing.add_M_stellar_and_r_hl_disk(
    #         #     slit_num, z_spec, correct_Slit_LPA=False, save=True) # False 
    #         # print('slitLPA:', test['spec'][0]['par_meta']['slitLPA'].value)
            
    #         print(f'\nSlit {slit_num} processed.')
            
    print('Done.')
    
    
    
    
    