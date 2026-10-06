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


# ==============================
#    2 = SpecProcessing
# ==============================

class SpecProcessing:
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

        return data_info

            
            
if __name__ == '__main__':
    for slit_num in [5,23,55,72,95,138]: #range(1,143), 3, 4, 90, 7, 8
            SpecProcessing.clear_cont(
                slit_num= 7, save=True, fig_verbose=True)
                # slit_num=69, save=False, fig_verbose=True)
                # slit_num=91, save=False, fig_verbose=True)
            
    print('Done.')
    
    
    
    
    