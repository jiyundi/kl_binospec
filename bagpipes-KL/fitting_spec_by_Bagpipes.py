import numpy  as np 
import pandas as pd
import bagpipes as pipes
import os
import argparse

from src.load_photo     import load_photo
from src.load_spec      import load_spec
from src.convert_s_to_p import convert_spec_to_eq_photo
from src.load_config    import load_config
from src.safe_plot import setup; setup() # must before plt

def load_spec_photo(slit_num_str, 
                    spectrum_exists, photometry_exists):
    if   spectrum_exists and photometry_exists==False:
            spectrum = load_spec(slit_num_str)
            return spectrum
    elif spectrum_exists==False and photometry_exists:
            photometry = load_photo(slit_num_str)
            return photometry
    elif spectrum_exists and photometry_exists:
            spectrum = load_spec(slit_num_str)
            photometry = load_photo(slit_num_str)
            return spectrum, photometry
    else:
            return None



if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--slitID', default=5, type=int)
    slit_num = parser.parse_args().slitID
    
    photometry_exists = True
    spectrum_exists   = False
    convert_spec_to_photo = True
            
    # From: https://sdss-mangadap.readthedocs.io/en/latest/emissionlines.html
    all_emilines_supported = {  'O2': (3727.092 + 3729.875)/2,
                                'Ha':  6564.608,
                                'Hb':  4862.683, 
                                'Hg':  4341.684, 
                                'O3a': 4960.295, 
                                'O3b': 5008.240, 
                                'N2a': 6549.86 , 
                                'N2b': 6585.27 ,
                                }
    
    filter_list = [ "../photo/filters/B.txt",
                    "../photo/filters/V.txt",
                    "../photo/filters/R.txt",
                    "../photo/filters/I.txt",
                    "../photo/filters/sdss-i.txt",
                    "../photo/filters/sdss-z.txt"]
    
    # arr_Mstellar = np.zeros((1,6))
    # arr_Mstellar = np.append(arr_Mstellar,
    #                          [['slit', 'median', 'std', 
    #                            'err_lo', 'mean', 'err_hi']], axis=0)
            
    # for slit_num in tqdm([7]): # np.arange(1, 142+1)
    df     = pd.read_excel("redshift_table.xlsx", header=None, engine='openpyxl')
    array  = df.to_numpy()
    ztable = array[1:, 0:2]
    z_spec = ztable[slit_num, 1]
    ltable = array[1:, [0,6,9,12]]
    
    if np.isnan(z_spec)==False:
        if convert_spec_to_photo:
            # Look up emission line wavelengths
            emilines = set()
            for idxcol in range(1,len(ltable[0])):
                linename = ltable[slit_num,idxcol]
                if pd.notna(linename):
                    emilines.update(ltable[slit_num,idxcol].split(","))
            emilines = {k: all_emilines_supported[k] for k in emilines}
            emilines = dict(sorted(emilines.items(), key=lambda x: x[1]))
            
            x_flux, x_filt = convert_spec_to_eq_photo(slit_num, 
                                                      z_spec, emilines)
            filter_list.extend(x_filt) # must be same unit: uJy

        fit_instructions = load_config(z_spec=z_spec)
        
        galaxy = pipes.galaxy(ID=slit_num, load_data=load_spec_photo, 
                              spectrum_exists=spectrum_exists, 
                              spec_units="ergscma", 
                              photometry_exists=photometry_exists, 
                              phot_units="mujy", 
                              filt_list=filter_list,
                              spec_converted_photos=x_flux) # unit: uJy
        galaxy.plot(save=True, savepath=f'input_plots/{slit_num:03d}.png')
        
        fit = pipes.fit(galaxy, fit_instructions, run=".")
        fit.fit(verbose=True, sampler='nautilus')
        
        Ms_arr = fit.posterior.samples["stellar_mass"]
        
        x123 = np.percentile(Ms_arr, [16, 50, 84])
        median, std = np.median(Ms_arr), np.std(Ms_arr)
        err_lo, mean, err_hi = x123[1]-x123[0], x123[1], x123[2]-x123[1]
        
        if not os.path.exists("Mstellar_table.txt") \
            or os.path.getsize("Mstellar_table.txt") == 0:
            with open("Mstellar_table.txt", "w") as f:
                f.write("slit median std err_lo mean err_hi\n")
        
        with open("Mstellar_table.txt", "a+") as f:
            f.seek(0, 2)  # 移动到文件末尾
            f.write(f'{slit_num:>4d} {median:>6.3f} {std   :>6.3f} '+ 
                    f'{err_lo:>6.3f} {mean  :>6.3f} {err_hi:>6.3f}'+"\n")
        
        with open("Mstellar_result.txt", "a+") as f:
            f.seek(0, 2)  # 移动到文件末尾
            if f.tell() > 0:
                f.write("\n")  # 仅当文件非空时添加换行
            f.write(f' ========== Slit: {slit_num} ========== '+"\n")
            f.write(f'log10(M*) = {median:5.2f} +/- {std:5.2f}'+"\n")
            f.write(f'          = {mean:5.2f} +{err_hi:5.2f}/-{err_lo:5.2f}'+"\n")
        
        fig = fit.plot_spectrum_posterior(save=True, show=False)
        fig = fit.plot_sfh_posterior(     save=True, show=False)
        fig = fit.plot_corner(            save=True, show=False)
        
