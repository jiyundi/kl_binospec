import numpy as np
import pandas as pd

import bagpipes as pipes
from bagpipes.fitting import posterior

# from src.load_photo     import load_photo
# from src.load_spec      import load_spec
from src.convert_s_to_p import convert_spec_to_eq_photo
from src.load_config    import load_config
from fitting_spec_by_Bagpipes import load_spec_photo


filter_list = [ "../photo/filters/B.txt",
                "../photo/filters/V.txt",
                "../photo/filters/R.txt",
                "../photo/filters/I.txt",
                "../photo/filters/sdss-i.txt",
                "../photo/filters/sdss-z.txt"]

slit_num = 95
folder   = ''
spectrum_exists   = False
photometry_exists = True
convert_spec_to_photo = True

if convert_spec_to_photo:
    bonus_phs, bonus_fts = convert_spec_to_eq_photo(slit_num)
    filter_list.extend(bonus_fts) # must be same unit: uJy

galaxy = pipes.galaxy(ID=slit_num, load_data=load_spec_photo, 
                      spectrum_exists=spectrum_exists, 
                      spec_units="ergscma", 
                      photometry_exists=photometry_exists, 
                      phot_units="mujy", 
                      filt_list=filter_list,
                      spec_converted_photos=bonus_phs)

df     = pd.read_excel("redshift_table.xlsx", header=None, engine='openpyxl')
array  = df.to_numpy()
ztable = array[1:, 0:2]
z_spec = ztable[slit_num, 1]

fit_instructions = load_config(z_spec=z_spec)

fit = pipes.fit(galaxy, fit_instructions, run=".")
# fit.fit(verbose=True, sampler='nautilus')

posterior = posterior(galaxy)

Ms_arr = fit.posterior.samples["stellar_mass"]

x123 = np.percentile(Ms_arr, [16, 50, 84])
err_lo, mean, err_hi = x123[1]-x123[0], x123[1], x123[2]-x123[1]

# with open(f"Mstellar_result_{folder}.txt", "w") as file_object:
print(f' ========== Slit: {slit_num} ========== ')
print(f'log10(M*) = {np.median(Ms_arr):5.2f} +/- {np.std(Ms_arr):5.2f}')
print(f'          = {mean:5.2f} +{err_hi:5.2f}/-{err_lo:5.2f}')

