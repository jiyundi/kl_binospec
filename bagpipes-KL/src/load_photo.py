import numpy  as np
import pandas as pd


def load_photo(slit_num_str):
    slit_num = int(slit_num_str)
    
    txtname   = '../photo/slits_Subaru_matched.txt'
    mag_table = pd.read_csv(txtname, sep=" ", skiprows=0, # including this line
                            comment="#", engine="python")
    mag_table = mag_table.sort_values(by='slit_num', ascending=True)
    
    for index, row in mag_table.iterrows():
        if slit_num == row['slit_num']:
            # Extract the object we want from the catalogue.
            fluxes   = np.array([row['B'],  row['V'],  row['R'], 
                                 row['I'],  row['i'],  row['Z']  ])
            fluxerrs = np.array([row['dB'], row['dV'], row['dR'], 
                                 row['dI'], row['di'], row['dZ'] ])
            break

    # Turn these into a 2D array and apply conversion: AB mag --> microJy.
    photometry = np.c_[fluxes, fluxerrs]
    photometry = ABmag_to_microJy(photometry)

    return photometry

def ABmag_to_microJy(ABmag_arr):
    for i in range(len(ABmag_arr)):
        ABmag, ABmag_err = ABmag_arr[i,0], ABmag_arr[i,1]
        flux0 = 10**((23.947 - ABmag) / 2.5) # README: zeropoint=23.947
        flux1 = 10**((23.947 - ABmag + ABmag_err) / 2.5) 
        flux_err = flux1 - flux0
        ABmag_arr[i,0] = flux0
        ABmag_arr[i,1] = flux_err
        microJy_arr = ABmag_arr
    return microJy_arr

