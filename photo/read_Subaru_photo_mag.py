import numpy  as np
import pandas as pd
from tqdm import tqdm

def readinfodat(infdatfilepath):
    infile_dat = open(infdatfilepath)
    dat_dict   = {}
    for sen in infile_dat:
        if sen[:2] != 'ID':
            dat_dict[sen.split('       ')[0]] = float(sen.split('       ')[1][:-1])
    return dat_dict

"""
 1 id      Object ID Number
 2 RA      Right Ascension in decimal degrees
 3 Dec     Declination in decimal degrees
 4 x       x pixel coordinate
 5 y       y pixel coordinate
 6 area    Segment size
 7 B       B_Subaru band magnitude
 8 dB      Uncertainty in B
 9 V       V_Subaru band magnitude
10 dV      Uncertainty in V
11 R       R_Subaru band magnitude
12 dR      Uncertainty in R
13 I       I_Subaru band magnitude
14 dI      Uncertainty in I
15 i       i_Subaru band magnitude
16 di      Uncertainty in i
17 Z       z_Subaru band magnitude
18 dZ      Uncertainty in Z
19 zb      BPZ most likely redshift
20 zbmin   Lower limit (95% confidence)
21 zbmax   Upper limit (95% confidence)
22 tb      BPZ most likely spectal type
23 odds    P(z) contained within 2*0.06*(1+z)
24 zml     Maximum Likelihood most likely redshift
25 tml     Maximum Likelihood most likely spectral type
26 chisq   Poorness of BPZ fit: observed vs. model fluxes
27 chisq2  Modified chisq: model fluxes given error bars
28 nfdet   Number of filters in which object is detected
29 nfobs   " in which object is observed (within FOV, no bad pixels, etc.)
"""


if __name__ == '__main__':
    txtname = "hlsp_clash_subaru_suprimecam_a383_photoz-cat.txt"
    
    with open(txtname, "r") as f:
        for i, line in enumerate(f):
            if i == 32: # = Line# - 1
                colnames = line.strip().split()[1:]
                break

    print('Reading:', txtname)
    mag_table = pd.read_csv(txtname, sep=r"\s+", skiprows=32, # including this line
                            names=colnames, comment="#", engine="python")
    
    mag_table["slit_num"] =  -1
    mag_table["dist"]     = '-1'
    count_matched = 0
    slit_list     = range(1, 142 +1)
    
    for slit_num in tqdm(slit_list):
        spec1d2dfolder02 = '../../../../RSCH3/UAO-S156-23B-A383/psf/231019/1d2dspecfiles/'
        infdatfilename   = 'info.829.{:03d}.{:06d}.dat'.format(slit_num, slit_num+100305)
        dat_dict = readinfodat(spec1d2dfolder02 + infdatfilename)
        target_RA, target_DEC = dat_dict['RA'], dat_dict['DEC']
        threshold  = 0.5 # arcsec
        # mag_table["dist"] = round(((mag_table["RA"]  - target_RA )**2 + 
        #                            (mag_table["Dec"] - target_DEC)**2   )**0.5, 2)
        for index, row in mag_table.iterrows():
            ID   = int(row["id"])
            dRA  = float(row["RA"])  - target_RA
            dDEC = float(row["Dec"]) - target_DEC
            dist = np.sqrt(dRA**2 + dDEC**2) * 3600 # arcsec
            if dist < threshold:
                mag_table.loc[index, "slit_num"] = slit_num
                mag_table.loc[index, "dist"]     = f"{dist:3.1f}"
                count_matched += 1
                tqdm.write(f'\nSlit #{slit_num}, {dist:4.2f} arcsec: ID={ID:6d}')
                break
        
    matched = mag_table[mag_table["slit_num"] != -1].copy()
    matched.to_csv("matched.txt", sep=" ", index=False)
