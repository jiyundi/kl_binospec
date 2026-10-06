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






if __name__ == '__main__':
    with open("hlsp_clash_hst_acs-ir_a383_cat.txt", "r") as f:
        for i, line in enumerate(f):
            if i == 139:
                colnames = line.strip().split()[1:]
                break
    
    mag_table = pd.read_csv("hlsp_clash_hst_acs-ir_a383_cat.txt", 
                        sep=r"\s+", skiprows=139, 
                        names=colnames, comment="#")
    # mag_table = mag_table.values
    
    count_matched = 0
    
    for slit_num in tqdm(range(1, 142+1)):
        spec1d2dfolder02 = '../../../../RSCH3/UAO-S156-23B-A383/psf/231019/1d2dspecfiles/'
        infdatfilename   = 'info.829.{:03d}.{:06d}.dat'.format(slit_num, slit_num+100305)
        dat_dict = readinfodat(spec1d2dfolder02 + infdatfilename)
        target_RA, target_DEC = dat_dict['RA'], dat_dict['DEC']
        threshold  = 1 / 3600 # '' --> deg
        for index, row in mag_table.iterrows():
            dRA  = row["RA"]  - target_RA
            dDEC = row["Dec"] - target_DEC
            dist = np.sqrt(dRA**2 + dDEC**2)
            if dist < threshold:
                count_matched += 1
                print(f'\n {count_matched}, Slit #{slit_num}, {dist*3600:.2f} arcsec:', 
                        row['f225w_mag'],
                        # row['f225w_magerr'],
                        row['f275w_mag'],
                        # row['f275w_magerr'],
                        row['f336w_mag'],
                        # row['f336w_magerr'],
                        row['f390w_mag'],
                        # row['f390w_magerr'],
                        row['f435w_mag'],
                        # row['f435w_magerr'],
                        row['f475w_mag'],
                        # row['f475w_magerr'],
                        row['f606w_mag'],
                        # row['f606w_magerr'],
                        row['f625w_mag'],
                        # row['f625w_magerr'],
                        row['f775w_mag'],
                        # row['f775w_magerr'],
                        row['f814w_mag'],
                        # row['f814w_magerr'],
                        row['f850lp_mag'],
                        # row['f850lp_magerr'],
                        row['f105w_mag'],
                        # row['f105w_magerr'],
                        row['f110w_mag'],
                        # row['f110w_magerr'],
                        row['f125w_mag'],
                        # row['f125w_magerr'],
                        row['f140w_mag'],
                        # row['f140w_magerr'],
                        row['f160w_mag'],
                        # row['f160w_magerr']
                        )
                break