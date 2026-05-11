import joblib
import numpy as np
import matplotlib.pyplot as plt

def another_load_mock(pkl_folder='mock/', Ms_folder='./', slit_num=95, 
                      rescale_image=False):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as f:
        data_info = joblib.load(f)
    
    # assert data_info['galaxy']['log10_Mstar'] != None, \
    #     "Cannot find corresponing stallar mass M*"  # if not, error
    
    # Recover wcs(galsim.wcs) from ap_wcs
    # ap_wcs  = data_info['image']['par_meta']['ap_wcs']
    # data_info['image']['par_meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)
    return data_info

if __name__ == '__main__':
    # Load
    data_info  = another_load_mock(
        pkl_folder='../scripts_beta/binospec_pkl/', 
        Ms_folder='../../bagpipes-KL/', 
        slit_num=95)
    
    # p1_list = data_info['spec']
    # p2_dict = data_info['image']
    p3_dict = data_info['galaxy']
    
    # p1_1_dict = data_info['spec'][0]
    # p1_2_dict = data_info['spec'][1]
    # p1_3_dict = data_info['spec'][2]
    
    spec_idx = 2
    p1_1_1_arr  = data_info['spec'][spec_idx]['data']
    p1_1_2_arr  = data_info['spec'][spec_idx]['var' ]
    p1_1_3_arr  = data_info['spec'][spec_idx]['cont_model']
    # p1_1_4_dict = data_info['spec'][spec_idx]['par_meta']
    p1_1_5_arr  = data_info['spec'][spec_idx]['mask']
    p1_1_6_arr  = data_info['spec'][spec_idx]['gauss_back']
    # p1_1_7_arr  = data_info['spec'][spec_idx]['shot_noise']
    
    p1_1_4_01_str  = data_info['spec'][spec_idx]['par_meta']['line_species']
    p1_1_4_02_tupl = data_info['spec'][spec_idx]['par_meta']['ngrid']
    p1_1_4_03_arr  = data_info['spec'][spec_idx]['par_meta']['lambda_grid']
    p1_1_4_04_num  = data_info['spec'][spec_idx]['par_meta']['pixScale']
    p1_1_4_05_num  = data_info['spec'][spec_idx]['par_meta']['rhl']
    p1_1_4_06_dict = data_info['spec'][spec_idx]['par_meta']['line_sig_amps']
    p1_1_4_07_unum = data_info['spec'][spec_idx]['par_meta']['slitRA']
    p1_1_4_08_unum = data_info['spec'][spec_idx]['par_meta']['slitDec']
    p1_1_4_09_num  = data_info['spec'][spec_idx]['par_meta']['slitWidth']
    p1_1_4_10_num  = data_info['spec'][spec_idx]['par_meta']['slitLen']
    p1_1_4_11_unum = data_info['spec'][spec_idx]['par_meta']['slitLPA']
    p1_1_4_12_unum = data_info['spec'][spec_idx]['par_meta']['slitWPA']
    
    p2_1_arr  = data_info['image']['data']
    p2_2_arr  = data_info['image']['var' ]
    # p2_3_dict = data_info['image']['par_meta']
    p2_4_arr  = data_info['image']['varr']
    p2_5_arr  = data_info['image']['mask']
    
    p2_3_1_tupl = data_info['image']['par_meta']['ngrid']
    p2_3_2_num  = data_info['image']['par_meta']['pixScale']
    p2_3_3_num  = data_info['image']['par_meta']['psfFWHM']
    p2_3_4_wcs  = data_info['image']['par_meta']['ap_wcs']
    p2_3_5_num  = data_info['image']['par_meta']['RA']
    p2_3_6_num  = data_info['image']['par_meta']['Dec']
    p2_3_7_awcs = None # data_info['image']['par_meta']['wcs']
    
    
    # for key, arr in data_info['spec'][spec_idx].items():
    #     if key != 'par_meta':
    #         fig, ax = plt.subplots(figsize=(3,3))
    #         im = ax.imshow(
    #             np.where(p1_1_5_arr, arr, np.nan), 
    #             aspect='auto', cmap='bone')
    #         plt.colorbar(im, ax=ax)
    #         plt.title(key)
    #         plt.show()
    
    # S2_arr1 = np.loadtxt('../../../../_Archived/2026-1-spring-UA/ASTR 599 Spectroscopy/086.A2552.txt')
    # S2_arr2 = np.loadtxt('../../../../_Archived/2026-1-spring-UA/ASTR 599 Spectroscopy/058.A2261_2014A.txt')
    
    # plt.style.use('default')
    # plt.rcParams.update({
    #     "font.family": "serif",
    #     "font.serif": ["Times New Roman"],
    # })
    # specs_data, specs_mask, slit_nums = [], [], []
    # for slit_num in [15, 21, 75, 95, 112, 113]:
    #     # Load
    #     data_info  = another_load_mock(pkl_folder='../scripts_beta/binospec_pkl/', 
    #                                    Ms_folder='../../bagpipes-KL/', 
    #                                    slit_num=slit_num)
    #     for i in range(len(data_info['spec'])):
    #         if data_info['spec'][i]['par_meta']['line_species'] == 'O2':
    #             specs_data.append(data_info['spec'][i]['data'])
    #             specs_mask.append(data_info['spec'][i]['mask'])
    #             slit_nums.append(slit_num)
    
    # fig = plt.figure(figsize=(len(specs_data)*0.8, 4))  # (length, height)
    # plt.subplots_adjust(hspace=0, wspace=0.1) # h=height
    # plt.tight_layout()
    # gs = fig.add_gridspec(nrows=4, ncols=len(specs_data)//3+2)
    # for j in range(len(specs_data)):
    #     ax1 = fig.add_subplot(gs[j%3, j//3])
    #     ax1.imshow(np.where(specs_mask[j], specs_data[j], 0),
    #                cmap='bone', aspect='auto', origin='lower',
    #                vmin=-20)
    #     ax1.set_axis_off()
        
    #     if j%3==0: 
    #         ax1.set_title(f'# {slit_nums[j]}')
    
    # ax2 = fig.add_subplot(gs[:2, -2:])
    # ax3 = fig.add_subplot(gs[2:, -2:])
    # ax2.imshow(S2_arr1, cmap='gray', aspect='auto', origin='lower')
    # ax3.imshow(S2_arr2, cmap='gray', aspect='auto', origin='lower')
    # ax2.set_title(r'[S II] $\lambda\lambda$6718, 6733')
    # ax2.set_axis_off()
    # ax3.set_axis_off()
    
    # plt.savefig("O2_examples.png", dpi=150, bbox_inches='tight', transparent=True)