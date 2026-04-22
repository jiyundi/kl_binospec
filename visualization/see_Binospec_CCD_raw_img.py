import numpy as np
from astropy.io  import fits
# import matplotlib.patches as patches
import matplotlib.lines   as mlines
import matplotlib.pyplot  as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Inter",
    "font.serif": "Inter",
})


def manual_correct(image):
    plt.close('all')  # 防止旧窗口残留
    idxs = []
    
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(image, cmap='bone', aspect='auto', 
              vmax=np.percentile(image, 90))
    
    # 新增：创建横线（初始 invisible）
    center_line = mlines.Line2D([], [], linewidth=1, ls='--', 
                                color='white', alpha=0.5, visible=False)
    ax.add_line(center_line)
    
    # 鼠标移动事件
    def on_move(event):
        if event.inaxes != ax \
            or event.xdata is None \
                or event.ydata is None: # 确保点击发生在图像区域内
            center_line.set_visible(False) # ← 新增：隐藏横线
            fig.canvas.draw_idle()
            return
    
        # 中间横线：从左边到右边
        x_center_left  = 0
        x_center_right = 200
        y_center       = event.ydata
        center_line.set_data([x_center_left, x_center_right],
                             [y_center,      y_center])
        center_line.set_visible(True)
        
        fig.canvas.draw_idle()
    
    # 鼠标点击事件
    def onclick(event):
        if event.inaxes == ax and event.button == 1: 
            x, y = event.xdata, event.ydata
            idxs.append([x,y])
            print(f'Saved: index (vertical, horizontal) = ({int(round(y,0))}, {x:.1f}) 👍')
            fig.canvas.draw_idle()  # 立即刷新显示
            
    # 绑定鼠标移动事件
    move = fig.canvas.mpl_connect('motion_notify_event', on_move)
    clic = fig.canvas.mpl_connect('button_press_event',  onclick)
    
    # 真正阻塞式显示
    plt.tight_layout()
    plt.show(block=True)
    
    # 清理事件绑定
    fig.canvas.mpl_disconnect(move)
    fig.canvas.mpl_disconnect(clic)
    
    return idxs




if __name__ == '__main__':
    spec_CCD_root_dir = '../../../RSCH3/UAO-S156-23B-A383-raw/'
    set_A_dir = 'Abell383_psf_offset_m1/raw/2023.1023/'
    set_B_dir = 'Abell383_psf_offset_p1/raw/2023.1019/'
    fits_nameA = 'sci_img_2023.1023.091355.fits'
    fits_nameB = 'sci_img_2023.1019.082429.fits'
    
    set_dir, fits_name = set_B_dir, fits_nameB
    
    science_hdul = fits.open(spec_CCD_root_dir + set_dir + fits_name)
    
    full_CCD_nDEC = science_hdul[1].header.get('DETSIZE')[1:-1].split(',')[1][2:]
    full_CCD_nRA  = science_hdul[1].header.get('DETSIZE')[1:-1].split(',')[0][2:]
    full_CCD_size = (int(full_CCD_nDEC), int(full_CCD_nRA))
    full_CCD_imge = np.zeros(full_CCD_size)
    
    bin_table_1 = {name: science_hdul[ 9].data[name][0] \
                   for name in science_hdul[ 9].data.columns.names}
    bin_table_2 = {name: science_hdul[10].data[name][0] \
                   for name in science_hdul[10].data.columns.names}
    
    science_meta = {}
    for i in range(1, 11):
        science_ar = science_hdul[i].data
        science_h1 = science_hdul[i].header
        science_hdr1 = [{'key':     card.keyword,
                         'value':   card.value, 
                         'comment': card.comment
                         }
                        for card in science_h1.cards ]
        science_meta[f'{i}'] = {'data':   science_ar,
                                'header': science_hdr1,
                                }
    
    # Join CCD extensions
    for i in range(1, 9):
        # Configuration:
        #     |    Binospec Side A    |    Binospec Side B    |
        # NE  0 -------- 2048 ------ 4096 ------ 6144 ------ 8192 NW
        #     | *         |         * | *         |         * |
        #     |    [4]    |    [3]    |    [6]    |    [5]    |
        #     |           |           |           |           |
        #    2056 -------- ----------- ----------- -----------
        #     |           |           |           |           |
        #     |    [1]    |    [2]    |    [7]    |    [8]    |
        #     | *         |         * | *         |         * |
        # SE 4112 ----------------------------------------------- SW
        # * = Extension i's array origin (0, 0), i = 1 - 8.
        
        # Image locating default: Origin at upper left
        this_ext_up_idx, this_ext_dn_idx = \
            np.array(
                science_hdul[i].header.get('CCDSEC')[1:-1].split(',')[1].split(':'),
                dtype=int)
        this_ext_lf_idx, this_ext_rt_idx = \
            np.array(
                science_hdul[i].header.get('CCDSEC')[1:-1].split(',')[0].split(':'),
                dtype=int)
        this_ext_x1_idx = min(this_ext_lf_idx, this_ext_rt_idx) - 1
        this_ext_x2_idx = max(this_ext_lf_idx, this_ext_rt_idx)
        this_ext_y1_idx = min(this_ext_up_idx, this_ext_dn_idx) - 1
        this_ext_y2_idx = max(this_ext_up_idx, this_ext_dn_idx)
        
        this_ext_data = science_hdul[i].data
        if this_ext_up_idx > this_ext_dn_idx: 
            this_ext_data = this_ext_data[::-1, :] # Flip DEC for each RA axis
        if this_ext_lf_idx > this_ext_rt_idx: 
            this_ext_data = this_ext_data[:, ::-1] # Flip RA for each DEC axis
        
        # Rescaling flux
        this_ext_data = np.log10(this_ext_data)
        this_ext_data = this_ext_data - np.percentile(this_ext_data, 5)
        
        # Crop data's over-scan and Valuate full_CCD_imge
        y1, Y2 = np.array(
            science_hdul[i].header.get('DATASEC')[1:-1].split(',')[1].split(':'),
            dtype=int)
        x1, X2 = np.array(
            science_hdul[i].header.get('DATASEC')[1:-1].split(',')[0].split(':'),
            dtype=int)
        Y1, X1 = y1 - 1, x1 - 1
        full_CCD_imge[this_ext_y1_idx: this_ext_y2_idx, 
                      this_ext_x1_idx: this_ext_x2_idx] = this_ext_data[Y1: Y2, 
                                                                        X1: X2]
    
    # Need manually identify slits?
    manual = False #  True
    if manual:
        # idxs = manual_correct(full_CCD_imge[:, :100])
        # print(np.array(idxs)[:,1])
        idxs = manual_correct(full_CCD_imge[:, -100:])
        print(np.array(idxs)[:,1])
    else:
        sideA_slitxs = [ 
            232, 305, 354, 413, 466, 520, 564, 603, 657, 696, 
            755, 814, 867, 911, 951, 999, 1043, 1097, 1151, 1220,
            1293, 1342, 1386, 1430, 1469, 1503, 1542, 1591, 1640, 1689, 
            1738, 1797, 1841, 1904, 1948, 1992, 2031, 2090, 2134, 2188,
            2237, 2286, 2330, 2379, 2423, 2457, 2496, 2550, 2603, 2652,
            2711, 2755, 2794, 2838, 2897, 2956, 3009, 3068, 3122, 3166,
            3225, 3278, 3322, 3381, 3420, 3464, 3503, 3552, 3596, 3645,
            3709, 3767, 3860
            ]
        sideB_slitxs = [ 
            227, 276, 320, 374, 418, 462, 510, 550, 598, 647, 
            701, 750, 809, 867, 916, 965, 1029, 1068, 1131, 1180,
            1234, 1283, 1327, 1405, 1498, 1562, 1625, 1699, 1757, 1826,
            1880, 1929, 1987, 2051, 2100, 2154, 2237, 2300, 2359, 2408,
            2447, 2491, 2540, 2594, 2652, 2716, 2765, 2824, 2863, 2907,
            2956, 2995, 3034, 3083, 3141, 3180, 3234, 3269, 3322, 3381,
            3410, 3459, 3508, 3572, 3626, 3665, 3728, 3792, 3870
            ]
        sideA_nums = np.arange(1, len(sideA_slitxs)+1)
        sideB_nums = np.arange(1, len(sideB_slitxs)+1) + len(sideA_slitxs)
    
    # CCD PLotting
    fig = plt.figure(figsize=(full_CCD_imge.shape[1]/150*1.2, 
                              full_CCD_imge.shape[0]/150), 
                     dpi=200)
    gs = fig.add_gridspec(nrows=1, ncols=1)
    ax1 = fig.add_subplot(gs[0, 0])
    high = np.percentile(full_CCD_imge, 85)
    imshow1 = ax1.imshow(full_CCD_imge, cmap='bone', 
                         vmax=np.percentile(full_CCD_imge, 90)
                         )
    cbar = fig.colorbar(imshow1, ax=ax1, pad=0.02)
    for iA in range(len(sideA_nums)):
        ax1.text(0-2, sideA_slitxs[iA], sideA_nums[iA], 
                 color='slategray', fontsize=7, ha='right', va='center')
    for iB in range(len(sideB_nums)):
        ax1.text(full_CCD_imge.shape[1]+2, sideB_slitxs[iB], sideB_nums[iB], 
                 color='slategray', fontsize=7, ha='left',  va='center')
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label(label=r'$log_{10}F$ - detector sky (=5th percentile)', size=24)
    ax1.minorticks_on() # enable minor ticks
    ax1.tick_params(right=True, labelright=True)
    ax1.tick_params(axis='both', which='major', length=12, width=2.5)
    ax1.tick_params(axis='both', which='minor', length=9,  width=1, right=True)
    ax1.set_title(f'{set_dir+fits_name}, '+'Side A (left) & B (right), 8 square regions, '+
                  f'{full_CCD_imge.shape[1]} px × {full_CCD_imge.shape[0]} px', loc='left', fontsize=36)
    plt.savefig(f'binospec_CCD_{fits_name.split("_")[2]}.jpg', bbox_inches='tight')