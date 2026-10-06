import joblib
import numpy as np


def another_load_mock(pkl_folder='mock/', slit_num=95):
    with open(f'{pkl_folder}pkl/slit_{slit_num:03d}.pkl', "rb") as f:
        data_info = joblib.load(f)
    return data_info


def find_not_overlaps(pkl_folder='../scripts/binospec_pkl/'):
    slitnums  = []
    spec_idxs = []
    not_overlaps = []
    
    for slit_num in range(1, 143):
        try:
            data_info = another_load_mock(
                pkl_folder=pkl_folder, 
                slit_num=slit_num)
            
        except FileNotFoundError:
            print( "\033[43m" + 'WARNING:' + "\033[0m " + 
                  f'Slit {slit_num} skipped because no PKL found.')
            continue
        
        for spec_idx in range(len(data_info['spec'])):
            mask = data_info['spec'][spec_idx]['mask']
            line = data_info['spec'][spec_idx]['meta']['line_species']
            
            if line == 'O2':
                left, right = mask.shape[1]//4, mask.shape[1] - mask.shape[1]//4
            else:
                left, right = mask.shape[1]//3, mask.shape[1] - mask.shape[1]//3
            
            accept = True
            for x in range(left, right):
                accept_this_col = np.any(mask[:, x])
                accept &= accept_this_col
            
            slitnums.append(slit_num)
            not_overlaps.append(bool(accept))
            spec_idxs.append(spec_idx)
                
        
    return slitnums, spec_idxs, not_overlaps # Slits are GOOD, without sky masks


if __name__ == '__main__':
    slitnums, spec_idxs, not_overlaps = find_not_overlaps()
    
    import matplotlib.pyplot as plt
    import imageio # GIF
    frames = []    # GIF
    n = 0
    for slitnum in np.unique(slitnums):
        select_slit = (np.array(slitnums) == slitnum)
        goods_or_bads = np.array(not_overlaps)[select_slit]
        
        # If all are bads...
        all_specs_rejected   = ~np.any(goods_or_bads) 
        
        # If >= 50% are good, < 50% specs are bad...
        major_specs_accepted = (np.sum(goods_or_bads) >= 0.5 * len(goods_or_bads))
        
        if not major_specs_accepted:
            n += 1
            if all_specs_rejected:
                print(f'Slit {slitnum:03d}:  NO  spec   is not overlapped by sky line masks. ({n:3d})')
            else:
                print(f'Slit {slitnum:03d}: <50% specs are not overlapped by sky line masks. ({n:3d})')
            
        # these_spec_idxs = np.array(spec_idxs)[select_slit]
        # for spec_idx in these_spec_idxs:
            
        #     if goods_or_bads[spec_idx]:# == False:
                
        #         data_info = another_load_mock(
        #             pkl_folder='../scripts/binospec_pkl/', 
        #             slit_num=slitnum)
                
        #         mask = data_info['spec'][spec_idx]['mask']
        #         data = data_info['spec'][spec_idx]['data']
        #         line = data_info['spec'][spec_idx]['meta']['line_species']
        #         if line == 'O2':
        #             left, right = mask.shape[1]//4, mask.shape[1] - mask.shape[1]//4
        #         else:
        #             left, right = mask.shape[1]//3, mask.shape[1] - mask.shape[1]//3
                
        #         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
        #         im = ax.imshow(np.where(mask, data, np.nan),
        #                        aspect='auto', cmap='viridis', 
        #                        origin='lower')
        #         ax.axvline(left-0.5,  color='red')
        #         ax.axvline(right+0.5, color='red')
        #         plt.colorbar(im, ax=ax)
        #         plt.title(f'Slit {slitnum:3d} spec_idx {spec_idx} line {line}', loc="left")
                
        #         # GIF
        #         fig.canvas.draw()
        #         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        #         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #         frames.append(image)
        #         plt.close(fig)
    
    # # --- 结束循环后统一保存为 GIF ---
    # if frames:
    #     print(f"正在生成 GIF，总计 {len(frames)} 帧...")
    #     # duration 代表每帧之间的时间间隔（单位：毫秒）。500ms = 0.5秒
    #     # loop=0 表示无限循环播放
    #     imageio.mimsave('output_spectras.gif', frames, duration=50, loop=0)
    #     print("GIF 保存成功！文件名为: output_spectras.gif")
    # else:
    #     print("没有符合条件的帧，未生成 GIF。")
    