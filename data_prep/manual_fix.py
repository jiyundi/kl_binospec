# Ignore this syntax check since it's an iPython command:
# %matplotlib qt
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def manual_correct(spec2d, spec_width=20, spec_height=20, tag='A', cont_margin=5):
    """
    打开交互窗口：
      - 鼠标移动时显示矩形
      - 左键点击记录坐标并关闭窗口
    
    Inputs:
        spec2d: dict
            It contains 'flux', 'wave', 'ivar' keys.
            
    Returns:
        返回 [y, x] indices
    """
    print(f'\nManually cutting for this spectrum {tag}...')
    plt.close('all')  # 防止旧窗口残留
    idxs = []
    
    fig, ax = plt.subplots(figsize=(4, 4))
    spec  = np.where(spec2d['mask'], spec2d['flux'], np.nan)
    noise = np.std(spec[spec > -500])
    pl_spec = ax.imshow(spec,
                        cmap='viridis', aspect='auto', origin='lower', 
                        vmin=0-noise, vmax=0 + 5*noise,
                        )
    wave_low = spec2d['wave'][0,0]
    wave_med = np.mean(spec2d['wave'][0])
    wave_hih = spec2d['wave'][0,-1]
    fig.colorbar(pl_spec, ax=ax)
    ax.text(0.02, 0.02, f"{wave_low:.1f}"+r" $\AA$", color='orangered', 
            fontsize=10, ha='left',  va='bottom', transform=ax.transAxes)
    ax.text(0.5 , 0.02, f"{wave_med:.1f}"+r" $\AA$", color='orangered', 
            fontsize=10, ha='center',  va='bottom', transform=ax.transAxes)
    ax.text(0.98, 0.02, f"{wave_hih:.1f}"+r" $\AA$", color='orangered', 
            fontsize=10, ha='right', va='bottom', transform=ax.transAxes)
    plt.title(tag)
    
    # 创建矩形补丁（初始隐藏）
    rect1 = patches.Rectangle(
        (0, 0), spec_width, spec_height,
        linewidth=1.5, edgecolor='red', facecolor='none', visible=False
    )
    rect2 = patches.Rectangle(
        (0, 0), spec_width-cont_margin*2, spec_height, ls=':',
        linewidth=1.5, edgecolor='white', facecolor='none', visible=False
    )
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    # 新增：创建横线（初始 invisible）
    center_line = mlines.Line2D([], [], linewidth=1, ls='--', 
                                color='white', alpha=0.5, visible=False)
    ax.add_line(center_line)
    
    # 用于保存所有文字对象的列表（方便后续删除或更新）
    texts = []
    
    # 鼠标移动事件
    def on_move(event):
        if event.inaxes != ax \
            or event.xdata is None \
                or event.ydata is None: # 确保点击发生在图像区域内
            rect1.set_visible(False)
            rect2.set_visible(False)
            center_line.set_visible(False) # ← 新增：隐藏横线
            fig.canvas.draw_idle()
            return
    
        # 计算矩形左下角坐标（以鼠标为中心）
        x0_rect1 = event.xdata - spec_width  / 2
        y0_rect1 = event.ydata - spec_height / 2
        x0_rect2 = x0_rect1 + cont_margin
        y0_rect2 = y0_rect1
        rect1.set_xy((x0_rect1, y0_rect1))
        rect2.set_xy((x0_rect2, y0_rect2))
        rect1.set_visible(True)
        rect2.set_visible(True)
        
        # 中间横线：从左边到右边
        x_center_left  = x0_rect1
        x_center_right = x0_rect1 + spec_width
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
            if 'O2' in tag: print('This is the center of O2 doublet lines\' index ⬆️')
            print('Click again to retry, or close this matplotlib window to continue...')
            
            # 在点击位置添加文字
            txt = ax.text(0.5, 0.95, 
                          f"index y = {y:.1f}, x = {x:.1f}\n"+
                          f'wave = {wave_low+x*(wave_hih-wave_low)/spec.shape[1]:.1f}\n'+
                          'Click again to retry, or '+
                          'Close this window to continue.', 
                          color='darkred', fontsize=10,
                          ha='center', va='top', 
                          backgroundcolor='white', 
                          transform=ax.transAxes)
            texts.append(txt)
            fig.canvas.draw_idle()  # 立即刷新显示
            
    # 绑定鼠标移动事件
    move = fig.canvas.mpl_connect('motion_notify_event', on_move)
    clic = fig.canvas.mpl_connect('button_press_event',  onclick)
    
    # 真正阻塞式显示
    plt.show(block=True)
    
    # 清理事件绑定
    fig.canvas.mpl_disconnect(move)
    fig.canvas.mpl_disconnect(clic)
    
    if not idxs:
        print("⚠️ No mouse click recorded. Returned None as index.")
        return None
    
    return idxs[0] # [ix, iy]

if __name__ == '__main__':
    import joblib
    with open('spec2d.pkl', "rb") as f:
        spec2d = joblib.load(f)
    for i in range(3):
        print(f'\nCutting for emission line {i+1}...')
        result = manual_correct(spec2d)
        print('Result:', result)