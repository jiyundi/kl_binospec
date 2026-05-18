import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})

def merge_intervals(wav_min, wav_max):
    # 将输入波段组合为 (min, max) 的元组并按起点排序
    intervals = sorted(zip(wav_min, wav_max), key=lambda x: x[0])

    merged = []
    for interval in intervals:
        if not merged:
            merged.append(list(interval))
        else:
            # 如果当前区间与前一个有重叠，则合并
            if interval[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], interval[1])
            else:
                merged.append(list(interval))
    return merged

def plot_sky(sky_bands):
    fig, ax = plt.subplots(1, 1, figsize=(4, 6)) # (length, height)
    
    for i in range(len(sky_bands)):
        wav_min = sky_bands[i,0]
        wav_max = sky_bands[i,1]
        wav_0   = np.mean([wav_min, wav_max])
        ax.errorbar(0, i+1, 
                    xerr=wav_0-wav_min, 
                    fmt=' ', capsize=5, capthick=2, elinewidth=4, 
                    marker=None, markersize=6, color="#C82423")
        ax.text(wav_min-wav_0, i+1, 
                f'{str(int(wav_min)) if float(wav_min).is_integer() else str(wav_min)} ', 
                fontsize=10, color="#C82423", ha='right', va='center')
        ax.text(wav_max-wav_0, i+1, f' {str(int(wav_max)) if float(wav_max).is_integer() else str(wav_max)}', 
                fontsize=10, color="#C82423", ha='left', va='center')
        
    ax.minorticks_on()
    ax.ticklabel_format(useOffset=False)
    ax.set_xlabel(r'$\Delta\lambda$'+r' ($\AA$)')
    ax.set_xlim(-5, 5)
    ax.set_ylim( 0, len(sky_bands)+1)
    ax.grid(linestyle=':', alpha=0.5)
    ax.set_title('Sky line subtraction bands')
    plt.savefig("sky_line_bands.jpg", dpi=150, bbox_inches='tight')
    plt.show()
    return

# 示例输入
sky_wav_min  = [8378,   8785, 8428, 8411, 8427, 8835,   8463, 8999, 8825, 8834]
sky_wav_max  = [8382.5, 8789, 8432, 8416, 8432, 8837.5, 8467, 9003, 8828, 8838]

sky_wav_min += [8834, 8824, 7339, 8884, 9304, 9311, 9321, 8833, 8765, 8757]
sky_wav_max += [8838, 8829, 7342, 8888, 9308, 9315, 9326, 8838, 8770, 8763]

sky_wav_min += [7327, 7314, 8883, 8917, 8941]
sky_wav_max += [7330, 7318, 8887, 8921, 8944]

sky_bands = merge_intervals(sky_wav_min, sky_wav_max)
        
plot_sky(np.array(sky_bands))
