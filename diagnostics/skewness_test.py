import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

data = {}
for _ in range(100):
    for a in np.arange(0, 20):
        # loc 为位置参数（类似于峰值对应的位置），scale 为缩放参数（标准差）
        data_smooth = stats.skewnorm.rvs(a, loc=0, scale=15, size=1000)
        
        skewness = stats.skew(data_smooth)
        data[f'{skewness:.5f}'] = data_smooth
        
        print("当前数组偏度: "+f'{skewness:.5f}') # 结果约为 -0.98 ~ -1.02
    


# 使用 np.digitize 找出每个 skewness 属于哪个 bin
# bin_indices 的值范围是 1 到 len(bins)-1
# 初始化一个列表，用来存放每个 bin 拼接后的结果
# 共有 len(bins)-1 个区间（这里 7 个边界对应 6 个区间）
skewnesses  = np.array([float(skewness) for skewness in data.keys()])
bins        = np.linspace(0, 1.4, num=8, endpoint=True)
bin_indices = np.digitize(skewnesses, bins)

# 5. 分组并拼接
bin_results = [np.array([]) for _ in range(len(bins) - 1)]
for idx, bin_num in enumerate(bin_indices):
    # np.digitize 默认左闭右开，若数据刚好等于最大边界值，会归入 len(bins) 
    # 我们需要把它修正到最后一个有效的 bin 里
    if bin_num == len(bins):
        bin_num -= 1
    
    if 1 <= bin_num <= len(bins) - 1:
        str_key = list(data.keys())[idx]
        bin_results[bin_num - 1] = np.append(
            bin_results[bin_num - 1], 
            data[str_key]
            )


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,4)) # (width, height)
for i in range(len(bins)-1):
    if len(bin_results[i]) != 0:
        this_skew_left  = bins[i]
        this_skew_right = bins[i+1]
        RVs_shifted     = \
            (bin_results[i] - np.percentile(bin_results[i], 5)) /        \
            (np.percentile(bin_results[i], 95) - np.percentile(bin_results[i], 5))
        yy_abs, xx_left = np.histogram(RVs_shifted, bins=50)
        ax.plot((xx_left[:-1] + xx_left[1:]) / 2, 
                yy_abs/np.sum(yy_abs), 
                label=f'{this_skew_left:.1f}'+r'$< \gamma <$'+f'{this_skew_right:.1f}',
                linestyle='-', linewidth=1+0.5*i, 
                color=(1-0.08*i, 0, 0.15*i, 0.4+0.08*i)
                )
ax.set_xlabel('Random variables (normalized)')
ax.set_ylabel('Density')
ax.set_title('Distributions vs. skewness'+r' ($\gamma$)'+'\n(Random generator: stats.skewnorm.rvs)')
ax.set_xlim(-0.25, 1.25)
ax.legend(prop={'size': 8})
plt.savefig("distribution_by_skewnesses.jpg", bbox_inches='tight')

