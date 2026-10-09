import joblib
import numpy as np
from klm.safe_plot import setup; setup() # must before plt

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})



def analyze_percentile(slit_folder, pars_to_plot):
    # Read posterior sample points
    samples = np.loadtxt(f'{slit_folder}post.txt', dtype=str, skiprows=0)

    idxs = []
    for plot_par in pars_to_plot:
        for i in range(len(samples[0])):
            if plot_par in samples[0][i]:
                idxs.append(i)

    results = {}; i = 0
    for idx in idxs:
        weights     = samples[2:,   0].astype(float)
        samp_points = samples[2:, idx].astype(float) # of title rows: 2
        
        q16, median, q84 = weighted_quantile(samp_points, [0.16, 0.50, 0.84], weights)
        err_lo = q16 - median
        err_hi = q84 - median
        # mean = np.mean(samp_points)

        results[pars_to_plot[i]] = np.around([median, err_lo, err_hi], decimals=4)
        i += 1

    return results

def weighted_quantile(x, quantiles, weights):
    x = np.asarray(x)
    weights = np.asarray(weights)
    quantiles = np.asarray(quantiles)

    valid = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
    x = x[valid]
    weights = weights[valid]

    order = np.argsort(x)
    x = x[order]
    weights = weights[order]

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    return np.interp(quantiles, cdf, x)



colors = ['royalblue', 'darkorange', 'crimson', 'limegreen']
ylims  = [(-0.25, 0.25), (-0.8, 0.8)]

pars_to_plot = ['gamma_t', 'theta_int'] #
runs_folders = ['runs_noisy_snr20/', 'runs_noisy_snr15/', 'runs_noisy_snr10/']
slit_names  = np.arange(0, 11)

# Specify x-axis
mock_folder = '008b_vary_thetaint_slitLPA_major/'
special_idxs = ['0.0', '0.3', '0.7', '1.0', '1.4', '1.6', '1.7', '2.1', '2.4', '2.8', '3.1']
data_infos = []
for slit_name in slit_names:
    data_info_path = f'{mock_folder}/slit_002_{special_idxs[slit_name]}.pkl'
    with open(data_info_path, "rb") as f:
        data_infos.append(joblib.load(f))

trues_to_plot = np.array(
    [
    [data_infos[s]['fid_params']['shared_params']['gamma_t'  ] for s in slit_names], 
    [data_infos[s]['fid_params']['shared_params']['theta_int'] for s in slit_names]
    ]
    )

xaxis_points  = [data_infos[s]['fid_params']['shared_params']['theta_int'] for s in slit_names]
xaxis_values = np.array(xaxis_points) / np.pi * 180
xaxis_label  = r'$\theta_{\rm int}$ (deg)'

# Find y-axis values
post_1sigma_ranges = {}
for runs_folder in runs_folders:
    this_folder_post_1sigma_ranges = []
    for slit_name in slit_names:
        slit_folder = runs_folder + f'Slit_{slit_name:03d}/'
        this_folder_post_1sigma_ranges.append(
            analyze_percentile(slit_folder, pars_to_plot)
            )
    post_1sigma_ranges[runs_folder] = this_folder_post_1sigma_ranges

# Plot
fig = plt.figure(figsize=(6, 3*len(pars_to_plot)))  # (length, height)
plt.subplots_adjust(hspace=0.5, wspace=0.2) # h=height
gs = fig.add_gridspec(nrows=len(pars_to_plot), ncols=1, 
                      height_ratios=[1]*len(pars_to_plot), 
                      width_ratios=[1]*1)
        
for idx_par, par_to_plot in enumerate(pars_to_plot):
    ax1 = fig.add_subplot(gs[idx_par, 0])
    ax1.set_title(par_to_plot, fontsize=12)
    
    ax1.set_ylabel('Fit - Mock param')
    ax1.set_xlabel(xaxis_label, labelpad=0)

    ax1.axhline(y=0, color='forestgreen', linestyle='--', linewidth=2)
    ax1.grid(linestyle=':', color='black', alpha=0.5)

    # for each run folders
    run_num = 0
    for run_num, (folder_name, ranges_all_xs) in enumerate(post_1sigma_ranges.items()):
        medians = np.array([ranges_this_x[par_to_plot][0] for ranges_this_x in ranges_all_xs])
        err_los = np.array([ranges_this_x[par_to_plot][1] for ranges_this_x in ranges_all_xs])
        err_his = np.array([ranges_this_x[par_to_plot][2] for ranges_this_x in ranges_all_xs])
        yerrs   = np.array([-err_los, err_his])
        color   = colors[run_num % len(colors)]

        # Special periodic correction for theta_int
        for t in range(len(medians)): 
            if medians[t] > 6: medians[t] -= 2*np.pi

        ax1.errorbar(
            xaxis_values, medians - trues_to_plot[idx_par], 
            yerr=yerrs, 
            fmt=':', 
            capsize   =5-1.5*run_num, 
            capthick  =2-0.5*run_num, 
            linewidth =3-0.5*run_num,
            elinewidth=2-0.5*run_num, 
            markersize=6-1.5*run_num, 
            marker='o', color=color, alpha=1.0, 
            label=folder_name
            )

        ax1.text(0.01, 0.01 + run_num * 0.05, 
                f'Avg offset = {np.mean(medians - trues_to_plot[idx_par]):.2f}   '+
                f'Avg error = {np.mean(yerrs.flatten()):.2f}', 
                fontsize=8, color=color, ha='left', va='bottom', 
                transform=ax1.transAxes)

    ax1.legend(prop={'size': 10}, loc='upper right')
    ax1.set_ylim(ylims[idx_par])

fig.savefig(f'shape_noise_errorbars.png', dpi=150, bbox_inches='tight')
plt.close()

