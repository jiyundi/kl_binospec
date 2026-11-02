import joblib
import numpy as np

from klm.parameters import Parameters

# This setup must be called before importing matplotlib.pyplot.
from klm.safe_plot import setup
setup()
import matplotlib.pyplot as plt
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.serif": ["Helvetica", "Arial", "DejaVu Sans"],
    }
)



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



# ===========================================================================
# Main workflow
# ===========================================================================

if __name__ == "__main__":

    # User input: Specify x-axis: 'cosi' or 'theta_int'
    xaxis_param = 'theta_int' # , 'cosi'

    # User input: ylim
    ylims  = [(-0.12, 0.12), (-0.12, 0.12)] #

    # User input: which params
    pars_to_plot = ['g1', 'g2']

    # User input: mock and runs folder
    mock_folder = './mock_pkl/pkl/'
    runs_folders = ['runs_noisy_imgSNR40_specSNR30/'] # 'runs_no_noise/', 
    colors = ['crimson', 'black', 'royalblue', 'darkorange', 'limegreen']
    
    # User input: slit num definition
    slit_names = np.arange(1, 30+1)
    special_idxs = [
            "001", "002", "003", "004", "005", 
            "011", "012", "013", "014", "015",
            "021", "022", "023", "024", "025",
            "031", "032", "033", "034", "035",
            "041", "042", "043", "044", "045",
            "051", "052", "053", "054", "055"
        ]

    # read files
    data_infos = []
    for slit_name in slit_names:
        data_info_path = f'{mock_folder}mock_000_{special_idxs[slit_name-1]}.pkl'
        with open(data_info_path, "rb") as f:
            data_info = joblib.load(f)

        # Historic error: par_fit -> fid_params. Rename for clarity.
        if "par_fit" in data_info:
            data_info["fid_params"] = Parameters._unflatten(data_info.pop("par_fit"))

        data_infos.append(data_info)

    trues_to_plot = np.array(
        [
            [
                data_infos[s-1]['fid_params']['shared_params'][par] 
                for s in slit_names
            ] for par in pars_to_plot
        ]
        )

    if xaxis_param == 'cosi':
        xaxis_values = np.array([
            info['fid_params']['shared_params']['cosi'] for info in data_infos
        ], dtype=float)
        xaxis_label = r'$\cos(i)$'
    elif xaxis_param == 'theta_int':
        xaxis_values = np.rad2deg([
            info['fid_params']['shared_params']['theta_int'] for info in data_infos
        ])
        xaxis_label = r'$\theta_{\rm int}$ (deg)'
    else:
        raise ValueError("xaxis_param must be 'cosi' or 'theta_int'")

    # Round only for grouping, so floating-point noise does not split equal values.
    xaxis_groups = np.round(xaxis_values, decimals=8)
    xaxis_unique = np.unique(xaxis_groups)
    group_masks = [xaxis_groups == x for x in xaxis_unique]

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
    plt.subplots_adjust(hspace=0.4, wspace=0.2) # h=height
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
            if par_to_plot == 'theta_int':
                for t in range(len(medians)):
                    if medians[t] > 6: medians[t] -= 2*np.pi

            # Average Fit - Mock and each side of the error bar separately.
            # These are mean individual uncertainties, not errors on the mean.
            offsets = medians - trues_to_plot[idx_par]
            mean_offsets = np.array([offsets[mask].mean() for mask in group_masks])
            mean_yerrs = np.column_stack([
                yerrs[:, mask].mean(axis=1) for mask in group_masks
            ])

            ax1.errorbar(
                xaxis_unique, mean_offsets,
                yerr=mean_yerrs, 
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
        ax1.set_xticks(xaxis_unique)
        ax1.set_ylim(ylims[idx_par])

    fig.savefig(f'shape_noise_errorbars_{xaxis_param}.png', dpi=150, bbox_inches='tight')
    plt.close()

