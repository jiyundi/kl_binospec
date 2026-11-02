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

    # User input: mock and runs folder
    mock_folder = './mock_pkl_noisy_imgSNR40_specSNR30/pkl/'
    runs_folders = ['runs_noisy_imgSNR40_specSNR30/', 'runs_noisy_imgSNR40_specSNR05/'] # 'runs_no_noise/', 
    colors = ['crimson', 'darkorange', 'black', 'royalblue', 'limegreen']

    # User input: Specify x-axis: 'cosi' or 'theta_int'
    xaxis_param = 'cosi' # 'theta_int', 

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
    print('Mock data reading finished.')

    # User input: which params
    pars_to_plot = ['g1', 'g2']

    # User input: ylim
    ylims  = [(-0.12, 0.12), (-0.12, 0.12)] #

    # Check if cosi and theta_int in mock files
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

    # # ===============================================================
    # trues_to_plot = np.array(
    #     [
    #         [
    #             data_infos[s-1]['fid_params']['shared_params'][par] 
    #             for s in slit_names
    #         ] for par in pars_to_plot
    #     ]
    #     )

    # # Find y-axis values
    # post_1sigma_ranges = {}
    # for runs_folder in runs_folders:
    #     this_folder_post_1sigma_ranges = []
    #     for slit_name in slit_names:
    #         slit_folder = runs_folder + f'Slit_{slit_name:03d}/'
    #         this_folder_post_1sigma_ranges.append(
    #             analyze_percentile(slit_folder, pars_to_plot)
    #             )
    #     post_1sigma_ranges[runs_folder] = this_folder_post_1sigma_ranges

    # # Plot
    # fig = plt.figure(figsize=(6, 3*len(pars_to_plot)))  # (length, height)
    # plt.subplots_adjust(hspace=0.4, wspace=0.2) # h=height
    # gs = fig.add_gridspec(nrows=len(pars_to_plot), ncols=1, 
    #                     height_ratios=[1]*len(pars_to_plot), 
    #                     width_ratios=[1]*1)
            
    # for idx_par, par_to_plot in enumerate(pars_to_plot):
    #     ax1 = fig.add_subplot(gs[idx_par, 0])
    #     ax1.set_title(par_to_plot, fontsize=12)
        
    #     ax1.set_ylabel('Fit - Mock param')
    #     ax1.set_xlabel(xaxis_label, labelpad=0)

    #     ax1.axhline(y=0, color='forestgreen', linestyle='--', linewidth=2)
    #     ax1.grid(linestyle=':', color='black', alpha=0.5)

    #     # for each run folders
    #     run_num = 0
    #     for run_num, (folder_name, ranges_all_xs) in enumerate(post_1sigma_ranges.items()):
    #         medians = np.array([ranges_this_x[par_to_plot][0] for ranges_this_x in ranges_all_xs])
    #         err_los = np.array([ranges_this_x[par_to_plot][1] for ranges_this_x in ranges_all_xs])
    #         err_his = np.array([ranges_this_x[par_to_plot][2] for ranges_this_x in ranges_all_xs])
    #         yerrs   = np.array([-err_los, err_his])
    #         color   = colors[run_num % len(colors)]

    #         # Special periodic correction for theta_int
    #         if par_to_plot == 'theta_int':
    #             for t in range(len(medians)):
    #                 if medians[t] > 6: medians[t] -= 2*np.pi

    #         # Average Fit - Mock and each side of the error bar separately.
    #         # These are mean individual uncertainties, not errors on the mean.
    #         offsets = medians - trues_to_plot[idx_par]
    #         mean_offsets = np.array([offsets[mask].mean() for mask in group_masks])
    #         mean_yerrs = np.column_stack([
    #             yerrs[:, mask].mean(axis=1) for mask in group_masks
    #         ])

    #         ax1.errorbar(
    #             xaxis_unique, mean_offsets,
    #             yerr=mean_yerrs, 
    #             fmt=':', 
    #             capsize   =5-1.5*run_num, 
    #             capthick  =2-0.5*run_num, 
    #             linewidth =3-0.5*run_num,
    #             elinewidth=2-0.5*run_num, 
    #             markersize=6-1.5*run_num, 
    #             marker='o', color=color, alpha=1.0, 
    #             label=folder_name
    #             )

    #         ax1.text(0.01, 0.01 + run_num * 0.05, 
    #                 f'Avg offset = {np.mean(medians - trues_to_plot[idx_par]):.2f}   '+
    #                 f'Avg error = {np.mean(yerrs.flatten()):.2f}', 
    #                 fontsize=8, color=color, ha='left', va='bottom', 
    #                 transform=ax1.transAxes)

    #     ax1.legend(prop={'size': 10}, loc='upper right')
    #     ax1.set_xticks(xaxis_unique)
    #     ax1.set_ylim(ylims[idx_par])

    # fig.savefig(f'shape_noise_errorbars_{xaxis_param}.png', dpi=150, bbox_inches='tight')
    # plt.close()


    # ===============================================================
    # Optional - Sanity check: 
    # See possible bias from averaging theta_int/cosi over multiple cosi/theta_int's
    from pathlib import Path
    from scipy.stats import gaussian_kde
    from scipy.integrate import trapezoid

    # 必须改成实际使用的均匀剪切先验区间
    prior_bounds = {'g1': (-0.2, 0.2), 'g2': (-0.2, 0.2)}

    for runs_folder in runs_folders:
        output_dir = Path(runs_folder) / 'posterior_groups'
        output_dir.mkdir(parents=True, exist_ok=True)

        for par in pars_to_plot:
            if par not in prior_bounds:
                continue

            grid = np.linspace(*prior_bounds[par], 2001)

            for xvalue, group_mask in zip(xaxis_unique, group_masks):
                members = np.flatnonzero(group_mask)
                truths = np.array([
                    data_infos[s]['fid_params']['shared_params'][par]
                    for s in members
                ])

                print(f"Plotting for all {par}'s posterior at {xaxis_param} = {xvalue}...")

                # 原始后验相乘要求这些模拟具有相同的真实剪切
                if not np.allclose(truths, truths[0], rtol=0, atol=1e-10):
                    raise ValueError(
                        f'{par}, {xaxis_param}={xvalue}: mock truths differ'
                    )

                fig, ax = plt.subplots(figsize=(5, 4))
                log_product = np.zeros_like(grid)

                idx_s = 0
                for s in members:
                    idx_s += 1
                    slit_folder = (
                        Path(runs_folder) / f'Slit_{slit_names[s]:03d}'
                    )
                    samples = np.loadtxt(
                        slit_folder / 'post.txt', dtype=str
                    )

                    # 沿用现有脚本的参数列匹配方式，并检查唯一性
                    cols = [
                        i for i, name in enumerate(samples[0])
                        if par in name
                    ]
                    if len(cols) != 1:
                        raise ValueError(
                            f'{slit_folder}: ambiguous column for {par}'
                        )

                    weights = samples[2:, 0].astype(float)
                    values = samples[2:, cols[0]].astype(float)
                    valid = (
                        np.isfinite(values)
                        & np.isfinite(weights)
                        & (weights > 0)
                    )

                    kde = gaussian_kde(
                        values[valid], weights=weights[valid]
                    )
                    log_pdf = kde.logpdf(grid)
                    log_product += log_pdf

                    # 每条曲线在绘图区间内归一化
                    pdf = np.exp(log_pdf - log_pdf.max())
                    # pdf /= trapezoid(pdf, grid)

                    params = data_infos[s]['fid_params']['shared_params']
                    if xaxis_param == 'cosi':
                        label = (
                            rf'$p({par}|D_{idx_s})$: '
                            rf'$\theta_{{\rm int}}='
                            rf'{np.rad2deg(params["theta_int"]):g}^\circ$'
                        )
                    else:
                        label = (
                            rf'$p({par}|D_{idx_s})$: '
                            rf'$\cos(i)={params["cosi"]:g}$'
                        )

                    ax.plot(grid, pdf, lw=1.5, alpha=0.75, label=label, 
                            ls=('--' if idx_s%2==0 else ':'))

                log_geomean = log_product / len(members) # take n-th root
                product = np.exp(log_geomean - log_geomean.max()) # peak normalization
                # product /= trapezoid(product, grid)

                ax.plot(
                    grid, product, color='black', lw=2.5,
                    label='Multiplied '+rf'$p({par}|D_1, ..., D_{idx_s})$'
                )

                # Find the peak's location and spread
                x_peak = grid[np.argmax(product)]
                p = product / trapezoid(product, grid) # 临时转成面积归一化，用于计算；不改变原来的 product
                mu = trapezoid(grid * p, grid)
                sigma = np.sqrt(trapezoid((grid - mu)**2 * p, grid))

                ax.text(-0.2, 0,
                    rf'$\mu={x_peak:.3f},\ \sigma={sigma:.3f}$',
                    ha='left', va='bottom', fontsize=8,
                    color='black'
                )

                ax.axvline(
                    truths[0], color='forestgreen', ls='-', lw=2, 
                    label='Mock truth'
                )
                ax.set(
                    xlabel=f'{par}',
                    ylabel='Peak-normalized posterior probability '+r'$p_i/\mathrm{max}(p_i)$',
                    title=f'{xaxis_param} = {xvalue:g}'
                )
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(
                    output_dir / f'{par}_{xaxis_param}_{xvalue:g}.png',
                    dpi=150, bbox_inches='tight'
                )
                plt.close(fig)
        
        print(f'Finished the plots in {output_dir}')
