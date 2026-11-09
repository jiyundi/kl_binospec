import joblib
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde

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


def cosi_to_deg(cosi):
    return np.degrees(np.arccos(np.clip(cosi, 0.0, 1.0)))

def deg_to_cosi(i_deg):
    return np.cos(np.radians(i_deg))

def posterior_log_densities(slit_folder, pars_to_plot, grid):
    samples = np.loadtxt(f'{slit_folder}post.txt', dtype=str, ndmin=2)
    weights = samples[2:, 0].astype(float)
    log_densities = []

    for par in pars_to_plot:
        cols = [i for i, name in enumerate(samples[0]) if par in name]
        if len(cols) != 1:
            raise ValueError(f'{slit_folder}post.txt: expected one column for {par}')
        values = samples[2:, cols[0]].astype(float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        kde = gaussian_kde(values[valid], weights=weights[valid])
        log_densities.append(kde.logpdf(grid))

    return np.array(log_densities)

def geometric_mean_quantiles(log_densities, grid):
    # Average log densities across fits before exponentiating.
    log_geomean = log_densities.mean(axis=0)
    density = np.exp(log_geomean - log_geomean.max())
    cdf = cumulative_trapezoid(density, grid, initial=0)
    cdf /= cdf[-1]
    return np.interp([0.158655, 0.5, 0.841345], cdf, grid)


# ===========================================================================
# Main workflow
# ===========================================================================

if __name__ == "__main__":

    # User input: mock and runs folder
    plot_title_prefix = ''
    mock_folder = './mock_pkl_noisy_imgSNR40_specSNR05_idealLPF/pkl/'
    runs_folders = [
        'imgSNR40_specSNR30_idealLPF/',
        'imgSNR40_specSNR30/', 
        'imgSNR40_specSNR05_idealLPF/', 
        'imgSNR40_specSNR05/'
    ]
    pretty_folder_name = {
        'imgSNR40_specSNR05_idealLPF/': 
        r'Spec SNR = 5, constant line width', 

        'imgSNR40_specSNR05/':          
        r'Spec SNR = 5, real-like LPF', 

        'imgSNR40_specSNR30_idealLPF/': 
        r'Spec SNR = 30, constant line width',

        'imgSNR40_specSNR30/':          
        r'Spec SNR = 30, real-like LPF'
    }

    colors = ['crimson', 'lightcoral', 'purple', 'violet', 'black', 'royalblue', 'limegreen']

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
    latex_strs = {'g1': r'$g_1$', 'g2': r'$g_2$'}

    # User input: prior range and ylim
    posterior_grid = np.linspace(-0.2, 0.2, 201)
    ylims  = [(-0.17, 0.17), (-0.17, 0.17)] #

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

    # Calculate the geometric mean and error for every error bar
    grouped_quantiles = {}
    for runs_folder in runs_folders:
        slit_log_densities = np.array([
            posterior_log_densities(
                runs_folder + f'Slit_{slit_name:03d}/', pars_to_plot, posterior_grid
            )
            for slit_name in slit_names
        ])
        # The array below consists of:
        # *   axis=0: corresponds to one cosi (or theta_int)
        # *   axis=1: g1, g2
        # *   axis=2: q16, q50, q84 (percentiles)
        grouped_quantiles[runs_folder] = np.array([
            [
                geometric_mean_quantiles(slit_log_densities[mask, idx_par], posterior_grid)
                for idx_par in range(len(pars_to_plot))
            ]
            for mask in group_masks
        ])
        print(f'Finished reading {runs_folder}.')

    P23F6_gp = np.array([ # sin(i), bias(g+), bias(g+) + err
        [0.051, -0.008, 0.006],
        [0.155,  0.001, 0.017],
        [0.257, -0.009, 0.007],
        [0.361, -0.005, 0.014],
        [0.463, -0.014, 0.006],
        [0.567, -0.017, 0.019],
        [0.670, -0.016, 0.021], 
        [0.774, -0.008, 0.038],
        [0.877, -0.004, 0.049],
        [0.979,  0.020, 0.069]
    ])
    P23F6_gx = np.array([ # sin(i), bias(gx), bias(gx) + err
        [0.051,  0.003, 0.016],
        [0.155, -0.003, 0.015],
        [0.257,  0.033, 0.068],
        [0.361,  0.028, 0.062],
        [0.463,  0.008, 0.037],
        [0.567,  0.010, 0.058],
        [0.670,  0.033, 0.106], 
        [0.774,  0.016, 0.090],
        [0.877,  0.007, 0.092],
        [0.979, -0.019, 0.047]
    ])

    # ===============================================================
    # Option 1 - Error bars of g1/g2 vs cosi/theta_int: 
    # Match the grid used for the geometric-mean posterior plots.
    mock_truths = np.array([
        [info['fid_params']['shared_params'][par] for par in pars_to_plot]
        for info in data_infos
    ])

    group_truths = np.array([mock_truths[mask][0] for mask in group_masks])
    for xvalue, mask, truth in zip(xaxis_unique, group_masks, group_truths):
        if not np.allclose(mock_truths[mask], truth, rtol=0, atol=1e-10):
            raise ValueError(f'{xaxis_param}={xvalue:g}: mock shear truths differ')

    # Plot
    fig = plt.figure(figsize=(6, 3*len(pars_to_plot)), layout='constrained') # (length, height)
    # plt.subplots_adjust(hspace=0.4, wspace=0.2) # h=height
    gs = fig.add_gridspec(nrows=len(pars_to_plot), ncols=1, 
                        height_ratios=[1]*len(pars_to_plot), 
                        width_ratios=[1]*1)
            
    for idx_par, par_to_plot in enumerate(pars_to_plot):
        ax1 = fig.add_subplot(gs[idx_par, 0])
        
        ax1.set_ylabel(
            f'{plot_title_prefix}' + 
            r'$\leftangle$' + 
            latex_strs[par_to_plot] + r'$^{\rm fit}$' + 
            r' $-$ ' + 
            latex_strs[par_to_plot] + r'$^{\rm true}$' + 
            r'$\rightangle$',
            fontsize=14
            )
        ax1.set_xlabel(xaxis_label, labelpad=0)

        ax1.axhline(y=0, color='forestgreen', linestyle='-', linewidth=1.5)
        ax1.grid(linestyle=':', alpha=0.5)

        for run_num, (folder_name, quantiles) in enumerate(grouped_quantiles.items()):
            q16, medians, q84 = quantiles[:, idx_par, :].T
            offsets = medians - group_truths[:, idx_par]
            yerrs = np.array([medians - q16, q84 - medians])
            color = colors[run_num % len(colors)]

            ax1.errorbar(
                xaxis_unique, offsets,
                yerr=yerrs, 
                fmt='-' if run_num % 2 == 0 else ':', 
                capsize   =1+2*run_num, 
                capthick  =2-0.5*run_num, 
                linewidth =1 if run_num % 2 == 0 else 3,
                elinewidth=2-0.25*run_num, 
                markersize=6-1.5*run_num, 
                marker='o', 
                color=color, alpha=1.0, 
                label=pretty_folder_name[folder_name]
                )

            ax1.text(0.01, 0.01 + run_num * 0.05, 
                    f'Geometric mean offset = {offsets.mean():.2f}   '+
                    f'Avg error = {yerrs.mean():.2f}', 
                    fontsize=8, color=color, ha='left', va='bottom', 
                    transform=ax1.transAxes)

        if xaxis_param == 'cosi':
            ax1.text(
                0.99, 0.01, r'More face-on $\rightarrow$', 
                color='gray', ha='right', va='bottom', 
                transform=ax1.transAxes
            )

            # Secondary axis: i (inclination angle; deg)
            ax1.set_xlim(0.0, 1.0)  # 让上轴显示完整的 90° 到 0°
            ax_top = ax1.secondary_xaxis(
                'top', functions=(cosi_to_deg, deg_to_cosi)
            )
            ax_top.set_xlabel(r'Inclination angle $i$')
            ax_top.set_xticks(cosi_to_deg(xaxis_unique))
            ax_top.xaxis.set_major_formatter('{x:.1f}°')

            # Compare with Pranjal+2023 Figure 6
            if par_to_plot == 'g1':
                ax1.fill_between(
                    x=(1 - P23F6_gp[:, 0]**2)**0.5, 
                    y1=P23F6_gp[:, 2], 
                    y2=(2 * P23F6_gp[:, 1] - P23F6_gp[:, 2]), 
                    color='gray', alpha=0.2, 
                    label=r'Bias and error for 2-spec case from P23'
                )
            elif par_to_plot == 'g2':
                ax1.fill_between(
                    x=(1 - P23F6_gx[:, 0]**2)**0.5, 
                    y1=P23F6_gx[:, 2], 
                    y2=(2 * P23F6_gx[:, 1] - P23F6_gx[:, 2]), 
                    color='gray', alpha=0.2, 
                    label=r'Bias and error for 2-spec case from P23'
                )

        if idx_par == 0:
            ax1.legend(prop={'size': 8}, loc='upper right')
        
        ax1.set_xticks(xaxis_unique)
        ax1.set_ylim(ylims[idx_par])

    fig.savefig(f'shear_bias_vs_{xaxis_param}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ===============================================================
    # Option 2 - Shear uncertainty vs. inclination
    if xaxis_param == 'cosi':
        shear_cols = [pars_to_plot.index(par) for par in ('g1', 'g2')]

        # Each array has shape (inclination groups, shear components).
        half_widths_by_run = {
            folder: (quantiles[:, shear_cols, 2] - quantiles[:, shear_cols, 0]) / 2
            for folder, quantiles in grouped_quantiles.items()
        }

        def add_inclination_axis(ax):
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks(xaxis_unique)
            ax_top = ax.secondary_xaxis(
                'top', functions=(cosi_to_deg, deg_to_cosi)
            )
            ax_top.set_xlabel(r'Inclination $i$')
            ax_top.set_xticks(cosi_to_deg(xaxis_unique))
            ax_top.xaxis.set_major_formatter('{x:.1f}°')

        # -----------------------------------------------------------
        # 2A - Combined shape noise (RMS across g1 and g2)
        fig_eps, ax_eps = plt.subplots(figsize=(5, 4))

        # Plot Pranjal's error bars
        gp_errs = P23F6_gp[:, 2] - P23F6_gp[:, 1]
        gx_errs = P23F6_gx[:, 2] - P23F6_gx[:, 1]
        ax_eps.plot(
            (1 - P23F6_gp[:, 0]**2)**0.5, 
            np.sqrt((gp_errs**2 + gx_errs**2) / 2), 
            color='gray', marker='s', linestyle='--', alpha=0.5, 
            label=r'2-spec case from P23'
        )

        for run_num, (folder, half_widths) in enumerate(half_widths_by_run.items()):
            epsilon_err = np.sqrt(np.mean(half_widths**2, axis=1))
            ax_eps.plot(
                xaxis_unique, epsilon_err, marker='o', 
                linestyle='-' if run_num % 2 == 0 else ':',
                color=colors[run_num % len(colors)], 
                label=pretty_folder_name[folder],
            )

        add_inclination_axis(ax_eps)
        ax_eps.set_xlabel(xaxis_label)
        ax_eps.set_ylabel(r'Shape noise $\sigma_\epsilon = \sqrt{(\sigma^2_{g1} + \sigma^2_{g2})/2}$')
        ax_eps.set_ylim(bottom=0, top=0.11)
        ax_eps.grid(linestyle=':', alpha=0.5)
        ax_eps.legend(fontsize=8)
        fig_eps.savefig('shape_noise_vs_cosi.png', dpi=150, bbox_inches='tight')
        plt.close(fig_eps)

        # -----------------------------------------------------------
        # 2B - Separate shape noises for g1 and g2
        fig_width, width_axes = plt.subplots(
            nrows=2, sharex=True, sharey=True, figsize=(5, 5),
            layout='constrained',
        )

        # Plot Pranjal's error bars
        width_axes[0].plot(
            (1 - P23F6_gp[:, 0]**2)**0.5, 
            (P23F6_gp[:, 2] - P23F6_gp[:, 1]), 
            color='gray', marker='s', linestyle='--', alpha=0.5, 
            label=r'2-spec case from P23'
        )
        width_axes[1].plot(
            (1 - P23F6_gx[:, 0]**2)**0.5, 
            (P23F6_gx[:, 2] - P23F6_gx[:, 1]), 
            color='gray', marker='s', linestyle='--', alpha=0.5, 
            label=r'2-spec case from P23'
        )

        for component_idx, ax in enumerate(width_axes):
            for run_num, (folder, half_widths) in enumerate(half_widths_by_run.items()):
                ax.plot(
                    xaxis_unique, half_widths[:, component_idx],
                    marker='o', 
                    linestyle='-' if run_num % 2 == 0 else ':',
                    color=colors[run_num % len(colors)], 
                    label=pretty_folder_name[folder],
                )

            ax.set_ylabel(
                rf'Shape noise $\sigma$('+latex_strs[f"g{component_idx+1}"]+')'
            )
            ax.grid(linestyle=':', alpha=0.5)

        add_inclination_axis(width_axes[0])
        width_axes[0].set_ylim(bottom=0, top=0.11)
        width_axes[0].legend(fontsize=7)
        width_axes[-1].set_xlabel(xaxis_label)
        fig_width.savefig(
            'shape_noise_g1g2_vs_cosi.png', dpi=150, bbox_inches='tight',
        )
        plt.close(fig_width)

    else:
        print('Skipped the shear uncertainty plots vs. cosi.')

    # # ===============================================================
    # # Option 3 - Sanity check: 
    # # See possible bias from averaging theta_int/cosi over multiple cosi/theta_int's
    # from pathlib import Path
    # from scipy.stats import gaussian_kde
    # from scipy.integrate import trapezoid

    # # 必须改成实际使用的均匀剪切先验区间
    # prior_bounds = {'g1': (-0.2, 0.2), 'g2': (-0.2, 0.2)}

    # for runs_folder in runs_folders:
    #     output_dir = Path(runs_folder) / 'posterior_groups'
    #     output_dir.mkdir(parents=True, exist_ok=True)

    #     for par in pars_to_plot:
    #         if par not in prior_bounds:
    #             continue

    #         grid = np.linspace(*prior_bounds[par], 2001)

    #         for xvalue, group_mask in zip(xaxis_unique, group_masks):
    #             members = np.flatnonzero(group_mask)
    #             truths = np.array([
    #                 data_infos[s]['fid_params']['shared_params'][par]
    #                 for s in members
    #             ])

    #             print(f"Plotting for all {par}'s posterior at {xaxis_param} = {xvalue}...")

    #             # 原始后验相乘要求这些模拟具有相同的真实剪切
    #             if not np.allclose(truths, truths[0], rtol=0, atol=1e-10):
    #                 raise ValueError(
    #                     f'{par}, {xaxis_param}={xvalue}: mock truths differ'
    #                 )

    #             fig, ax = plt.subplots(figsize=(5, 4))
    #             log_product = np.zeros_like(grid)

    #             idx_s = 0
    #             for s in members:
    #                 idx_s += 1
    #                 slit_folder = (
    #                     Path(runs_folder) / f'Slit_{slit_names[s]:03d}'
    #                 )
    #                 samples = np.loadtxt(
    #                     slit_folder / 'post.txt', dtype=str
    #                 )

    #                 # 沿用现有脚本的参数列匹配方式，并检查唯一性
    #                 cols = [
    #                     i for i, name in enumerate(samples[0])
    #                     if par in name
    #                 ]
    #                 if len(cols) != 1:
    #                     raise ValueError(
    #                         f'{slit_folder}: ambiguous column for {par}'
    #                     )

    #                 weights = samples[2:, 0].astype(float)
    #                 values = samples[2:, cols[0]].astype(float)
    #                 valid = (
    #                     np.isfinite(values)
    #                     & np.isfinite(weights)
    #                     & (weights > 0)
    #                 )

    #                 kde = gaussian_kde(
    #                     values[valid], weights=weights[valid]
    #                 )
    #                 log_pdf = kde.logpdf(grid)
    #                 log_product += log_pdf

    #                 # 每条曲线在绘图区间内归一化
    #                 pdf = np.exp(log_pdf - log_pdf.max())
    #                 # pdf /= trapezoid(pdf, grid)

    #                 params = data_infos[s]['fid_params']['shared_params']
    #                 if xaxis_param == 'cosi':
    #                     label = (
    #                         rf'$p({par}|D_{idx_s})$: '
    #                         rf'$\theta_{{\rm int}}='
    #                         rf'{np.rad2deg(params["theta_int"]):g}^\circ$'
    #                     )
    #                 else:
    #                     label = (
    #                         rf'$p({par}|D_{idx_s})$: '
    #                         rf'$\cos(i)={params["cosi"]:g}$'
    #                     )

    #                 ax.plot(grid, pdf, lw=1.5, alpha=0.75, label=label, 
    #                         ls=('--' if idx_s%2==0 else ':'))

    #             log_geomean = log_product / len(members) # take n-th root
    #             product = np.exp(log_geomean - log_geomean.max()) # peak normalization
    #             # product /= trapezoid(product, grid)

    #             ax.plot(
    #                 grid, product, color='black', lw=2.5,
    #                 label='Multiplied '+rf'$p({par}|D_1, ..., D_{idx_s})$'
    #             )

    #             # Find the peak's location and spread
    #             x_peak = grid[np.argmax(product)]
    #             p = product / trapezoid(product, grid) # 临时转成面积归一化，用于计算；不改变原来的 product
    #             mu = trapezoid(grid * p, grid)
    #             sigma = np.sqrt(trapezoid((grid - mu)**2 * p, grid))

    #             ax.text(-0.2, 0,
    #                 rf'$\mu={x_peak:.3f},\ \sigma={sigma:.3f}$',
    #                 ha='left', va='bottom', fontsize=8,
    #                 color='black'
    #             )

    #             ax.axvline(
    #                 truths[0], color='forestgreen', ls='-', lw=2, 
    #                 label='Mock truth'
    #             )
    #             ax.set(
    #                 xlabel=f'{par}',
    #                 ylabel='Peak-normalized posterior probability '+r'$p_i/\mathrm{max}(p_i)$',
    #                 title=f'{xaxis_param} = {xvalue:g}'
    #             )
    #             ax.legend(fontsize=8)
    #             fig.tight_layout()
    #             fig.savefig(
    #                 output_dir / f'{par}_{xaxis_param}_{xvalue:g}.png',
    #                 dpi=150, bbox_inches='tight'
    #             )
    #             plt.close(fig)
        
    #     print(f'Finished the plots in {output_dir}')

    print('Done.')
