# import os
# os.chdir('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from pathlib import Path
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde

from data_prep.image.read_and_cutout import read_subaru_img_wcs, cutoutimg

from klm.safe_plot import setup; setup() # must before plt

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def lensing_efficiency(z_lens, z_source):
    """D_A(lens, source) / D_A(observer, source) in a flat cosmology."""
    if not np.isfinite(z_source) or z_source <= z_lens:
        raise ValueError(f'Source redshift {z_source} must exceed lens redshift {z_lens}')
    return (COSMO.angular_diameter_distance_z1z2(z_lens, z_source) /
            COSMO.angular_diameter_distance(z_source)).value


def weighted_quantiles(values, weights):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp([0.158655, 0.5, 0.841345], cdf, values)


def read_shear_posterior(path):
    with open(path, encoding='utf-8') as source:
        names = source.readline().split()
    columns = [names.index(name) for name in ('weight', 'shared_params-g1', 'shared_params-g2')]
    data = np.loadtxt(path, skiprows=1, usecols=columns, ndmin=2)
    weights, g1, g2 = data.T
    valid = np.isfinite(weights) & (weights > 0) & np.isfinite(g1) & np.isfinite(g2)
    if valid.sum() < 3:
        raise ValueError(f'{path}: too few valid weighted shear samples')
    return weights[valid], g1[valid], g2[valid]


def transformed_shear(weights, g1, g2, theta, factor):
    cos2, sin2 = np.cos(2 * theta), np.sin(2 * theta)
    return weights, -factor * (g1 * cos2 + g2 * sin2), factor * (g1 * sin2 - g2 * cos2)


def geometric_mean_posterior_quantiles(samples_and_weights):
    """Quantiles of the normalized geometric mean of slit posteriors."""
    values = np.concatenate([values for values, _ in samples_and_weights])
    low, high = np.min(values), np.max(values)
    if not high > low:
        raise ValueError('Posterior samples have no finite range')
    padding = 0.05 * (high - low)
    grid = np.linspace(low - padding, high + padding, 1001)
    log_density = np.zeros_like(grid)
    for slit_values, weights in samples_and_weights:
        log_density += gaussian_kde(slit_values, weights=weights).logpdf(grid)
    log_density /= len(samples_and_weights)
    density = np.exp(log_density - np.max(log_density))
    cdf = cumulative_trapezoid(density, grid, initial=0)
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError('Combined posterior cannot be normalized')
    return np.interp([0.158655, 0.5, 0.841345], cdf / cdf[-1], grid)


plt.style.use('default')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def shear_plot(arr_shears, cluster_centerRA, cluster_centerDEC, wcs=None, 
               image_size=None, fig_centerRA=None, fig_centerDEC=None, 
               plot_gt=False, plot_gx=False, savefig=True, posterior_by_slit=None):
    
    assert (plot_gt or plot_gx)
    
    common_kwargs = dict(
        arr_shears=arr_shears, 
        cluster_centerRA=cluster_centerRA, 
        cluster_centerDEC=cluster_centerDEC, 
        wcs=wcs, 
        image_size=image_size, 
        fig_centerRA=fig_centerRA, 
        fig_centerDEC=fig_centerDEC, 
        posterior_by_slit=posterior_by_slit,
    )
    
    if plot_gt and plot_gx:
        fig = plt.figure(figsize=(20, 12 if wcs is not None else 8), dpi=300)
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        gs = fig.add_gridspec(3 if wcs is not None else 2, 2,
                              height_ratios=[2, 1, 1] if wcs is not None else [1, 1],
                              width_ratios=[1, 1])
        
        ax1, ax2, ax3 = _draw_shear_column(fig, gs, col=0, kind='gt', **common_kwargs)
        
        plot_together = True
        if plot_together:
            common_kwargs['share_axs'] = (ax1, ax2, ax3)
            
        _draw_shear_column(fig, gs, col=1, kind='gx', **common_kwargs)
        
        if savefig:
            fig.savefig(Path(__file__).resolve().parent / "gt_gx_distribution.jpg",
                        dpi=300, bbox_inches='tight')
        
        return fig
    
    elif plot_gt:
        fig = plt.figure(figsize=(8, 16 if wcs is not None else 10), dpi=300)
        plt.subplots_adjust(hspace=0.2, wspace=0) # h=height
        gs = fig.add_gridspec(3 if wcs is not None else 2, 1,
                              height_ratios=[3,1,1] if wcs is not None else [1,1],
                              width_ratios=[1])
        
        _draw_shear_column(fig, gs, col=0, kind='gt', **common_kwargs)
        
        if savefig:
            fig.savefig(Path(__file__).resolve().parent / "gt_distribution.jpg",
                        dpi=300, bbox_inches='tight')
        
        return fig
    
    else: # plot_gx only
        fig = plt.figure(figsize=(8, 15 if wcs is not None else 10), dpi=300)
        plt.subplots_adjust(hspace=0.2, wspace=0) # h=height
        gs = fig.add_gridspec(3 if wcs is not None else 2, 1,
                              height_ratios=[3,1,1] if wcs is not None else [1,1],
                              width_ratios=[1])
        
        _draw_shear_column(fig, gs, col=0, kind='gx', **common_kwargs)
        
        if savefig:
            fig.savefig(Path(__file__).resolve().parent / "gx_distribution.jpg",
                        dpi=300, bbox_inches='tight')
        
        return fig


def _draw_shear_column(fig, gs, col, kind, 
                       arr_shears, cluster_centerRA, cluster_centerDEC, wcs, 
                       share_axs=None, 
                       image_size=None, fig_centerRA=None, fig_centerDEC=None,
                       posterior_by_slit=None):
    assert kind in ('gt', 'gx')
    
    if kind == 'gt':
        main_color  = 'firebrick' # 'darkcyan'
        ylabel_full = r'$g_+$'
        # ylabel_bin  = r'Binned $g_+$'
    else: # kind == 'gx'
        main_color  = 'mediumblue' # 'royalblue'
        ylabel_full = r'$g_\times$'
        # ylabel_bin  = r'Binned $g_\times$'
    
    if share_axs is not None:
        ax1, ax2, ax3 = share_axs
        while ax2.texts:
            ax2.texts[0].remove()
        while ax3.texts:
            ax3.texts[0].remove()
        
    else:
        ax1 = fig.add_subplot(gs[0, col], projection=wcs) if wcs is not None else None
        offset = 1 if wcs is not None else 0
        ax2 = fig.add_subplot(gs[offset, col])
        ax3 = fig.add_subplot(gs[offset + 1, col])
    
    # ax2_ymin, ax2_ymax = ax2.get_ylim()
    ax2_ymin, ax2_ymax = -0.30, 0.43
    ax2_xmin, ax2_xmax =     0, 16.5
    
    # Umetsu+14 CLASH (A383) WL profile - degitized
    U14_N_bkg_total = 7602 # (A383)
    U14_beta_avg = 0.79 # (A383)
    n_bins = 10
    R_min  =  0.9
    R_max  = 16.0
    delta_ln = (np.log(R_max) - np.log(R_min)) / n_bins
    ln_edges = np.log(R_min) + np.arange(n_bins + 1) * delta_ln # log-scale edges
    
    # Midpoint of linear edges
    lin_edges = np.exp(ln_edges)
    U14_Rs    = (lin_edges[1:] + lin_edges[:-1]) / 2
    U14_area_each_bin = (lin_edges[1:]**2 - lin_edges[:-1]**2) * np.pi
    
    # Mean surface number density (arcmin-2, A383), should = 9.3
    U14_surf_den_bkg   = U14_N_bkg_total / np.sum(U14_area_each_bin)
    U14_N_bkg_each_bin = U14_area_each_bin * U14_surf_den_bkg
    
    if share_axs is not None:
        U14_gts = np.array([0.265, 0.229, 0.120, 0.082, 0.070, 
                            0.067, 0.036, 0.000, 0.022, 0.029 ])
        U14_gts_hi    = np.array([0.381, 0.291,  0.162, 0.123, 0.097, 
                                  0.084, 0.050,  0.012, 0.031, 0.038])
        U14_gts_lo    = np.array([0.150, 0.169,  0.075, 0.044, 0.043, 
                                  0.046, 0.019, -0.012, 0.014, 0.021])
        U14_gts_CL_hi = np.array([0.338, 0.291,  0.183, 0.133, 0.084, 
                                  0.079, 0.043,  0.012, 0.031, 0.034])
        U14_gts_CL_lo = np.array([0.120, 0.171,  0.099, 0.062, 0.031, 
                                  0.044, 0.017, -0.010, 0.015, 0.022])
        ax2.fill_between(x=U14_Rs, 
                         y1=U14_gts_CL_hi, y2=U14_gts_CL_lo, 
                         color='mistyrose', alpha=0.75, zorder=0.1)
        ax3.fill_between(x=U14_Rs, 
                         y1=U14_gts_CL_hi, y2=U14_gts_CL_lo, 
                         color='mistyrose', alpha=0.75, zorder=0.1)
        ax2.errorbar(U14_Rs, U14_gts, 
                     yerr=[U14_gts_hi - U14_gts, 
                           U14_gts    - U14_gts_lo ], 
                     xerr=[U14_Rs        - lin_edges[:-1],
                           lin_edges[1:] - U14_Rs         ],
                     fmt='', capsize=3, capthick=1, elinewidth=1, 
                     ecolor=(0.5, 3/8, 3/8), marker='s', markersize=3, 
                     markerfacecolor=(0.5, 3/8, 3/8), 
                     markeredgecolor=(0.5, 3/8, 3/8), 
                     color=(0.5, 3/8, 3/8), lw=1, linestyle=':', 
                     label=r'$g_+$: Umetsu+14 (A383, $\langle \beta \rangle =$'+f'{U14_beta_avg}'+')', 
                     zorder=1.5)
        ax3.errorbar(U14_Rs, U14_gts, 
                     yerr=[U14_gts_hi - U14_gts, 
                           U14_gts    - U14_gts_lo ], 
                     xerr=[U14_Rs        - lin_edges[:-1],
                           lin_edges[1:] - U14_Rs         ],
                     fmt='', capsize=3, capthick=1, elinewidth=1, 
                     ecolor=(0.5, 3/8, 3/8), marker='s', markersize=3,  
                     markerfacecolor=(0.5, 3/8, 3/8), 
                     markeredgecolor=(0.5, 3/8, 3/8), 
                     color=(0.5, 3/8, 3/8), lw=1, linestyle=':', 
                     label=r'$g_+$: Umetsu+14 (A383, $\langle \beta \rangle =$'+f'{U14_beta_avg}'+')', 
                     zorder=1.5)
        for U14_R, U14_gt_lo, U14_N_bkg in \
            zip(U14_Rs, U14_gts_lo, U14_N_bkg_each_bin):
            ax3.text(U14_R, U14_gt_lo-0.01, f'({int(round(U14_N_bkg, 0))})', 
                     color=(0.5, 3/8, 3/8), fontsize=6, va='top', ha='center',
                     bbox=dict(
                        facecolor="white",  # Background color
                        edgecolor="none",       # Border color
                        alpha=0.9,              # Transparency (0 to 1)
                        boxstyle="round,pad=0.0" # Rounded corners with padding
                    ), 
                    zorder=0.5)
    
    # Pranjal+24 results (A2261) -- only overplotted on the tangential (gt) panel
    if (kind == 'gt') or (share_axs is not None):
        P24_names  = ['b007', 'b008', 'c007']
        P24_zs     = np.array([ 0.623,  0.585,  0.594])
        P24_Rs     = np.array([ 1.60 ,  5.05 ,  0.55 ])
        P24_gts    = np.array([ 0.208,  0.041,  0.144])
        P24_gterrs = np.array([ 0.020,  0.038,  0.030])
        U14_beta_2261 = U14_beta_avg # 0.70
        
        # Calibrate gt and gx
        P24_facs = np.array([])
        for i in range(len(P24_gts)):
            beta = lensing_efficiency(0.224, P24_zs[i])
            fac  = U14_beta_2261 / beta
            P24_facs = np.append(P24_facs, fac)
        
        for N_, R_, Gt_, Gte_, Fac_ \
            in zip(P24_names, P24_Rs, P24_gts, P24_gterrs, P24_facs):
            ax2.text(R_, (Gt_ + Gte_) * Fac_ + 0.01, N_, 
                     color='magenta', fontsize=8, zorder=4, ha='center')
            if share_axs is None:
                ax2.errorbar( R_, Gt_ * Fac_, yerr=Gte_ * Fac_, 
                             capsize=4, marker='^', markersize=1, 
                             color='magenta', zorder=0.75)
                ax3.errorbar( R_, Gt_ * Fac_, yerr=Gte_ * Fac_, 
                             capsize=4, marker='^', markersize=1, 
                             color='magenta', zorder=0.75)
                
        if share_axs is None:
            ax2.scatter(P24_Rs, P24_gts * P24_facs, 
                        marker='^', s=20, color='magenta', 
                        label=r'$g_+$: Pranjal+24 (A2261, rescaled to $\beta =$'+f'{U14_beta_avg}'+')', 
                        zorder=0.75)
            ax3.scatter(P24_Rs, P24_gts * P24_facs, 
                        marker='^', s=20, color='magenta', 
                        label=r'$g_+$: Pranjal+24 (A2261, $\rightarrow \beta =$'+f'{U14_beta_avg}'+')', 
                        zorder=0.75)
        
    # In this work...
    Rs, betas, facs, gts, gt_errs, gxs, gx_errs = [], [], [], [], [], [], []
    transformed_posteriors = []
    for i in range(1, len(arr_shears)):
        slitID, z, RA, DEC, R, theta_rad, \
            g1, g1_err, g2, g2_err, \
            gt, gt_err, gx, gx_err = arr_shears[i].astype(float)
        
        if ax1 is not None:
            ax1.scatter(RA, DEC, 
                        marker='o',
                        s=16, facecolors='none', edgecolors='darkcyan',
                        transform=ax1.get_transform('world'), zorder=4)
        
            ax1.text(RA-7/3600, DEC+7/3600, int(slitID), 
                     color='darkcyan', fontsize=8, zorder=5, 
                     transform=ax1.get_transform('world'))
        
            ax1.plot([RA,  cluster_centerRA ], 
                     [DEC, cluster_centerDEC], 
                     color='darkcyan', ls=':', alpha=0.23, 
                     transform=ax1.get_transform('world'), zorder=3)
        
            if kind == 'gt':
                if gt >= 0: 
                    add_an_arrow(ax1, RA, DEC,  gt*10000, theta_rad+np.pi*1/2, color='firebrick', lw=2, zorder=4)
                    add_an_arrow(ax1, RA, DEC,  gt*10000, theta_rad+np.pi*3/2, color='firebrick', lw=2, zorder=4, text=rf'${gt:.2f}$')
                else:
                    add_an_arrow(ax1, RA, DEC, -gt*10000, theta_rad      , color='firebrick', lw=2, zorder=4, text=rf'${gt:.2f}$')
                    add_an_arrow(ax1, RA, DEC, -gt*10000, theta_rad+np.pi, color='firebrick', lw=2, zorder=4)
            else: # kind == 'gx'
                if gx >= 0: 
                    add_an_arrow(ax1, RA, DEC,  gx*10000, theta_rad+np.pi*1/4, color='mediumblue', lw=2, zorder=4)
                    add_an_arrow(ax1, RA, DEC,  gx*10000, theta_rad+np.pi*5/4, color='mediumblue', lw=2, zorder=4, text=rf'${gx:.2f}$')
                else:
                    add_an_arrow(ax1, RA, DEC, -gx*10000, theta_rad+np.pi*3/4, color='mediumblue', lw=2, zorder=4, text=rf'${gx:.2f}$')
                    add_an_arrow(ax1, RA, DEC, -gx*10000, theta_rad+np.pi*7/4, color='mediumblue', lw=2, zorder=4)
        
            if g1 >= 0: 
                add_an_arrow(ax1, RA, DEC,  g1*10000,         0, color='thistle', lw=1)
                add_an_arrow(ax1, RA, DEC,  g1*10000, np.pi*  1, color='thistle', lw=1, text=rf'${g1:.2f}$')
            else:
                add_an_arrow(ax1, RA, DEC, -g1*10000, np.pi*1/2, color='thistle', lw=1, text=rf'${g1:.2f}$')
                add_an_arrow(ax1, RA, DEC, -g1*10000, np.pi*3/2, color='thistle', lw=1)
            if g2 >= 0: 
                add_an_arrow(ax1, RA, DEC,  g2*10000, np.pi*1/4, color='paleturquoise', lw=1, text=rf'${g2:.2f}$')
                add_an_arrow(ax1, RA, DEC,  g2*10000, np.pi*5/4, color='paleturquoise', lw=1)
            else:
                add_an_arrow(ax1, RA, DEC, -g2*10000, np.pi*3/4, color='paleturquoise', lw=1, text=rf'${g2:.2f}$')
                add_an_arrow(ax1, RA, DEC, -g2*10000, np.pi*7/4, color='paleturquoise', lw=1)
        
        # Calibrate gt and gx
        beta = lensing_efficiency(0.1883, z)
        fac  = U14_beta_avg / beta
        
        Rs.append(R); betas.append(beta); facs.append(fac)
        gts.append(gt * fac); gt_errs.append(gt_err * fac)
        gxs.append(gx * fac); gx_errs.append(gx_err * fac)
        if posterior_by_slit is not None:
            weights, g1_samples, g2_samples = posterior_by_slit[int(slitID)]
            transformed_posteriors.append(
                transformed_shear(weights, g1_samples, g2_samples, theta_rad, fac))
        
        # add texts
        y_val = gt * fac if kind == 'gt' else gx * fac
        
        ax2.text(R+1/8, y_val, int(slitID), 
                 color='black', fontsize=8, va='center', zorder=5)
    
    print(f'[INFO] Shears {kind} have been rescaled by factors of: \n'+
          f'{[round(fac, 2) for fac in facs]} \nfor Slits \n'+
          f'{[int(s) for s in arr_shears[1:, 0].astype(float)]}')
    
    y_vals = gts     if kind == 'gt' else gxs
    y_errs = gt_errs if kind == 'gt' else gx_errs
    marker = '^'     if kind == 'gt' else 'o'
    ax2.errorbar(Rs, y_vals, yerr=y_errs, 
                 fmt=' ', capsize=4, capthick=1, elinewidth=1, 
                 ecolor=main_color, marker=marker, markersize=4, 
                 markerfacecolor='white', markeredgecolor=main_color, 
                 zorder=2)
    
    # Label objects, plot them at once
    y_vals_all = gts if kind == 'gt' else gxs
    marker = '^'     if kind == 'gt' else 'o'
    ax2.scatter(Rs, y_vals_all, 
                marker=marker, s=16, 
                facecolors='white', edgecolors=main_color, zorder=2, 
                label=ylabel_full+r': Binospec (A383, $\beta =$'+f'{U14_beta_avg}'+')')
    
    # Error information
    if share_axs is not None:
        ax2.text(0.01, 0.01, 
                 r'Arithmetic error $\sigma_{\langle g_+ \rangle}$ = '+f'{np.mean(np.array(gt_errs)):.3f}\n'+
                 r'Arithmetic $\langle g_\times \rangle$ = '+f'{np.mean(np.array(gxs)):.3f}'+
                 r' $\pm$ '+f'{np.mean(np.array(gx_errs)):.3f}',
                 fontsize=7, color='black', ha='left', va='bottom', 
                 transform=ax2.transAxes)
    elif kind == 'gt':
        ax2.text(0.01, 0.01, 
                 f'Arithmetic error = {np.mean(np.array(gt_errs)):.3f}', 
                 fontsize=7, color=main_color, ha='left', va='bottom', 
                 transform=ax2.transAxes)
    elif kind == 'gx':
        ax2.text(0.01, 0.01, 
                 f'Arithmetic offset = {np.mean(np.array(gxs)):.3f}   \n'+
                 f'Arithmetic error = { np.mean(np.array(gx_errs)):.3f}', 
                 fontsize=7, color=main_color, ha='left', va='bottom', 
                 transform=ax2.transAxes)
    
    # initial binned dict
    binsize = 2
    groups = {}
    for r in Rs: 
        r_low = int(r // binsize) * binsize
        groups[f'{r_low}'] = []
    
    # count gt/gx in bins
    for i, (r, gt, gt_err, gx, gx_err) in enumerate(zip(Rs, gts, gt_errs, gxs, gx_errs)):
        r_low = int(r // binsize) * binsize
        posterior = transformed_posteriors[i] if posterior_by_slit is not None else None
        groups[f'{r_low}'].append((gt, gt_err, gx, gx_err, posterior))
    
    # calculate binned average
    binned_R_lows, binned_Nslits, binned_yvals, binned_yerrs = [], [], [], []
    binned_yerrs_lo, binned_yerrs_hi = [], []
    binned_gts, binned_gterrs, binned_gxs, binned_gxerrs = [], [], [], []
    binned_rows = []
    sorted_keys = sorted(groups.keys(), key=lambda x: float(x))
    for r_low in sorted_keys:
        vals_and_errs = groups[r_low]
        gts    = np.array([v[0] for v in vals_and_errs])
        gterrs = np.array([v[1] for v in vals_and_errs])
        gxs    = np.array([v[2] for v in vals_and_errs])
        gxerrs = np.array([v[3] for v in vals_and_errs])
        if posterior_by_slit is None:
            gt_weights = 1 / gterrs**2
            gx_weights = 1 / gxerrs**2
            gt_avg = np.average(gts, weights=gt_weights)
            gx_avg = np.average(gxs, weights=gx_weights)
            gterr_avg = np.sum(gt_weights)**(-0.5)
            gxerr_avg = np.sum(gx_weights)**(-0.5)
            gt_q = [gt_avg - gterr_avg, gt_avg, gt_avg + gterr_avg]
            gx_q = [gx_avg - gxerr_avg, gx_avg, gx_avg + gxerr_avg]
        else:
            gt_q = geometric_mean_posterior_quantiles([(v[4][1], v[4][0]) for v in vals_and_errs])
            gx_q = geometric_mean_posterior_quantiles([(v[4][2], v[4][0]) for v in vals_and_errs])
            gt_avg, gx_avg = gt_q[1], gx_q[1]
            gterr_avg = 0.5 * (gt_q[2] - gt_q[0])
            gxerr_avg = 0.5 * (gx_q[2] - gx_q[0])
        val_avg, err_avg = (gt_avg, gterr_avg) if kind == 'gt' else (gx_avg, gxerr_avg)
        q = gt_q if kind == 'gt' else gx_q
        
        binned_R_lows.append(int(r_low)); binned_Nslits.append(len(gts))
        binned_yvals.append(val_avg); binned_yerrs.append(err_avg)
        binned_yerrs_lo.append(q[1] - q[0]); binned_yerrs_hi.append(q[2] - q[1])
        binned_gts.append(gt_avg); binned_gterrs.append(gterr_avg)
        binned_gxs.append(gx_avg); binned_gxerrs.append(gxerr_avg)
        binned_rows.append({
            'radius_low_arcmin': int(r_low), 'radius_high_arcmin': int(r_low) + binsize,
            'n_slits': len(gts),
            'gt_median': gt_q[1], 'gt_err_minus': gt_q[1] - gt_q[0],
            'gt_err_plus': gt_q[2] - gt_q[1],
            'gx_median': gx_q[1], 'gx_err_minus': gx_q[1] - gx_q[0],
            'gx_err_plus': gx_q[2] - gx_q[1],
        })
    if posterior_by_slit is not None and (kind == 'gt' or share_axs is None):
        pd.DataFrame(binned_rows).to_csv(
            Path(__file__).resolve().parent / 'binned_shears.csv', index=False)
    
    # y_vals = gts    if kind == 'gt' else gxs
    # y_errs = gt_err if kind == 'gt' else gx_err
    marker = '^' if kind == 'gt' else 'o'
    # va = 'bottom' if kind == 'gt' else 'top'
    ax3.errorbar(np.array(binned_R_lows)+binsize/2, 
                 np.array(binned_yvals), 
                 yerr=np.array([binned_yerrs_lo, binned_yerrs_hi]), 
                 fmt='', capsize=5, capthick=1.5, elinewidth=1.5, 
                 ecolor=main_color, marker=marker, markersize=4, 
                 markerfacecolor='white', markeredgecolor=main_color, 
                 color=main_color, lw=0.5, linestyle='-', 
                 zorder=2)
    ax3.scatter(np.array(binned_R_lows)+binsize/2, 
                np.array(binned_yvals), 
                marker=marker, s=16, 
                facecolors='white', edgecolors=main_color, zorder=2, 
                label=ylabel_full+r': Binospec (A383, $\rightarrow \beta =$'+f'{U14_beta_avg}'+f'), bin size = {binsize}\'')
    
    for i in range(len(binned_R_lows)):
        if share_axs is None:
            ax3.text(binned_R_lows[i]+binsize/2, binned_yvals[i]+0.1, 
                     '('+r'$N_\mathrm{KL}$'+f' = {binned_Nslits[i]})',
                     fontsize=8, color='black', ha='center', va='bottom', 
                     zorder=4)
        else:
            ax3.text(binned_R_lows[i]+binsize/2, 
                     np.max([binned_gts[i], binned_gxs[i]])+0.1, 
                     '('+r'$N_\mathrm{KL}$'+f' = {binned_Nslits[i]})\n'+
                     f'{binned_gts[i] / binned_gterrs[i]:.1f}'+r'$\sigma$',
                     fontsize=8, color='firebrick', ha='center', va='bottom', 
                     zorder=4)
            ax3.text(binned_R_lows[i]+binsize/2, 
                     np.max([binned_gts[i], binned_gxs[i]])+0.1, 
                     f'{binned_gxs[i] / binned_gxerrs[i]:.1f}'+r'$\sigma$',
                     fontsize=8, color='mediumblue', ha='center', va='top', 
                     zorder=4)
    
    # Binned regions (e.g. 0--2, 2--4, 4--6, ...)
    R_low_max = np.max([binned_R_lows[i] for i in range(len(binned_R_lows))])
    for r_low in np.arange(0, R_low_max+2*binsize, 2*binsize):
        ax2.fill_between([r_low, r_low + binsize], 
                         ax2_ymin, ax2_ymax, 
                         color='gray', alpha=0.04)
        ax3.fill_between([r_low, r_low + binsize], 
                         ax2_ymin, ax2_ymax, 
                         color='gray', alpha=0.04)
    
    # Error information (binned)
    if share_axs is not None:
        weighted_binned = np.average(np.array(binned_gxs), weights=np.array(binned_gxerrs)**(-2))
        weighted_binned_err = (np.sum(np.array(binned_gxerrs)**(-2)))**(-0.5)
        ax3.text(0.01, 0.01, 
                 r'Arithmetic error $\sigma_{\langle g_+ \rangle}$ = '+f'{np.mean(np.array(binned_gterrs)):.3f} (binned)\n'+
                 r'Arithmetic $\langle g_\times \rangle$ = '+f'{np.mean(np.array(binned_gxs)):.3f}'+
                 r' $\pm$ '+f'{np.mean(np.array(binned_gxerrs)):.3f} (binned)\n'+
                 r'Weighted $\langle g_\times \rangle$ = '+f'{weighted_binned:.3f}'+
                 r' $\pm$ '+f'{weighted_binned_err:.3f} (binned)',
                 fontsize=7, color='black', ha='left', va='bottom', 
                 transform=ax3.transAxes)
    elif kind == 'gt':
        ax3.text(0.01, 0.01, 
                 f'Arithmetic error = {np.mean(np.array(binned_yerrs)):.3f} (binned)', 
                 fontsize=7, color=main_color, ha='left', va='bottom', 
                 transform=ax3.transAxes)
    elif kind == 'gx':
        ax3.text(0.01, 0.01, 
                 f'Arithmetic offset = {np.mean(np.array(binned_yvals)):.3f} (binned)\n'+
                 f'Arithmetic error = { np.mean(np.array(binned_yerrs)):.3f} (binned)', 
                 fontsize=7, color=main_color, ha='left', va='bottom', 
                 transform=ax3.transAxes)
    
    # R_500, R_200(=R_500/0.65)
    R_500 = 944 # kpc, Vikhlinin et al. (2006) arXiv:astro-ph/0507092, Table 4
    d_A   = scale_calc(0.1883, 70, 0.3) # kpc/arcsec
    r_500 = R_500 / d_A
    r_200 = r_500 / 0.65
    if ax1 is not None:
        draw_r500_r200(plt, ax1, r_500, r_200, color='limegreen')
    for ax in (ax2, ax3):
        ax.text(r_500/60+1/8, ax2_ymin+1/50, r'$r_{500,\mathrm{ A383}}$', color='limegreen', fontsize=12, zorder=5)
        ax.text(r_200/60+1/8, ax2_ymin+1/50, r'$r_{200,\mathrm{ A383}}$', color='limegreen', fontsize=12, zorder=5)
        ax.vlines(r_500/60, ymin=ax2_ymin, ymax=ax2_ymax, ls='--', color='limegreen',
                  zorder=0.3)
        ax.vlines(r_200/60, ymin=ax2_ymin, ymax=ax2_ymax, ls=':',  color='limegreen',
                  zorder=0.3)
    
    if ax1 is not None and image_size is not None: # image_size must in pixels
        assert (fig_centerRA is not None) & (fig_centerDEC is not None)
        
        x0, y0 = ax1.wcs.world_to_pixel_values(fig_centerRA, fig_centerDEC)
        ax1.set_xlim(x0-image_size/2, x0+image_size/2)
        ax1.set_ylim(y0-image_size/2, y0+image_size/2)
    
    if ax1 is not None:
        ax1.set_xlabel('RA (deg)')
        ax1.set_ylabel('DEC (deg)')
        ax1.minorticks_on()
        ax1.set_aspect(1)
    
    ax2.set_xlim(ax2_xmin, ax2_xmax)
    ax2.set_ylim(ax2_ymin, ax2_ymax)
    if kind == 'gx':
        ax2.axhline(y=0, linestyle='-', lw=1, color='gray', zorder=0)
    ax2.set_xlabel(r'$R$'+' (arcmin)', fontsize=15)
    ax2.set_ylabel(r'Reduced shear', fontsize=15)
    ax2.minorticks_on()
    ax2.legend(prop={'size': 7})
    ax2.grid(linestyle='-', axis='y', color='gray', alpha=0.25, zorder=0)
    
    ax3.set_xlim(ax2_xmin, ax2_xmax)
    ax3.set_ylim(ax2_ymin, ax2_ymax)
    if kind == 'gx':
        ax3.axhline(y=0, linestyle='-', lw=1, color='gray', zorder=0)
    ax3.set_xlabel(r'$R$'+' (arcmin)', fontsize=15)
    ax3.set_ylabel(r'Binned reduced shear', fontsize=15)
    ax3.minorticks_on()
    ax3.legend(prop={'size': 6})
    ax3.grid(linestyle='-', axis='y', color='gray', alpha=0.25, zorder=0)
    
    return ax1, ax2, ax3
 
 
def add_an_arrow(ax, x0, y0, length, angle_radians, 
                 color='black', lw=2, text='', zorder=3):
    x0, y0 = ax.wcs.world_to_pixel_values(x0, y0)
    
    # Calculate the end point of the arrow 
    # (assuming it points from the center point to the end point)
    x_end = x0 + length * np.cos(np.pi-angle_radians) # pixel
    y_end = y0 + length * np.sin(np.pi-angle_radians) # pixel
    
    # 使用 annotate 画箭头
    ax.text(x_end, y_end, text, size=4, color=color, zorder=zorder)
    ax.annotate('', 
                xy=(x_end, y_end),      # where the arrow points (the destination)
                xytext=(x0, y0),        # starting point/center point of the arrow
                arrowprops=dict(arrowstyle="->", # "->", "-|>", "<->"
                                mutation_scale=10,  # the size of the arrow tip
                                color=color, 
                                lw=lw),
                transform=ax.get_transform('world'), zorder=zorder
                )
    return


def draw_r500_r200(plt, ax, r_500, r_200, color='greenyellow'):
    circ_500 = plt.Circle(( (( 2)+(48)/60+( 3.3)/3600)*15,
                           -(( 3)+(31)/60+(46.4)/3600)   ), 
                          r_500/3600, 
                          color=color, fill=False, linestyle='--', # 'dashdot'
                          linewidth=1.2, zorder=1, 
                          transform=ax.get_transform('world'))
    circ_200 = plt.Circle(( (( 2)+(48)/60+( 3.3)/3600)*15,
                           -(( 3)+(31)/60+(46.4)/3600)   ), 
                          r_200/3600, 
                          color=color, fill=False, linestyle=':', 
                          zorder=1, 
                          transform=ax.get_transform('world'))
    ax.text( (( 2)+(48)/60+( 3.3)/3600)*15,
             -(( 3)+(31)/60+(46.4)/3600) - r_500/3600,
             r'$r_{\rm 500}$', color=color, size=18, 
             verticalalignment='top', 
             transform=ax.get_transform('world'))
    ax.text( (( 2)+(48)/60+( 3.3)/3600)*15,
             -(( 3)+(31)/60+(46.4)/3600) - r_200/3600,
             r'$r_{\rm 200}$', color=color, size=18, 
             verticalalignment='top', 
             transform=ax.get_transform('world'))
    ax.add_patch(circ_500)
    ax.add_patch(circ_200)
    return circ_500, circ_200


def scale_calc(z, H0, Omega_M):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0, Om0=Omega_M) # a selected cosmological model
    d_A = cosmo.kpc_proper_per_arcmin(z)
    d_A = d_A.value/60 # kpc/arcmin --> kpc/arcsec
    return d_A


def proper_a_d_distance_calc(z, H0, Omega_M):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_M)
    d_A = cosmo.angular_diameter_distance(z) # Mpc
    return d_A.value


if __name__ == '__main__':
    # img_dir  = '../../RSCH3/HSC_img_A383/'
    # sci_fn_G = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz.fits'
    spec1d2dfolder02 = '../../RSCH3/UAO-S156-23B-A383/psf/231019/1d2dspecfiles/'
    
    # Read shear catalog
    df     = pd.read_excel("shear_table_20261120.xlsx", header=None, engine='openpyxl')
    array  = df.to_numpy()
    
    headers = {name: i for i, name in enumerate(array[1]) if isinstance(name, str)}
    if all(name in headers for name in ('skyline_status', 'g1_status', 'g2_status', 'fit_status')):
        mask_good_g1g2 = (
            (array[2:, headers['skyline_status']] == 'ok') &
            (array[2:, headers['g1_status']] == 'ok') &
            (array[2:, headers['g2_status']] == 'ok') &
            (array[2:, headers['fit_status']] == 'ok'))
    else:
        # Older catalogs encoded quality in the error signs and magnitude.
        g1err1 = np.array(array[2:, 21], dtype=float)
        g1err2 = np.array(array[2:, 22], dtype=float)
        g2err1 = np.array(array[2:, 25], dtype=float)
        g2err2 = np.array(array[2:, 26], dtype=float)
        mask_good_g1 = ~np.isnan(g1err1) & (g1err1 < 0) & (g1err2 > 0) & (np.abs(g1err1) <= 1e3)
        mask_good_g2 = ~np.isnan(g2err1) & (g2err1 < 0) & (g2err2 > 0) & (np.abs(g2err1) <= 1e3)
        mask_good_g1g2 = mask_good_g1 & mask_good_g2
    
    # slits_to_delete = [ 57, 135,  42,  58, 141,  49,  79]
    # mask_deleted = [(num not in slits_to_delete) for num in array[2:, 0].astype(int)]
    
    g_cat_good_g1g2 = array[np.r_[True, True,  mask_good_g1g2]]
    
    cluster_centerRA  =  ( ( 2)+(48)/60+( 3.3)/3600 )*15
    cluster_centerDEC = -( ( 3)+(31)/60+(46.4)/3600 )
    
    columns    = ['slitID', 'z', 'RA', 'DEC', 'radius', 'theta_rad', 
                  'g1_median', 'g1_err', 'g2_median', 'g2_err', 
                  'gt',      'gt_err', 'gx',      'gx_err']
    arr_shears = np.zeros((1, len(columns))).astype(str)
    arr_shears[0] = np.array(columns)
    posterior_by_slit = {}
    
    for i in range(len(g_cat_good_g1g2[2:, 0])):
        # Calculate galaxy's position angle w.r.t. CLUSTER CENTER
        slit_id = g_cat_good_g1g2[2+i,  0]
        z       = np.nanmean(g_cat_good_g1g2[2+i, [5, 8, 11]])
        RA, DEC = g_cat_good_g1g2[2+i, 15], g_cat_good_g1g2[2+i, 16]
        post_path = (Path(__file__).resolve().parent.parent / 'scripts' /
                     f'Slit_{int(slit_id):03d}' / 'post.txt')
        weights, g1_samples, g2_samples = read_shear_posterior(post_path)
        posterior_by_slit[int(slit_id)] = (weights, g1_samples, g2_samples)
        g1_q = weighted_quantiles(g1_samples, weights)
        g2_q = weighted_quantiles(g2_samples, weights)
        g1, g2 = g1_q[1], g2_q[1]
        g1_err = 0.5 * (g1_q[2] - g1_q[0])
        g2_err = 0.5 * (g2_q[2] - g2_q[0])
        
        # Polar coordinates
        R = 60 * np.sqrt((RA - cluster_centerRA)**2 + 
                         (DEC - cluster_centerDEC)**2)
        theta_cl_rad = np.arctan2(DEC - cluster_centerDEC, 
                                  RA  - cluster_centerRA)
        
        _, gt_samples, gx_samples = transformed_shear(
            weights, g1_samples, g2_samples, theta_cl_rad, 1.0)
        gt_q = weighted_quantiles(gt_samples, weights)
        gx_q = weighted_quantiles(gx_samples, weights)
        gt, gx = gt_q[1], gx_q[1]
        gt_err = 0.5 * (gt_q[2] - gt_q[0])
        gx_err = 0.5 * (gx_q[2] - gx_q[0])
        
        arr_shears = np.append(arr_shears, 
                               [[slit_id, z, RA, DEC, R, theta_cl_rad, 
                                 g1, g1_err, g2, g2_err, 
                                 gt, gt_err, gx, gx_err]],
                               axis=0)
    
    # Save shear catalog
    import pandas as pd
    headers = arr_shears[0] 
    data    = arr_shears[1:]
    df = pd.DataFrame(data, columns=headers)
    df = df.apply(pd.to_numeric)
    df.iloc[:, 1]   = df.iloc[:, 1].round(5) # 5 digits
    df.iloc[:, 2:6] = df.iloc[:, 2:6].round(7) # 7 digits
    df.iloc[:, 6:]  = df.iloc[:, 6:].round(5) # 7 digits
    with open('arr_shears.txt', 'w') as f:
        f.write(df.to_string(index=False))
        
    def sort_arr_by_col(arr, col_idx=0, skip_row0=True):
        # If column titles at the 1st row
        if skip_row0:
            sorted_idx = np.append([0], np.argsort(arr[1:, col_idx].astype(float)) + 1)
        else:
            sorted_idx = np.argsort(arr[1:, col_idx].astype(float))
        return arr[sorted_idx]
    
    sorted_arr = sort_arr_by_col(arr_shears, col_idx=3)
    
    # Read WCS
    scale = 0.2 # arcsec/pix: 0.2 for image
    image_size = 25 # arcmin
    # img_data_G = read_subaru_img_wcs(img_dir+sci_fn_G, None)
    # imgG, wcsG = img_data_G['science_data'], img_data_G['science_wcs' ]
    
    # _, _, _, wcsGc = cutoutimg(
    #     imgG, wcsG, maskCenterRA, maskCenterDEC, 
    #     img_width=image_size*60/scale, img_height=image_size*60/scale, 
    #     outputWCS=True)
    wcsGc = None
    
    # tan_shear_plot(arr_shears, cluster_centerRA, cluster_centerDEC, wcsGc, 
    #                image_size*60/scale, maskCenterRA, maskCenterDEC)
    
    # cross_shear_plot(arr_shears, cluster_centerRA, cluster_centerDEC, wcsGc, 
    #                  image_size*60/scale, maskCenterRA, maskCenterDEC)
    
    shear_plot(arr_shears, cluster_centerRA, cluster_centerDEC, wcsGc,
               plot_gt=True, plot_gx=True, posterior_by_slit=posterior_by_slit)
