import os
os.chdir('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

from data_prep.image.read_and_cutout import read_subaru_img_wcs, cutoutimg

from klm.safe_plot import setup; setup() # must before plt
plt.style.use('default')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


def shear_plot(arr_shears, cluster_centerRA, cluster_centerDEC, wcs, 
               image_size=None, fig_centerRA=None, fig_centerDEC=None, 
               plot_gt=False, plot_gx=False, savefig=True):
    
    assert (plot_gt or plot_gx)
    
    common_kwargs = dict(
        arr_shears=arr_shears, 
        cluster_centerRA=cluster_centerRA, 
        cluster_centerDEC=cluster_centerDEC, 
        wcs=wcs, 
        image_size=image_size, 
        fig_centerRA=fig_centerRA, 
        fig_centerDEC=fig_centerDEC, 
    )
    
    if plot_gt and plot_gx:
        fig = plt.figure(figsize=(20, 12), dpi=300)
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        gs = fig.add_gridspec(3, 2, 
                              height_ratios=[2,1,1], 
                              width_ratios=[1,1])
        
        ax1, ax2, ax3 = _draw_shear_column(fig, gs, col=0, kind='gt', **common_kwargs)
        
        plot_together = True
        if plot_together:
            common_kwargs['share_axs'] = (ax1, ax2, ax3)
            
        _draw_shear_column(fig, gs, col=1, kind='gx', **common_kwargs)
        
        if savefig:
            plt.savefig("gt_gx_distribution.jpg", dpi=300, bbox_inches='tight')
        
        return fig
    
    elif plot_gt:
        fig = plt.figure(figsize=(8, 16), dpi=300)
        plt.subplots_adjust(hspace=0.2, wspace=0) # h=height
        gs = fig.add_gridspec(3, 1, 
                              height_ratios=[3,1,1], 
                              width_ratios=[1])
        
        _draw_shear_column(fig, gs, col=0, kind='gt', **common_kwargs)
        
        if savefig:
            plt.savefig("gt_distribution.jpg", dpi=300, bbox_inches='tight')
        
        return fig
    
    else: # plot_gx only
        fig = plt.figure(figsize=(8, 15), dpi=300)
        plt.subplots_adjust(hspace=0.2, wspace=0) # h=height
        gs = fig.add_gridspec(3, 1, 
                              height_ratios=[3,1,1], 
                              width_ratios=[1])
        
        _draw_shear_column(fig, gs, col=0, kind='gx', **common_kwargs)
        
        if savefig:
            plt.savefig("gx_distribution.jpg", dpi=300, bbox_inches='tight')
        
        return fig


def _draw_shear_column(fig, gs, col, kind, 
                       arr_shears, cluster_centerRA, cluster_centerDEC, wcs, 
                       share_axs=None, 
                       image_size=None, fig_centerRA=None, fig_centerDEC=None):
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
        ax1 = fig.add_subplot(gs[0, col], projection=wcs)
        ax2 = fig.add_subplot(gs[1, col])
        ax3 = fig.add_subplot(gs[2, col])
    
    # ax2_ymin, ax2_ymax = ax2.get_ylim()
    ax2_ymin, ax2_ymax = -0.30, 0.42
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
                         color='mistyrose', alpha=0.5, zorder=0.5)
        ax3.fill_between(x=U14_Rs, 
                         y1=U14_gts_CL_hi, y2=U14_gts_CL_lo, 
                         color='mistyrose', alpha=0.5, zorder=0.5)
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
                     label=r'$g_+$: Umetsu+14 (A383)', 
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
                     label=r'$g_+$: Umetsu+14 (A383)', 
                     zorder=1.5)
        for U14_R, U14_gt_lo, U14_N_bkg in \
            zip(U14_Rs, U14_gts_lo, U14_N_bkg_each_bin):
            # ax2.text(U14_R, U14_gt_lo, f'{int(round(U14_N_bkg, 0))}', 
            #          color=(0.5, 3/8, 3/8), fontsize=8, zorder=4, va='center')
            ax3.text(U14_R, U14_gt_lo-0.01, f'({int(round(U14_N_bkg, 0))})', 
                     color=(0.5, 3/8, 3/8), fontsize=6, va='top', ha='center',
                     zorder=4)
    
    # Pranjal+24 results (A2261) -- only overplotted on the tangential (gt) panel
    if (kind == 'gt') or (share_axs is not None):
        P24_names  = ['b007', 'b008', 'c007']
        P24_zs     = np.array([ 0.623,  0.585,  0.594])
        P24_Rs     = np.array([ 1.60 ,  5.05 ,  0.55 ])
        P24_gts    = np.array([ 0.208,  0.041,  0.144])
        P24_gterrs = np.array([ 0.020,  0.038,  0.030])
        P24_d_L    = proper_a_d_distance_calc(0.224, 70, 0.3)
        U14_beta_2261 = U14_beta_avg # 0.70
        
        # Calibrate gt and gx
        P24_facs = np.array([])
        for i in range(len(P24_gts)):
            d_S  = proper_a_d_distance_calc(P24_zs[i], 70, 0.3)
            d_LS = d_S - P24_d_L
            beta = d_LS / d_S
            fac  = U14_beta_2261 / beta
            P24_facs = np.append(P24_facs, fac)
        
        for N_, R_, Gt_, Gte_, Fac_ \
            in zip(P24_names, P24_Rs, P24_gts, P24_gterrs, P24_facs):
            # ax2.text(1/8+ R_, Gt_ * Fac_, N_, 
            #          color='magenta', fontsize=8, zorder=4, va='center')
            ax3.text(1/8+ R_, Gt_ * Fac_, N_, 
                     color='magenta', fontsize=8, zorder=4, va='center')
            if share_axs is None:
                ax2.errorbar( R_, Gt_ * Fac_, yerr=Gte_, 
                             capsize=4, marker='^', markersize=1, 
                             color='magenta', zorder=0.75)
                ax3.errorbar( R_, Gt_ * Fac_, yerr=Gte_ * Fac_, 
                             capsize=4, marker='^', markersize=1, 
                             color='magenta', zorder=0.75)
        if share_axs is None:
            ax2.scatter(P24_Rs, P24_gts * Fac_, 
                        marker='^', s=20, color='magenta', 
                        label=r'$g_+$: Pranjal+24 (A2261)', zorder=0.75)
            ax3.scatter(P24_Rs, P24_gts * Fac_, 
                        marker='^', s=20, color='magenta', 
                        label=r'$g_+$: Pranjal+24 (A2261)', zorder=0.75)
        
    # In this work...
    d_L  = proper_a_d_distance_calc(0.1883, 70, 0.3)
    Rs, betas, gts, gt_errs, gxs, gx_errs = [], [], [], [], [], []
    for i in range(1, len(arr_shears)):
        slitID, z, RA, DEC, R, theta_rad, \
            g1, g1_err, g2, g2_err, \
            gt, gt_err, gx, gx_err = arr_shears[i].astype(float)
        
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
        d_S  = proper_a_d_distance_calc(z, 70, 0.3)
        d_LS = d_S - d_L
        beta = d_LS / d_S
        fac  = U14_beta_avg / beta
        
        Rs.append(R); betas.append(beta)
        gts.append(gt * fac); gt_errs.append(gt_err * fac)
        gxs.append(gx * fac); gx_errs.append(gx_err * fac)
        
        # add texts
        y_val = gt * fac if kind == 'gt' else gx * fac
        
        ax2.text(R+1/8, y_val, int(slitID), 
                 color='black', fontsize=8, va='center', zorder=5)
    
    y_vals = gts    if kind == 'gt' else gxs
    y_errs = gt_err if kind == 'gt' else gx_err
    marker = '^'    if kind == 'gt' else 'o'
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
                label=ylabel_full+': Binospec (A383)')
    
    # Error information
    if share_axs is not None:
        ax2.text(0.01, 0.01, 
                 r'Arithmetic error $\sigma_{\langle g_+ \rangle}$ = '+f'{np.mean(np.array(gt_errs)):.3f}\n'+
                 r'Arithmetic $\langle g_\times \rangle$ = '+f'{np.mean(np.array(gxs)):.3f}'+
                 r' $\pm$ '+f'{np.mean(np.array(gx_errs)):.3f}',
                 fontsize=8, color='black', ha='left', va='bottom', 
                 transform=ax2.transAxes)
    elif kind == 'gt':
        ax2.text(0.01, 0.01, 
                 f'Arithmetic error = {np.mean(np.array(gt_errs)):.3f}', 
                 fontsize=9, color=main_color, ha='left', va='bottom', 
                 transform=ax2.transAxes)
    elif kind == 'gx':
        ax2.text(0.01, 0.01, 
                 f'Arithmetic offset = {np.mean(np.array(gxs)):.3f}   \n'+
                 f'Arithmetic error = { np.mean(np.array(gx_errs)):.3f}', 
                 fontsize=9, color=main_color, ha='left', va='bottom', 
                 transform=ax2.transAxes)
    
    # initial binned dict
    binsize = 2
    groups = {}
    for r in Rs: 
        r_low = int(r // binsize) * binsize
        groups[f'{r_low}'] = []
    
    # count gt/gx in bins
    for r, gt, gt_err, gx, gx_err \
        in zip(Rs, gts, gt_errs, gxs, gx_errs):
        r_low = int(r // binsize) * binsize
        groups[f'{r_low}'].append((gt, gt_err, gx, gx_err))
    
    # calculate binned average
    binned_R_lows, binned_Nslits, binned_yvals, binned_yerrs = [], [], [], []
    binned_gts, binned_gterrs, binned_gxs, binned_gxerrs = [], [], [], []
    for r_low in sorted(np.array(list(groups.keys())).astype(int)):
        vals_and_errs = groups[f'{int(r_low)}']
        gts    = np.array([v[0] for v in vals_and_errs])
        gterrs = np.array([v[1] for v in vals_and_errs])
        gxs    = np.array([v[2] for v in vals_and_errs])
        gxerrs = np.array([v[3] for v in vals_and_errs])
        gt_weights  = 1 / gterrs**2
        gx_weights  = 1 / gxerrs**2
        gt_avg  = np.average(gts, weights=gt_weights)
        gx_avg  = np.average(gxs, weights=gx_weights)
        gterr_avg  = (np.sum(gt_weights))**(-0.5)
        gxerr_avg  = (np.sum(gx_weights))**(-0.5)
        val_avg, err_avg = (gt_avg, gterr_avg) if kind == 'gt' else (gx_avg, gxerr_avg)
        
        binned_R_lows.append(int(r_low)); binned_Nslits.append(len(gts))
        binned_yvals.append(val_avg); binned_yerrs.append(err_avg)
        binned_gts.append(gt_avg); binned_gterrs.append(gterr_avg)
        binned_gxs.append(gx_avg); binned_gxerrs.append(gxerr_avg)
    
    # y_vals = gts    if kind == 'gt' else gxs
    # y_errs = gt_err if kind == 'gt' else gx_err
    marker = '^' if kind == 'gt' else 'o'
    # va = 'bottom' if kind == 'gt' else 'top'
    ax3.errorbar(np.array(binned_R_lows)+binsize/2, 
                 np.array(binned_yvals), 
                 yerr=np.array(binned_yerrs), 
                 fmt='', capsize=5, capthick=1.5, elinewidth=1.5, 
                 ecolor=main_color, marker=marker, markersize=4, 
                 markerfacecolor='white', markeredgecolor=main_color, 
                 color=main_color, lw=0.5, linestyle='-', 
                 zorder=2)
    ax3.scatter(np.array(binned_R_lows)+binsize/2, 
                np.array(binned_yvals), 
                marker=marker, s=16, 
                facecolors='white', edgecolors=main_color, zorder=2, 
                label=ylabel_full+f': Binospec (A383), bin size = {binsize}\'')
    
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
    
    # Error information (binned)
    if share_axs is not None:
        weighted_binned = np.average(np.array(binned_gxs), weights=np.array(binned_gxerrs)**(-2))
        weighted_binned_err = (np.sum(np.array(binned_gxerrs)**(-2)))**(-0.5)
        ax3.text(0.01, 0.01, 
                 r'Arithmetic error $\sigma_{\langle g_+ \rangle}$ = '+f'{np.mean(np.array(binned_gterrs)):.3f}\n'+
                 r'Arithmetic $\langle g_\times \rangle$ = '+f'{np.mean(np.array(binned_gxs)):.3f}'+
                 r' $\pm$ '+f'{np.mean(np.array(binned_gxerrs)):.3f}\n'+
                 r'Weighted $\langle g_\times \rangle$ = '+f'{weighted_binned:.3f}'+
                 r' $\pm$ '+f'{weighted_binned_err:.3f}',
                 fontsize=8, color='black', ha='left', va='bottom', 
                 transform=ax3.transAxes)
    elif kind == 'gt':
        ax3.text(0.01, 0.01, 
                 f'Arithmetic error = {np.mean(np.array(binned_yerrs)):.3f}', 
                 fontsize=8, color=main_color, ha='left', va='bottom', 
                 transform=ax3.transAxes)
    elif kind == 'gx':
        ax3.text(0.01, 0.01, 
                 f'Arithmetic offset = {np.mean(np.array(binned_yvals)):.3f}   \n'+
                 f'Arithmetic error = { np.mean(np.array(binned_yerrs)):.3f}', 
                 fontsize=8, color=main_color, ha='left', va='bottom', 
                 transform=ax3.transAxes)
    
    # R_500, R_200(=R_500/0.65)
    R_500 = 944 # kpc, Vikhlinin et al. (2006) arXiv:astro-ph/0507092, Table 4
    d_A   = scale_calc(0.1883, 70, 0.3) # kpc/arcsec
    r_500 = R_500 / d_A
    r_200 = r_500 / 0.65
    c500, c200 = draw_r500_r200(plt, ax1, r_500, r_200, color='limegreen')
    
    for ax in (ax2, ax3):
        ax.text(r_500/60+1/8, ax2_ymin+1/50, r'$r_{500,\mathrm{ A383}}$', color='limegreen', fontsize=12, zorder=5)
        ax.text(r_200/60+1/8, ax2_ymin+1/50, r'$r_{200,\mathrm{ A383}}$', color='limegreen', fontsize=12, zorder=5)
        ax.vlines(r_500/60, ymin=ax2_ymin, ymax=ax2_ymax, ls='--', color='limegreen')
        ax.vlines(r_200/60, ymin=ax2_ymin, ymax=ax2_ymax, ls=':',  color='limegreen')
    
    if image_size is not None: # image_size must in pixels
        assert (fig_centerRA is not None) & (fig_centerDEC is not None)
        
        x0, y0 = ax1.wcs.world_to_pixel_values(fig_centerRA, fig_centerDEC)
        ax1.set_xlim(x0-image_size/2, x0+image_size/2)
        ax1.set_ylim(y0-image_size/2, y0+image_size/2)
    
    ax1.set_xlabel('RA (deg)')
    ax1.set_ylabel('DEC (deg)')
    ax1.minorticks_on()
    ax1.set_aspect(1)
    
    ax2.set_xlim(ax2_xmin, ax2_xmax)
    ax2.set_ylim(ax2_ymin, ax2_ymax)
    if kind == 'gx':
        ax2.axhline(y=0, linestyle='--', color='gray')
    ax2.set_xlabel(r'$R$'+' (arcmin)', fontsize=15)
    ax2.set_ylabel(r'Reduced shear', fontsize=15)
    ax2.minorticks_on()
    ax2.legend(prop={'size': 8})
    ax2.grid(linestyle=':', color='black', alpha=0.5, zorder=0)
    
    ax3.set_xlim(ax2_xmin, ax2_xmax)
    ax3.set_ylim(ax2_ymin, ax2_ymax)
    if kind == 'gx':
        ax3.axhline(y=0, linestyle='--', color='gray')
    ax3.set_xlabel(r'$R$'+' (arcmin)', fontsize=15)
    ax3.set_ylabel(r'Binned reduced shear', fontsize=15)
    ax3.minorticks_on()
    ax3.legend(prop={'size': 7})
    ax3.grid(linestyle=':', color='black', alpha=0.5, zorder=0)
    
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
    img_dir  = '../../RSCH3/HSC_img_A383/'
    sci_fn_G = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz.fits'
    spec1d2dfolder02 = '../../RSCH3/UAO-S156-23B-A383/psf/231019/1d2dspecfiles/'
    
    # Read shear catalog
    df     = pd.read_excel("summary/redshift_table_with_shear_notedge_constrained.xlsx", header=None, engine='openpyxl')
    array  = df.to_numpy()
    
    # Slits with z & posterior solved & valid but extreme g1/g2
    g1err1 = np.array(array[2:, 21], dtype=float) # should be -
    g1err2 = np.array(array[2:, 22], dtype=float) # should be +
    g2err1 = np.array(array[2:, 25], dtype=float) # should be -
    g2err2 = np.array(array[2:, 26], dtype=float) # should be +
    
    mask_good_g1 = ~np.isnan(g1err1) & (g1err1 < 0) & (g1err2 > 0)
    mask_good_g2 = ~np.isnan(g2err1) & (g2err1 < 0) & (g2err2 > 0)
    mask_good_g1g2  = mask_good_g1 & mask_good_g2
    g_cat_good_g1g2 = array[np.r_[True, True,  mask_good_g1g2]]
    
    cluster_centerRA  =  ( ( 2)+(48)/60+( 3.3)/3600 )*15
    cluster_centerDEC = -( ( 3)+(31)/60+(46.4)/3600 )
    
    columns    = ['slitID', 'z', 'RA', 'DEC', 'radius', 'theta_rad', 
                  'g1_best', 'g1_err', 'g2_best', 'g2_err', 
                  'gt',      'gt_err', 'gx',      'gx_err']
    arr_shears = np.zeros((1, len(columns))).astype(str)
    arr_shears[0] = np.array(columns)
    
    for i in range(len(g_cat_good_g1g2[2:, 0])):
        # Calculate galaxy's position angle w.r.t. CLUSTER CENTER
        slit_id = g_cat_good_g1g2[2+i,  0]
        z       = np.nanmean(g_cat_good_g1g2[2+i, [5, 8, 11]])
        RA, DEC = g_cat_good_g1g2[2+i, 15], g_cat_good_g1g2[2+i, 16]
        g1, g2  = g_cat_good_g1g2[2+i, 20], g_cat_good_g1g2[2+i, 24]
        g1_err_nega, g1_err_posi = g_cat_good_g1g2[2+i, 21], g_cat_good_g1g2[2+i, 22]
        g2_err_nega, g2_err_posi = g_cat_good_g1g2[2+i, 25], g_cat_good_g1g2[2+i, 26]
        
        # Error propagation
        g1_err = 0.5 * (g1_err_posi - g1_err_nega)
        g2_err = 0.5 * (g2_err_posi - g2_err_nega)
        assert (g1_err > 0) & (g2_err > 0)
        
        # Polar coordinates
        R = 60 * np.sqrt((RA - cluster_centerRA)**2 + 
                         (DEC - cluster_centerDEC)**2)
        theta_cl_rad = np.arctan2(DEC - cluster_centerDEC, 
                                  RA  - cluster_centerRA)
        
        # Calculate tangential shear g+ by using Pranjal+22
        gt = -(g1 * np.cos(2*theta_cl_rad) + g2 * np.sin(2*theta_cl_rad))
        gx =  (g1 * np.sin(2*theta_cl_rad) - g2 * np.cos(2*theta_cl_rad))
        
        # σ^​2(g+) ​= (cos2ϕ)^2 σ^2(g1) ​+ (sin2ϕ)^2 σ^2(g2)
        # σ^​2(gx) ​= (sin2ϕ)^2 σ^2(g1) ​+ (cos2ϕ)^2 σ^2(g2)
        gt_err = (np.cos(2*theta_cl_rad)**2 * g1_err**2 + 
                  np.sin(2*theta_cl_rad)**2 * g2_err**2 )**0.5
        gx_err = (np.sin(2*theta_cl_rad)**2 * g1_err**2 + 
                  np.cos(2*theta_cl_rad)**2 * g2_err**2 )**0.5
        
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
    img_data_G = read_subaru_img_wcs(img_dir+sci_fn_G, None)
    imgG, wcsG = img_data_G['science_data'], img_data_G['science_wcs' ]
    
    # Read mask RA/DEC
    hdulist01 = fits.open(spec1d2dfolder02+'../obj_abs_slits_extr.fits')
    slitsInfo01 = np.append(hdulist01[4].data, hdulist01[5].data, axis=0)
    maskCenterRA, maskCenterDEC = slitsInfo01['MASK_RA'][0], slitsInfo01['MASK_DEC'][0]
    
    _, _, _, wcsGc = cutoutimg(
        imgG, wcsG, maskCenterRA, maskCenterDEC, 
        img_width=image_size*60/scale, img_height=image_size*60/scale, 
        outputWCS=True)
    
    # tan_shear_plot(arr_shears, cluster_centerRA, cluster_centerDEC, wcsGc, 
    #                image_size*60/scale, maskCenterRA, maskCenterDEC)
    
    # cross_shear_plot(arr_shears, cluster_centerRA, cluster_centerDEC, wcsGc, 
    #                  image_size*60/scale, maskCenterRA, maskCenterDEC)
    
    shear_plot(arr_shears, cluster_centerRA, cluster_centerDEC, wcsGc, 
               image_size*60/scale, maskCenterRA, maskCenterDEC, 
               plot_gt=True, plot_gx=True)