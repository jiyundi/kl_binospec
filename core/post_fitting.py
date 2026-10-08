"""
post_fitting.py  —  Observation / fit / residual diagnostic plots
==================================================================
Purely plotting.  All parameter-building, statistics, and I/O logic
lives in ``fitting_utils.py``; doublet constants live in ``doublet_utils.py``.

Public API
----------
ax_compass        : draw a N/E compass rose on a matplotlib axes
plot_obs_fit_res  : full obs|fit|residual figure (imaging + spectra)
"""

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.pyplot import MultipleLocator

# ── project ───────────────────────────────────────────────────────────────────
from klm.safe_plot import setup; setup()   # must come before any plt call
from core.doublet_utils import deduplicate_ordered
from core.fitting_result_utils import complete_fit_params

# ── matplotlib font config ────────────────────────────────────────────────────
plt.style.use('default')
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['font.sans-serif']  = ['Helvetica']
plt.rcParams['mathtext.it']  = 'Helvetica:italic'
plt.rcParams['mathtext.bf']  = 'Helvetica:bold'
plt.rcParams['mathtext.cal'] = 'Helvetica'
plt.rcParams['mathtext.rm']  = 'Helvetica'
plt.rcParams['mathtext.sf']  = 'Helvetica'

# ── Plot-specific constants ───────────────────────────────────────────────────

_LINE_DISPLAY: dict[str, str] = {
    'O2':  r"[O II] $\lambda\lambda$3726,3729",
    'Ha':  r"H$\alpha$",
    'Hb':  r"H$\beta$",
    'Hg':  r"H$\gamma$",
    'O3a': r"[O III] $\lambda$4959",
    'O3b': r"[O III] $\lambda$5007",
    'N2a': r"[N II] $\lambda$6549",
    'N2b': r"[N II] $\lambda$6583",
}

# set_num (1-based) → display letter.  Order is A/C/B by observing convention.
_SET_LABELS: dict[int, str] = {1: 'A', 2: 'C', 3: 'B'}


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

class _PanelPlotter:
    """
    Draws one horizontal obs | fit | residual triplet of imshow panels.

    Encapsulates the repeated pattern:
      • imshow with shared colour limits derived from noise
      • colorbar on each axes
      • large stroked χ²/ν label
      • smaller χ² + dof text block
      • white dotted grid
    """

    _STROKE = [
        path_effects.Stroke(linewidth=5, foreground='black'),
        path_effects.Normal(),
    ]

    def __init__(self, fig, ax_obs, ax_fit, ax_res):
        self.fig    = fig
        self.ax_obs = ax_obs
        self.ax_fit = ax_fit
        self.ax_res = ax_res

    def draw(
            self,
            obs: np.ndarray,
            fit: np.ndarray,
            mask: np.ndarray | None,
            chi2: float,
            dof:  float,
            extent: list,
            cmap_data: str = 'cividis',
            cmap_res:  str = 'coolwarm',
            origin:    str = 'lower',
            aspect:    str = 'equal',
    ) -> None:
        noise = (np.nanstd(obs[mask]) if mask is not None
                 else np.nanstd(obs))
        vmin = np.nanmin(obs[mask] if mask is not None else 0)
        vmax = 5 * noise
        vmax = np.nanmax(obs[mask] if mask is not None else 1)

        obs_show = np.where(mask, obs, np.nan) if mask is not None else obs
        res_show = np.where(mask, obs - fit, np.nan) if mask is not None else obs - fit

        kw = dict(extent=extent, origin=origin, aspect=aspect)
        im1 = self.ax_obs.imshow(obs_show,  vmin=vmin,  vmax=vmax,  cmap=cmap_data, **kw)
        im2 = self.ax_fit.imshow(fit,       vmin=vmin,  vmax=vmax,  cmap=cmap_data, **kw)
        im3 = self.ax_res.imshow(res_show,  vmin=-vmax, vmax=vmax,  cmap=cmap_res,  **kw)

        for im, ax in ((im1, self.ax_obs), (im2, self.ax_fit), (im3, self.ax_res)):
            self.fig.colorbar(im, ax=ax)

        self._annotate_chi2(chi2, dof)
        self._set_grid()

    def _annotate_chi2(self, chi2: float, dof: float) -> None:
        txt = self.ax_res.text(
            1, 1,
            r'$\chi^2_\nu=$' + f'{chi2 / dof:.1f}',
            c='yellow', fontsize=30, weight='bold',
            ha='right', va='top', transform=self.ax_res.transAxes,
        )
        txt.set_path_effects(self._STROKE)
        self.ax_res.text(
            0.98, 0.8,
            r'$\chi^2$' + f' = {chi2:.0f}\ndof = {dof:.0f}',
            fontsize=18, color='black', ha='right', va='top',
            transform=self.ax_res.transAxes,
        )

    def _set_grid(self) -> None:
        for ax in (self.ax_obs, self.ax_fit, self.ax_res):
            ax.grid(linestyle=':', color='white', alpha=0.5)


class _AnnotationHelper:
    """Builds best-fit parameter annotation strings for imshow panels."""

    @staticmethod
    def shared_params_text(fitting_par: dict, best_fit_dict: dict) -> str:
        lines = []
        for key, subdict in fitting_par.items():
            if key.split('-')[0] == 'shared_params':
                par_name = key.split('-')[1]
                value    = best_fit_dict['shared_params'][par_name]
                lines.append(f"{subdict['latex_name']} = {value:.2g}")
        return '\n'.join(lines)

    @staticmethod
    def line_params_text(
            fitting_par: dict,
            best_fit_dict: dict,
            line: str,
            set_num: int,
    ) -> str:
        all_lines = []
        for key, subdict in fitting_par.items():
            if key.split('-')[0] == f'{line}_params':
                par_name = key.split('-')[1]
                value    = best_fit_dict[f'{line}_params'][par_name]
                all_lines.append(f"{subdict['latex_name']} = {value:.1f}")

        set_label1, set_label2 = f'Spec {set_num}', f'Spec\,{set_num}'
        return '\n'.join(
            s for s in all_lines
            if (set_label1 in s) or (set_label2 in s) or ('Spec' not in s)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ax_compass(
        ax,
        x0: float, y0: float,
        dx: float, dy: float,
        color: str = 'black',
) -> None:
    """
    Draw a N/E compass rose on *ax* using axes-fraction coordinates.

    East points in the −x direction (astronomical convention: E left, N up).
    """
    kw = dict(head_width=0.02, head_length=0.02,
              fc=color, ec=color, linewidth=1.5,
              transform=ax.transAxes)
    ax.arrow(x0,  y0,  0,   dy, **kw)
    ax.arrow(x0,  y0, -dx,  0,  **kw)
    ax.text(x0 + 0.02, y0 + dy, 'N',
            color=color, ha='left',   va='center', fontsize=12,
            transform=ax.transAxes)
    ax.text(x0 - dx,   y0 + 0.02, 'E',
            color=color, ha='center', va='bottom',  fontsize=12,
            transform=ax.transAxes)


def plot_tilted_wavelength_grid(ax, lambda_grid, model_type='spec', **kwargs):
    wave_min = int(np.min(lambda_grid.value))
    wave_max = int(np.max(lambda_grid.value))
    waves    = np.arange(wave_min, wave_max, step=5)
    cs = ax.contour(
        lambda_grid.value,
        levels=waves,
        **kwargs,
    )

    if model_type == 'rotation_curve':
        ax.clabel(cs, fmt=r'%d $\AA$', fontsize=10)


def plot_obs_fit_res(
        data_info: dict,
        inference,
        best_fit_dict: dict,
        fitting_par: dict,
        slit_name: str,
        save_path: str | None = None,
        other_path_filename: str | None = None,
) -> None:
    """
    Produce the full obs / fit / residual diagnostic figure.

    Layout
    ------
    Row 0       : imaging triplet (obs | fit | residual)
    Rows 1..N   : one row per slit spectrum

    Parameters
    ----------
    data_info : dict
        Contains ``'image'`` and ``'spec'`` sub-dicts from the KLM pipeline.
    inference : klm Inference object
    best_fit_dict : dict
    fitting_par : dict  (flat sorted, from ``fitting_utils.complete_flattened_fit_params``)
    slit_name : str
        Used in the figure title.
    save_path : str, optional
        Directory for ``best_fit_spec.png``.  Ignored when *other_path_filename* is set.
    other_path_filename : str, optional
        Full output path (overrides *save_path*).
    """
    # (0.1) Find all (and duplicated) emission lines in data specs
    lines_all = [
        inference.meta_spec[i]['line_species']
        for i in range(len(inference.meta_spec))
    ]

    # (0.2) Resolve unique ordered line list from best_fit_dict keys
    lines = deduplicate_ordered(
        k.split('_')[0]
        for k in best_fit_dict
        if k.split('_')[0] != 'shared'
    )

    # (0.3) Normalise fitting_par to flat form if a nested dict was passed in
    if 'shared_params' in fitting_par:
        fitting_par = complete_fit_params(
            fitting_par,
            inference.config.galaxy_params.line_species,
        )

    nspec      = len(data_info['spec'])
    pix_scale  = data_info['image']['meta']['pixScale']
    annotator  = _AnnotationHelper()

    # (0.4) Free-parameter count for DOF calculation
    n_par = sum(
        len(fitting_par[k]) if len(k.split('-')) == 1 else 1
        for k in fitting_par
    )

    # ── (0) Figure / GridSpec ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 4 * (1 + nspec)))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    gs  = fig.add_gridspec(
        nrows=1 + nspec, ncols=3,
        height_ratios=[1] * (1 + nspec),
        width_ratios=[1, 1, 1],
    )

    # ── (1) Imaging ────────────────────────────────────────────────────────
    # (1.1) Axis flip note:
    #   WCS arrays have [ΔDEC=0, ΔRA=0] at the SW corner.
    #   Flip the horizontal axis so East (ΔRA > 0) is on the left; use
    #   origin='lower' so DEC increases upward.
    #   The residual panel uses the *reversed* RA extent to preserve the
    #   coolwarm colour-scale direction after the flip.
    if hasattr(inference, 'data_image'): # if fitting imaging data
        image_obs  = np.flip(inference.data_image, axis=1)
        image_msk  = np.flip(inference.mask_image, axis=1)
        image_var  = np.flip(inference.var_image,  axis=1)
        image_fit  = np.flip(
            inference.image_model.get_image(best_fit_dict['shared_params']),
            axis=1,
        )
        image_chi2 = inference.calc_image_loglike(best_fit_dict, reduce=True)
    else:
        image_obs  = np.zeros((10, 10))
        image_msk  = np.ones((10, 10), dtype=bool)
        image_var  = np.ones((10, 10))
        image_fit  = np.zeros((10, 10))
        image_chi2 = 0

    ny_img, nx_img = image_obs.shape
    img_dof  = np.sum(image_msk) - n_par
    half_ra  = nx_img * pix_scale / 2
    half_dec = ny_img * pix_scale / 2

    img_extent     = [-half_ra,  half_ra, -half_dec, half_dec]
    # res_img_extent = [ half_ra, -half_ra, -half_dec, half_dec]  # reversed RA

    # (1.2) Plot the imaging
    ax_img_obs = fig.add_subplot(gs[0, 0])
    ax_img_fit = fig.add_subplot(gs[0, 1])
    ax_img_res = fig.add_subplot(gs[0, 2])

    _PanelPlotter(fig, ax_img_obs, ax_img_fit, ax_img_res).draw(
                obs=image_obs, fit=image_fit, mask=image_msk,
                chi2=image_chi2, dof=img_dof,
                extent=img_extent,
                cmap_data='cividis', cmap_res='coolwarm',
                origin='lower', 
                aspect='auto',
            )

    # (1.4) Shared-params text on fit panel
    fsize = 12
    N_shared_pars = len([p for p in fitting_par if 'shared_params' in p])
    if N_shared_pars > 10:
        fsize = 120/N_shared_pars
    ax_img_fit.text(
        1, 1,
        annotator.shared_params_text(fitting_par, best_fit_dict),
        fontsize=fsize, color='white', ha='right', va='top',
        transform=ax_img_fit.transAxes,
    )

    # (1.5) SNR info on obs panel
    image_snr = np.sum(image_obs[image_msk]) / np.sqrt(np.sum(image_obs[image_msk] + image_var[image_msk]))
    ax_img_obs.text(
        0.98, 0.97,
        f'N_RA = {nx_img} px\nN_DEC = {ny_img} px\nSNR = {image_snr:.0f}',
        fontsize=12, color='white', ha='right', va='top',
        transform=ax_img_obs.transAxes,
        bbox=dict(facecolor='black', alpha=0.75),
    )

    # (1.6) Compass roses
    for ax in (ax_img_obs, ax_img_fit):
        ax_compass(ax, x0=0.05, y0=0.05, dx=-0.12, dy=0.12, color='white')
    ax_compass(ax_img_res, x0=0.05, y0=0.05, dx=-0.12, dy=0.12, color='black')

    # (1.7) Labels and titles
    ax_img_obs.set_ylabel(
        r'${\bf Imaging}$' + '\n' + r'$\Delta$ DEC (arcsec)', fontsize=18)
    for ax in (ax_img_obs, ax_img_fit, ax_img_res):
        ax.set_xlabel(r'$\Delta$ RA (arcsec)')
    ax_img_obs.set_title(f'#{slit_name} Observation', fontsize=18)
    ax_img_fit.set_title('Best fit model',             fontsize=18)
    ax_img_res.set_title('Residual (= obs − model)',   fontsize=18)

    # ── (2) spectra ────────────────────────────────────────────────────
    for i in range(nspec):
        # (2.1) Prepare params for model spec
        line    = lines[i // 3]
        set_num = i % 3 + 1

        line    = lines_all[i]
        counts  = lines_all[: i + 1].count(line) # occurrences of this line
        set_num = counts

        inference.spec_model[i]._init_observable(
            data_info['galaxy'],
            data_info['spec'][i]['meta'],
        )

        # (2.2) Assign (I01_specN → I01) for this spec number
        best_one_level = {
            **best_fit_dict['shared_params'],
            **best_fit_dict[f'{line}_params'],
        }
        for k in ('I01', 'I02', 'bkg_level', 'f1_1', 'f1_2', 'f2_1', 'f2_2'):
            if k in best_one_level.keys():
                best_one_level[k] = best_one_level.get(f'{k}_spec{set_num}')

        # (2.3) Get observed spec data
        spec_obs  = inference.data_spec[i]
        spec_msk  = inference.mask_spec[i]
        spec_var  = inference.var_spec[i]

        # (2.4) Get model spec, or rotation curve
        if ('data_k' in dir(inference)) and ('data_k_err' in dir(inference)) and ('mask_k' in dir(inference)):
            model_type = 'rotation_curve'
            data_k     = inference.data_k[i].astype(float)
            data_k_err = inference.data_k_err[i].astype(float)
            mask_k     = inference.mask_k[i].astype(bool)

            model_k    = inference.spec_model[i].get_observable(best_one_level, return_type='rotation_curve')

            spec_chi2  = inference._loglike_one_rotation_curve(
                data_k, mask_k, data_k_err, model_k
                )
        else:
            model_type = 'spec'
            spec_fit   = inference.spec_model[i].get_observable(best_one_level)
            spec_chi2  = inference._loglike_one_slit(
                spec_obs, spec_msk, spec_var, spec_fit, 
                reduce=True
                )

        # (2.5) DOF calculation and other plotting params
        ny_spec, nx_spec = spec_obs.shape
        if model_type == 'spec':
            spec_dof = np.sum(spec_msk) - len(fitting_par)
        else:
            spec_dof = np.sum(mask_k) - len(fitting_par)

        lambda_grid = data_info['spec'][i]['meta']['lambda_grid']
        spec_extent = [
            lambda_grid[0][0].value,
            lambda_grid[0][-1].value,
            inference.spec_model[i].slit_x[0],
            inference.spec_model[i].slit_x[-1],
        ]

        # (2.6) Plot the spec/rotation curves
        ax_spe_obs = fig.add_subplot(gs[i + 1, 0])
        ax_spe_fit = fig.add_subplot(gs[i + 1, 1])
        ax_spe_res = fig.add_subplot(gs[i + 1, 2])

        if model_type == 'spec':
            _PanelPlotter(fig, ax_spe_obs, ax_spe_fit, ax_spe_res).draw(
                obs=spec_obs, fit=spec_fit, mask=spec_msk,
                chi2=spec_chi2, dof=spec_dof,
                extent=spec_extent,
                cmap_data='viridis', cmap_res='coolwarm',
                origin='lower', # spectra: wavelength increases left→right
                aspect='auto',
            )

        else:
            noise_spec = np.nanstd(spec_obs[spec_msk])
            vmax_spec  = 5 * noise_spec

            # (2.6.1) Observation
            im1 = ax_spe_obs.imshow(
                np.where(spec_msk, spec_obs, np.nan),
                vmin=0, vmax=vmax_spec, 
                cmap='viridis', origin='lower', aspect='auto',
            )
            ln1 = ax_spe_obs.errorbar(
                data_k[mask_k], 
                np.arange(len(data_k ))[mask_k], 
                color='deeppink',
                xerr=data_k_err[mask_k], 
                fmt='',
                ecolor='crimson',
                elinewidth=1,
                capsize=2,
                capthick=1, label='Extracted from observation'
                )
            fig.colorbar(im1, ax=ax_spe_obs)
            fake = cm.ScalarMappable(cmap="viridis", norm=colors.Normalize(vmin=0, vmax=1)); fake.set_array([])
            fig.colorbar(fake, ax=ax_spe_fit)
            fig.colorbar(fake, ax=ax_spe_res)

            # (2.6.2) Model
            ln2 = ax_spe_fit.plot(
                model_k, 
                np.arange(len(model_k)), 
                color='deepskyblue', linewidth=1.5,
                label='Model rotation curve'
                )
            ln3 = ax_spe_fit.errorbar(
                data_k[mask_k], 
                np.arange(len(data_k ))[mask_k], 
                color='deeppink',
                xerr=data_k_err[mask_k], 
                fmt='',
                ecolor='crimson',
                elinewidth=1,
                capsize=2,
                capthick=1, label='Extracted from observation'
                )
            
            # (2.6.3) Residual
            ln4 = ax_spe_res.axvline(
                x=0, color='gray', linestyle='--', linewidth=2, 
                label='Systematic wavelength'
                )
            ln5 = ax_spe_res.errorbar(
                data_k[mask_k] - model_k[mask_k], 
                np.arange(len(data_k ))[mask_k], 
                color='deeppink',
                xerr=data_k_err[mask_k], 
                fmt='',
                ecolor='crimson',
                elinewidth=1,
                capsize=2,
                capthick=1, label='Extracted from observation'
                )
            ax_spe_res.grid(alpha=0.3)

            ax_spe_fit.set_xlim(0, lambda_grid.shape[1])
            ax_spe_fit.set_ylim(0, lambda_grid.shape[0])
            ax_spe_res.set_xlim(-lambda_grid.shape[1]/2, lambda_grid.shape[1]/2)
            ax_spe_res.set_ylim(0, lambda_grid.shape[0])
            ax_spe_obs.legend(handles=[ln1], loc='lower right', fontsize=8)
            ax_spe_fit.legend(handles=[ln2[0], ln3], loc='lower right', fontsize=8)
            ax_spe_res.legend(handles=[ln4, ln5], loc='lower right', fontsize=8)

            # (2.6.4) Tilted wavelength grid
            plot_tilted_wavelength_grid(
                ax_spe_obs, lambda_grid, model_type=model_type,
                colors='white',
                linestyles='dotted', 
                linewidths=1, 
                alpha=0.25
                )
            plot_tilted_wavelength_grid(
                ax_spe_fit, lambda_grid, model_type=model_type,
                colors='gray',
                linestyles='dotted', 
                linewidths=1, 
                alpha=0.5
                )

            # (2.6.5) chi2 annotation + grid
            spec_panel = _PanelPlotter(fig, ax_spe_obs, ax_spe_fit, ax_spe_res)
            spec_panel._annotate_chi2(spec_chi2, spec_dof)

        # (2.7) Line params text on fit panel
        ax_spe_fit.text(
            1, 1,
            annotator.line_params_text(fitting_par, best_fit_dict, line, set_num),
            fontsize=12, 
            color='white' if model_type == 'spec' else 'steelblue', 
            ha='right', va='top',
            transform=ax_spe_fit.transAxes,
        )

        # (2.8) SNR info on obs panel
        bg = np.nanmedian(spec_obs[spec_msk])       # = -2.9597
        spec_obs_zero_bg = np.where(spec_msk, spec_obs - bg, np.nan)
        spec_snr = (
            np.nansum(spec_obs_zero_bg) /
            np.sqrt(np.nansum(spec_var[spec_msk] + spec_obs_zero_bg[spec_msk]))
        )
        ax_spe_obs.text(
            0.98, 0.97,
            'N_' + r'$\lambda$' + f' = {nx_spec} px\n'
            f'N_slit = {ny_spec} px\n'
            f'SNR = {spec_snr:.0f}',
            fontsize=12, color='white', ha='right', va='top',
            transform=ax_spe_obs.transAxes,
            bbox=dict(facecolor='black', alpha=0.75),
        )

        # (2.9) Line label on obs panel
        ax_spe_obs.text(
            0.02, 0.98,
            _LINE_DISPLAY.get(line, line),
            fontsize=18, color='white', ha='left', va='top',
            transform=ax_spe_obs.transAxes,
            bbox=dict(facecolor='black', alpha=0.75),
        )

        # (2.10) Wavelength axis formatting
        for ax in (ax_spe_obs, ax_spe_fit, ax_spe_res):
            ax.xaxis.get_major_formatter().set_useOffset(False)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))

        if model_type == 'spec':
            ax_spe_obs.set_ylabel(
                r'${\bf Spec}$' + f' set {_SET_LABELS[set_num]}\n'
                'slit position (arcsec)',
                fontsize=18,
            )
        else:
            ax_spe_obs.set_ylabel(
                r'${\bf Spec}$' + f' set {_SET_LABELS[set_num]}\n'
                'slit position (pixels)',
                fontsize=18,
            )

    # (2.12) Bottom x-axis labels (last spec row axes are still in scope)
    if model_type == 'spec':
        ax_spe_obs.set_xlabel(r'Wavelength ($\AA$)')
        ax_spe_fit.set_xlabel(r'Wavelength ($\AA$)')
        ax_spe_res.set_xlabel(r'Wavelength ($\AA$)')
    else:
        ax_spe_obs.set_xlabel('Wavelength position (pixels)')
        ax_spe_fit.set_xlabel('Wavelength position (pixels)')
        ax_spe_res.set_xlabel('Wavelength position (pixels)')

    # ── (3) Save ──────────────────────────────────────────────────────────────────
    fig_path = (other_path_filename
                if other_path_filename is not None
                else f'{save_path}/best_fit_spec.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
