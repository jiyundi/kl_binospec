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
        vmax = 5 * noise

        obs_show = np.where(mask, obs, np.nan) if mask is not None else obs
        res_show = np.where(mask, obs - fit, np.nan) if mask is not None else obs - fit

        kw = dict(extent=extent, origin=origin, aspect=aspect)
        im1 = self.ax_obs.imshow(obs_show,  vmin=0,     vmax=vmax,  cmap=cmap_data, **kw)
        im2 = self.ax_fit.imshow(fit,       vmin=0,     vmax=vmax,  cmap=cmap_data, **kw)
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

        set_label = f'Spec {set_num}'
        return '\n'.join(
            s for s in all_lines
            if (set_label in s) or ('Spec' not in s)
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
    # Resolve unique ordered line list from best_fit_dict keys
    lines = deduplicate_ordered(
        k.split('_')[0]
        for k in best_fit_dict
        if k.split('_')[0] != 'shared'
    )

    # Normalise fitting_par to flat form if a nested dict was passed in
    if 'shared_params' in fitting_par:
        fitting_par = complete_fit_params(
            fitting_par,
            inference.config.galaxy_params.line_species,
        )

    nspec      = len(data_info['spec'])
    pix_scale  = data_info['image']['meta']['pixScale']
    annotator  = _AnnotationHelper()

    # Free-parameter count for DOF calculation
    n_par = sum(
        len(fitting_par[k]) if len(k.split('-')) == 1 else 1
        for k in fitting_par
    )

    # ── Figure / GridSpec ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 4 * (1 + nspec)))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    gs  = fig.add_gridspec(
        nrows=1 + nspec, ncols=3,
        height_ratios=[1] * (1 + nspec),
        width_ratios=[1, 1, 1],
    )

    # ── Row 0: imaging ────────────────────────────────────────────────────────
    ax_img_obs = fig.add_subplot(gs[0, 0])
    ax_img_fit = fig.add_subplot(gs[0, 1])
    ax_img_res = fig.add_subplot(gs[0, 2])

    # Axis flip note:
    #   WCS arrays have [ΔDEC=0, ΔRA=0] at the SW corner.
    #   Flip the horizontal axis so East (ΔRA > 0) is on the left; use
    #   origin='lower' so DEC increases upward.
    #   The residual panel uses the *reversed* RA extent to preserve the
    #   coolwarm colour-scale direction after the flip.
    image_obs  = np.flip(inference.data_image, axis=1)
    image_msk  = np.flip(inference.mask_image, axis=1)
    image_var  = np.flip(inference.var_image,  axis=1)
    image_fit  = np.flip(
        inference.image_model.get_image(best_fit_dict['shared_params']),
        axis=1,
    )
    image_chi2 = inference.calc_image_loglike(best_fit_dict)

    ny_img, nx_img = image_obs.shape
    img_dof  = np.sum(image_msk) - n_par
    half_ra  = nx_img * pix_scale / 2
    half_dec = ny_img * pix_scale / 2

    img_extent     = [-half_ra,  half_ra, -half_dec, half_dec]
    res_img_extent = [ half_ra, -half_ra, -half_dec, half_dec]  # reversed RA

    noise_img = np.nanstd(image_obs[image_msk])
    vmax_img  = 5 * noise_img

    im1 = ax_img_obs.imshow(
        np.where(image_msk, image_obs, np.nan),
        vmin=0, vmax=vmax_img, extent=img_extent,
        cmap='cividis', origin='lower', aspect='equal',
    )
    im2 = ax_img_fit.imshow(
        image_fit,
        vmin=0, vmax=vmax_img, extent=img_extent,
        cmap='cividis', origin='lower', aspect='equal',
    )
    im3 = ax_img_res.imshow(
        np.where(image_msk, image_obs - image_fit, np.nan),
        vmin=-vmax_img, vmax=vmax_img, extent=res_img_extent,
        cmap='coolwarm', origin='lower', aspect='equal',
    )
    for im, ax in ((im1, ax_img_obs), (im2, ax_img_fit), (im3, ax_img_res)):
        fig.colorbar(im, ax=ax)

    # Reuse _PanelPlotter only for chi2 annotation + grid
    img_panel = _PanelPlotter(fig, ax_img_obs, ax_img_fit, ax_img_res)
    img_panel._annotate_chi2(image_chi2, img_dof)
    img_panel._set_grid()

    # Shared-params text on fit panel
    ax_img_fit.text(
        1, 1,
        annotator.shared_params_text(fitting_par, best_fit_dict),
        fontsize=12, color='white', ha='right', va='top',
        transform=ax_img_fit.transAxes,
    )

    # SNR info on obs panel
    image_snr = np.sum(image_obs[image_msk]) / np.sqrt(np.sum(image_obs[image_msk] + image_var[image_msk]))
    ax_img_obs.text(
        0.98, 0.97,
        f'N_RA = {nx_img} px\nN_DEC = {ny_img} px\nSNR = {image_snr:.0f}',
        fontsize=12, color='white', ha='right', va='top',
        transform=ax_img_obs.transAxes,
        bbox=dict(facecolor='black', alpha=0.75),
    )

    # Compass roses
    for ax in (ax_img_obs, ax_img_fit):
        ax_compass(ax, x0=0.05, y0=0.05, dx=-0.12, dy=0.12, color='white')
    ax_compass(ax_img_res, x0=0.05, y0=0.05, dx=-0.12, dy=0.12, color='black')

    # Labels and titles
    ax_img_obs.set_ylabel(
        r'${\bf Imaging}$' + '\n' + r'$\Delta$ DEC (arcsec)', fontsize=18)
    for ax in (ax_img_obs, ax_img_fit, ax_img_res):
        ax.set_xlabel(r'$\Delta$ RA (arcsec)')
    ax_img_obs.set_title(f'#{slit_name} Observation', fontsize=18)
    ax_img_fit.set_title('Best fit model',             fontsize=18)
    ax_img_res.set_title('Residual (= obs − model)',   fontsize=18)

    # ── Rows 1..N: spectra ────────────────────────────────────────────────────
    for i in range(nspec):
        line    = lines[i // 3]
        set_num = i % 3 + 1

        inference.spec_model[i]._init_observable(
            data_info['galaxy'],
            data_info['spec'][i]['meta'],
        )

        # Collapse per-set intensity keys (I01_specN → I01) for this row
        best_one_level = {
            **best_fit_dict['shared_params'],
            **best_fit_dict[f'{line}_params'],
        }
        for k in ('I01', 'I02'):
            best_one_level[k] = best_one_level.get(f'{k}_spec{set_num}')

        spec_obs  = inference.data_spec[i]
        spec_msk  = inference.mask_spec[i]
        spec_fit  = inference.spec_model[i].get_observable(best_one_level)
        spec_var  = inference.var_spec[i]
        spec_chi2 = inference._loglike_one_slit(
            spec_obs, spec_msk, spec_var, spec_fit
            )

        ny_spec, nx_spec = spec_obs.shape
        spec_dof = np.sum(spec_msk) - len(fitting_par)

        lambda_grid = data_info['spec'][i]['meta']['lambda_grid']
        spec_extent = [
            lambda_grid[0][0].value,
            lambda_grid[0][-1].value,
            inference.spec_model[i].slit_x[0],
            inference.spec_model[i].slit_x[-1],
        ]

        ax_spe_obs = fig.add_subplot(gs[i + 1, 0])
        ax_spe_fit = fig.add_subplot(gs[i + 1, 1])
        ax_spe_res = fig.add_subplot(gs[i + 1, 2])

        _PanelPlotter(fig, ax_spe_obs, ax_spe_fit, ax_spe_res).draw(
            obs=spec_obs, fit=spec_fit, mask=spec_msk,
            chi2=spec_chi2, dof=spec_dof,
            extent=spec_extent,
            cmap_data='viridis', cmap_res='coolwarm',
            origin='upper',   # spectra: wavelength increases left→right
            aspect='auto',
        )

        # Line params text on fit panel
        ax_spe_fit.text(
            1, 1,
            annotator.line_params_text(fitting_par, best_fit_dict, line, set_num),
            fontsize=12, color='white', ha='right', va='top',
            transform=ax_spe_fit.transAxes,
        )

        # SNR info on obs panel
        spec_snr = (
            np.sum(spec_obs[spec_msk]) /
            np.sqrt(np.sum(spec_obs[spec_msk] + spec_var[spec_msk]))
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

        # Line label on obs panel
        ax_spe_obs.text(
            0.02, 0.02,
            _LINE_DISPLAY.get(line, line),
            fontsize=18, color='white', ha='left', va='bottom',
            transform=ax_spe_obs.transAxes,
            bbox=dict(facecolor='black', alpha=0.75),
        )

        # Wavelength axis formatting
        for ax in (ax_spe_obs, ax_spe_fit, ax_spe_res):
            ax.xaxis.get_major_formatter().set_useOffset(False)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=5))

        ax_spe_obs.set_ylabel(
            r'${\bf Spec}$' + f' set {_SET_LABELS[set_num]}\n'
            'Slit Position (arcsec)',
            fontsize=18,
        )

    # Bottom x-axis labels (last spec row axes are still in scope)
    ax_spe_obs.set_xlabel(r'Wavelength ($\AA$)')
    ax_spe_fit.set_xlabel(r'Wavelength ($\AA$)')
    ax_spe_res.set_xlabel(r'Wavelength ($\AA$)')

    # ── Save ──────────────────────────────────────────────────────────────────
    fig_path = (other_path_filename
                if other_path_filename is not None
                else f'{save_path}/best_fit_spec.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
