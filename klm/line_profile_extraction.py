import warnings
import numpy as np
import astropy.units as u
from scipy.optimize import curve_fit


def two_half_gaussians(x, amp, mu, sigma_left, sigma_right):
    """Evaluate a split Gaussian using wavelength values in angstroms."""
    sigma = np.where(x < mu, sigma_left, sigma_right)
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def asymmetric_doublet(x,
                       amp1, mu1, sigma1_left, sigma1_right,
                       amp2, dmu, sigma2_left, sigma2_right):
    """Evaluate an asymmetric doublet with mu2 = mu1 + dmu."""
    mu2 = mu1 + dmu

    return (
        two_half_gaussians(
            x, amp1, mu1, sigma1_left, sigma1_right
        )
        + two_half_gaussians(
            x, amp2, mu2, sigma2_left, sigma2_right
        )
    )


def _as_angstrom_values(value):
    """Convert a wavelength quantity to angstroms or accept values in angstroms."""
    if isinstance(value, u.Quantity):
        return value.to_value(u.Angstrom)
    return np.asarray(value, dtype=float)


def _average_neighboring_rows(data, wave, y):
    """Interpolate three neighboring rows onto row y's grid and average them."""
    row_indices = np.clip([y - 1, y, y + 1], 0, data.shape[0] - 1)
    x_reference = wave[y]
    interpolated_rows = []

    for row in row_indices:
        wave_row = wave[row]
        flux_row = data[row]
        valid = np.isfinite(wave_row) & np.isfinite(flux_row)
        if np.count_nonzero(valid) < 2:
            continue

        wave_valid = wave_row[valid]
        flux_valid = flux_row[valid]
        # Sort the wavelengths before dropping duplicates for interpolation.
        order = np.argsort(wave_valid)
        wave_valid, unique_idx = np.unique(wave_valid[order], return_index=True)
        flux_valid = flux_valid[order][unique_idx]
        interpolated_rows.append(
            np.interp(x_reference, wave_valid, flux_valid, left=np.nan, right=np.nan)
        )

    if not interpolated_rows:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(interpolated_rows, axis=0)


def fit_asymmetric_line_profiles(
        data,
        wavelength,
        line,
        doublets=("O2",),
        sigma_bounds_pix=(1.0, 20.0),
        separation_bounds_pix=(4.0, 12.0),
        initial_sigma_pix=3.0):
    """Fit an asymmetric single line or doublet at each spatial row.

    Each fit averages three rows on the central row's wavelength grid.
    At the edges, the outermost row is used twice.

    Parameters
    ----------
    data : ndarray, shape (ny, nx)
        Two-dimensional spectrum.
    wavelength : Quantity or ndarray, shape (ny, nx)
        Wavelength at each spatial and spectral pixel. Plain arrays use angstroms.
    line : str
        Line name. Names in doublets use the doublet model.
    doublets : sequence of str
        Line names that use the doublet model.
    sigma_bounds_pix, separation_bounds_pix : tuple of float
        Bounds on the wing widths and doublet separation in spectral pixels.
        They are converted to angstroms using the median wavelength spacing.
    initial_sigma_pix : float
        Initial width for each wing, in spectral pixels.

    Returns
    -------
    amp : ndarray, shape (ny, 2)
        Peak amplitudes.
    mu, sigma_left, sigma_right : Quantity, shape (ny, 2)
        Peak centers and left and right wing widths in angstroms.
        For a successful single-line fit, the unused center is NaN and the
        unused amplitude and widths are zero. Failed rows are all NaN.
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional array with shape (ny, nx)")

    ny, nx = data.shape

    wave = _as_angstrom_values(wavelength)

    if wave.shape != (ny, nx):
        raise ValueError("wavelength must have the same shape (ny, nx) as data")

    # Use the typical wavelength spacing across all rows.
    delta_lambda = np.nanmedian(np.abs(np.diff(wave, axis=1)))

    if not np.isfinite(delta_lambda) or delta_lambda <= 0:
        raise ValueError("Cannot determine a finite, positive wavelength spacing")

    min_sigma = sigma_bounds_pix[0] * delta_lambda
    max_sigma = sigma_bounds_pix[1] * delta_lambda
    sigma0 = initial_sigma_pix * delta_lambda

    min_separation = separation_bounds_pix[0] * delta_lambda
    max_separation = separation_bounds_pix[1] * delta_lambda

    if min_sigma <= 0 or max_sigma <= min_sigma:
        raise ValueError("sigma_bounds_pix must be positive and strictly increasing")

    if min_separation <= 0 or max_separation <= min_separation:
        raise ValueError("separation_bounds_pix must be positive and strictly increasing")

    use_doublet = line in doublets
    function_this_line = (
        asymmetric_doublet if use_doublet
        else two_half_gaussians
    )

    Amp = np.full((ny, 2), np.nan)
    Mu = np.full((ny, 2), np.nan)
    sigma_left = np.full((ny, 2), np.nan)
    sigma_right = np.full((ny, 2), np.nan)

    for y in range(ny):

        x_reference = wave[y]
        profile = _average_neighboring_rows(data, wave, y)
        if profile is None:
            continue

        valid = np.isfinite(x_reference) & np.isfinite(profile)

        if np.count_nonzero(valid) < 8:
            continue

        x = x_reference[valid]
        flux = profile[valid]

        # Sort by wavelength so the wings follow increasing wavelength.
        order = np.argsort(x)
        x = x[order]
        flux = flux[order]

        # Subtract the median background before fitting the line-only model.
        background = np.nanmedian(flux)
        flux = flux - background

        if np.nanmax(flux) <= 0:
            continue

        wave_min = np.nanmin(x)
        wave_max = np.nanmax(x)

        # Keep the line center away from the edges of the fitting window.
        center_margin = 2.0 * delta_lambda
        center_low = wave_min + center_margin
        center_high = wave_max - center_margin

        if center_low >= center_high:
            continue

        if not use_doublet:
            # Single line
            peak_idx = np.nanargmax(flux)

            amp0 = max(flux[peak_idx], 1e-3)
            mu0 = np.clip(x[peak_idx], center_low, center_high)

            p0 = [amp0, mu0, sigma0, sigma0]
            lower_bounds = [0.0, center_low, min_sigma, min_sigma]
            upper_bounds = [np.inf, center_high, max_sigma, max_sigma]

        else:
            # Doublet
            peak1_idx = np.nanargmax(flux)
            peak1_wave = x[peak1_idx]

            # Exclude pixels near the strongest peak before finding the other peak.
            flux_masked = flux.copy()
            suppress = np.abs(x - peak1_wave) < min_separation
            flux_masked[suppress] = -np.inf

            if not np.any(np.isfinite(flux_masked)):
                continue

            peak2_idx = np.argmax(flux_masked)
            # Order the peaks by wavelength.
            blue_peak, red_peak = sorted((peak1_idx, peak2_idx))
            mu1_0, mu2_0 = x[[blue_peak, red_peak]]
            amp1_0 = max(flux[blue_peak], 1e-3)
            amp2_0 = max(flux[red_peak], 1e-3)

            dmu0 = np.clip(
                mu2_0 - mu1_0,
                min_separation,
                max_separation
            )

            # Leave room for at least the minimum doublet separation.
            mu1_high = center_high - min_separation

            if center_low >= mu1_high:
                continue

            mu1_0 = np.clip(mu1_0, center_low, mu1_high)

            p0 = [
                amp1_0, mu1_0, sigma0, sigma0,
                amp2_0, dmu0, sigma0, sigma0,
            ]
            lower_bounds = [
                0.0, center_low, min_sigma, min_sigma,
                0.0, min_separation, min_sigma, min_sigma,
            ]
            upper_bounds = [
                np.inf, mu1_high, max_sigma, max_sigma,
                np.inf, max_separation, max_sigma, max_sigma,
            ]

        # Keep the initial parameters strictly inside the bounds.
        p0 = np.clip(
            np.asarray(p0, dtype=float),
            np.asarray(lower_bounds) + 1e-10,
            np.asarray(upper_bounds) - 1e-10,
        )

        try:
            popt, _ = curve_fit(
                function_this_line,
                x,
                flux,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue

        if not use_doublet:
            amp, mu, sig_l, sig_r = popt

            Amp[y] = [amp, 0.0]
            Mu[y] = [mu, np.nan]
            sigma_left[y] = [sig_l, 0.0]
            sigma_right[y] = [sig_r, 0.0]

        else:
            amp1, mu1, sig1_l, sig1_r, amp2, dmu, sig2_l, sig2_r = popt

            mu2 = mu1 + dmu

            # Discard fits whose second peak falls outside the fitting window.
            if mu2 > center_high:
                continue

            Amp[y] = [amp1, amp2]
            Mu[y] = [mu1, mu2]
            sigma_left[y] = [sig1_l, sig2_l]
            sigma_right[y] = [sig1_r, sig2_r]

    return (
        Amp,
        Mu * u.Angstrom,
        sigma_left * u.Angstrom,
        sigma_right * u.Angstrom,
    )


def _reconstruct_profiles(wave, amp, mu, sigma_left, sigma_right, valid_fit):
    """Evaluate each valid peak on its own row's wavelength grid."""
    restored = np.full(wave.shape, np.nan, dtype=float)
    min_sigma = np.finfo(float).eps
    for y, wavelengths in enumerate(wave):
        valid_wave = np.isfinite(wavelengths)
        peaks = np.flatnonzero(valid_fit[y])
        if not valid_wave.any() or not peaks.size:
            continue

        model = np.zeros(wavelengths.shape, dtype=float)
        for peak in peaks:
            model[valid_wave] += two_half_gaussians(
                wavelengths[valid_wave], amp[y, peak], mu[y, peak],
                max(sigma_left[y, peak], min_sigma),
                max(sigma_right[y, peak], min_sigma),
            )
        restored[y] = model
    return restored


def _profile_residuals(data, restored, spec_var):
    """Use the global median background for the diagnostic residuals."""
    background = np.nanmedian(data)
    residual = (data - background) - restored
    if spec_var is None:
        return residual, residual, f"Data - bkg - restored\n(bkg: {background:.1f})"

    valid_var = np.isfinite(spec_var) & (spec_var > 0)
    spec_err = np.full(data.shape, np.nan)
    spec_err[valid_var] = np.sqrt(spec_var[valid_var])
    title = r"(Data - bkg - restored) / $\sqrt{\rm var}$"
    return residual, residual / spec_err, f"{title}\n(bkg: {background:.1f})"


def plot_lpf_diagnostics(
        LPFs,
        spec_data,
        spec_mask,
        wavelength,
        line,
        spec_var=None,
        doublets=("O2",),
        filename="test_LPF_A.jpg",
        plot_with=None, label_of_plot_with=None, plot_with_color='red'
    ):
    """Reconstruct line profiles, save diagnostic plots, and return residuals.

    LPFs is the tuple (amp, mu, sigma_left, sigma_right), with each array
    shaped (ny, 2). Plain wavelength, center, and width arrays are in angstroms.
    The wavelength grid and spec_mask must match spec_data. True mask entries
    mark valid data. If spec_var is given, the residual panel shows the
    residual divided by the square root of the variance.

    Returns
    -------
    restored : ndarray, shape (ny, nx)
        Reconstructed emission lines without a background.
    residual : ndarray, shape (ny, nx)
        Data minus the global median background and reconstructed lines.
        The returned residual is in flux units.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    Amp, Mu, sigma_l, sigma_r = LPFs

    Amp = np.asarray(Amp, dtype=float)
    Mu = _as_angstrom_values(Mu)
    sigma_l = _as_angstrom_values(sigma_l)
    sigma_r = _as_angstrom_values(sigma_r)
    wave = _as_angstrom_values(wavelength)

    spec_data = np.asarray(spec_data, dtype=float)
    spec_mask = np.asarray(spec_mask, dtype=bool)

    if spec_data.ndim != 2:
        raise ValueError("spec_data must be a two-dimensional array")

    ny, nx = spec_data.shape

    if spec_mask.shape != (ny, nx):
        raise ValueError("spec_mask must have the same shape as spec_data")

    if wave.shape != (ny, nx):
        raise ValueError(
            "wavelength must have the same shape (ny, nx) as spec_data"
        )

    for name, parameter in (
        ("Amp", Amp),
        ("Mu", Mu),
        ("sigma_l", sigma_l),
        ("sigma_r", sigma_r),
    ):
        if parameter.shape != (ny, 2):
            raise ValueError(
                f"{name} must have shape ({ny}, 2), "
                f"got {parameter.shape}"
            )

    if spec_var is not None:
        spec_var = np.asarray(spec_var, dtype=float)

        if spec_var.shape != (ny, nx):
            raise ValueError(
                "spec_var must have the same shape as spec_data"
            )

    use_doublet = line in doublets
    n_lines = 2 if use_doublet else 1
    valid_fit = (
        np.isfinite(Amp) & np.isfinite(Mu)
        & np.isfinite(sigma_l) & np.isfinite(sigma_r)
        & (Amp >= 0) & (sigma_l > 0) & (sigma_r > 0)
    )[:, :n_lines]

    arr = np.where(spec_mask, spec_data, np.nan)
    restored = _reconstruct_profiles(wave, Amp, Mu, sigma_l, sigma_r, valid_fit)
    restored = np.where(spec_mask, restored, np.nan)
    if spec_var is not None:
        spec_var = np.where(spec_mask, spec_var, np.nan)
    residual, residual_for_plot, residual_title = _profile_residuals(
        arr, restored, spec_var
    )

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    panels = [
        (axes[0, 0], arr, "Data", "viridis"),
        (
            axes[0, 1], restored,
            "Asymmetric LPF restored\n(from data, not from velocity model)",
            "viridis",
        ),
        (axes[1, 1], residual_for_plot, residual_title, "coolwarm"),
    ]
    ax_lpf = axes[1, 0]
    divider = make_axes_locatable(ax_lpf)
    ax_sigma = divider.append_axes("left", size="60%", pad=0.02, sharey=ax_lpf)

    all_rows = np.arange(ny)
    y_grid = np.broadcast_to(all_rows[:, None].astype(float), wave.shape)

    for ax_image, image, title, cmap in panels:
        plot_kwargs = {
            "shading": "auto",
            "cmap": cmap,
            "rasterized": True,
        }

        # Center the residual color scale on zero.
        if cmap == "coolwarm":
            finite = image[np.isfinite(image)]

            if finite.size:
                vmax = np.nanpercentile(np.abs(finite), 99)

                if np.isfinite(vmax) and vmax > 0:
                    plot_kwargs.update(vmin=-vmax, vmax=vmax)

        # Plot each pixel at its wavelength, which can vary by spatial row.
        im = ax_image.pcolormesh(
            wave,                        # shape (ny, nx), in angstroms
            y_grid,                      # shape (ny, nx)
            np.ma.masked_invalid(image),
            **plot_kwargs
        )

        ax_image.set_title(title)
        ax_image.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
        ax_image.set_ylabel("Spatial row")
        ax_image.grid(linestyle=':', color='gray', alpha=1)
        fig.colorbar(im, ax=ax_image)

    # Scale opacity by the largest amplitude across the active peaks.
    amp_used = Amp[:, :n_lines]
    finite_amps = amp_used[np.isfinite(amp_used) & (amp_used >= 0)]
    max_alpha = finite_amps.max() if finite_amps.size else np.nan
    markers = ("|", ".")
    colors = ("blue", "tab:orange")

    for num in range(n_lines):
        valid = valid_fit[:, num]
        ys = all_rows[valid]
        centers = Mu[valid, num]
        amps = Amp[valid, num]
        xerr_left = sigma_l[valid, num]
        xerr_right = sigma_r[valid, num]

        # NaNs break the width curves at rows with invalid fits.
        sigma_left_plot = np.where(valid, sigma_l[:, num], np.nan)
        sigma_right_plot = np.where(valid, sigma_r[:, num], np.nan)
        for side, widths, linestyle in (
            ("left", sigma_left_plot, "--"),
            ("right", sigma_right_plot, "-"),
        ):
            ax_sigma.plot(
                widths, all_rows, color=colors[num], linestyle=linestyle,
                linewidth=1.2, label=rf"L#{num + 1}: {side} $\sigma$",
            )

        if centers.size == 0:
            continue

        if np.isfinite(max_alpha) and max_alpha > 0:
            alphas = np.clip(
                amps / max_alpha,
                0.0,
                1.0,
            )
        else:
            alphas = np.ones_like(amps)

        # Use the highest-amplitude point for the legend entry.
        label_index = int(np.nanargmax(amps))

        points = zip(centers, ys, xerr_left, xerr_right, alphas)
        for i, (center, y, xl, xr, alpha) in enumerate(points):
            ax_lpf.errorbar(
                center,
                y,
                xerr=np.array([[xl], [xr]]),
                alpha=float(alpha),
                fmt=markers[num],
                capsize=5,
                color=colors[num],
                label=f"Line {num + 1}" if i == label_index else None,
            )

    # The overlay is a single profile, shown on the last active peak's rows.
    if plot_with is not None:
        overlay_rows = valid_fit[:, -1]
        overlay_centers, overlay_left, overlay_right = (
            values[overlay_rows].to_value(u.Angstrom) for values in plot_with[1:4]
        )
        ax_lpf.errorbar(
            overlay_centers,
            all_rows[overlay_rows],
            xerr=np.array([overlay_left, overlay_right]),
            fmt='x',
            color=plot_with_color,
            alpha=0.5,
            capsize=5,
            label=label_of_plot_with,
            zorder=-1,
        )

        for side, values, color, linestyle in (
            ("left", overlay_left, plot_with_color, "--"),
            ("right", overlay_right, "magenta", "-"),
        ):
            widths = np.full(ny, np.nan)
            widths[overlay_rows] = values
            ax_sigma.plot(
                widths, all_rows, color=color, alpha=0.5, linestyle=linestyle,
                linewidth=1.5, label=rf"L#{n_lines}: {side} $\sigma$",
            )

    ax_lpf.set_title(
        f"{line}: asymmetric "
        + ("double-line" if use_doublet else "single-line")
        + " fit"
    )
    ax_lpf.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    ax_lpf.set_ylabel("Spatial row")
    ax_lpf.set_ylim(-0.5, ny - 0.5)
    ax_lpf.grid(linestyle=':', color='gray', alpha=1)

    # Show shared y tick labels on the leftmost panel.
    ax_lpf.tick_params(axis="y", labelleft=False)

    ax_sigma.set_xlabel(r"$\sigma$ [$\mathrm{\AA}$]")
    ax_sigma.set_ylabel("Spatial row")
    ax_sigma.grid(linestyle=":", color="gray", alpha=1)
    ax_sigma.legend(fontsize=8)

    handles, _ = ax_lpf.get_legend_handles_labels()
    if handles:
        ax_lpf.legend(fontsize=10)

    if np.isfinite(max_alpha) and max_alpha > 0:
        add_colorbar_by_alpha(
            plt,
            fig,
            ax_lpf,
            "blue",
            label="amp",
            bar_low=0,
            bar_high=max_alpha,
        )

    fig.tight_layout()
    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return restored, residual


def add_colorbar_by_alpha(plt, fig, ax, color, label="label", bar_low=0, bar_high=1):
    """Add a colorbar with opacity proportional to the value."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

    rgb = to_rgb(color)
    cmap = LinearSegmentedColormap.from_list("AlphaMap", [(*rgb, 0), (*rgb, 1)])
    norm = Normalize(vmin=bar_low, vmax=bar_high)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    return fig.colorbar(mappable, ax=ax, label=label)


if __name__ == "__main__":
    import joblib
    from pathlib import Path

    slitnum, spec_idx = 109, 3

    data_info_path = Path(f"../scripts/binospec_pkl/pkl/slit_{slitnum}.pkl")
    with data_info_path.open("rb") as file:
        data_info = joblib.load(file)

    # Line profile extracted from the noisy spectrum.
    amp, mean, sigma_left, sigma_right = fit_asymmetric_line_profiles(
        data_info["spec"][spec_idx]["data"],
        data_info["spec"][spec_idx]["meta"]["lambda_grid"],
        line="Ha",
    )

    plot_lpf_diagnostics(
        (amp, mean, sigma_left, sigma_right),
        data_info["spec"][spec_idx]["data"],
        data_info["spec"][spec_idx]["mask"],
        wavelength=data_info["spec"][spec_idx]["meta"]["lambda_grid"],
        line="Ha",
        filename=f"../scripts/binospec_pkl/LPF_slit_{slitnum}_{spec_idx}.png",
    )

    # First asymmetric Gaussian component.
    amp_1 = amp[:, 0].copy()
    mean_1 = mean[:, 0]
    sigma_left_1 = sigma_left[:, 0]
    sigma_right_1 = sigma_right[:, 0]

    # Second asymmetric Gaussian component.
    amp_2 = amp[:, 1].copy()
    mean_2 = mean[:, 1]
    sigma_left_2 = sigma_left[:, 1]
    sigma_right_2 = sigma_right[:, 1]

    # Normalize the amplitudes.
    if np.any(np.isfinite(amp_1)):
        maximum_amp_1 = np.nanmax(amp_1)

        if maximum_amp_1 > 0:
            amp_1 /= maximum_amp_1

    if np.any(np.isfinite(amp_2)):
        maximum_amp_2 = np.nanmax(amp_2)

        if maximum_amp_2 > 0:
            amp_2 /= maximum_amp_2

    # Rows with invalid amplitudes are marked as unreliable.
    reliability_mask = (
        np.isfinite(amp_1)
        & np.isfinite(amp_2)
    )

    amp_1[~reliability_mask] = 0.0
    amp_2[~reliability_mask] = 0.0

    line_profile_1 = {
        "amp": amp_1,
        "mean": mean_1,
        "std_left": sigma_left_1,
        "std_right": sigma_right_1,
        "reliability": reliability_mask,
        "bkg": np.zeros_like(amp_1),
    }

    line_profile_2 = {
        "amp": amp_2,
        "mean": mean_2,
        "std_left": sigma_left_2,
        "std_right": sigma_right_2,
        "reliability": reliability_mask,
        "bkg": np.zeros_like(amp_2),
    }

    with open(f'../scripts/binospec_pkl/LPF_slit_{slitnum}_{spec_idx}.pkl', "wb") as f:
        joblib.dump((line_profile_1, line_profile_2), f)