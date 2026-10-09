#!/usr/bin/env python3
"""Fit mock Binospec slit observations with the Nautilus sampler.

The script performs the following steps:

1. Load a mock observation.
2. Add image and spectral noise.
3. Apply synthetic masks to the image.
4. Extract asymmetric H-beta line profiles.
5. Run the Nautilus sampler.
6. Save posterior summaries.
7. Plot the best-fitting model and posterior distributions.

Examples
--------
Standard run:

    python main_fitting_nautilus.py --slit-id 1 --run 1

Short test run:

    python main_fitting_nautilus.py --slit-id 1 --test

Parallel run:

    python main_fitting_nautilus.py --slit-id 1 --parallel
"""

import os

# Set this before importing NumPy and other numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import astropy.units as u
import joblib
import numpy as np
import yaml
from scipy.stats import gaussian_kde

from core.fitting_result_utils import complete_flattened_fit_params
from core.make_config_dic import make_config_dic
from core.post_fitting import plot_obs_fit_res
from klm.line_profile_extraction import (
    exam_line_profile_A,
    extract_asymmetric_lpf_A,
)
from klm.nautilus_sampler import NautilusSampler
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


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_image(
    data,
    mask,
    plotname="image.png",
    increasing_ra=False,
    increasing_dec=True,
):
    """Plot a masked two-dimensional image.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional image data.
    mask : numpy.ndarray
        Boolean mask in which True indicates a valid pixel.
    plotname : str or pathlib.Path
        Output filename.
    increasing_ra : bool
        Whether rightward movement corresponds to increasing RA.
    increasing_dec : bool
        Whether upward movement corresponds to increasing Dec.
    """

    fig, ax = plt.subplots(figsize=(4, 3))

    image = ax.imshow(
        np.where(mask, data, np.nan),
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )

    ax.set_xlabel(
        "Increasing RA →" if increasing_ra else "Decreasing RA →"
    )
    ax.set_ylabel(
        "Increasing Dec →" if increasing_dec else "Decreasing Dec →"
    )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(plotname, dpi=150)
    plt.close(fig)


def plot_spectrum(data, mask, wave=None, plotname="spectrum.png"):
    """Plot a masked two-dimensional slit spectrum.

    Parameters
    ----------
    data : numpy.ndarray
        Two-dimensional spectral data.
    mask : numpy.ndarray
        Boolean mask in which True indicates a valid pixel.
    wave : numpy.ndarray or astropy.units.Quantity, optional
        Two-dimensional wavelength grid.
    plotname : str or pathlib.Path
        Output filename.
    """

    fig, ax = plt.subplots(figsize=(4, 3))

    if wave is None:
        image = ax.imshow(
            np.where(mask, data, np.nan),
            aspect="auto",
            cmap="viridis",
            origin="lower",
        )
        ax.set_xlabel("Wavelength pixels")

    else:
        wave_values = wave.value if isinstance(wave, u.Quantity) else wave

        if wave_values.shape != data.shape:
            raise ValueError(
                "The wavelength grid and spectrum must have the same shape. "
                f"Received {wave_values.shape} and {data.shape}."
            )

        spatial_grid = np.broadcast_to(
            np.arange(data.shape[0])[:, None],
            data.shape,
        )

        masked_data = np.ma.array(
            data,
            mask=(~np.isfinite(data) | ~mask),
        )

        color_map = plt.colormaps["viridis"].copy()
        color_map.set_bad("white")

        image = ax.pcolormesh(
            wave_values,
            spatial_grid,
            masked_data,
            cmap=color_map,
            shading="auto",
        )

        ax.set_xlabel(f"Wavelength (N={wave_values.shape[1]} px)")

    ax.set_ylabel("Slit spatial pixels")
    ax.grid(
        linestyle="--",
        color="silver",
        alpha=0.25,
    )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(plotname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Image masking utility
# ---------------------------------------------------------------------------

def circular_maskout(
    image_data,
    image_mask=None,
    n_circles=1,
    radius_range=(2, 16),
    avoid_center=5,
    random_generator=None,
    max_attempts=1000,
):
    """Mask randomly positioned circular regions in an image.

    Each circular region is kept inside the image boundary and away from the
    image center.

    Parameters
    ----------
    image_data : numpy.ndarray
        Two-dimensional image data.
    image_mask : numpy.ndarray, optional
        Existing boolean mask. True indicates a valid pixel.
    n_circles : int
        Number of circular regions to mask.
    radius_range : tuple of int
        Inclusive minimum and maximum radii, in pixels.
    avoid_center : float
        Additional minimum separation from the image center.
    random_generator : numpy.random.Generator, optional
        Random-number generator.
    max_attempts : int
        Maximum number of placement attempts per circle.

    Returns
    -------
    numpy.ndarray
        Updated valid-pixel mask.
    """

    if image_data.ndim != 2:
        raise ValueError("image_data must be a two-dimensional array.")

    ny, nx = image_data.shape
    radius_min, radius_max = radius_range

    if radius_min <= 0 or radius_max < radius_min:
        raise ValueError(
            f"Invalid radius range: {radius_range}"
        )

    if image_mask is None:
        circular_mask = np.zeros((ny, nx), dtype=bool)
    else:
        circular_mask = ~image_mask.copy()

    if random_generator is None:
        random_generator = np.random.default_rng()

    yy, xx = np.ogrid[:ny, :nx]

    image_center_x = (nx - 1) / 2
    image_center_y = (ny - 1) / 2

    for _ in range(n_circles):
        radius = int(
            random_generator.integers(
                radius_min,
                radius_max + 1,
            )
        )

        # The upper limits of Generator.integers are exclusive.
        x_min = radius
        x_max = nx - radius
        y_min = radius
        y_max = ny - radius

        if x_min >= x_max or y_min >= y_max:
            raise ValueError(
                f"A circle with radius {radius} cannot fit inside an image "
                f"with shape {image_data.shape}."
            )

        for _ in range(max_attempts):
            center_x = int(
                random_generator.integers(x_min, x_max)
            )
            center_y = int(
                random_generator.integers(y_min, y_max)
            )

            distance_to_center = np.hypot(
                center_x - image_center_x,
                center_y - image_center_y,
            )

            if distance_to_center >= radius + avoid_center:
                break

        else:
            raise RuntimeError(
                "Could not place a circular mask after "
                f"{max_attempts} attempts."
            )

        circular_region = (
            (xx - center_x) ** 2
            + (yy - center_y) ** 2
            <= radius**2
        )

        circular_mask |= circular_region

    return ~circular_mask


# ---------------------------------------------------------------------------
# Posterior statistics
# ---------------------------------------------------------------------------

def weighted_median(values, weights):
    """Calculate a one-dimensional weighted median."""

    order = np.argsort(values)

    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative_weights = np.cumsum(sorted_weights)

    return np.interp(
        0.5,
        cumulative_weights,
        sorted_values,
    )


def weighted_kde_mode(values, weights, grid_size=200):
    """Estimate a marginal posterior mode using a weighted KDE."""

    values = np.asarray(values)

    # gaussian_kde fails if all values are identical.
    if np.allclose(values, values[0]):
        return values[0]

    kde = gaussian_kde(
        values,
        weights=weights,
        bw_method="scott",
    )

    evaluation_grid = np.linspace(
        values.min(),
        values.max(),
        grid_size,
    )

    density = kde(evaluation_grid)

    return evaluation_grid[np.argmax(density)]
















# ===========================================================================
# Main workflow
# ===========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # 0. Parse command-line arguments
    # -----------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Fit a mock DEIMOS slit observation."
    )

    parser.add_argument(
        "--slit-id",
        "--slitID",
        dest="slit_id",
        type=int,
        default=1,
        help="Mock slit index. Default: 1.",
    )

    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Run number. Default: 1.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a short test and skip final plots.",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel sampling.",
    )

    args = parser.parse_args()

    slit_id    = args.slit_id
    run_number = args.run
    is_test    = args.test
    use_parallel = args.parallel

    # -----------------------------------------------------------------------
    # Paths: edit these directly when necessary
    # -----------------------------------------------------------------------

    mock_folder = Path("008b_vary_thetaint_slitLPA_major")

    fiducial_yaml = Path(
        "binospec_fid_params_DEIMOS.yaml"
    )

    fitting_yaml = Path(
        "binospec_fitting_params_DEIMOS.yaml"
    )

    output_root = Path(".")

    special_indices = [
        "0.0",
        "0.3",
        "0.7",
        "1.0",
        "1.4",
        "1.6",
        "1.7",
        "2.1",
        "2.4",
        "2.8",
        "3.1",
    ]

    if not 0 <= slit_id < len(special_indices):
        raise ValueError(
            f"slit-id must be between 0 and "
            f"{len(special_indices) - 1}. Received {slit_id}."
        )

    slit_folder = output_root / f"Slit_{slit_id:03d}"
    run_folder = slit_folder / f"run0.{run_number:02d}"
    diagnostic_folder = slit_folder / "diagnostics"

    slit_folder.mkdir(parents=True, exist_ok=True)
    run_folder.mkdir(parents=True, exist_ok=True)
    diagnostic_folder.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load mock observation
    # -----------------------------------------------------------------------

    print()
    print(
        f"Fitting Slit {slit_id:03d} "
        "............................................................."
    )
    print("Current time:", time.ctime())

    data_info_path = (
        mock_folder
        / f"slit_002_{special_indices[slit_id]}.pkl"
    )

    with data_info_path.open("rb") as file:
        data_info = joblib.load(file)

    print(
        "Mock setting: slit_LPA  =",
        data_info["spec"][0]["meta"]["slitLPA"],
    )
    print(
        "Mock setting: theta_int =",
        data_info["fid_params"]["shared_params"]["theta_int"],
    )
    print(
        "Mock setting: cosi      =",
        data_info["fid_params"]["shared_params"]["cosi"],
    )

    # Save the original spectrum before adding observational noise.
    spec_data_no_noise = data_info["spec"][0]["data"].copy()

    # -----------------------------------------------------------------------
    # 1.1 Add realistic image conditions
    # -----------------------------------------------------------------------

    print("CAUTION: adding realistic observational effects to raw data.")

    image_random_generator = np.random.default_rng(seed=slit_id)
    spec_random_generator = np.random.default_rng(seed=slit_id + 1)

    image_data = data_info["image"]["data"]
    image_mask = data_info["image"]["mask"]
    image_var = data_info["image"]["var"]

    # Determine the displayed sky-coordinate directions from the image WCS.
    image_wcs = data_info["image"]["meta"]["wcs"]
    increasing_ra = image_wcs.cd[0, 0] > 0
    increasing_dec = image_wcs.cd[1, 1] > 0

    plot_image(
        image_data,
        image_mask,
        plotname=diagnostic_folder / "image_before_noise.png",
        increasing_ra=increasing_ra,
        increasing_dec=increasing_dec,
    )

    # Add independent Gaussian noise to the image.
    image_noise = np.sqrt(
        np.clip(image_var, a_min=0.0, a_max=None)
    )

    image_data += image_random_generator.normal(
        loc=0.0,
        scale=image_noise,
        size=image_data.shape,
    )

    plot_image(
        image_data,
        image_mask,
        plotname=diagnostic_folder / "image_after_noise.png",
        increasing_ra=increasing_ra,
        increasing_dec=increasing_dec,
    )

    # Mask five synthetic neighboring objects.
    image_mask[:] = circular_maskout(
        image_data,
        image_mask,
        n_circles=5,
        radius_range=(2, 16),
        avoid_center=5,
        random_generator=image_random_generator,
    )

    # Remove non-finite pixels and pixels that fail the positivity condition.
    image_mask &= np.isfinite(image_data)
    image_mask &= image_data + image_var > 0

    plot_image(
        image_data,
        image_mask,
        plotname=diagnostic_folder / "image_after_masking.png",
        increasing_ra=increasing_ra,
        increasing_dec=increasing_dec,
    )

    # -----------------------------------------------------------------------
    # 1.2 Add realistic spectral conditions
    # -----------------------------------------------------------------------

    spec_data = data_info["spec"][0]["data"]
    spec_mask = data_info["spec"][0]["mask"]
    spec_var = data_info["spec"][0]["var"]
    spec_wave = data_info["spec"][0]["meta"]["lambda_grid"]

    plot_spectrum(
        spec_data,
        spec_mask,
        wave=spec_wave,
        plotname=diagnostic_folder / "spectrum_before_noise.png",
    )

    spec_noise = np.sqrt(
        np.clip(spec_var, a_min=0.0, a_max=None)
    )

    white_noise = spec_random_generator.normal(
        loc=0.0,
        scale=spec_noise,
        size=spec_data.shape,
    )

    spec_noisy = spec_data + white_noise

    # Estimate the S/N of the noisy spectrum.
    from core.spec_snr_estimate import bkg_estimate

    background = bkg_estimate(spec_noisy)[0]

    spec_obs_zero_background = np.where(
        spec_mask,
        spec_noisy - background,
        np.nan,
    )

    snr_numerator = np.nansum(spec_obs_zero_background)
    snr_denominator_squared = np.nansum(
        spec_var + spec_obs_zero_background
    )

    if snr_denominator_squared > 0:
        spec_snr = (
            snr_numerator
            / np.sqrt(snr_denominator_squared)
        )
        print("Spectrum S/N after adding noise =", spec_snr)
    else:
        print(
            "WARNING: spectrum S/N could not be calculated because "
            "the denominator is non-positive."
        )

    # Install the noisy spectrum in data_info.
    data_info["spec"][0]["data"] = spec_noisy
    spec_data = data_info["spec"][0]["data"]

    # Update the spectral mask.
    spec_mask &= np.isfinite(spec_data)
    spec_mask &= spec_data + spec_var > 0

    plot_spectrum(
        spec_data,
        spec_mask,
        wave=spec_wave,
        plotname=(
            diagnostic_folder
            / "spectrum_after_noise_and_masking.png"
        ),
    )

    # Remove the existing line-profile representation before replacing it.
    data_info["spec"][0]["meta"].pop("line_profile", None)

    # -----------------------------------------------------------------------
    # 1.3 Extract the asymmetric H-beta line profiles
    # -----------------------------------------------------------------------

    # Reference line profile from the noise-free spectrum.
    amp_reference, mean_reference, sigma_left_reference, \
        sigma_right_reference = extract_asymmetric_lpf_A(
            spec_data_no_noise,
            spec_wave,
            line="Hb",
        )

    # Line profile extracted from the noisy spectrum.
    amp, mean, sigma_left, sigma_right = extract_asymmetric_lpf_A(
        spec_data,
        spec_wave,
        line="Hb",
    )

    _restored, _residual = exam_line_profile_A(
        (amp, mean, sigma_left, sigma_right),
        spec_data,
        spec_mask,
        wavelength=spec_wave,
        line="Hb",
        filename=str(
            diagnostic_folder / f"LPF_mock_{slit_id:03d}.png"
        ),
        plot_with=(
            amp_reference[:, 0],
            mean_reference[:, 0],
            sigma_left_reference[:, 0],
            sigma_right_reference[:, 0],
        ),
        label_of_plot_with="No noise",
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

    data_info["spec"][0]["meta"]["line_profile"] = (
        line_profile_1,
        line_profile_2,
    )

    # -----------------------------------------------------------------------
    # 2. Load the fitting configuration
    # -----------------------------------------------------------------------

    with fiducial_yaml.open("r", encoding="utf-8") as file:
        fiducial_params = yaml.safe_load(file)

    with fitting_yaml.open("r", encoding="utf-8") as file:
        fitting_params = yaml.safe_load(file)

    line_species = [
        spectrum["meta"]["line_species"]
        for spectrum in data_info["spec"]
    ]

    config_dict = make_config_dic(
        line_species,
        fitting_params,
        fiducial_params,
        log10_Mstar=data_info["galaxy"]["log10_Mstar"],
        log10_Mstar_err=data_info["galaxy"]["log10_Mstar_err"],
        use_line_profile="extracted",
    )

    nautilus_sampler = NautilusSampler(
        data_info,
        config_dict,
        not is_test,
    )

    print("\nSampler priors:")

    for parameter, prior in nautilus_sampler.config.params.prior.items():
        print(f"  {parameter}: {prior}")

    print(
        "Slit position angle:",
        data_info["spec"][0]["meta"]["slitLPA"],
    )

    # -----------------------------------------------------------------------
    # 3. Run the Nautilus sampler
    # -----------------------------------------------------------------------

    print("\nStarting Nautilus sampling.")

    start_time = time.perf_counter()

    sampler = nautilus_sampler.run(
        output_dir=str(run_folder),
        test_run=is_test,
        run_num=run_number,
        parallel=use_parallel,
    )

    points, log_weights, log_likelihoods = sampler.posterior()

    elapsed_time = time.perf_counter() - start_time

    print(f"Total fitting time: {elapsed_time:.1f} s")

    # -----------------------------------------------------------------------
    # 4. Save the highest-weight posterior samples
    # -----------------------------------------------------------------------

    # Shift log weights before exponentiation for numerical stability.
    sample_weights = np.exp(
        log_weights - np.max(log_weights)
    )

    output_samples = np.column_stack(
        (
            sample_weights,
            log_likelihoods,
            points,
        )
    )

    # Save samples in the top 5% by posterior weight.
    weight_threshold = np.percentile(
        sample_weights,
        95,
    )

    high_weight_mask = sample_weights >= weight_threshold
    selected_samples = output_samples[high_weight_mask]

    posterior_header = (
        "weight logl "
        + " ".join(nautilus_sampler.config.params.names)
    )

    posterior_path = slit_folder / "post.txt"

    np.savetxt(
        posterior_path,
        selected_samples,
        header=posterior_header,
        comments="",
    )

    print(
        f"Saved {selected_samples.shape[0]} of "
        f"{output_samples.shape[0]} posterior samples to "
        f"{posterior_path}."
    )

    # -----------------------------------------------------------------------
    # 5. Calculate and save posterior summaries
    # -----------------------------------------------------------------------

    if not is_test:
        parameter_names = nautilus_sampler.config.params.names

        # Maximum-likelihood point.
        maximum_likelihood_point = points[
            np.argmax(log_likelihoods)
        ]

        maximum_likelihood_dict = dict(
            zip(parameter_names, maximum_likelihood_point)
        )

        # Normalize posterior weights.
        posterior_weights = sample_weights / sample_weights.sum()

        # Weighted marginal median of each parameter.
        median_point = np.array(
            [
                weighted_median(
                    points[:, parameter_index],
                    posterior_weights,
                )
                for parameter_index in range(points.shape[1])
            ]
        )

        median_dict = dict(
            zip(parameter_names, median_point)
        )

        # Weighted marginal KDE mode of each parameter.
        #
        # This is a collection of one-dimensional marginal modes. It is not
        # necessarily the joint posterior mode.
        mode_point = np.array(
            [
                weighted_kde_mode(
                    points[:, parameter_index],
                    posterior_weights,
                )
                for parameter_index in range(points.shape[1])
            ]
        )

        mode_dict = dict(
            zip(parameter_names, mode_point)
        )

        output_summary = {
            "fid_params": fiducial_params,
            "fitting_params": fitting_params,
            "maximum_likelihood": {
                "point": {
                    key: float(value)
                    for key, value in maximum_likelihood_dict.items()
                },
                "log_likelihood": float(
                    np.max(log_likelihoods)
                ),
            },
            "posterior_median": {
                "point": {
                    key: float(value)
                    for key, value in median_dict.items()
                },
            },
            "posterior_marginal_mode": {
                "point": {
                    key: float(value)
                    for key, value in mode_dict.items()
                },
            },
        }

        best_fit_path = slit_folder / "best_fit.json"

        with best_fit_path.open("w", encoding="utf-8") as file:
            json.dump(
                output_summary,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(f"Saved posterior summary to {best_fit_path}.")

    # -----------------------------------------------------------------------
    # 6. Plot a representative posterior model
    # -----------------------------------------------------------------------

    if not is_test:
        print(
            "Plotting posterior marginal-mode comparison with "
            "the observed spectrum/image."
        )

        fitting_params_flat = Parameters._flatten(
            fitting_params,
            level=1,
        )

        fitting_parameter_template = complete_flattened_fit_params(
            fitting_params_flat,
            line_species=(
                nautilus_sampler
                .config
                .galaxy_params
                .line_species
            ),
        )

        representative_fit_dict = (
            nautilus_sampler.params.gen_param_dict(
                fitting_parameter_template.keys(),
                mode_dict.values(),
            )
        )

        plot_obs_fit_res(
            data_info,
            nautilus_sampler,
            representative_fit_dict,
            fitting_params,
            slit_id,
            save_path=str(slit_folder),
        )

        print("Model-comparison plotting done.")

    else:
        print(
            "Model-comparison plotting skipped because this is "
            "a test run."
        )

    # -----------------------------------------------------------------------
    # 7. Plot the posterior corner diagram
    # -----------------------------------------------------------------------

    if not is_test:
        from core.plot_corner import plot_corner

        # Calibrate v_0 because DEIMOS and Binospec use different
        # definitions of the H-beta reference wavelength.
        SPEED_OF_LIGHT_KMS = 299_792.458

        HB_WAVELENGTH_DEIMOS = 486.1333 * u.nm
        HB_WAVELENGTH_BINOSPEC = 4862.683 * u.Angstrom

        wavelength_shift = (
            HB_WAVELENGTH_DEIMOS
            - HB_WAVELENGTH_BINOSPEC
        )

        velocity_shift_kms = (
            SPEED_OF_LIGHT_KMS
            * wavelength_shift
            / HB_WAVELENGTH_BINOSPEC
        ).decompose().value

        v0_deimos = data_info["fid_params"]["Hb_params"]["v_0"]

        corrected_v0 = v0_deimos + velocity_shift_kms

        print(
            "Corrected DEIMOS v_0 using the Binospec H-beta "
            f"definition: {corrected_v0} km/s"
        )

        true_values = [
            data_info["fid_params"]["shared_params"]["gamma_t"],
            data_info["fid_params"]["shared_params"]["theta_int"],
            data_info["fid_params"]["shared_params"]["vcirc"],
            data_info["fid_params"]["shared_params"]["cosi"],
            data_info["fid_params"]["shared_params"]["r_hl_disk"],
            data_info["fid_params"]["shared_params"]["flux"],
            data_info["fid_params"]["shared_params"]["dx_disk"],
            data_info["fid_params"]["shared_params"]["dy_disk"],
            data_info["fid_params"]["shared_params"]["vscale"],
            corrected_v0,
            data_info["fid_params"]["Hb_params"]["I01"],
            data_info["fid_params"]["Hb_params"]["bkg_level"],
            data_info["fid_params"]["Hb_params"]["dx_vel"],
            1,
            1,
        ]

        plot_corner(
            [str(posterior_path)],
            [f"#{slit_id}"],
            contour_color="dimgray",
            read_latex_from=fitting_params,
            true_values=true_values,
            corner_name=str(slit_folder / "corner_all.png"),
        )

        print("Corner plotting done.")

    else:
        print(
            "Corner plotting skipped because this is a test run."
        )

    print("Run finished.")