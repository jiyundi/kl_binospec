#!/usr/bin/env python3
"""Fit real Binospec slit observations with the Nautilus sampler.

The script performs the following steps:

1. Load data from dictionaries packed from observation.
2. Extract asymmetric line profiles.
3. Run the Nautilus sampler.
4. Save posterior summaries.
5. Plot the best-fitting model and posterior distributions.

Examples
--------
Standard run:

    python main_fitting.py --slit-id 1 --run 1

Short test run:

    python main_fitting.py --slit-id 1 --test

Parallel run:

    python main_fitting.py --slit-id 1 --parallel
"""

import os

# Set this before importing NumPy and other numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import galsim
import joblib
import numpy as np
import yaml

from core.fitting_result_utils import complete_flattened_fit_params
from core.make_config_dic import make_config_dic
from core.plot_corner import plot_corner
from core.post_fitting import plot_obs_fit_res
from core.find_post_mode import weighted_kde_mode
from core.find_post_median import weighted_median

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





# ===========================================================================
# Main workflow
# ===========================================================================

# -----------------------------------------------------------------------
# 0. Parse command-line arguments
# -----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Fit a mock Binospec slit observation."
)

parser.add_argument(
    "--slit-id",
    "--slitID",
    dest="slit_id",
    type=int,
    default=1,
    help="Mock slit index. Default: 1",
)

parser.add_argument(
    "--run",
    type=int,
    default=1,
    help="Run number. Default: 1",
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

pkl_folder = Path("binospec_pkl/pkl")
use_what_line_proflie = "meta" #"analytic", 

fiducial_yaml = Path(
    "../config/binospec_fid_params.yaml"
)

fitting_yaml = Path(
    "../config/binospec_fitting_params.yaml"
)

output_root = Path(".")
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

data_info_path = pkl_folder / f"slit_{slit_id:03d}.pkl"

with data_info_path.open("rb") as file:
    data_info = joblib.load(file)

# Historic error: par_fit -> fid_params. Rename for clarity.
if "par_fit" in data_info:
    data_info["fid_params"] = Parameters._unflatten(data_info.pop("par_fit"))

# Recover wcs(galsim.wcs) from ap_wcs
ap_wcs  = data_info['image']['meta']['ap_wcs']
data_info['image']['meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)

# -----------------------------------------------------------------------
# 1.1 Extract the asymmetric line profiles for all lines
# -----------------------------------------------------------------------

for spec_idx in range(len(data_info["spec"])):

    # Line profile extracted from the noisy spectrum.
    amp, mean, sigma_left, sigma_right = extract_asymmetric_lpf_A(
        data_info["spec"][spec_idx]["data"],
        data_info["spec"][spec_idx]["meta"]["lambda_grid"],
        line=data_info["spec"][spec_idx]["meta"]["line_species"],
    )

    exam_line_profile_A(
        (amp, mean, sigma_left, sigma_right),
        data_info["spec"][spec_idx]["data"],
        data_info["spec"][spec_idx]["mask"],
        wavelength=data_info["spec"][spec_idx]["meta"]["lambda_grid"],
        line=data_info["spec"][spec_idx]["meta"]["line_species"],
        filename=str(
            diagnostic_folder / f"LPF_slit{slit_id:03d}_line{spec_idx}.png"
        )
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
    use_what_line_proflie=use_what_line_proflie,
)

nautilus_sampler = NautilusSampler(
    data_info,
    config_dict,
    not is_test,
)

print("\nSampler priors:")

for parameter, prior in nautilus_sampler.config.params.prior.items():
    print(f"  {parameter}: {prior}")

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
    plot_corner(
        [str(posterior_path)],
        [f"#{slit_id}"],
        contour_color="dimgray",
        read_latex_from=fitting_params,
        corner_name=str(slit_folder / "corner_all.png"),
    )

    print("Corner plotting done.")

else:
    print(
        "Corner plotting skipped because this is a test run."
    )

print("Run finished.")