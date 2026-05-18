"""
Public API
----------
complete_fit_params            : nested config  → flat sorted param dict
complete_flattened_fit_params  : flat config    → flat sorted param dict
analyze_percentile             : MCMC samples  → 16/50/84 percentile dict
load_best_fit_json             : Nautilus JSON → (estimates, best_fit_dict, fitting_par)
"""

import copy
import json

import numpy as np
import yaml

from klm.parameters import Parameters
from core.doublet_utils import (
    DOUBLET_LINES,
    # DOUBLET_PAR_MAP,
    DoubletExpander,
    deduplicate_ordered,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal: flat-param sorter
# ─────────────────────────────────────────────────────────────────────────────

class _FlatParamBuilder:
    """
    Sorts and interleaves a nested fitting-param dict into the canonical
    flat form  ``'{line}_params-{param_name}'``.

    Used exclusively by ``complete_fit_params``.
    """

    def __init__(self, lines: list[str], fitting_par: dict):
        self.lines       = lines
        self.fitting_par = fitting_par

    def build(self) -> dict:
        all_keys = self._collect_shared() + self._collect_and_sort_line()
        return {
            k: self.fitting_par[k.split('-')[0]][k.split('-')[1]]
            for k in all_keys
        }

    def _collect_shared(self) -> list[str]:
        return [
            f'{k1}-{k2}'
            for k1, subdict in self.fitting_par.items()
            if k1 == 'shared_params'
            for k2 in subdict
        ]

    def _collect_and_sort_line(self) -> list[str]:
        line_param_keys = [f'{line}_params' for line in self.lines]
        flat_pairs = [
            (f'{k1}-{k2}', k2)
            for k1, subdict in self.fitting_par.items()
            if k1 in line_param_keys
            for k2 in subdict
        ]
        param_order = deduplicate_ordered(p for _, p in flat_pairs)
        sorted_pairs = sorted(
            flat_pairs,
            key=lambda x: (
                line_param_keys.index(x[0].split('-')[0]),
                param_order.index(x[1]),
            ),
        )
        return self._interleave_doublet([k for k, _ in sorted_pairs])

    @staticmethod
    def _interleave_doublet(keys: list[str]) -> list[str]:
        """
        Insert each O2 doublet partner key immediately after its primary key,
        so corner plots display paired parameters side-by-side.
        """
        result:   list[str] = []
        skip_set: set[str]  = set()

        for key in keys:
            if key in skip_set:
                continue
            result.append(key)

            if not key.startswith('O2_params-'):
                continue

            base    = key.split('-', 1)[1]
            partner = DoubletExpander.partner_name_with_spec(base)
            if partner:
                partner_key = f'O2_params-{partner}'
                if partner_key in keys and partner_key not in skip_set:
                    result.append(partner_key)
                    skip_set.add(partner_key)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Public: parameter builders
# ─────────────────────────────────────────────────────────────────────────────

def complete_fit_params(
        fitting_params: dict,
        line_species: list[str],
        need_sorted_flattened: bool = True,
) -> dict:
    """
    Build the per-line parameter dictionary from a *nested* config dict.

    Handles doublet lines (e.g. O2) by automatically inserting partner
    parameters (``v_0_2``, ``I02``, …) alongside their primary counterparts.

    Parameters
    ----------
    fitting_params : dict
        Nested config dict with keys ``'shared_params'`` and ``'line_params'``.
    line_species : list[str]
        Ordered list of line identifiers, possibly with repeats
        (e.g. ``['O2','O2','O2','Hg','Hg','Hg']``).
    need_sorted_flattened : bool
        When ``True`` (default) return a flat dict keyed
        ``'{line}_params-{param}'``; when ``False`` return the nested form.

    Returns
    -------
    dict
    """
    lines = deduplicate_ordered(line_species)

    # Initialise per-line containers
    fitting_par: dict = {}
    for line in lines:
        fitting_par[f'{line}_params'] = {}

    for key, dic in fitting_params.items():
        if key == 'shared_params':
            fitting_par[key] = dic

        elif key == 'line_params':
            for pname_w_spec, prior_dic in dic.items():
                for line in lines:
                    fitting_par[f'{line}_params'][pname_w_spec] = prior_dic
                    if line in DOUBLET_LINES:
                        partner = DoubletExpander.partner_name_with_spec(pname_w_spec)
                        if partner:
                            fitting_par[f'{line}_params'][partner] = prior_dic

    if not need_sorted_flattened:
        return fitting_par

    return _FlatParamBuilder(lines, fitting_par).build()


def complete_flattened_fit_params(
        fitting_params_flat: dict,
        line_species: list[str],
) -> dict:
    """
    Build the per-line parameter dictionary from an *already-flat* config dict.

    Accepts keys of the form ``'shared_params-re'``, ``'line_params-v_0'``,
    or ``'{line}_params-v_0'``.

    Parameters
    ----------
    fitting_params_flat : dict
    line_species : list[str]

    Returns
    -------
    dict  ``{'{line}_params-{param}': latex_prior_dict, …}``
    """
    lines       = deduplicate_ordered(line_species)
    fitting_par: dict = {}
    # pname_w_spec → {line: {standard_key, latex_prior}}
    pars_this_line: dict = {}

    for key, subdic in fitting_params_flat.items():
        prefix, _, pname_w_spec = key.partition('-')

        # ── shared ────────────────────────────────────────────────────────────
        if prefix == 'shared_params':
            fitting_par[key] = subdic

        # ── generic line params: broadcast to every line ──────────────────────
        elif prefix == 'line_params':
            pars_this_line.setdefault(pname_w_spec, {})
            for line in lines:
                std_key = f'{line}_params-{pname_w_spec}'
                pars_this_line[pname_w_spec][line] = {
                    'standard_key': std_key,
                    'latex_prior':  subdic,
                }
                if line in DOUBLET_LINES:
                    partner_pname = DoubletExpander.partner_name_with_spec(pname_w_spec)
                    if partner_pname:
                        partner_latex = DoubletExpander.partner_latex(subdic['latex_name'])
                        pars_this_line.setdefault(partner_pname, {})
                        pars_this_line[partner_pname][line] = {
                            'standard_key': f'{line}_params-{partner_pname}',
                            'latex_prior': {
                                'latex_name': partner_latex,
                                'prior':      subdic['prior'],
                            },
                        }

        # ── already per-line  e.g. 'O2_params-v_0' ───────────────────────────
        elif prefix.split('_')[0] in lines:
            line    = prefix.split('_')[0]
            std_key = f'{line}_params-{pname_w_spec}'
            subdic_ = copy.deepcopy(subdic)
            fitting_par[std_key] = subdic_

            if line in DOUBLET_LINES:
                partner_pname = DoubletExpander.partner_name_with_spec(pname_w_spec)
                if partner_pname:
                    partner_subdic = copy.deepcopy(subdic)
                    partner_subdic['latex_name'] = DoubletExpander.partner_latex(
                        subdic['latex_name'])
                    fitting_par[f'{line}_params-{partner_pname}'] = partner_subdic

    # Append broadcast params, sorted by line then param
    if pars_this_line:
        for line in lines:
            for par, line_entries in pars_this_line.items():
                try:
                    entry = line_entries[line]
                    fitting_par[entry['standard_key']] = entry['latex_prior']
                except KeyError:
                    pass   # doublet-only secondary params absent for non-doublet lines

    return fitting_par


# ─────────────────────────────────────────────────────────────────────────────
# Public: statistics
# ─────────────────────────────────────────────────────────────────────────────

def analyze_percentile(
        samples: np.ndarray,
        config_filename: str,
        line_species: list[str] | None = None,
) -> dict | str:
    """
    Compute 16th / 50th / 84th percentiles for each sampled parameter.

    Parameters
    ----------
    samples : (N_samples, N_params) ndarray
    config_filename : str
        Path to the YAML fitting config.
    line_species : list[str], optional
        Defaults to ``['O2','O2','O2','Hg','Hg','Hg']``.

    Returns
    -------
    dict  ``{param_key: {'median': …, 'err_lo': …, 'err_hi': …}}``
    or the string ``'ERROR'`` on a parameter-count mismatch.
    """
    if line_species is None:
        line_species = ['O2', 'O2', 'O2', 'Hg', 'Hg', 'Hg']

    with open(config_filename, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)

    config_par  = complete_fit_params(config, line_species)
    param_names = list(config_par.keys())
    n_params    = samples.shape[1]

    # (N_params, 3) array of [p16, p50, p84]
    percentiles = np.percentile(samples, [16, 50, 84], axis=0).T
    result: dict = {}

    for j in range(n_params):
        lo, med, hi = percentiles[j]
        values = np.around([med, lo - med, hi - med], decimals=4)
        try:
            result[param_names[j]] = {
                'median': values[0],
                'err_lo': values[1],
                'err_hi': values[2],
            }
        except IndexError:
            return 'ERROR'

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public: I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_best_fit_json(
        inference,
        fitting_params: dict,
        filename: str,
        config_to_check=None,
) -> tuple:
    """
    Load maximum-likelihood point estimates from a Nautilus JSON results file.

    Parameters
    ----------
    inference : klm Inference object
    fitting_params : dict
        Fitting parameter config (nested or flat).
    filename : str
        Path to the ``.json`` results file produced by Nautilus.
    config_to_check : optional
        If supplied, validates that the JSON parameter count matches the
        config's parameter count; raises ``ValueError`` if they differ.

    Returns
    -------
    estimates : list[float]
    best_fit_dict : dict
    fitting_par : dict  (flat, sorted)
    """
    with open(filename, 'r', encoding='utf-8') as fh:
        estimates = json.load(fh)['maximum_likelihood']['point']

    if isinstance(estimates, dict):
        estimates = list(estimates.values())

    if config_to_check is not None:
        config_params = Parameters._flatten(config_to_check['params'], level=1)
        n_est, n_cfg  = len(estimates), len(config_params)
        if n_est != n_cfg:
            raise ValueError(
                f'JSON file "{filename}" has {n_est} parameters, '
                f'but the config defines {n_cfg}. '
                'Make sure you are using the matching config YAML.'
            )

    fitting_params_flat = Parameters._flatten(fitting_params, level=1)
    fitting_par = complete_flattened_fit_params(
        fitting_params_flat,
        line_species=inference.config.galaxy_params.line_species,
    )
    best_fit_dict = inference.params.gen_param_dict(
        fitting_par.keys(), estimates)

    return estimates, best_fit_dict, fitting_par
