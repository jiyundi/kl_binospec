from dataclasses import dataclass
import numpy as np


@dataclass
class ImageData:
    data:     np.ndarray
    mask:     np.ndarray
    var:      np.ndarray  # Total variance including source Poisson noise.
    var_bkg:  np.ndarray
    meta: dict


@dataclass
class SpecData:
    data:       np.ndarray
    var:        np.ndarray
    cont:       np.ndarray
    mask:       np.ndarray
    meta:   dict
    mask_base: np.ndarray | None = None
    coverage_mask: np.ndarray | None = None
    # Pipeline total variance; var retains the legacy data + var convention.
    var_total: np.ndarray | None = None


@dataclass
class RealDataInfo:
    image:  ImageData
    spec:   list[SpecData]
    galaxy: dict
