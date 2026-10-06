from dataclasses import dataclass
import numpy as np


@dataclass
class ImageData:
    data:     np.ndarray
    mask:     np.ndarray
    var:      np.ndarray
    var_raw:  np.ndarray
    meta: dict


@dataclass
class SpecData:
    data:       np.ndarray
    var:        np.ndarray
    cont:       np.ndarray
    mask:       np.ndarray
    meta:   dict


@dataclass
class RealDataInfo:
    image:  ImageData
    spec:   list[SpecData]
    galaxy: dict