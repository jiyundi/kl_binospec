import numpy as np

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