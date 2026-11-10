import numpy as np
from scipy.stats import gaussian_kde

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