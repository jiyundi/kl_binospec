import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

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