import numpy as np
import matplotlib.pyplot as plt

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
        "Increasing RA -->" if increasing_ra else "Decreasing RA -->"
    )
    ax.set_ylabel(
        "Increasing Dec -->" if increasing_dec else "Decreasing Dec -->"
    )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(plotname, dpi=150)
    plt.close(fig)