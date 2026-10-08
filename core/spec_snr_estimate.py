import numpy as np
from scipy import ndimage


def bkg_estimate(data):
    # -------------------------------------------------
    # 1. Robustly estimate the background and per-pixel noise
    # -------------------------------------------------
    def robust_background(values, niter=8, clip=3.0):
        x = np.asarray(values, dtype=float).ravel()
        x = x[np.isfinite(x)]

        for _ in range(niter):
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            sigma = 1.4826 * mad

            if sigma <= 0:
                break

            keep = np.abs(x - med) < clip * sigma

            if keep.sum() == x.size:
                break

            x = x[keep]

        bkg = np.median(x)
        sigma_pix = np.std(x, ddof=1)

        return bkg, sigma_pix, x.size

    # Initial estimate using the entire image
    bkg0, sigma0, _ = robust_background(data)

    # -------------------------------------------------
    # 2. Identify significant positive emission structures
    # -------------------------------------------------
    # Smoothing is used only for detection, not for flux measurement
    smooth = ndimage.gaussian_filter(data, sigma=1.0)

    detect_mask = smooth > bkg0 + 3.0 * sigma0

    # Label connected components
    labels, nlabels = ndimage.label(detect_mask)

    if nlabels == 0:
        raise RuntimeError("No significant emission structure was detected")

    # Select the connected component with the largest total excess flux
    scores = np.zeros(nlabels)

    for label_id in range(1, nlabels + 1):
        component = labels == label_id
        scores[label_id - 1] = np.sum(
            np.clip(data[component] - bkg0, 0, None)
        )

    source_label = np.argmax(scores) + 1
    core_mask = labels == source_label

    # Dilate outward to include the emission-line wings
    source_mask = ndimage.binary_dilation(core_mask, iterations=4)

    # Define a larger guard region to prevent line-wing contamination
    # of the background estimate
    guard_mask = ndimage.binary_dilation(source_mask, iterations=4)

    # -------------------------------------------------
    # 3. Re-estimate the background using off-source pixels
    # -------------------------------------------------
    bkg, sigma_pix, n_bkg = robust_background(data[~guard_mask])

    return bkg, source_mask, sigma_pix, n_bkg


if __name__ == "__main__":
    data = np.loadtxt("/xdisk/timeifler/jiyundi/kl_binospec/scripts_Pranjal/spec_data.txt")
    assert data.shape == (63, 95)

    bkg, source_mask, sigma_pix, n_bkg = bkg_estimate(data)

    # -------------------------------------------------
    # 4. Integrate the emission-line signal
    # -------------------------------------------------
    source_pixels = data[source_mask]
    n_source = source_pixels.size

    signal = np.sum(source_pixels - bkg)

    # Approximate aperture extent, for inspection only
    yy, xx = np.where(source_mask)

    print("Data shape:", data.shape)
    print(f"Final background estimate:   {bkg:.4f}")
    print(f"Per-pixel noise:              {sigma_pix:.4f}")
    print(f"Number of background pixels: {n_bkg}")
    print(f"Number of source pixels:     {n_source}")
    print(f"Approximate source row range: {yy.min()}:{yy.max() + 1}")
    print(f"Approximate source col range: {xx.min()}:{xx.max() + 1}")
    print(f"Integrated signal:            {signal:.4f}")