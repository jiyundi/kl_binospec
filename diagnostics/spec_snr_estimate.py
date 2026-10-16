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
        print("\033[43m" + 'WARNING:' + "\033[0m " + 
              "No significant emission structure was detected")
        return 0, None, None, 0

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
    # -------------------------------------------------
    # 1. Test DEIMOS
    # -------------------------------------------------
    # data = np.loadtxt("/xdisk/timeifler/jiyundi/kl_binospec/scripts_Pranjal/spec_data.txt")
    # assert data.shape == (63, 95)

    # bkg, source_mask, sigma_pix, n_bkg = bkg_estimate(data)

    # source_pixels = data[source_mask]
    # n_source = source_pixels.size

    # signal = np.sum(source_pixels - bkg)

    # # Approximate aperture extent, for inspection only
    # yy, xx = np.where(source_mask)

    # print("Data shape:", data.shape)
    # print(f"Final background estimate:   {bkg:.4f}")
    # print(f"Per-pixel noise:              {sigma_pix:.4f}")
    # print(f"Number of background pixels: {n_bkg}")
    # print(f"Number of source pixels:     {n_source}")
    # print(f"Approximate source row range: {yy.min()}:{yy.max() + 1}")
    # print(f"Approximate source col range: {xx.min()}:{xx.max() + 1}")
    # print(f"Integrated signal:            {signal:.4f}")





    # -------------------------------------------------
    # 2. Test Binospec
    # -------------------------------------------------

    # import numpy as np
    # import astropy.units as u
    # from astropy.io import fits
    # from astropy.wcs import WCS
    # hdul = fits.open("/xdisk/timeifler/jiyundi/kl_binospec/diagnostics/obj_counts_slits_lin.fits")

    # slit_num = 109
    # hdu = hdul[slit_num]
    # spec2d = np.asarray(hdu.data)
    # hdr = hdu.header.copy()
    # wcs = WCS(hdr)

    # print("pixel axes:", wcs.pixel_n_dim)
    # print("world axes:", wcs.world_n_dim)
    # print("world physical types:", wcs.world_axis_physical_types)
    # print("world units:", wcs.world_axis_units)
    # print("spectral world-axis index:", wcs.wcs.spec)

    # ny, nx = spec2d.shape

    # # FITS/WCS 使用 (x, y)，而 NumPy 数组索引是 [y, x]
    # yy, xx = np.indices((ny, nx))

    # # 最后的 0 表示输入像素坐标采用 Python 的 0-based convention
    # world = wcs.all_pix2world(xx, yy, 0)

    # spec_axis = wcs.wcs.spec
    # if spec_axis < 0:
    #     raise ValueError(
    #         "Astropy 没有识别出 spectral axis；请检查 CTYPE1/CTYPE2。"
    #     )

    # wave_values = world[spec_axis]
    # wave_unit = wcs.world_axis_units[spec_axis]

    # if wave_unit:
    #     wave2d = u.Quantity(wave_values, wave_unit).to(u.AA)
    # else:
    #     # 不建议静默假设单位；先检查 CUNIT1/CUNIT2
    #     raise ValueError(
    #         "WCS 中没有波长单位。请检查 header 的 CUNIT1/CUNIT2。"
    #     )

    # print(wave2d.shape)
    # print("wavelength range:", np.nanmin(wave2d), np.nanmax(wave2d))

    # # 因为文件名含有 lin，而且数据已经 rectified/linearized，很可能每一行使用相同的波长采样。这时只需要取中间一行对应的波长：
    # y0 = ny // 2
    # wave1d = wave2d[y0, :]

    # print(wave1d.shape)
    # print(wave1d[:10])

    # max_difference = np.nanmax(
    #     np.abs(wave2d - wave2d[y0:y0+1, :])
    # )

    # print("Maximum wavelength variation along slit:", max_difference)

    # # Binospec linearized 2D spectrum:
    # # NumPy data[y, x]，其中 x 通常是波长方向
    # x = np.arange(nx, dtype=float)
    # y = np.full(nx, (ny - 1) / 2.0)

    # # 将像素坐标转换为世界坐标
    # world = wcs.all_pix2world(x, y, 0)

    # # 找到波长对应的 world axis
    # spec_axis = wcs.wcs.spec
    # if spec_axis < 0:
    #     physical_types = wcs.world_axis_physical_types
    #     candidates = [
    #         i for i, value in enumerate(physical_types)
    #         if value is not None and (
    #             "em.wl" in value or "spect" in value.lower()
    #         )
    #     ]
    #     if not candidates:
    #         raise RuntimeError(
    #             f"无法识别波长轴；CTYPE={wcs.wcs.ctype}"
    #         )
    #     spec_axis = candidates[0]

    # wave_value = np.asarray(world[spec_axis])

    # # 读取波长单位并转换为 Angstrom
    # wave_unit_string = wcs.world_axis_units[spec_axis]
    # wave_unit = u.Unit(wave_unit_string) if wave_unit_string else u.AA
    # wave = (wave_value * wave_unit).to(u.AA)

    # # 选择 7310–7350 Angstrom
    # wave_min = 7310.0 * u.AA
    # wave_max = 7350.0 * u.AA
    # mask = (
    #     np.isfinite(wave.value)
    #     & (wave >= wave_min)
    #     & (wave <= wave_max)
    # )

    # if not np.any(mask):
    #     raise RuntimeError(
    #         f"数据波长范围为 {np.nanmin(wave):.2f} 到 "
    #         f"{np.nanmax(wave):.2f}，不包含 7310–7350 Angstrom"
    #     )

    # indices = np.where(mask)[0]
    # i0, i1 = indices.min(), indices.max() + 1

    # sub_image = spec2d[:, i0:i1]
    # sub_wave = wave[i0:i1].to_value(u.AA)

    # # Add sky masks
    # sub_image[:,  7:13] = np.nan
    # sub_image[:, 29:33] = np.nan
    # sub_image[:, 47:53] = np.nan

    # # imshow 的横坐标使用像素边界，而不是像素中心
    # if len(sub_wave) > 1:
    #     dw_left = sub_wave[1] - sub_wave[0]
    #     dw_right = sub_wave[-1] - sub_wave[-2]
    # else:
    #     dw_left = dw_right = 1.0

    # extent = [
    #     sub_wave[0] - 0.5 * dw_left,
    #     sub_wave[-1] + 0.5 * dw_right,
    #     -0.5,
    #     ny - 0.5,
    # ]

    # # 使用 asinh 拉伸显示较弱的谱线结构
    # # from astropy.visualization import simple_norm
    # # finite_data = sub_image[np.isfinite(sub_image)]
    # # norm = simple_norm(
    # #     finite_data,
    # #     stretch="asinh",
    # #     percent=99.5
    # # )

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(11, 5))

    # im = ax.imshow(
    #     sub_image,
    #     origin="lower",
    #     aspect="auto",
    #     extent=extent,
    #     cmap="gray",
    #     # norm=norm,
    #     interpolation="nearest",
    # )

    # ax.set_xlim(7310, 7350)
    # ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    # ax.set_ylabel("Spatial pixel along slit")
    # ax.set_title(f"Binospec slit extension {slit_num}: 7310–7350 Å")

    # cbar = fig.colorbar(im, ax=ax, pad=0.02)
    # cbar.set_label("Counts")

    # fig.tight_layout()
    # plt.savefig('test.jpg')

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,3)) # (width, height)
    # ax.hist(sub_image[np.isfinite(sub_image)], bins=50)
    # plt.tight_layout()
    # plt.savefig('test1.jpg')




    # -------------------------------------------------
    # 3. Test PKL
    # -------------------------------------------------
    import joblib
    import matplotlib.pyplot as plt
    
    try:
        result_array = np.loadtxt('spec_snr_table.csv', delimiter=',')
    
    except FileNotFoundError:
        records = []
    
        for slit_num in range(143):
            filepath = f"../scripts/binospec_pkl/pkl/slit_{slit_num:03d}.pkl"
    
            try:
                with open(filepath, "rb") as file:
                    data_info = joblib.load(file)
            except FileNotFoundError:
                print(f"文件不存在，跳过：{filepath}")
                continue
    
            for spec_idx, spec in enumerate(data_info["spec"]):
                pkl_flux    = spec["data"]
                pkl_bkg_var = spec["var"]
                pkl_mask    = spec["mask"]
    
                background = bkg_estimate(pkl_flux[pkl_mask])[0]
    
                spec_snr = (
                    np.nansum(pkl_flux[pkl_mask])
                    / np.sqrt(np.nansum((pkl_flux[pkl_mask] - background) + pkl_bkg_var[pkl_mask]))
                )
    
                records.append(
                    (slit_num, spec_idx, round(spec_snr, 1), round(background, 1), round(np.mean(pkl_bkg_var**0.5), 1))
                )
    
        # 转换为带字段名的 NumPy 结构化数组
        result_array = np.array(
            records,
            dtype=[
                ("slit_num", np.int32),
                ("spec_idx", np.int32),
                ("spec_snr", np.float64),
                ("spec_bkg", np.float64),
                ("spec_var", np.float64),
            ],
        )

        # 可选：保存数组
        np.savetxt('spec_snr_table.csv', result_array, delimiter=',')

    print(result_array)
    print(f"总光谱数量：{len(result_array)}")

    # 删除 NaN 和 Inf，避免影响直方图
    try:
        snr_values = result_array["spec_snr"]
    except IndexError:
        snr_values = result_array[:, 2]
    snr_values = snr_values[np.isfinite(snr_values)]

    # 绘制 spec_snr 直方图
    plt.figure(figsize=(3, 3))
    plt.hist(
        snr_values,
        bins=30,
        edgecolor="black",
        alpha=0.75,
    )

    plt.xlabel("Spectrum SNR")
    plt.ylabel("Number of spectra")
    plt.title("Distribution of Spectrum SNR")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("spec_snr_histogram.png", dpi=300)
    plt.show()