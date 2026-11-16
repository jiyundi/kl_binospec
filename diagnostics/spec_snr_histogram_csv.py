import numpy as np
from scipy import ndimage

from core.find_bkg import bkg_estimate



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
    # 3. Test PKL by surveying SNRs 
    # -------------------------------------------------
    import joblib
    import matplotlib.pyplot as plt

    # Specify which SNR do you want:
    which_snr = 'spec' # 'image', 

    try:
        result_array = np.loadtxt('snr_table.csv', delimiter=',')
    
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

            # Option 1: Estimate for image
            if which_snr == 'image':
                pkl_flux    = data_info["image"]["data"]
                pkl_bkg_var = data_info["image"]["var"]
                pkl_mask    = data_info["image"]["mask"]

                background = 0

                image_snr = (
                    np.nansum(pkl_flux[pkl_mask] - background)
                    / np.sqrt(np.nansum((pkl_flux[pkl_mask] - background) + pkl_bkg_var[pkl_mask]))
                )

                records.append(
                    (slit_num, -1, round(image_snr, 1), round(background, 1), round(np.mean(pkl_bkg_var**0.5), 1))
                )

            # Option 2: Estimate for spec
            if which_snr == 'spec':
                for spec_idx, spec in enumerate(data_info["spec"]):
                    pkl_flux    = spec["data"]
                    pkl_bkg_var = spec["var"]
                    pkl_mask    = spec["mask"]
        
                    background = bkg_estimate(pkl_flux[pkl_mask])[0]
        
                    spec_snr = (
                        np.nansum(pkl_flux[pkl_mask] - background)
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
                ("idx", np.int32),
                ("snr", np.float64),
                ("bkg", np.float64),
                ("var", np.float64),
            ],
        )

        # 可选：保存数组
        np.savetxt('snr_table.csv', result_array, delimiter=',')

    print(result_array)
    print(f"总成像/光谱数量：{len(result_array)}")

    # 删除 NaN 和 Inf，避免影响直方图
    try:
        snr_values = result_array["snr"]
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

    plt.xlabel("SNR")
    plt.ylabel("Number of images/spectra")
    plt.title("Distribution of SNR")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"snr_histogram_{which_snr}.png", dpi=300)
    plt.show()