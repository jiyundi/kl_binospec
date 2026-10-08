import warnings
import numpy as np
import astropy.units as u
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def two_half_gaussians(x, amp, mu, sigma_left, sigma_right):
    """单个非对称 Gaussian；x、mu、sigma 均以 Å 的数值表示。"""
    sigma = np.where(x < mu, sigma_left, sigma_right)
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def asymmetric_doublet(x,
                       amp1, mu1, sigma1_left, sigma1_right,
                       amp2, dmu, sigma2_left, sigma2_right):
    """非对称双峰，其中 mu2 = mu1 + dmu。"""
    mu2 = mu1 + dmu

    return (
        two_half_gaussians(
            x, amp1, mu1, sigma1_left, sigma1_right
        )
        + two_half_gaussians(
            x, amp2, mu2, sigma2_left, sigma2_right
        )
    )


def extract_asymmetric_lpf_A(
        data,
        wavelength,
        line,
        doublets=("O2",),
        sigma_bounds_pix=(1.0, 20.0),
        separation_bounds_pix=(4.0, 12.0),
        initial_sigma_pix=3.0):
    """
    每个空间位置取相邻三行平均，并根据 line 进行单峰或双峰拟合。

    Parameters
    ----------
    data : ndarray, shape (ny, nx)
        二维光谱数据。

    wavelength : Quantity or ndarray, shape (nx,)
        每个光谱像素对应的波长。若不是 Quantity，默认单位为 Å。

    line : str
        谱线名称；位于 doublets 中时使用双峰，否则使用单峰。

    doublets : sequence of str
        使用双峰模型的谱线名称。

    sigma_bounds_pix : tuple
        左右翼 sigma 的像素尺度上下限，内部自动转换为 Å。

    separation_bounds_pix : tuple
        双峰间距的像素尺度上下限，内部自动转换为 Å。

    Returns
    -------
    Amp : ndarray, shape (ny, 2)
        两个峰的振幅。单峰拟合的第二列为 0。

    Mu : Quantity, shape (ny, 2)
        峰中心，单位为 u.Angstrom。单峰第二列为 NaN。

    sigma_left, sigma_right : Quantity, shape (ny, 2)
        两个峰的左右翼宽度，单位为 u.Angstrom。
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("data 必须是 shape (ny, nx) 的二维数组")

    ny, nx = data.shape

    # 波长统一转换为 Å 的浮点数，curve_fit 内部不直接处理 Quantity
    if isinstance(wavelength, u.Quantity):
        wave = wavelength.to_value(u.Angstrom)
    else:
        wave = np.asarray(wavelength, dtype=float)

    if wave.shape != (ny, nx):
        raise ValueError("wavelength 必须是与 data 相同的 shape (ny, nx)")

    # 所有行的典型波长采样间隔
    delta_lambda = np.nanmedian(np.abs(np.diff(wave, axis=1)))

    if not np.isfinite(delta_lambda) or delta_lambda <= 0:
        raise ValueError("无法确定有效的波长采样间隔")

    min_sigma = sigma_bounds_pix[0] * delta_lambda
    max_sigma = sigma_bounds_pix[1] * delta_lambda
    sigma0 = initial_sigma_pix * delta_lambda

    min_separation = separation_bounds_pix[0] * delta_lambda
    max_separation = separation_bounds_pix[1] * delta_lambda

    if min_sigma <= 0 or max_sigma <= min_sigma:
        raise ValueError("sigma_bounds_pix 设置不正确")

    if min_separation <= 0 or max_separation <= min_separation:
        raise ValueError("separation_bounds_pix 设置不正确")

    use_doublet = line in doublets
    function_this_line = (
        asymmetric_doublet if use_doublet
        else two_half_gaussians
    )

    Amp = np.full((ny, 2), np.nan)
    Mu = np.full((ny, 2), np.nan)
    sigma_left = np.full((ny, 2), np.nan)
    sigma_right = np.full((ny, 2), np.nan)

    for y in range(ny):

        # 始终使用三行；边界处重复最外侧行
        row_indices = np.clip([y - 1, y, y + 1], 0, ny - 1)

        # 以当前行的波长轴作为三行平均的公共网格
        x_reference = wave[y]
        interpolated_rows = []

        for row in row_indices:
            wave_row = wave[row]
            flux_row = data[row]

            valid_row = np.isfinite(wave_row) & np.isfinite(flux_row)
            if np.count_nonzero(valid_row) < 2:
                continue

            wave_valid = wave_row[valid_row]
            flux_valid = flux_row[valid_row]

            # np.interp 要求插值波长递增
            order = np.argsort(wave_valid)
            wave_valid = wave_valid[order]
            flux_valid = flux_valid[order]

            # 删除重复的波长点
            wave_valid, unique_idx = np.unique(
                wave_valid, return_index=True
            )
            flux_valid = flux_valid[unique_idx]

            interpolated = np.interp(
                x_reference,
                wave_valid,
                flux_valid,
                left=np.nan,
                right=np.nan,
            )
            interpolated_rows.append(interpolated)

        if not interpolated_rows:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            profile = np.nanmean(interpolated_rows, axis=0)

        valid = np.isfinite(x_reference) & np.isfinite(profile)

        if np.count_nonzero(valid) < 8:
            continue

        x = x_reference[valid]
        flux = profile[valid]

        # 按波长排序，保证左右翼定义稳定
        order = np.argsort(x)
        x = x[order]
        flux = flux[order]

        # 当前模型不含背景项，因此先减去中值背景
        background = np.nanmedian(flux)
        flux = flux - background

        if np.nanmax(flux) <= 0:
            continue

        wave_min = np.nanmin(x)
        wave_max = np.nanmax(x)

        # 避免中心紧贴拟合范围边缘
        center_margin = 2.0 * delta_lambda
        center_low = wave_min + center_margin
        center_high = wave_max - center_margin

        if center_low >= center_high:
            continue

        if not use_doublet:
            # ---------------- 单峰 ----------------
            peak_idx = np.nanargmax(flux)

            amp0 = max(flux[peak_idx], 1e-3)
            mu0 = np.clip(x[peak_idx], center_low, center_high)

            p0 = [
                amp0,
                mu0,
                sigma0,
                sigma0,
            ]

            lower_bounds = [
                0.0,
                center_low,
                min_sigma,
                min_sigma,
            ]

            upper_bounds = [
                np.inf,
                center_high,
                max_sigma,
                max_sigma,
            ]

        else:
            # ---------------- 双峰 ----------------
            peak1_idx = np.nanargmax(flux)
            peak1_wave = x[peak1_idx]

            # 排除第一峰附近，再寻找第二峰
            flux_masked = flux.copy()
            suppress = np.abs(x - peak1_wave) < min_separation
            flux_masked[suppress] = -np.inf

            if not np.any(np.isfinite(flux_masked)):
                continue

            peak2_idx = np.argmax(flux_masked)
            peak2_wave = x[peak2_idx]

            # mu1 始终为短波侧峰，mu2 为长波侧峰
            if peak1_wave <= peak2_wave:
                mu1_0 = peak1_wave
                mu2_0 = peak2_wave
                amp1_0 = max(flux[peak1_idx], 1e-3)
                amp2_0 = max(flux[peak2_idx], 1e-3)
            else:
                mu1_0 = peak2_wave
                mu2_0 = peak1_wave
                amp1_0 = max(flux[peak2_idx], 1e-3)
                amp2_0 = max(flux[peak1_idx], 1e-3)

            dmu0 = np.clip(
                mu2_0 - mu1_0,
                min_separation,
                max_separation
            )

            # 为第二峰至少预留最小间距
            mu1_high = center_high - min_separation

            if center_low >= mu1_high:
                continue

            mu1_0 = np.clip(mu1_0, center_low, mu1_high)

            p0 = [
                amp1_0,
                mu1_0,
                sigma0,
                sigma0,
                amp2_0,
                dmu0,
                sigma0,
                sigma0,
            ]

            lower_bounds = [
                0.0,
                center_low,
                min_sigma,
                min_sigma,
                0.0,
                min_separation,
                min_sigma,
                min_sigma,
            ]

            upper_bounds = [
                np.inf,
                mu1_high,
                max_sigma,
                max_sigma,
                np.inf,
                max_separation,
                max_sigma,
                max_sigma,
            ]

        # 确保初始值严格位于 bounds 内
        p0 = np.clip(
            np.asarray(p0, dtype=float),
            np.asarray(lower_bounds) + 1e-10,
            np.asarray(upper_bounds) - 1e-10,
        )

        try:
            popt, _ = curve_fit(
                function_this_line,
                x,
                flux,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue

        if not use_doublet:
            amp, mu, sig_l, sig_r = popt

            Amp[y] = [amp, 0.0]
            Mu[y] = [mu, np.nan]
            sigma_left[y] = [sig_l, 0.0]
            sigma_right[y] = [sig_r, 0.0]

        else:
            (
                amp1,
                mu1,
                sig1_l,
                sig1_r,
                amp2,
                dmu,
                sig2_l,
                sig2_r,
            ) = popt

            mu2 = mu1 + dmu

            # 排除第二峰落到波长范围之外的结果
            if mu2 > center_high:
                continue

            Amp[y] = [amp1, amp2]
            Mu[y] = [mu1, mu2]
            sigma_left[y] = [sig1_l, sig2_l]
            sigma_right[y] = [sig1_r, sig2_r]

    return (
        Amp,
        Mu * u.Angstrom,
        sigma_left * u.Angstrom,
        sigma_right * u.Angstrom,
    )


def exam_line_profile_A(
        LPFs,
        spec_data,
        spec_mask,
        wavelength,
        line,
        spec_var=None,
        doublets=("O2",),
        filename="test_LPF_A.jpg",
        plot_with=None, label_of_plot_with=None, plot_with_color='red'
    ):
    """
    LPFs = (Amp, Mu, sigma_left, sigma_right)

    Parameters
    ----------
    Amp : ndarray, shape (ny, 2)
        两个峰的振幅；单峰时仅使用第 0 列。

    Mu : ndarray or Quantity, shape (ny, 2)
        线心，单位为 Angstrom。

    sigma_left, sigma_right : ndarray or Quantity, shape (ny, 2)
        两个峰的左右翼 sigma，单位为 Angstrom。

    wavelength : ndarray or Quantity, shape (ny, nx)
        每行每个像素对应的波长，单位为 Angstrom。

    line : str
        若位于 doublets 中，则按双峰重建，否则按单峰重建。

    spec_var : ndarray, optional
        光谱方差。提供时绘制 residual / sqrt(var)。

    Returns
    -------
    restored : ndarray, shape (ny, nx)
        重建的发射线，不包含背景。

    residual : ndarray, shape (ny, nx)
        (data - background) - restored，未除以误差。
    """
    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt

    def as_angstrom_value(value):
        """Quantity 转换为 Å 数值；普通数组默认已经是 Å。"""
        if isinstance(value, u.Quantity):
            return value.to_value(u.Angstrom)
        return np.asarray(value, dtype=float)

    Amp, Mu, sigma_l, sigma_r = LPFs

    Amp = np.asarray(Amp, dtype=float)
    Mu = as_angstrom_value(Mu)
    sigma_l = as_angstrom_value(sigma_l)
    sigma_r = as_angstrom_value(sigma_r)
    wave = as_angstrom_value(wavelength)

    spec_data = np.asarray(spec_data, dtype=float)
    spec_mask = np.asarray(spec_mask, dtype=bool)

    if spec_data.ndim != 2:
        raise ValueError("spec_data 必须是二维数组")

    ny, nx = spec_data.shape

    if spec_mask.shape != (ny, nx):
        raise ValueError("spec_mask 必须与 spec_data 形状相同")

    if wave.shape != (ny, nx):
        raise ValueError(
            "wavelength 必须是 shape (ny, nx) 的二维数组"
        )

    for name, parameter in (
        ("Amp", Amp),
        ("Mu", Mu),
        ("sigma_l", sigma_l),
        ("sigma_r", sigma_r),
    ):
        if parameter.shape != (ny, 2):
            raise ValueError(
                f"{name} 必须是 shape ({ny}, 2)，"
                f"当前为 {parameter.shape}"
            )

    if spec_var is not None:
        spec_var = np.asarray(spec_var, dtype=float)

        if spec_var.shape != (ny, nx):
            raise ValueError(
                "spec_var 必须与 spec_data 形状相同"
            )

    use_doublet = line in doublets
    n_lines = 2 if use_doublet else 1

    arr = np.where(spec_mask, spec_data, np.nan)

    # ---------------------------------------------------------
    # 非对称单峰
    # ---------------------------------------------------------
    def half_gaussian(x, amp, mu, sig_l, sig_r):
        sigma = np.where(x < mu, sig_l, sig_r)
        sigma = np.maximum(sigma, np.finfo(float).eps)

        return amp * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

    # ---------------------------------------------------------
    # 使用每行自己的波长坐标重建
    # ---------------------------------------------------------
    restored = np.full((ny, nx), np.nan, dtype=float)
    
    for y in range(ny):
        x = wave[y]
        valid_wave = np.isfinite(x)

        if not np.any(valid_wave):
            continue

        model = np.zeros(nx, dtype=float)
        has_valid_fit = False

        for num in range(n_lines):
            parameters = np.array([
                Amp[y, num],
                Mu[y, num],
                sigma_l[y, num],
                sigma_r[y, num],
            ])

            if not np.all(np.isfinite(parameters)):
                continue

            amp, mu, sig_l, sig_r = parameters

            if amp < 0 or sig_l <= 0 or sig_r <= 0:
                continue

            model[valid_wave] += half_gaussian(
                x[valid_wave],
                amp,
                mu,
                sig_l,
                sig_r,
            )
            has_valid_fit = True

        if has_valid_fit:
            restored[y] = model

    restored = np.where(spec_mask, restored, np.nan)

    # ---------------------------------------------------------
    # 背景、残差及方差归一化
    # ---------------------------------------------------------
    bkg = np.nanmedian(arr)
    residual = (arr - bkg) - restored

    if spec_var is not None:
        valid_var = (
            spec_mask
            & np.isfinite(spec_var)
            & (spec_var > 0)
        )

        spec_err = np.full((ny, nx), np.nan)
        spec_err[valid_var] = np.sqrt(spec_var[valid_var])

        residual_for_plot = residual / spec_err
        residual_title = (
            r"(Data - bkg - restored) / $\sqrt{\rm var}$"
            + "\n"
            + f"(bkg: {bkg:.1f})"
        )
    else:
        residual_for_plot = residual
        residual_title = (
            "Data - bkg - restored"
            + "\n"
            + f"(bkg: {bkg:.1f})"
        )

    # ---------------------------------------------------------
    # 绘制图像
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        2, # rows
        2,
        figsize=(9, 6),
        sharex=True, 
        sharey=True,
    )

    panels = [
        (
            axes[0, 0],
            arr,
            "Data",
            "viridis",
        ),
        (
            axes[0, 1],
            restored,
            "Asymmetric LPF restored"+"\n"+"(from data, not from velocity model)",
            "viridis",
        ),
        (
            axes[1, 1],
            residual_for_plot,
            residual_title,
            "coolwarm",
        ),
        # (
        #     axes[1, 2],
        #     plot_with,
        #     "Model"+"\n"+"(line centers by velocity field)",
        #     "coolwarm",
        # ),
    ]
    ax_lpf = axes[1, 0]
    
    # 在 ax_lpf 左侧添加一个共用 y 轴的 sigma 面板
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_lpf)
    ax_sigma = divider.append_axes(
        "left",
        size="60%",       # sigma 面板宽度
        pad=0.02,
        sharey=ax_lpf,
        )

    # 每个光谱像素对应的空间坐标
    y_grid = np.broadcast_to(
        np.arange(ny, dtype=float)[:, None],
        wave.shape
    )

    for ax_image, image, title, cmap in panels:
        plot_kwargs = {
            "shading": "auto",
            "cmap": cmap,
            "rasterized": True,
        }

        # 残差图使用以 0 为中心的对称色标
        if cmap == "coolwarm":
            finite = image[np.isfinite(image)]

            if finite.size:
                vmax = np.nanpercentile(np.abs(finite), 99)

                if np.isfinite(vmax) and vmax > 0:
                    plot_kwargs.update(vmin=-vmax, vmax=vmax)
        
        # 高级extent - 第k列不要简单画在横坐标k，横坐标也与y有关！
        #              第(y, k)像素画在横坐标wave[y, k]上。
        im = ax_image.pcolormesh(
            wave,                        # shape (ny, nx)，单位 Å
            y_grid,                      # shape (ny, nx)
            np.ma.masked_invalid(image),
            **plot_kwargs
        )

        ax_image.set_title(title)
        ax_image.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
        ax_image.set_ylabel("Spatial row")
        ax_image.grid(linestyle=':', color='gray', alpha=1)
        fig.colorbar(im, ax=ax_image)

    # ---------------------------------------------------------
    # 绘制 LPF 中心和非对称宽度
    # ---------------------------------------------------------
    amp_used = Amp[:, :n_lines]
    finite_amps = amp_used[
        np.isfinite(amp_used) & (amp_used >= 0)
    ]

    if finite_amps.size > 0:
        max_alpha = np.nanmax(finite_amps)
    else:
        max_alpha = np.nan

    markers = ("|", ".")
    colors = ("blue", "tab:orange")

    for num in range(n_lines):
        valid = (
            np.isfinite(Amp[:, num])
            & np.isfinite(Mu[:, num])
            & np.isfinite(sigma_l[:, num])
            & np.isfinite(sigma_r[:, num])
            & (Amp[:, num] >= 0)
            & (sigma_l[:, num] > 0)
            & (sigma_r[:, num] > 0)
        )

        ys = np.arange(ny)[valid]
        centers = Mu[valid, num]
        amps = Amp[valid, num]
        xerr_left = sigma_l[valid, num]
        xerr_right = sigma_r[valid, num]

        # -----------------------------------------------------
        # 在左侧面板画每一行的左右 sigma
        # 使用完整数组并将无效位置设为 NaN，避免跨无效行连线
        # -----------------------------------------------------
        sigma_left_plot = np.full(ny, np.nan)
        sigma_right_plot = np.full(ny, np.nan)

        sigma_left_plot[valid] = sigma_l[valid, num]
        sigma_right_plot[valid] = sigma_r[valid, num]

        all_rows = np.arange(ny)

        ax_sigma.plot(
            sigma_left_plot,
            all_rows,
            color=colors[num],
            linestyle="--",
            linewidth=1.2,
            label=rf"L#{num + 1}: left $\sigma$",
        )

        ax_sigma.plot(
            sigma_right_plot,
            all_rows,
            color=colors[num],
            linestyle="-",
            linewidth=1.2,
            label=rf"L#{num + 1}: right $\sigma$",
        )

        if centers.size == 0:
            continue

        if np.isfinite(max_alpha) and max_alpha > 0:
            alphas = np.clip(
                amps / max_alpha,
                0.0,
                1.0,
            )
        else:
            alphas = np.ones_like(amps)

        # 将标签放在当前峰振幅最大的点上
        label_index = int(np.nanargmax(amps))

        for i, (center, y, xl, xr, alpha) in enumerate(
            zip(
                centers,
                ys,
                xerr_left,
                xerr_right,
                alphas,
            )
            ):
            ax_lpf.errorbar(
                center,
                y,
                xerr=np.array([[xl], [xr]]),
                alpha=float(alpha),
                fmt=markers[num],
                capsize=5,
                color=colors[num],
                label=(
                    f"Line {num + 1}"
                    if i == label_index
                    else None
                    ),
                )
            
    if plot_with is not None:
        ax_lpf.errorbar(
            plot_with[1][valid].to(u.Angstrom).value, 
            ys, 
            xerr=np.array([plot_with[2][valid].to(u.Angstrom).value, 
                           plot_with[3][valid].to(u.Angstrom).value]),
            fmt='x',
            color=plot_with_color, 
            alpha=0.5,
            capsize=5,
            label=label_of_plot_with,
            zorder=-1
            )
        
        sigma_left_plot[valid]  = plot_with[2][valid].to(u.Angstrom).value
        sigma_right_plot[valid] = plot_with[3][valid].to(u.Angstrom).value

        all_rows = np.arange(ny)

        ax_sigma.plot(
            sigma_left_plot,
            all_rows,
            color=plot_with_color,
            alpha=0.5,
            linestyle="--",
            linewidth=1.5,
            label=rf"L#{num + 1}: left $\sigma$",
        )

        ax_sigma.plot(
            sigma_right_plot,
            all_rows,
            color='magenta',
            alpha=0.5,
            linestyle="-",
            linewidth=1.5,
            label=rf"L#{num + 1}: right $\sigma$",
        )

    ax_lpf.set_title(
        f"{line}: asymmetric "
        + ("double-line" if use_doublet else "single-line")
        + " fit"
    )
    ax_lpf.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    ax_lpf.set_ylabel("Spatial row")
    ax_lpf.set_ylim(-0.5, ny - 0.5)
    ax_lpf.grid(linestyle=':', color='gray', alpha=1)

    # y 轴标签只放在最左边的 sigma 面板上
    ax_lpf.tick_params(axis="y", labelleft=False)

    ax_sigma.set_xlabel(r"$\sigma$ [$\mathrm{\AA}$]")
    ax_sigma.set_ylabel("Spatial row")
    ax_sigma.grid(linestyle=":", color="gray", alpha=1)
    ax_sigma.legend(fontsize=8)

    handles, labels = ax_lpf.get_legend_handles_labels()
    if handles:
        ax_lpf.legend(fontsize=10)

    if np.isfinite(max_alpha) and max_alpha > 0:
        add_colorbar_by_alpha(
            plt,
            fig,
            ax_lpf,
            "blue",
            label="amp",
            bar_low=0,
            bar_high=max_alpha,
        )

    fig.tight_layout()
    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return restored, residual


def extract_asymmetric_LPF(data, lambda_scale, line='Hb', 
                           min_separation=3.0, # in Angstrom
                           max_separation=15.0, # in Angstrom
                           min_sigma=1, # in pixels
                           max_sigma=10.0 # in pixels
                           ):
    """
    对每个空间位置取相邻三行的平均谱，并拟合非对称双峰。

    Parameters
    ----------
    data : ndarray, shape (ny, nx)
        二维光谱数据，允许包含 NaN。

    lambda_scale : float
        像素到波长的转换系数。

    Returns
    -------
    Amp : ndarray, shape (ny, 2)
        两个峰的振幅。

    Mu : ndarray, shape (ny, 2)
        两个峰的中心，单位为像素。

    sigma1 : ndarray, shape (ny, 2)
        两个峰的左翼 sigma，单位为波长。

    sigma2 : ndarray, shape (ny, 2)
        两个峰的右翼 sigma，单位为波长。
    """
    doublets = ['O2']
    function_this_line = asymmetric_doublet if line in doublets else two_half_gaussians

    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError("data 必须是二维数组")

    ny, nx = data.shape
    x_all = np.arange(nx, dtype=float)

    Amp = np.full((ny, 2), np.nan)
    Mu = np.full((ny, 2), np.nan)
    sigma1 = np.full((ny, 2), np.nan)
    sigma2 = np.full((ny, 2), np.nan)

    for y in range(ny):

        # 相邻三行；边界处重复边界行
        if y == 0:
            rows = data[[0, 0, 1]]
        elif y == ny - 1:
            rows = data[[ny - 2, ny - 1, ny - 1]]
        else:
            rows = data[y - 1:y + 2]

        with np.errstate(invalid="ignore"):
            profile = np.nanmean(rows, axis=0)

        valid = np.isfinite(profile)
        x = x_all[valid]
        flux = profile[valid]

        if x.size < 10:
            continue

        # 估计并减去背景
        background = np.nanmedian(flux)
        flux = flux - background

        if not np.any(np.isfinite(flux)) or np.nanmax(flux) <= 0:
            continue

        # 最高峰
        peak1_idx = np.nanargmax(flux)
        peak1_x = x[peak1_idx]

        # 屏蔽最高峰附近，再寻找第二峰
        flux_masked = flux.copy()
        suppress = np.abs(x - peak1_x) < (min_separation/lambda_scale)
        flux_masked[suppress] = -np.inf

        if not np.any(np.isfinite(flux_masked)):
            continue

        peak2_idx = np.argmax(flux_masked)
        peak2_x = x[peak2_idx]

        # 按波长位置区分左右峰
        if peak1_x <= peak2_x:
            mu1_0 = peak1_x
            mu2_0 = peak2_x
            amp1_0 = max(flux[peak1_idx], 1e-3)
            amp2_0 = max(flux[peak2_idx], 1e-3)
        else:
            mu1_0 = peak2_x
            mu2_0 = peak1_x
            amp1_0 = max(flux[peak2_idx], 1e-3)
            amp2_0 = max(flux[peak1_idx], 1e-3)

        dmu_0 = np.clip(
            mu2_0 - mu1_0,
            (min_separation/lambda_scale),
            (max_separation/lambda_scale)
        )

        p0 = [
            amp1_0,
            mu1_0,
            2.0,
            2.0,
            amp2_0,
            dmu_0,
            2.0,
            2.0
        ]

        lower_bounds = [
            0.0,
            1.0,
            min_sigma,
            min_sigma,
            0.0,
            (min_separation/lambda_scale),
            min_sigma,
            min_sigma
        ]

        upper_bounds = [
            np.inf,
            nx - 2.0,
            max_sigma,
            max_sigma,
            np.inf,
            (max_separation/lambda_scale),
            max_sigma,
            max_sigma
        ]

        # 防止初值落在边界之外
        p0 = np.clip(
            p0,
            np.asarray(lower_bounds) + 1e-8,
            np.asarray(upper_bounds) - 1e-8
        )

        try:
            popt, _ = curve_fit(
                function_this_line,
                x,
                flux,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue

        # popt 的正确参数顺序
        (
            amp1,
            mu1,
            sigma1_left,
            sigma1_right,
            amp2,
            dmu,
            sigma2_left,
            sigma2_right
        ) = popt

        mu2 = mu1 + dmu

        # 避免第二峰中心超出图像
        if mu2 >= nx:
            continue

        Amp[y] = [amp1, amp2]
        Mu[y] = [mu1, mu2]

        # sigma 从像素转换为波长单位
        sigma1[y] = np.array(
            [sigma1_left, sigma2_left]
        ) * lambda_scale

        sigma2[y] = np.array(
            [sigma1_right, sigma2_right]
        ) * lambda_scale

    return Amp, Mu, sigma1, sigma2


def exam_line_profile(LPFs, spec_data, spec_mask, lambda_scale,
                      spec_var=None, filename="test_LPF.jpg"):
    """
    LPFs = (Amp, Mu, sigma_left, sigma_right)

    Amp、Mu、sigma_left、sigma_right: shape (ny, 2)
    Mu 为像素单位；sigma 为波长单位。
    """
    Amp, Mu, sigma_l, sigma_r = [
        np.asarray(v, dtype=float) for v in LPFs
    ]

    arr = np.where(spec_mask, spec_data, np.nan)
    ny, nx = arr.shape

    if Amp.shape != (ny, 2):
        raise ValueError("LPF参数的形状必须为 (ny, 2)")

    # sigma：波长单位转换回像素，用于重建谱线 [1]
    sigma_l_pix = sigma_l / lambda_scale
    sigma_r_pix = sigma_r / lambda_scale

    def half_gaussian(x, amp, mu, sig_l, sig_r):
        sigma = np.where(x < mu, sig_l, sig_r)
        sigma = np.maximum(sigma, 1e-8)
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    x = np.arange(nx, dtype=float)
    restored = np.full((ny, nx), np.nan)

    for y in range(ny):
        if not np.all(np.isfinite(
            [*Amp[y], *Mu[y], *sigma_l_pix[y], *sigma_r_pix[y]]
        )):
            continue

        restored[y] = (
            half_gaussian(
                x, Amp[y, 0], Mu[y, 0],
                sigma_l_pix[y, 0], sigma_r_pix[y, 0]
            )
            + half_gaussian(
                x, Amp[y, 1], Mu[y, 1],
                sigma_l_pix[y, 1], sigma_r_pix[y, 1]
            )
        )

    restored = np.where(spec_mask, restored, np.nan)
    spec_err = np.where(spec_mask, spec_var**0.5, np.nan)
    bkg      = np.nanmedian(arr)
    residual = (arr - bkg) - restored

    fig, axes = plt.subplots(2, 2, figsize=(7, 7), 
                            #  sharex=True, 
                             sharey=True)

    panels = [
        (arr, "Data", "viridis"),
        (restored, "Asymmetric LPF restored", "viridis"),
        (residual/spec_err, r"(Data - bkg - restored) / $\sqrt{\rm var}$"+"\n"+f"(bkg: {bkg:.1f})", "coolwarm"),
    ]

    for ax, (image, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(
            image, origin="lower", aspect="auto", cmap=cmap
        )
        ax.set_title(title)
        ax.set_xlabel("Spectral pixel")
        ax.set_ylabel("Spatial row")
        fig.colorbar(im, ax=ax)

    # Plot of LPFs
    valid = (
        np.isfinite(Amp[:, 0])
        & np.isfinite(Mu[:, 0])
        & np.isfinite(sigma_l_pix[:, 0])
        & np.isfinite(sigma_r_pix[:, 0])
    )

    ys = np.arange(ny)[valid]
    mu = Mu[valid]
    amps = Amp[valid]
    xerr_left = sigma_l_pix[valid]
    xerr_right = sigma_r_pix[valid]

    max_alpha   = np.nanmax(Amp)
    
    if np.isfinite(max_alpha) and max_alpha > 0:
        alphas = np.clip(amps / max_alpha, 0.0, 1.0)
    else:
        alphas = np.ones_like(amps)
    where_max_a = [np.nanargmax(alphas[:,num]) 
                   for num in range(alphas.shape[1])]

    ax = axes[-1, -1]

    # For each non-NaN row
    for i, (x, y, xl, xr, alpha) in enumerate(
        zip(mu, ys, xerr_left, xerr_right, alphas)
        ):
        
        # For each line
        for num in range(len(amps[0])):
            ax.errorbar(
                x[num],
                y,
                xerr=[[xl[num]], [xr[num]]],  # 非对称左右误差
                alpha=float(alpha[num]),      # 必须是标量
                fmt="|" if num==0 else '.',
                capsize=5,
                color="blue",
                label=f"Line {num+1}" if i==where_max_a[num] 
                                      else None # Only add once
                )

    add_colorbar_by_alpha(
        plt,
        fig,
        ax,
        "blue",
        label="amp",
        bar_low=0,
        bar_high=max_alpha
    )

    plt.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return restored, residual


def add_colorbar_by_alpha(plt, fig, ax, color, label='label', bar_low=0, bar_high=1):
    # 创建一个只改变 alpha 的自定义 cmap，转换成 RGB，然后定义渐变
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    cdict = {
        'red':   [(0.0, rgb[0], rgb[0]), (1.0, rgb[0], rgb[0])],
        'green': [(0.0, rgb[1], rgb[1]), (1.0, rgb[1], rgb[1])],
        'blue':  [(0.0, rgb[2], rgb[2]), (1.0, rgb[2], rgb[2])],
        'alpha': [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] # 这里控制透明度渐变
    }
    alpha_cmap = mcolors.LinearSegmentedColormap('AlphaMap', cdict)

    # 添加色条，创建归一化映射
    norm = mcolors.Normalize(vmin=bar_low, vmax=bar_high)
    
    # 创建 ScalarMappable 并传给 colorbar
    sm = plt.cm.ScalarMappable(cmap=alpha_cmap, norm=norm)
    sm.set_array([]) # 必须设置一个空数组

    # 在指定的子图 ax 旁边添加色条
    cbar = fig.colorbar(sm, ax=ax, label=label)
    
    return cbar
