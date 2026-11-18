import math
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from core.find_snr import snr_estimate

def plot_2d_spectra_grid(
    records,
    title,
    output_file,
    cmap_name="viridis",
    ncols=None,
    max_figsize=None,
    panel_width=1.0,
    panel_height=1.0,
    percentile_limits=(1, 99),
    dpi=150,
    show_figure=False,
):
    """
    将二维光谱绘制成多子图网格。

    Parameters
    ----------
    records : list of dict
        每条记录包含 slit_num、spec_idx、spec_snr、flux。
    title : str
        整张图标题。
    output_file : str
        保存文件名。
    cmap_name : str
        imshow 使用的颜色表。
    ncols : int or None
        每行子图数量。None 时自动计算。
    max_figsize : float or None
        图像宽、高的最大英寸数。默认 None，不限制画布尺寸，
        按 panel_width、panel_height 和网格行列数增长。
    panel_width, panel_height : float
        每个小图（含标题和间距）分配的宽、高，单位为英寸。
    show_figure : bool
        默认 False，仅保存图片，不创建交互窗口。
    percentile_limits : tuple
        每幅光谱计算 vmin、vmax 时使用的百分位数。
    dpi : int
        输出分辨率。
    """

    n_spectra = len(records)

    if n_spectra == 0:
        print(f"{title}：没有符合条件的二维光谱。")
        return

    if ncols is None:
        # 二维光谱通常横向较长，因此网格略偏宽
        ncols = max(
            1,
            math.ceil(math.sqrt(n_spectra * 1.4))
        )

    ncols = min(ncols, n_spectra)
    nrows = math.ceil(n_spectra / ncols)

    fig_width = max(6, ncols * panel_width)
    fig_height = max(4, nrows * panel_height + 0.4)
    if max_figsize is not None:
        fig_width = min(max_figsize, fig_width)
        fig_height = min(max_figsize, fig_height)

    figure_options = dict(
        figsize=(fig_width, fig_height),
        facecolor="white",
        layout="constrained",
    )
    if show_figure:
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, **figure_options)
    else:
        # 独立的 Agg 画布保证保存时不弹窗，也不修改其他脚本的全局后端。
        fig = Figure(**figure_options)
        FigureCanvasAgg(fig)
        axes = fig.subplots(nrows, ncols, squeeze=False)

    axes = axes.ravel()

    # copy() 避免修改 Matplotlib 的全局 colormap
    cmap = plt.get_cmap(cmap_name).copy()

    # masked、NaN 等 bad pixels 显示为白色
    cmap.set_bad(color="white", alpha=1.0)

    p_low, p_high = percentile_limits

    for ax, record in zip(axes, records):
        # 无论输入是 ndarray 还是 MaskedArray，都统一转换为 MaskedArray
        # 同时屏蔽 NaN 和 Inf
        flux = np.ma.masked_invalid(
            np.ma.asarray(record["flux"], dtype=float)
        )
        
        # 提取未被 mask 的有效像素
        valid_values = flux.compressed()

        if valid_values.size > 0:
            vmin, vmax = np.nanpercentile(
                valid_values,
                [p_low, p_high],
            )

            # 防止整幅图几乎是一个常数，导致 vmin == vmax
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin = None
                vmax = None
            elif np.isclose(vmin, vmax):
                delta = max(abs(vmin) * 0.01, 1e-12)
                vmin -= delta
                vmax += delta
        else:
            vmin = None
            vmax = None

        ax.imshow(
            flux,
            origin="lower",
            aspect="auto",
            # interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        panel_title = (
            f"{record['slit_num']}, {record['spec_idx']}: "
            f"SNR={record['spec_snr']:.1f}"
        )

        ax.set_title(
            panel_title,
            ha="left",
            loc="left",
            fontsize=6,
            color="black",
            pad=3,
        )

        # 小图很多时关闭刻度，使画面更清楚
        ax.set_xticks([])
        ax.set_yticks([])

        # 完全没有有效像素时进行提示
        if valid_values.size == 0:
            ax.text(
                0.5,
                0.5,
                "No valid pixels",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="red",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                },
            )

        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("0.5")

    # 隐藏没有使用的子图
    for ax in axes[n_spectra:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{title}  (N={n_spectra})",
        fontsize=12,
    )

    fig.savefig(
        output_file,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    if show_figure:
        plt.show()

    plt.close(fig)

    print(
        f"已保存：{output_file}\n"
        f"光谱数量：{n_spectra}\n"
        f"网格：{nrows} 行 × {ncols} 列\n"
        f"图像尺寸：{fig_width:.1f} × {fig_height:.1f} 英寸"
    )
    








if __name__ == "__main__":
    
    # --------------------------------------------------
    # 1. 读取所有光谱、mask 和 SNR
    # --------------------------------------------------
    spectra_records = []

    for slit_num in range(143):
        filepath = (
            f"../scripts/binospec_pkl_new/pkl/"
            f"slit_{slit_num:03d}.pkl"
        )

        try:
            with open(filepath, "rb") as file:
                data_info = joblib.load(file)
        except FileNotFoundError:
            print(f"文件不存在，跳过：{filepath}")
            continue

        for spec_idx, spec in enumerate(data_info["spec"]):
            flux = np.asarray(spec["data"])
            var  = np.asarray(spec["var"])
            mask = np.asarray(spec["mask"], dtype=bool)

            spec_snr = snr_estimate(
                np.where(mask, flux, np.nan),
                bkg_is_subtracted=False, 
                # restore_CCD_flux=True,
                var_given=np.where(mask, var, np.nan)
            )

            # mask=False 的位置设置成 NaN，画图时会显示为空白
            masked_flux = np.where(mask, flux, np.nan)

            spectra_records.append({
                "slit_num": slit_num,
                "spec_idx": spec_idx,
                "spec_snr": spec_snr,
                "flux": masked_flux,
            })


    print(f"成功读取的光谱总数：{len(spectra_records)}")

    # --------------------------------------------------
    # 2. 根据 SNR 分成两组，NaN 类型的 SNR 不会进入任何一组。
    # --------------------------------------------------

    positive_spectra = [
        record
        for record in spectra_records
        if np.isfinite(record["spec_snr"])
        and record["spec_snr"] >= 0
    ]

    negative_spectra = [
        record
        for record in spectra_records
        if np.isfinite(record["spec_snr"])
        and record["spec_snr"] < 0
    ]

    # 按照 SNR 从大到小排序
    positive_spectra.sort(
        key=lambda record: record["spec_snr"],
        reverse=True,
    )

    negative_spectra.sort(
        key=lambda record: record["spec_snr"],
        reverse=True,
    )

    nan_snr_count = sum(
        not np.isfinite(record["spec_snr"])
        for record in spectra_records
    )

    print(f"SNR >= 0：{len(positive_spectra)}")
    print(f"SNR < 0：{len(negative_spectra)}")
    print(f"SNR 为 NaN 或 Inf：{nan_snr_count}")

    # --------------------------------------------------
    # 3. 绘制包含很多小光谱的整张图片
    # --------------------------------------------------

    plot_2d_spectra_grid(
        positive_spectra,
        title="2D Spectra with SNR >= 0",
        output_file="spectra_2d_positive_snr.png",
        ncols=20,
        dpi=100,
    )

    plot_2d_spectra_grid(
        negative_spectra,
        title="2D Spectra with SNR < 0",
        output_file="spectra_2d_negative_snr.png",
        dpi=100,
    )
