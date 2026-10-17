import math
import joblib
import numpy as np
import matplotlib.pyplot as plt

from spec_snr_estimate import bkg_estimate

# --------------------------------------------------
# 1. 读取所有光谱、mask 和 SNR
# --------------------------------------------------
spectra_records = []

for slit_num in range(143):
    filepath = (
        f"../scripts/binospec_pkl/pkl/"
        f"slit_{slit_num:03d}.pkl"
    )

    try:
        with open(filepath, "rb") as file:
            data_info = joblib.load(file)
    except FileNotFoundError:
        print(f"文件不存在，跳过：{filepath}")
        continue

    for spec_idx, spec in enumerate(data_info["spec"]):
        pkl_flux = np.asarray(spec["data"]).squeeze()
        pkl_bkg_var = np.asarray(spec["var"]).squeeze()

        # 如果你的 mask 在其他位置，请修改这一行
        mask = np.asarray(spec["mask"], dtype=bool).squeeze()

        if pkl_flux.shape != mask.shape:
            print(
                f"跳过 slit={slit_num}, spec={spec_idx}："
                f"flux shape={pkl_flux.shape}, "
                f"mask shape={mask.shape}"
            )
            continue

        background = bkg_estimate(pkl_flux[mask])[0]

        spec_snr = (
            np.nansum(pkl_flux[mask])
            / np.sqrt(np.nansum((pkl_flux[mask] - background) + pkl_bkg_var[mask]))
        )
        
        spec_snr = float(np.asarray(spec_snr).squeeze())

        # mask=False 的位置设置成 NaN，画图时会显示为空白
        masked_flux = np.where(mask, pkl_flux, np.nan)

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

def plot_2d_spectra_grid(
    records,
    title,
    output_file,
    cmap_name="viridis",
    ncols=None,
    max_figsize=30,
    panel_width=2.4,
    panel_height=1.8,
    percentile_limits=(1, 99),
    dpi=150,
    show_figure=True,
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
    max_figsize : float
        图像宽、高的最大英寸数。
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

    fig_width = min(
        max_figsize,
        max(6, ncols * panel_width)
    )
    fig_height = min(
        max_figsize,
        max(4, nrows * panel_height)
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        facecolor="white",
    )

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

        annotation = (
            f"slit={record['slit_num']:03d}, "
            f"spec={record['spec_idx']}\n"
            f"SNR={record['spec_snr']:.3f}"
        )

        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
                "pad": 1.0,
            },
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
        fontsize=15,
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.015,
        right=0.995,
        bottom=0.015,
        top=0.965,
        wspace=0.08,
        hspace=0.15,
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
    
plot_2d_spectra_grid(
    positive_spectra,
    title="2D Spectra with SNR >= 0",
    output_file="spectra_2d_snr_ge_0_standard.png",
    max_figsize=20,
    dpi=100,
)

plot_2d_spectra_grid(
    negative_spectra,
    title="2D Spectra with SNR < 0",
    output_file="spectra_2d_snr_lt_0_standard.png",
    max_figsize=20,
    dpi=100,
)