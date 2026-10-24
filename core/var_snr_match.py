import numpy as np
from scipy.optimize import brentq

from core.find_snr import snr_estimate
from core.find_bkg import bkg_estimate


def find_background_variance(
    data,
    background_subtracted,
    target_snr,
):
    """
    根据指定的全图 SNR，反求均匀背景噪声方差。

    Parameters
    ----------
    data : array-like, shape (height, width)
        无噪声二维数据。所有像素可能已经减去过同一个背景值。

    background_subtracted : bool
        True:
            输入数据已经减去背景，不再估计或减去背景。
        False:
            使用 bkg_estimate(data) 估计一个标量背景值，
            然后执行 signal = data - background。

    target_snr : float
        期望的全图信噪比，必须为有限正数。

    Returns
    -------
    variance : float
        为达到 target_snr 所需的均匀背景噪声方差。

    Notes
    -----
    假定：
    1. snr_estimate(signal_2d, variance_2d) 返回一个标量；
    2. 在信号固定时，SNR 随背景方差增大而单调减小；
    3. variance 表示方差 sigma**2，而不是标准差 sigma。
    """
    data = np.asarray(data, dtype=np.float64)

    # ---------- 输入检查 ----------
    if data.ndim != 2:
        raise ValueError(
            f"`data` 必须是二维数组，当前 shape={data.shape}。"
        )

    if data.size == 0:
        raise ValueError("`data` 不能为空。")

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "`data` 包含 NaN 或 inf。由于要求在整个二维数组上计算 SNR，"
            "请先处理这些无效像素。"
        )

    if not isinstance(background_subtracted, (bool, np.bool_)):
        raise TypeError("`background_subtracted` 必须是 bool。")

    target_snr = float(target_snr)
    if not np.isfinite(target_snr) or target_snr <= 0:
        raise ValueError("`target_snr` 必须是有限正数。")

    # 不原地修改输入
    signal = data.copy()

    # ---------- 可选背景扣除 ----------
    if not background_subtracted:
        background = float(bkg_estimate(signal))

        if not np.isfinite(background):
            raise ValueError(
                f"`bkg_estimate` 返回了无效背景值：{background}。"
            )

        signal = signal - background

    # 不要删除负像素：
    # 扣除背景之后出现负值通常是合法的。
    if not np.any(signal != 0):
        raise ValueError(
            "扣除背景后的二维信号全部为零，通常无法获得有限正的目标 SNR。"
        )

    # ---------- 定义 SNR(variance) ----------
    def _snr_at_variance(variance):
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("用于计算 SNR 的方差必须是有限正数。")

        variance_map = np.full(
            signal.shape,
            variance,
            dtype=np.float64,
        )

        value = snr_estimate(signal, variance_map)
        value = np.asarray(value)

        if value.size != 1:
            raise ValueError(
                "`snr_estimate` 必须返回一个标量，"
                f"当前返回值 shape={value.shape}。"
            )

        value = float(value.squeeze())

        if not np.isfinite(value):
            raise ValueError(
                f"`snr_estimate` 在 variance={variance:.6g} 时"
                f"返回了无效值：{value}。"
            )

        return value

    def root_function(variance):
        return _snr_at_variance(variance) - target_snr

    # ---------- 自动寻找求根区间 ----------
    signal_power_scale = float(np.mean(np.square(signal)))

    # 初始猜测仅用于提高搜索效率，不改变最终结果
    variance_guess = signal_power_scale / target_snr**2

    tiny = np.finfo(np.float64).tiny
    scale_epsilon = max(
        signal_power_scale * 1e-15,
        np.finfo(np.float64).eps,
    )

    lower = scale_epsilon
    upper = max(variance_guess, lower * 10.0, 1e-12)

    snr_lower = _snr_at_variance(lower)

    # 若极小正方差下的 SNR 都小于目标值，则无法仅通过增加噪声实现目标
    if snr_lower < target_snr:
        raise ValueError(
            "目标 SNR 无法达到：即使背景方差接近 0，"
            f"计算得到的 SNR={snr_lower:.6g}，"
            f"仍小于 target_snr={target_snr:.6g}。"
        )

    # 如果初始上界的 SNR 仍然过高，持续增大方差
    max_expansions = 100
    for _ in range(max_expansions):
        snr_upper = _snr_at_variance(upper)

        if snr_upper <= target_snr:
            break

        upper *= 10.0

        if not np.isfinite(upper):
            raise RuntimeError("搜索方差上界时发生浮点溢出。")
    else:
        raise RuntimeError(
            "自动扩大背景方差范围后仍未包围目标 SNR。"
            "请检查 snr_estimate 是否随方差单调减小，"
            "以及传入的量是否确实为方差而非标准差。"
        )

    # ---------- 求解方差 ----------
    variance = brentq(
        root_function,
        lower,
        upper,
        xtol=1e-12,
        rtol=1e-10,
        maxiter=200,
    )

    return float(variance)


if __name__ == "__main__":
    # variance = find_background_variance(
    #     data=image_2d,
    #     background_subtracted=True,
    #     target_snr=20.0,
    # )

    # print("目标背景方差：", variance)
    # print("对应背景标准差：", np.sqrt(variance))
    print("Found variance.")