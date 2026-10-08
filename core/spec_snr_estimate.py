import numpy as np
from scipy import ndimage

def bkg_estimate(data):
    # -------------------------------------------------
    # 1. 稳健估计背景和单像素噪声
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


    # 第一次从全图估计
    bkg0, sigma0, _ = robust_background(data)

    # -------------------------------------------------
    # 2. 找显著的正发射结构
    # -------------------------------------------------
    # 平滑只用于检测，不用于测量通量
    smooth = ndimage.gaussian_filter(data, sigma=1.0)

    detect_mask = smooth > bkg0 + 3.0 * sigma0

    # 连通区域标记
    labels, nlabels = ndimage.label(detect_mask)

    if nlabels == 0:
        raise RuntimeError("没有检测到显著发射结构")

    # 选择总超额通量最大的连通区域
    scores = np.zeros(nlabels)

    for label_id in range(1, nlabels + 1):
        component = labels == label_id
        scores[label_id - 1] = np.sum(
            np.clip(data[component] - bkg0, 0, None)
        )

    source_label = np.argmax(scores) + 1
    core_mask = labels == source_label

    # 向外扩张，包含发射线翼部
    source_mask = ndimage.binary_dilation(core_mask, iterations=4)

    # 再设置一个更大的保护区，避免线翼污染背景
    guard_mask = ndimage.binary_dilation(source_mask, iterations=4)

    # -------------------------------------------------
    # 3. 用源外像素重新估计背景
    # -------------------------------------------------
    bkg, sigma_pix, n_bkg = robust_background(data[~guard_mask])

    return bkg, source_mask, sigma_pix, n_bkg








if __name__ == '__main__':
    data = np.loadtxt("spec_data.txt")
    assert data.shape == (63, 95)

    bkg, source_mask, sigma_pix, n_bkg = bkg_estimate(data)

    # -------------------------------------------------
    # 4. 积分发射线信号
    # -------------------------------------------------
    source_pixels = data[source_mask]
    n_source = source_pixels.size

    signal = np.sum(source_pixels - bkg)

    # 像素噪声项 + 背景均值估计误差
    noise = sigma_pix * np.sqrt(
        n_source + n_source**2 / n_bkg
    )

    snr = signal / noise

    # 孔径范围，仅用于查看
    yy, xx = np.where(source_mask)

    print("数据形状:", data.shape)
    # print(f"第一次背景估计: {bkg0:.4f}")
    print(f"最终背景估计:   {bkg:.4f}")
    print(f"单像素噪声:     {sigma_pix:.4f}")
    print(f"背景像素数:     {n_bkg}")
    print(f"源孔径像素数:   {n_source}")
    print(f"源的大致行范围: {yy.min()}:{yy.max()+1}")
    print(f"源的大致列范围: {xx.min()}:{xx.max()+1}")
    print(f"积分信号:       {signal:.4f}")
    print(f"积分噪声:       {noise:.4f}")
    print(f"S/N:            {snr:.3f}")