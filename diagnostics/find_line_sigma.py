def find_line_sigma(arr, line, fit_mode='y0!=0',
                    lambda_scale=0.61, verbose=False):
    ny, nx = arr.shape
    noise   = np.std(arr[(arr < 3*np.nanstd(arr)) & (arr > -1*np.nanstd(arr))])

    fit_func = _double_gaussian if line == "O2" else _gaussian

    if fit_mode == 'y0!=0':
        x0_sigma_amp_1 = np.zeros((1+ny, 4)).astype(str)
        x0_sigma_amp_2 = np.zeros((1+ny, 4)).astype(str)
        x0_sigma_amp_1[0] = ['mean1', 'std1', 'amp1', 'shared_bkg']
        x0_sigma_amp_2[0] = ['mean2', 'std2', 'amp2', 'shared_bkg']

        # params:  mean1,   std1,    amp1,      y0,
        bound1 = ((    2,      1,   noise,  -noise),
                  ( nx-2,   nx/2, 9*noise,   noise))
        p0_1   = [  nx/2,   nx/8, 1+noise,       0]

        # params:  mean1,   std1,    amp1,   dmean,   std2,    amp2,      y0
        bound2 = ((    2,      1,   noise,       4,      1,   noise,  -noise),
                  ( nx-2,   nx/2, 9*noise,      10,   nx/2, 9*noise,   noise))
        p0_2   = [  nx/3,   nx/8, 1+noise,       5,   nx/8, 1+noise,       0]

        fit_func = _double_gaussian_nonzero if line == "O2" else _gaussian_nonzero
        bounds = bound2 if line == "O2" else bound1
        p0     = p0_2   if line == "O2" else p0_1

    else:
        x0_sigma_amp_1 = np.zeros((1+ny, 3)).astype(str)
        x0_sigma_amp_2 = np.zeros((1+ny, 3)).astype(str)
        x0_sigma_amp_1[0] = ['mean1', 'std1', 'amp1']
        x0_sigma_amp_2[0] = ['mean2', 'std2', 'amp2']

        # params:  mean1,   std1,    amp1
        bound1 = ((    2,      1,   noise),
                  ( nx-2,   nx/2, 9*noise))
        p0_1   = [  nx/2,  nx/10, 1+noise]

        # params:  mean1,   std1,    amp1,   dmean,   std2,    amp2
        bound2 = ((    2,      1,   noise,       4,      1,   noise),
                  ( nx-2,   nx/2, 9*noise,      10,   nx/2, 9*noise))
        p0_2   = [  nx/3,  nx/10, 1+noise,       5,  nx/10, 1+noise]

        bounds = bound2 if line == "O2" else bound1
        p0     = p0_2   if line == "O2" else p0_1

    # 用列方向求和粗估发射线初始 x0，作为第一行的 warm start 起点
    col_sum  = np.nansum(arr, axis=0)
    x0_init  = float(np.argmax(col_sum))
    p0[0]    = x0_init          # 用粗估 x0 覆盖固定初始猜测
    if line == "O2":
        p0[0] = x0_init * 0.6  # O2 双线：mean1 在左侧

    popt_prev = None  # 记录上一行成功拟合的结果，用于 warm start

    for y in range(ny):
        if verbose:
            print(f'Fitting row #{y}...')

        # ------------------------------------------------------------------
        # NaN 检查：若发射线预期位置附近全是 NaN，跳过本行拟合
        # ------------------------------------------------------------------
        expected_x0 = int(round(popt_prev[0])) if popt_prev is not None \
                      else int(round(x0_init))
        check_cols  = np.arange(max(0, expected_x0 - 2),
                                min(nx, expected_x0 + 3))
        if np.all(np.isnan(arr[y][check_cols])):
            print(f'[SKIP] Row #{y}: x0 region (x≈{expected_x0}) is all NaN, '
                  f'inheriting prev popt')
            if popt_prev is not None:
                if fit_mode == 'y0=0':
                    if line == "O2":
                        m1, s1, a1, dm, s2, a2 = popt_prev
                        m2 = m1 + dm
                        x0_sigma_amp_1[y+1] = m1, s1 * lambda_scale, a1
                        x0_sigma_amp_2[y+1] = m2, s2 * lambda_scale, a2
                    else:
                        m1, s1, a1 = popt_prev
                        x0_sigma_amp_1[y+1] = m1, s1 * lambda_scale, a1
                else:
                    if line == "O2":
                        m1, s1, a1, dm, s2, a2, y0 = popt_prev
                        m2 = m1 + dm
                        x0_sigma_amp_1[y+1] = m1, s1 * lambda_scale, a1, y0
                        x0_sigma_amp_2[y+1] = m2, s2 * lambda_scale, a2, y0
                    else:
                        m1, s1, a1, y0 = popt_prev
                        x0_sigma_amp_1[y+1] = m1, s1 * lambda_scale, a1, y0
            continue  # 跳过本行拟合，popt_prev 保持不变

        # ------------------------------------------------------------------
        # 正常拟合流程
        # ------------------------------------------------------------------
        nan_mask = np.isnan(arr[y])
        xx = np.arange(nx)[~nan_mask]
        yy_ =        arr[y][~nan_mask]

        # Penalty for outliers
        panelty_mask = (yy_ > 3 * np.std(yy_)) | (yy_ < -3 * np.std(yy_))
        panelty_idx  = [i for i, val in enumerate(panelty_mask) if val]
        xx = np.delete(xx,   panelty_idx)
        yy = np.delete(yy_,  panelty_idx)

        assert len(yy) != 0, \
            f'ValueError: Please check Row {y} and raw yy = \n{yy_}'

        if verbose and len(panelty_idx) != 0:
            print(f'[INFO] Outliers of #{y}: x = {panelty_idx}')

        # Warm start：用上一行结果作为初始猜测
        p0_use = popt_prev.tolist() if popt_prev is not None else p0

        try:
            popt, _ = curve_fit(
                fit_func, xx, yy,
                bounds=bounds, p0=p0_use, maxfev=10000
            )

        except RuntimeError:
            print(f'[INFO] Row #{y}: curve_fit failed. Try using least_squares...')

            def _residuals(p, x, y):
                return fit_func(x, *p) - y

            result = least_squares(
                _residuals, p0_use, args=(xx, yy),
                bounds=(bounds[0], bounds[1]),
                max_nfev=10000,
                loss='soft_l1',
                f_scale=1.0
            )
            popt = result.x

        # x0 合理性保护：跳变超过 5 pixel 则 fallback 到上一行
        if popt_prev is not None and abs(popt[0] - popt_prev[0]) > 5:
            print(f'[WARN] Row #{y}: x0 jumped '
                  f'{popt_prev[0]:.1f} -> {popt[0]:.1f}, fallback to prev')
            popt = popt_prev.copy()

        if verbose:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.scatter(xx, yy)
            ax.scatter(xx, fit_func(xx, *popt), label='fitted')
            plt.show()
            plt.close()

        # 存储结果
        if fit_mode == 'y0=0':
            if line == "O2":
                mean1, std1, amp1, dmean, std2, amp2 = popt
                mean2 = mean1 + dmean
                x0_sigma_amp_1[y+1] = mean1, std1 * lambda_scale, amp1
                x0_sigma_amp_2[y+1] = mean2, std2 * lambda_scale, amp2
            else:
                mean1, std1, amp1 = popt
                x0_sigma_amp_1[y+1] = mean1, std1 * lambda_scale, amp1
        else:
            if line == "O2":
                mean1, std1, amp1, dmean, std2, amp2, y0 = popt
                mean2 = mean1 + dmean
                x0_sigma_amp_1[y+1] = mean1, std1 * lambda_scale, amp1, y0
                x0_sigma_amp_2[y+1] = mean2, std2 * lambda_scale, amp2, y0
            else:
                mean1, std1, amp1, y0 = popt
                x0_sigma_amp_1[y+1] = mean1, std1 * lambda_scale, amp1, y0

        popt_prev = popt.copy()  # 更新 warm start

    # Rewrite outliers of sigma & amp with smoothed values
    _, sigmas1 = spike_outlier(x0_sigma_amp_1[1:, 1], window=10, threshold=5)
    _, amps1   = spike_outlier(x0_sigma_amp_1[1:, 2], window=10, threshold=5)
    x0_sigma_amp_1[1:, 1] = sigmas1
    x0_sigma_amp_1[1:, 2] = amps1

    if line == "O2":
        _, sigmas2 = spike_outlier(x0_sigma_amp_2[1:, 1], window=10, threshold=5)
        _, amps2   = spike_outlier(x0_sigma_amp_2[1:, 2], window=10, threshold=5)
        x0_sigma_amp_2[1:, 1] = sigmas2
        x0_sigma_amp_2[1:, 2] = amps2

    print('[INFO] Line width fitting finished.')

    return x0_sigma_amp_1, x0_sigma_amp_2
