import numpy as np
from scipy.optimize import curve_fit


# ==============================
#    1 = EmissionLineFitter
# ==============================
class EmissionLineFitter:
    """负责光谱发射线中心的1D与2D拟合。"""

    def __init__(self, redshift, line, this_line_waves):
        self.redshift = redshift
        self.line = line
        self.this_line_waves = this_line_waves

    # ---------- 基础 Gaussian ----------
    @staticmethod
    def _gaussian(xx, amp, mean, std):
        return amp * np.exp(-0.5 * ((xx - mean) / std) ** 2)

    def _double_gaussian(self, xx, amp1, amp2, mean1, std1):
        """双高斯模型，专为 OII 双线。"""
        mean2  = mean1 + self.delta_pix
        yy1 = amp1 * np.exp(-0.5 * ((xx - mean1) / std1) ** 2)
        yy2 = amp2 * np.exp(-0.5 * ((xx - mean2) / std1) ** 2)
        return yy1 + yy2

    # ---------- 一维拟合 ----------
    def fit_1d(self, flux_2d, x0_fixed=None, y0_fixed=None):
        """在每一行上进行1D Gaussian拟合以找到最亮位置。"""
        spat_len, wave_len = flux_2d.shape
        best_sum = 0
        best_row, best_params = None, None
        xx = np.arange(wave_len)
        
        if self.line == "O2":
            delta_wav = self.this_line_waves[-1] - self.this_line_waves[0]
            wave_per_pixel = np.mean(np.diff(xx))
            delta_pix = delta_wav * (1 + self.redshift) / wave_per_pixel
            self.delta_pix = delta_pix
        
        if x0_fixed==None or y0_fixed==None:
            for row in range(spat_len):
                spec = flux_2d[row, :]
                fit_func = self._double_gaussian if self.line == "O2" else self._gaussian
                p0 = (100, 99, wave_len / 2, 4) if self.line == "O2" else (100, wave_len / 2, 4)
                bound1 = ((0,    0, 1), (np.inf,         spec.shape[0], np.inf))
                bound2 = ((0, 0, 0, 1), (np.inf, np.inf, spec.shape[0], np.inf))
                bounds = bound2 if self.line == "O2" else bound1
                try:
                    popt, _ = curve_fit(fit_func, xx, spec, p0=p0, bounds=bounds, maxfev=100)
                    flux_sum = popt[0] if self.line != "O2" else popt[0] + popt[1]
                    if flux_sum > best_sum:
                        best_sum = flux_sum
                        best_row = row
                        best_params = popt
                except RuntimeError:
                    continue
            
        # 备用：根据用户手动修正覆写参数
        else:
            y0_fixed = int(round(y0_fixed,0))
            spec = flux_2d[y0_fixed, :]
            fit_func = self._double_gaussian if self.line == "O2" else self._gaussian
            
            # Correct double line's manual input
            if self.line == "O2":
                x0_fixed -= 0.5 * self.delta_pix
            
            p0 = (100, 99, x0_fixed, 4) if self.line == "O2" else (100, x0_fixed, 4)
            bound1 = ((     0,         x0_fixed-1,      1), 
                      (np.inf,         x0_fixed+1, np.inf))
            bound2 = ((     0,      0, x0_fixed-1,      1), 
                      (np.inf, np.inf, x0_fixed+1, np.inf))
            bounds = bound2 if self.line == "O2" else bound1
            best_row = y0_fixed
            try:
                popt, _ = curve_fit(fit_func, xx, spec, p0=p0, bounds=bounds, maxfev=100)
                flux_sum = popt[0] if self.line != "O2" else popt[0] + popt[1]
                best_params = popt
            except RuntimeError:
                pass
            
        return best_row, best_params

    # ---------- 二维高斯拟合 ----------
    def fit_2d_single(self, flux, mean_wave, row_guess, manual_override=False):
        """二维单高斯拟合（返回x0, y0, 拟合参数）"""
        ny, nx = flux.shape
        y, x = np.mgrid[:ny, :nx]

        def model(params):
            amp, x0, y0, sigma_x, sigma_y, theta = params
            a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
            b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
            c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
            return amp * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))

        # 粗略初始猜测
        amp0 = np.max(flux)
        init       =  (  amp0, mean_wave  , row_guess+0  ,      3,     3,          0)
        bounds     = ((amp0/8, mean_wave-2, row_guess-4  ,      0,     0, -np.pi / 2), 
                      (np.inf, mean_wave+2, row_guess+4  , np.inf, np.inf, np.pi / 2))
        if manual_override:
            bounds = ((amp0/8, mean_wave-1, row_guess-0.5,      0,     0, -np.pi / 2), 
                      (np.inf, mean_wave+1, row_guess+0.6, np.inf, np.inf, np.pi / 2))

        def fit_func(xy, amp, x0, y0, sx, sy, th):
            return model((amp, x0, y0, sx, sy, th)).ravel()
        
        try:
            popt, _ = curve_fit(fit_func, (x, y), flux.ravel(), p0=init, bounds=bounds, maxfev=5000)
            amp, x0, y0, sx, sy, th = popt
            params = {"Amp1": amp, "dx1": sx, "dy": sy, "PA": th}
            return x0, y0, params
        
        except RuntimeError:
            if manual_override==False:
                return init[1], init[2], {"Amp1": init[0], "dx1": init[3], "dy": init[4], "PA": init[5]}
            else:
                raise RuntimeError
        

    def fit_2d_double(self, flux, mean1, mean2, row_guess, manual_override=False):
        """二维双高斯拟合（返回x0_1, x0_2, y0, params字典）"""
        ny, nx = flux.shape
        y, x = np.mgrid[:ny, :nx]

        def model(params):
            A1, A2, x1, x2, y0, sx, sy, th = params
            a = (np.cos(th) ** 2) / (2 * sx ** 2) + (np.sin(th) ** 2) / (2 * sy ** 2)
            b = -(np.sin(2 * th)) / (4 * sx ** 2) + (np.sin(2 * th)) / (4 * sy ** 2)
            c = (np.sin(th) ** 2) / (2 * sx ** 2) + (np.cos(th) ** 2) / (2 * sy ** 2)
            g1 = A1 * np.exp(-(a * (x - x1) ** 2 + 2 * b * (x - x1) * (y - y0) + c * (y - y0) ** 2))
            g2 = A2 * np.exp(-(a * (x - x2) ** 2 + 2 * b * (x - x2) * (y - y0) + c * (y - y0) ** 2))
            return g1 + g2

        amp0 = np.max(flux)
        init       =  (amp0/2, amp0/2, mean1,   mean2  , row_guess    ,      3,      3,       0)
        bounds     = ((amp0/8, amp0/4, mean1-2, mean2-2, row_guess-4  ,      0,      0,-np.pi/2), 
                      (np.inf, np.inf, mean1+2, mean2+2, row_guess+4  , np.inf, np.inf, np.pi/2))
        if manual_override:
            bounds = ((amp0/8, amp0/4, mean1-2, mean2-2, row_guess-0.5,      0,      0,-np.pi/2), 
                      (np.inf, np.inf, mean1+2, mean2+2, row_guess+0.5, np.inf, np.inf, np.pi/2))

        def fit_func(xy, *p):
            return model(p).ravel()
        
        try:
            popt, _ = curve_fit(fit_func, (x, y), flux.ravel(), p0=init, bounds=bounds, maxfev=4000)
            A1, A2, x1, x2, y0, sx, sy, th = popt
            params = {"Amp1": round(A1, 2), 
                      "Amp2": round(A2, 2), 
                      "dx1":  round(sx, 2), 
                      "dx2":  round(sx, 2), 
                      "dy":   round(sy, 2), 
                      "PA":   round(th, 2)}
            return x1, x2, y0, params
        
        except RuntimeError:
            if manual_override==False:
                return init[2], init[3], init[4],{"Amp1": init[0], 
                                                  "Amp2": init[1], 
                                                  "dx1":  init[5], 
                                                  "dx2":  init[5], 
                                                  "dy":   init[6], 
                                                  "PA":   init[7]}
            else:
                raise RuntimeError


# ==============================
#    2 = SpectrumCropper
# ==============================
class SpectrumCropper:
    """根据拟合中心计算裁剪范围，并执行裁剪。"""

    def __init__(self, specs):
        self.specs = specs

    def crop(self, centers, line, spec_width, spec_height):
        assert len(self.specs)==len(centers), \
               f'# of centers != specs. Do you know why? centers = \n{centers}'
        
        results, how_cut = [], {}
        for i, spec2d in enumerate(self.specs):
            how_cut[f'Set{i}'] = {}
            ny, nx = spec2d['flux'].shape
            try:
                cy = int(round(centers[f'Set{i}']['line1']['y'], 0))
                cx = int(round(np.mean([centers[f'Set{i}']['line1']['x'], 
                                        centers[f'Set{i}']['line2']['x']]), 0))
            except IndexError:
                xs = [f"{x:.1f}" for row in centers for x, _ in row]
                ys = [f"{y:.1f}" for row in centers for _, y in row]
                raise IndexError(f"centers:\ncenters = {centers}\nx = {xs}\ny = {ys}")
            half_h = min(spec_height//2, cy, ny - cy)
            half_w = min(spec_width //2, cx, nx - cx)
            up, dn = int(round(cy - half_h, 0)), int(round(cy + half_h, 0))
            lf, rt = int(round(cx - half_w, 0)), int(round(cx + half_w, 0))
            for key in spec2d:
                spec2d[key] = spec2d[key][up:dn, lf:rt]
            results.append(spec2d)
            how_cut[f'Set{i}'] = {
                'UP': up, 'DN': dn, 
                'LF': lf, 'RT': rt, 
                'up_flt': cy - half_h, 'dn_flt': cy + half_h,  
                'lf_flt': cx - half_w, 'rt_flt': cx + half_w, 
                }
        return results, how_cut


# ==============================
#    3 = Mask 函数
# ==============================
# def mask_bulge_pixels(spec_list, n_lines, how_cut, fac_max):
#     """按发射线形状掩膜中心过亮区域。"""
#     masked_specs = []
#     for i, spec2d in enumerate(spec_list):
#         flux = spec2d['flux']
#         mask = np.ones_like(flux, dtype=bool)
#         for j in range(n_lines):
#             x0 = how_cut[f'Set{i}'][f'line{j+1}']['x0']
#             y0 = how_cut[f'Set{i}'][f'line{j+1}']['y0']
#             dy = how_cut[f'Set{i}'][f'line{j+1}']['dy']
#             dx = how_cut[f'Set{i}'][f'line{j+1}'][f'dx{j+1}']
#             amp= how_cut[f'Set{i}'][f'line{j+1}'][f'Amp{j+1}']
#             yy, xx = np.mgrid[:flux.shape[0], :flux.shape[1]]
#             gauss = amp * np.exp(-((xx - x0) ** 2 / (2 * dx ** 2) + (yy - y0) ** 2 / (2 * dy ** 2)))
#             mask &= (flux < fac_max * gauss)
#         flux_masked = np.where(mask, flux, 0)
#         spec2d['flux'] = flux_masked
#         masked_specs.append(spec2d)
#     return masked_specs


# ==============================
#    4 = EmissionProcessor 总控类
# ==============================
class EmissionProcessor:
    """从初步拟合到裁剪和mask的一站式处理类。"""

    def __init__(self, specA, specC, specB, redshift):
        self.specA, self.specB, self.specC = specA, specB, specC
        self.redshift = redshift

    def process(self, line, this_line_waves, 
                spec_width, spec_height, fac_max=1.0, 
                idx_xs=[None, None, None], 
                idx_ys=[None, None, None]):
        fitter = EmissionLineFitter(self.redshift, line, this_line_waves)

        specs = [self.specA, self.specC, self.specB]
        centers, d2g_params = {}, {}
        
        for i, spec in enumerate(specs):
            centers[f'Set{i}'] = {'line1': None, 'line2': None}
            
            # Step 1.1 - 1D Gaussian 初步拟合中心
            row, params = fitter.fit_1d(np.where(spec['mask'], spec['flux'], 
                                                 np.std(spec['flux'][spec['mask']])), 
                                        x0_fixed=idx_xs[i], 
                                        y0_fixed=idx_ys[i])
            if params is None:
                print('Warning:',
                      f'1D Gaussian fitting failed to find a center of Spec {i}')
                if idx_xs[i] is not None:
                    mean1 = idx_xs[i]
                else:
                    raise IndexError("Failed 1D fit needs a manual override to continue")
            else:
                mean1 = params[1] if line != "O2" else params[2]
            
            if line=='O2':
                print(f'1D fit {i}: index (vertical, horizontal) = ({row:.1f}, '+
                      f'{mean1-fitter.delta_pix/2:.1f}|{mean1+fitter.delta_pix/2:.1f})')
            else:
                print(f'1D fit {i}: index (vertical, horizontal) = ({row:.1f}, '+
                      f'{mean1:.1f})')

            # Step 1.2 - 2D Gaussian 深度拟合中心
            if idx_xs[0] is not None:
                manual_override = True
            else:
                manual_override = False
                        
            if line == "O2":
                delta_wav = this_line_waves[-1] - this_line_waves[0]
                wave_per_pixel = np.mean(np.diff(spec['wave'][0]))
                delta_pix = delta_wav * (1 + self.redshift) / wave_per_pixel
                
                mean2 = mean1 + delta_pix
                x1, x2, y0, ps = fitter.fit_2d_double(
                    np.where(spec['mask'], spec['flux'], 
                             np.std(spec['flux'][spec['mask']])), 
                    mean1, mean2, row, manual_override)
            else:
                x1, y0, ps = fitter.fit_2d_single(
                    np.where(spec['mask'], spec['flux'], 
                             np.std(spec['flux'][spec['mask']])), 
                    mean1, row, manual_override)
                x2 = x1
            centers[f'Set{i}']['line1'] = {'y': round(y0, 1), 'x': round(x1, 1)}
            centers[f'Set{i}']['line2'] = {'y': round(y0, 1), 'x': round(x2, 1)}
            d2g_params[f'Set{i}'] = ps
            
            if line=='O2':
                print(f'2D fit {i}: index (vertical, horizontal) = ({y0:.1f}, '+
                      f'{x1:.1f}|{x2:.1f})')
            else:
                print(f'2D fit {i}: index (vertical, horizontal) = ({y0:.1f}, '+
                      f'{x1:.1f})')
            
        # Step 2 - 裁剪
        cropper = SpectrumCropper(specs)
        cropped, how_cut = cropper.crop(centers, line, 
                                        spec_width, spec_height)
        
        for i in range(len(specs)):
            UP = how_cut[f'Set{i}']['UP']
            LF = how_cut[f'Set{i}']['LF']
            how_cut[f'Set{i}']['fit_par'] = d2g_params[f'Set{i}']
            for j_line in range(2): # support max = doublet line
                how_cut[f'Set{i}'][f'line{j_line+1}'] = {
'y0': round(centers[f'Set{i}'][f'line{j_line+1}']['y'] - UP, 1),
'x0': round(centers[f'Set{i}'][f'line{j_line+1}']['x'] - LF, 1),
}
        
        # Step 3 - mask 发射线中心隆起
        # cropped = mask_bulge_pixels(cropped, 1 if line != "O2" else 2, how_cut, fac_max)
        return cropped, how_cut
