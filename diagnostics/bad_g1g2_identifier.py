import numpy as np
from scipy.ndimage import gaussian_filter1d


def check_g1g2_post(samp_points, pmin=-0.15, pmax=0.15, verbose=False):
    # ---------- 1. peak near boundary ----------
    hist, edges = np.histogram(samp_points, bins=30)
    hist_smooth = gaussian_filter1d(hist, sigma=(pmax - pmin)/15)
    
    peak_bin    = np.argmax(hist_smooth)
    peak_center = (edges[peak_bin] + edges[peak_bin + 1]) / 2
    how_near    = min(peak_center - pmin, pmax - peak_center)
    peak_near_boundary = how_near < 1/10 * (pmax - pmin)
    
    # ---------- 2. posterior width ----------
    width_frac_thresh = 0.25
    posterior_std = np.std(samp_points)
    too_wide = posterior_std > width_frac_thresh * (pmax - pmin)
    
    # ---------- 3. skewness ----------
    # import scipy.stats as stats
    # skew_thresh = 0.8
    # skewness = stats.skew(samp_points)
    # too_skew = np.abs(skewness) > skew_thresh
    
    # ---------- 4. Single peak --------------
    # from scipy.signal import find_peaks
    # hist, edges = np.histogram(samp_points, bins=20)
    # i_peaks, _ = find_peaks(
    #     hist, prominence=1/10*np.max(hist_smooth)
    #     )
    # n_peaks = len(i_peaks)
    # too_many_peaks = n_peaks >= 2
    
    # ---------- final ----------
    is_it_bad = (peak_near_boundary 
                 or too_wide 
                 # or too_skew 
                 # or too_many_peaks
                 )
    
    # if verbose:
    #     print(f'Skewness = {skewness:.2f}, how_near = {how_near:.3f}, posterior_std = {posterior_std:.3f}, n_peaks = {n_peaks}')
    
    return is_it_bad


if __name__ == '__main__':
    for slit in [56]:
        try:
            runsample = np.loadtxt(f'../../../RSCH3/HPC_database/runs_20260601/Slit_{slit:03d}/post.txt', dtype=str, skiprows=0)
        except FileNotFoundError:
            raise FileNotFoundError("Perhaps fitting was not completed for this slit?")
        
        # g1: 2, g2: 3
        g1_points = runsample[1:, 2].astype(float)
        g2_points = runsample[1:, 3].astype(float)
        
        is_g1_bad = check_g1g2_post(g1_points, pmin=-0.15, pmax=0.15, 
                                    verbose=True)
        is_g2_bad = check_g1g2_post(g2_points, pmin=-0.15, pmax=0.15, 
                                    verbose=True)
        print(f'Slit {slit}: g1, g2 bad? - {is_g1_bad}, {is_g2_bad}')

