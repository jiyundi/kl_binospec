import numpy as np
import pickle

import yaml
from safe_plot import setup; setup() # must before plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.style.use('classic')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})
colors=["#C82423", # "darkred"
        "#F54C22", # "oranred"
        "#8fc4a5", # "greedrk"
        "#1D3557", # "blueblk"
        "#457B9D", # "blueash"
        "#74A9CF", # "blueook"
        "#A8DADC"] # "bluegre"


def analyze_percentile(samples, config_filename):
    # dic_percent = {}
    # nparams = len(samples[0])
    # arr = np.zeros((nparams, 3))
    
    # for j in range(nparams):
    #     samp_points = samples[:,j]
    #     x123 = np.percentile(samp_points, [16, 50, 84])
    #     err_lo, mean, err_hi = x123[0]-x123[1], x123[1], x123[2]-x123[1]
    #     arr[j] = np.around([mean, err_lo, err_hi], decimals=4)
    
    # dic_percent = {
    #     'g1': {        'mean': arr[0,0], 'err_lo': arr[0,1], 'err_hi': arr[0,2]},
    #     'g2': {        'mean': arr[1,0], 'err_lo': arr[1,1], 'err_hi': arr[1,2]},
    #     'vcirc': {     'mean': arr[2,0], 'err_lo': arr[2,1], 'err_hi': arr[2,2]},
    #     'cosi': {      'mean': arr[3,0], 'err_lo': arr[3,1], 'err_hi': arr[3,2]}, 
    #     'theta_int': { 'mean': arr[4,0], 'err_lo': arr[4,1], 'err_hi': arr[4,2]}, 
    #     'vscale': {    'mean': arr[5,0], 'err_lo': arr[5,1], 'err_hi': arr[5,2]}, 
    #     'r_hl_disk': { 'mean': arr[6,0], 'err_lo': arr[6,1], 'err_hi': arr[6,2]},
    #     'r_hl_bulge': {'mean': arr[7,0], 'err_lo': arr[7,1], 'err_hi': arr[7,2]},
    #     'flux': {      'mean': arr[8,0], 'err_lo': arr[8,1], 'err_hi': arr[8,2]},
    #     'flux_bulge': {'mean': arr[9,0], 'err_lo': arr[9,1], 'err_hi': arr[9,2]}, 
    #     'I01': {       'mean': arr[10,0], 'err_lo': arr[10,1], 'err_hi': arr[10,2]}, 
    #     'I02': {       'mean': arr[11,0], 'err_lo': arr[11,1], 'err_hi': arr[11,2]}, 
    #     'bkg_level': { 'mean': arr[12,0], 'err_lo': arr[12,1], 'err_hi': arr[12,2]}, 
    #     }
    # return dic_percent
    with open(config_filename, "r", encoding="utf-8") as file1:
        config = yaml.safe_load(file1)
    config_param_names = []
    for shared_or_line_key, shared_or_line_dict in config.items():
        for key, subsubdict in shared_or_line_dict.items():
            config_param_names.append(key)
    
    dic_percent = {}
    nparams = len(samples[0])
    arr = np.zeros((nparams, 3))
    
    for j in range(nparams):
        samp_points = samples[:,j]
        x123 = np.percentile(samp_points, [16, 50, 84])
        err_lo, mean, err_hi = x123[0]-x123[1], x123[1], x123[2]-x123[1]
        arr[j] = np.around([mean, err_lo, err_hi], decimals=4)
        
        dic_percent[config_param_names[j]] = {'mean':   arr[j,0], 
                                              'err_lo': arr[j,1], 
                                              'err_hi': arr[j,2]}

    return dic_percent

def make_a_dic_for_slits(iter_nums, runs_folder, if_add_noise, 
                         config_filename="../config/binospec_fitting_params.yaml",
                         cosis=None, thetas=None):
    dic_all_iter = {}
    for i in range(len(iter_nums)):
        for j in range(len(iter_nums[0])):
            iter_num = iter_nums[i][j]
            # if iter_num < 75.5:
            try:
                samples = np.load(runs_folder+f'run0.{iter_num:02d}/ultranest_sampler_samples.npy')
            except FileNotFoundError:
                with open(runs_folder+f'run0.{iter_num:02d}/ultranest_sampler_results.pkl', 'rb') as f:
                    sample_results = pickle.load(f) # read
                # sample_results = np.load(f'../../../../RSCH3/kl_github/runs_March_2025/run0.{iter_num}/ultranest_sampler_results.npy')
                samples        = sample_results['samples']
            percentile = analyze_percentile(samples, config_filename)
            dic_all_iter[iter_num] = {
                    "iter_num":   iter_num, 
                    "add_noise":  if_add_noise,
                    "spec_snr":   spec_snrs[i][j],
                    "percentile": percentile
                    }
            if cosis  is not None:
                dic_all_iter[iter_num]["cosi"] = cosis[i][j]
            if thetas is not None:
                dic_all_iter[iter_num]["theta_int"] = thetas[i][j]
    return dic_all_iter

def average_dicts(data_array, axis=0):
    """
    对字典数组在指定轴方向求平均。
    axis=0 表示对每列进行平均（输出长度为5）
    axis=1 表示对每行进行平均（输出长度为6）
    """
    result = []

    if axis == 0:
        # 沿列方向平均（对每列 i）
        num_cols = data_array.shape[1]
        for i in range(num_cols):
            col_dicts = [row[i] for row in data_array]  # 第 i 列的所有字典（共6个）
            keys = col_dicts[0].keys()
            avg_dict = {}
            for key in keys:
                values = [d[key] for d in col_dicts]
                avg_dict[key] = np.mean(values)
            result.append(avg_dict)

    elif axis == 1:
        # 沿行方向平均（对每行 row）
        for row in data_array:
            keys = row[0].keys()
            avg_dict = {}
            for key in keys:
                values = [d[key] for d in row]
                avg_dict[key] = np.mean(values)
            result.append(avg_dict)

    else:
        raise ValueError("axis must be 0 (columns) or 1 (rows)")

    return result

def plot_diagnostics(dic_all_iter, n_spec_snr, 
                     first_runs_are_changing_param_A, 
                     param_B_maybe_not_periodic_in_your_runs,
                     average_over_param_B=True,
                     runs_folder='Where_is_your_runs', 
                     compare=None, comparename=None, labelname='?',
                     add_covert_cosi=False):
    first_runs_are_changing_param_A         = first_runs_are_changing_param_A[0]
    param_B_maybe_not_periodic_in_your_runs = param_B_maybe_not_periodic_in_your_runs[:, 0]
    n_param_P  = len(first_runs_are_changing_param_A)
    n_param_NP = len(param_B_maybe_not_periodic_in_your_runs)
    
    if add_covert_cosi==True: 
        total_n_spec_snr = n_spec_snr * 2
    else:
        total_n_spec_snr = n_spec_snr
    
    fig = plt.figure(figsize=(6*n_param_to_see, 3*total_n_spec_snr))  # (length, height)
    plt.subplots_adjust(hspace=0.4, wspace=0.2) # h=height
    gs = fig.add_gridspec(nrows=total_n_spec_snr, ncols=n_param_to_see, 
                          height_ratios=[1]*total_n_spec_snr, 
                          width_ratios=[1]*n_param_to_see)
    all_ylims = [-0.2, 0.2] # default
    
    g12_mean_off = [[], []]
    g12_error    = [[[], []], [[], []]]
    for i in range(n_spec_snr):
        this_spec_snr = spec_snrs[i,0]
        for k in range(len(want_to_see)):
            param = want_to_see[k]
            
            g1_values   = [dic["percentile"][param] 
                           for i_num, dic in dic_all_iter.items() 
                           if dic["spec_snr"] == this_spec_snr]
            g1_values   = np.array(g1_values).reshape(n_param_NP, n_param_P)
            
            # Average
            if average_over_param_B == True:
                g1_values = average_dicts(g1_values, axis=0)
            elif average_over_param_B is not True:
                g1_values = average_dicts(g1_values, axis=1)
            
            g1_mean     = np.array([g1_this_i['mean'  ] for g1_this_i in g1_values])
            g1_error_lo = np.array([g1_this_i['err_lo'] for g1_this_i in g1_values])
            g1_error_hi = np.array([g1_this_i['err_hi'] for g1_this_i in g1_values])
            g1_mean_off = g1_mean - want_to_see_ref[k]
            g1_error    = [np.abs(g1_error_lo), g1_error_hi]
            g12_mean_off[k], g12_error[k] = g1_mean_off, g1_error
            
            ax1 = fig.add_subplot(gs[i, k])
            
            if average_over_param_B == True:
                xaxis_values = first_runs_are_changing_param_A
                xaxis_label  = r'$\cos{(i)}$'
            elif average_over_param_B is not True:
                xaxis_values = param_B_maybe_not_periodic_in_your_runs / np.pi * 180
                xaxis_label  = r'$\theta_{\rm int}$ (deg)'
            ax1.errorbar(xaxis_values, g1_mean_off, yerr=g1_error, 
                         fmt=' ', capsize=5, capthick=2, elinewidth=4, 
                         marker='o', markersize=6, color=colors[k+4], alpha=1.0, 
                         label=param+f' ({labelname})', zorder=2)
            if add_covert_cosi==True:
                ax5 = fig.add_subplot(gs[i+n_spec_snr, k])
                sini_values = np.sqrt(1-np.array(xaxis_values)**2)
                ax5.errorbar(sini_values, g1_mean_off, yerr=g1_error, 
                             fmt=' ', capsize=5, capthick=2, elinewidth=4, 
                             marker='o', markersize=6, color=colors[k+4], alpha=1.0, 
                             label=param+f' ({labelname})', zorder=2)
                
            if compare != None:
                cosi_compar = first_runs_are_changing_param_A
                sini_compar = sini_values
                g1_compare  = [dic["percentile"][param] 
                               for i_num, dic in compare.items() 
                               if dic["spec_snr"] == this_spec_snr]
                g1_compare  = np.array(g1_compare).reshape(n_param_NP, n_param_P)
                
                # Average
                if average_over_param_B == True:
                    g1_compare = average_dicts(g1_compare, axis=0)
                elif average_over_param_B is not True:
                    g1_compare = average_dicts(g1_compare, axis=1)
                
                g1_mean_com = np.array([g1_this_i['mean'  ] for g1_this_i in g1_compare])
                g1_err_lo_c = np.array([g1_this_i['err_lo'] for g1_this_i in g1_compare])
                g1_err_hi_c = np.array([g1_this_i['err_hi'] for g1_this_i in g1_compare])
                g1_meanoffc = g1_mean_com - want_to_see_ref[k]
                g1_error_co = [np.abs(g1_err_lo_c), g1_err_hi_c]
                ax1.errorbar(cosi_compar, g1_meanoffc, yerr=g1_error_co, 
                             fmt=' ', capsize=10, capthick=1, elinewidth=1, 
                             marker='x', markersize=10, color="#C82423", 
                             label=param+f' ({comparename})', zorder=1)
                ax1.text(0.01, 0.13, 
                         f'Avg offset = {np.mean(g1_meanoffc):.2f}   '+
                         f'Avg error = {np.mean(np.array(g1_error_co).flatten()):.2f}', 
                         fontsize=12, color='brown', ha='left', va='bottom', 
                         transform=ax1.transAxes)
                if add_covert_cosi==True:
                    ax5.errorbar(sini_compar, g1_meanoffc, yerr=g1_error_co, 
                                 fmt=' ', capsize=10, capthick=1, elinewidth=1, 
                                 marker='x', markersize=10, color="#C82423", 
                                 label=param+f' ({comparename})', zorder=1)
                    ax5.text(0.01, 0.13, 
                             f'Avg offset = {np.mean(g1_meanoffc):.2f}   '+
                             f'Avg error = {np.mean(np.array(g1_error_co).flatten()):.2f}', 
                             fontsize=12, color='brown', ha='left', va='bottom', 
                             transform=ax5.transAxes)
                    
            # all_ylims[0] = np.min([all_ylims[0], ax1.get_ylim()[0]]) # update
            # all_ylims[1] = np.max([all_ylims[1], ax1.get_ylim()[1]]) # update
            xlim_max = np.min(xaxis_values)+(np.max(xaxis_values)-np.min(xaxis_values))*1.0
            ax1.hlines(y=0, xmin=0, xmax=xlim_max, 
                       color='gray', linestyle='--', linewidth=1)
            ax1.text(0.01, 0.03, 
                     f'Avg offset = {np.mean(g1_mean_off):.2f}   '+
                     f'Avg error = {np.mean(np.array(g1_error).flatten()):.2f}', 
                     fontsize=12, color=colors[k+4], ha='left', va='bottom', 
                     transform=ax1.transAxes)
            if average_over_param_B == True:
                ax1.text(0.01, 0.97, 'Edge-on', fontsize=10, color='black', 
                         ha='left', va='top', transform=ax1.transAxes)
                ax1.text(0.99, 0.03, 'Face-on', fontsize=10, color='black', 
                         ha='right', va='bottom', transform=ax1.transAxes)
            # ax1.text(0.5, 0.5, 
            #          'PA = 115', 
            #          fontsize=60, color='black', ha='center', va='center', 
            #          transform=ax1.transAxes, alpha=0.1, zorder=-1)
            ax1.set_xlabel(xaxis_label, labelpad=0)
            ax1.set_ylabel(r'Offset = Fit - Mock setting')
            ax1.set_ylim(all_ylims)
            ax1.grid(linestyle=':', color='black', alpha=0.5)
            ax1.set_title('Mocks: SNR='+f'{this_spec_snr}', fontsize=12)
            ax1.legend(prop={'size': 10})
            ax1.yaxis.set_major_locator(MultipleLocator(0.1))
            if add_covert_cosi==True:
                ax5.hlines(y=0, xmin=0, xmax=xlim_max, 
                           color='gray', linestyle='--', linewidth=1)
                ax5.text(0.01, 0.03, 
                         f'Avg offset = {np.mean(g1_mean_off):.2f}   '+
                         f'Avg error = {np.mean(np.array(g1_error).flatten()):.2f}', 
                         fontsize=12, color=colors[k+4], ha='left', va='bottom', 
                         transform=ax5.transAxes)
                if average_over_param_B == True:
                    ax5.text(0.01, 0.97, 'Face-on', fontsize=10, color='black', 
                             ha='left', va='top', transform=ax5.transAxes)
                    ax5.text(0.99, 0.03, 'Edge-on', fontsize=10, color='black', 
                             ha='right', va='bottom', transform=ax5.transAxes)
                    
                # ax5.text(0.5, 0.5, 
                #          'PA = 115', 
                #          fontsize=60, color='black', ha='center', va='center',
                #          transform=ax5.transAxes, alpha=0.1, zorder=-1)
                ax5.set_xlabel(r'$\sin{(i)}$', labelpad=0)
                ax5.set_ylabel(r'Offset = Fit - Mock setting')
                ax5.set_ylim(all_ylims)
                ax5.grid(linestyle=':', color='black', alpha=0.5)
                ax5.set_title('Mocks: SNR='+f'{this_spec_snr}', fontsize=12)
                ax5.legend(prop={'size': 10})
                ax5.yaxis.set_major_locator(MultipleLocator(0.1))
        fig.savefig('binospec_errorbars_test_SNR20_1x.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    # if len(theta_int_values) >= 2: # add any supplemental plots
    #     fig2 = plt.figure(figsize=(6, 3))  # (length, height)
    #     gsS2 = fig2.add_gridspec(nrows=1, ncols=1, 
    #                             height_ratios=[1], width_ratios=[1])
    #     axS1 = fig2.add_subplot(gsS2[0, 0])
        
    #     [g1_mean_off, g2_mean_off] = g12_mean_off
    #     [g1_error,    g2_error   ] = g12_error
    #     [g1_error_up, g1_error_dn] = g1_error
    #     [g2_error_up, g2_error_dn] = g2_error
    #     g12_err_up = np.sqrt((g1_error_up**2 + g2_error_up**2) / 2)
    #     g12_err_dn = np.sqrt((g1_error_up**2 + g2_error_up**2) / 2)
    #     g12_err    = [g12_err_up, g12_err_dn]
        
    #     axS1.errorbar(xaxis_values, 
    #                   np.sqrt(g1_mean_off**2 + g2_mean_off**2), 
    #                   yerr=g12_err, 
    #                   fmt=' ', capsize=5, capthick=2, elinewidth=4, 
    #                   marker='x', markersize=6, color=colors[3], alpha=1.0, 
    #                   label=r'$\sqrt{g_1^2 + g_2^2}$ with shape noise $\sqrt{(\Delta g_1^2 + \Delta g_2^2) / 2}$', zorder=2)
        
    #     axS1.hlines(y=0, xmin=0, xmax=xlim_max, 
    #                color='gray', linestyle='--', linewidth=1)
    #     axS1.text(0.01, 0.03, 
    #              f'Avg offset = {np.mean(g1_mean_off):.2f}   '+
    #              f'Avg error = {np.mean(np.array(g1_error).flatten()):.2f}', 
    #              fontsize=12, color='brown', ha='left', va='bottom', 
    #              transform=axS1.transAxes)
    #     axS1.set_xlabel(xaxis_label, labelpad=0)
    #     axS1.set_ylabel(r'Total Shear $|\gamma|$')
    #     axS1.set_ylim(all_ylims)
    #     axS1.grid(linestyle=':', color='black', alpha=0.5)
    #     if if_add_noise:
    #         axS1.set_title('Mocks: SNR='+f'{this_spec_snr}, noise added', fontsize=12)
    #     else:
    #         axS1.set_title('Mocks: SNR='+f'{this_spec_snr}, noise-free', fontsize=12)
    #     axS1.legend(prop={'size': 10})
    #     axS1.yaxis.set_major_locator(MultipleLocator(0.1))
    #     fig2.savefig(f'./binospec_Noise_{str(if_add_noise)}_supplement.png', dpi=150, bbox_inches='tight')
            
    return





runs_folder = '../../../../RSCH3/kl_github/runs_Jun_17_2025_noise_free/'
runs_folder_noisy = '../../../../RSCH3/kl_github/runs_noisy_SNR20_sqrt(var)/'
iter_nums = [ [ 1,  2,  3,  4,  5],  
              [11, 12, 13, 14, 15], 
              [21, 22, 23, 24, 25],  
              [31, 32, 33, 34, 35], 
              [41, 42, 43, 44, 45],  
              [51, 52, 53, 54, 55] ] #,
# iter_nums_no_noise = [[63, 65, 67],]#
                      # [51, 52, 53, 54, 55, 56, 57, 58],
                      # [61, 62, 63, 64, 65, 66, 67, 68],
                      # [71, 72, 73, 74, 75, 76, 77, 78]]
sini_list  = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
cosi_list  = np.sqrt(1 - sini_list**2)
cosis      = np.tile(cosi_list,  
                     (len(iter_nums), 1))
thetas    = np.tile(np.linspace(0, np.pi, 6, endpoint=False)[:, np.newaxis],  
                    (1, len(iter_nums[0])))
spec_snrs = np.tile(np.array(20),                         
                    (len(iter_nums), len(iter_nums[0])))
n_cosi  =  cosis.shape[1]
n_theta = thetas.shape[0]
n_spec_snr = 1

want_to_see     = [ 'g1', 'g2']
want_to_see_ref = [ 0.05, 0.05]
n_param_to_see  = len(want_to_see)
# dic_all_iter_noise_yes = make_a_dic_for_slits(iter_nums, runs_folder, if_add_noise=True)
# plot_diagnostics(dic_all_iter_noise_yes, runs_folder, if_add_noise=True)
dic_all_iter_noise_no  = make_a_dic_for_slits(iter_nums, 
                                              runs_folder, if_add_noise=False, 
                                              cosis=cosis, thetas=thetas)
dic_all_iter_noisy     = make_a_dic_for_slits(iter_nums, 
                                              runs_folder_noisy, if_add_noise=True, 
                                              cosis=cosis, thetas=thetas)
plot_diagnostics(dic_all_iter_noisy, n_spec_snr, 
                 first_runs_are_changing_param_A=cosis, 
                 param_B_maybe_not_periodic_in_your_runs=thetas,
                 average_over_param_B=True,
                 runs_folder=runs_folder, 
                 compare=dic_all_iter_noise_no, 
                 comparename='No noise, SNR=40', 
                 labelname='Noise = sqrt(var)',
                 add_covert_cosi=True)
