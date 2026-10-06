import numpy as np
import json
import yaml

from post_fitting import complete_fit_params

from klm.safe_plot import setup; setup() # must before plt
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


def analyze_percentile(samples, config_filename, best_fit=None):
    # load fitting config
    with open(config_filename, "r", encoding="utf-8") as file1:
        config = yaml.safe_load(file1)
    
    lines = []
    for key in best_fit.keys():
        line = key.split('-')[0].split('_')[0]
        if (line != 'shared') and (line not in lines):
            lines.append(line)
    config_par = complete_fit_params(
        config, line_species=lines
        )
    
    config_param_names = [key for key, _ in config_par.items()]
    
    dic_percent = {}
    arr = np.zeros((len(samples[0]) - 2, 3))
    
    for j in range(len(samples[0])-2): # exclude log_w, log_l
        samp_points = samples[:,j+2]
        x123 = np.percentile(samp_points, [16, 50, 84])
        err_lo, mean, err_hi = x123[0]-x123[1], x123[1], x123[2]-x123[1]
        arr[j] = np.around([mean, err_lo, err_hi], decimals=4)
        
        this_dict = {'mean':   arr[j,0], 
                     'err_lo': arr[j,1], 
                     'err_hi': arr[j,2],
                     'best':   best_fit[config_param_names[j]]}
        
        dic_percent[config_param_names[j]] = this_dict

    return dic_percent

def make_a_dic_for_slits(slit_num, iter_nums, runs_folder, if_add_noise, 
                         config_filename="../config/binospec_fitting_params.yaml",
                         cosis=None, thetas=None):
    dic_all_iter = {}
    for i in range(len(iter_nums)):
        for j in range(len(iter_nums[0])):
            iter_num = iter_nums[i][j]
            samples = np.loadtxt(runs_folder+f'Slit_{slit_num:03d}_{iter_num:03d}/post.txt', skiprows=1)
            with open(runs_folder+f"Slit_{slit_num:03d}_{iter_num:03d}/best_fit.json", "r") as f:
                best_fit = json.load(f)
            percentile = analyze_percentile(samples, config_filename, 
                                            best_fit['maximum_likelihood']['point'])
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
    
    g12_best_off = [[], []]
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
            
            g1_best     = np.array([g1_this_i['best'  ] for g1_this_i in g1_values])
            g1_error_lo = np.array([g1_this_i['err_lo'] for g1_this_i in g1_values])
            g1_error_hi = np.array([g1_this_i['err_hi'] for g1_this_i in g1_values])
            g1_best_off = g1_best - want_to_see_ref[k]
            g1_error    = [np.abs(g1_error_lo), g1_error_hi]
            g12_best_off[k], g12_error[k] = g1_best_off, g1_error
            
            ax1 = fig.add_subplot(gs[i, k])
            
            if average_over_param_B == True:
                xaxis_values = first_runs_are_changing_param_A
                xaxis_label  = r'$\cos{(i)}$'
            elif average_over_param_B is not True:
                xaxis_values = param_B_maybe_not_periodic_in_your_runs / np.pi * 180
                xaxis_label  = r'$\theta_{\rm int}$ (deg)'
            ax1.errorbar(xaxis_values, g1_best_off, yerr=g1_error, 
                         fmt=' ', capsize=5, capthick=2, elinewidth=4, 
                         marker='o', markersize=6, color=colors[k+4], alpha=1.0, 
                         label=f'{labelname}', zorder=2)
            if add_covert_cosi==True:
                ax5 = fig.add_subplot(gs[i+n_spec_snr, k])
                sini_values = np.sqrt(1-np.array(xaxis_values)**2)
                ax5.errorbar(sini_values, g1_best_off, yerr=g1_error, 
                             fmt=' ', capsize=5, capthick=2, elinewidth=4, 
                             marker='o', markersize=6, color=colors[k+4], alpha=1.0, 
                             label=f'{labelname}', zorder=2)
                
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
                             label=f'{comparename}', zorder=1)
                ax1.text(0.01, 0.13, 
                         f'Avg offset = {np.mean(g1_meanoffc):.2f}   '+
                         f'Avg error = {np.mean(np.array(g1_error_co).flatten()):.2f}', 
                         fontsize=12, color='brown', ha='left', va='bottom', 
                         transform=ax1.transAxes)
                if add_covert_cosi==True:
                    ax5.errorbar(sini_compar, g1_meanoffc, yerr=g1_error_co, 
                                 fmt=' ', capsize=10, capthick=1, elinewidth=1, 
                                 marker='x', markersize=10, color="#C82423", 
                                 label=f'{comparename}', zorder=1)
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
                     f'Avg offset = {np.mean(g1_best_off):.2f}   '+
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
            ax1.set_title(f'{param} (SNR={this_spec_snr})', fontsize=12)
            ax1.legend(prop={'size': 10})
            ax1.yaxis.set_major_locator(MultipleLocator(0.1))
            if add_covert_cosi==True:
                ax5.hlines(y=0, xmin=0, xmax=xlim_max, 
                           color='gray', linestyle='--', linewidth=1)
                ax5.text(0.01, 0.03, 
                         f'Avg offset = {np.mean(g1_best_off):.2f}   '+
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
                ax5.set_title(f'{param} (SNR={this_spec_snr})', fontsize=12)
                ax5.legend(prop={'size': 10})
                ax5.yaxis.set_major_locator(MultipleLocator(0.1))
        fig.savefig(f'mock_shape_noise_SNR{this_spec_snr}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
               
    return

# def complete_fit_params(fitting_params, line_species, 
#                         need_sorted_flattened=True):
#     lines        = list(dict.fromkeys(line_species))
#     doublet_lines  = ['O2']
#     doub_pars      = ['v_0_2','dx_vel_2','dy_vel_2','I02','f2_1','f2_2']
#     doub_pars_twin = ['v_0',  'dx_vel',  'dy_vel',  'I01','f1_1','f1_2']
    
#     fitting_par = {}
#     for line in lines: fitting_par[f'{line}_params'] = {}
    
#     for key, dic in fitting_params.items():
#         if key == 'shared_params':
#             fitting_par[key] = dic
        
#         elif key == 'line_params':
#             for pname_w_spec, prior_dic in dic.items():
#                 for line in lines:
#                     line_p = f'{pname_w_spec}'
#                     fitting_par[f'{line}_params'][line_p] = prior_dic
                    
#                     if line in doublet_lines:
#                         if 'spec' in pname_w_spec.split('_')[1]: # I01_spec1
#                             pname_wo_spec = pname_w_spec.split('_')[0]
#                             i_q    = doub_pars_twin.index(pname_wo_spec)
#                             q2name = doub_pars[i_q]
#                             line_q = f'{q2name}_{pname_w_spec.split("_")[1]}'
#                             fitting_par[f'{line}_params'][line_q] = prior_dic
                            
#                         else: # v_0
#                             i_q    = doub_pars_twin.index(pname_w_spec)
#                             q2name = doub_pars[i_q]
#                             line_q = f'{q2name}'
#                             fitting_par[f'{line}_params'][line_q] = prior_dic
    
#     if need_sorted_flattened:
#         # --- flatten ---
#         flattened_shared_keys = []
#         for k1, subdict in fitting_par.items():
#             if k1 == 'shared_params':
#                 for k2 in subdict.keys():
#                     flattened_shared_keys.append(f"{k1}-{k2}")
        
#         # --- flatten ---
#         flattened_line_keys = []
#         for k1, subdict in fitting_par.items():
#             if k1 in [f'{line}_params' for line in lines]:
#                 for k2 in subdict.keys():
#                     flattened_line_keys.append((f"{k1}-{k2}", k2))
        
#         # 按二级 key 的首次出现顺序排序
#         seen_order = []
#         for _, k2 in flattened_line_keys:
#             if k2 not in seen_order:
#                 seen_order.append(k2)
#         flattened = sorted(flattened_line_keys, 
#                            key=lambda x: seen_order.index(x[1]))
#         flattened = [x[0] for x in flattened]
        
#         # 对O2等双线的第二线参数进行插队
#         flattened_linekeys = []
#         skip_next = set()
        
#         for i, key in enumerate(flattened):
#             if key in skip_next:
#                 continue
        
#             flattened_linekeys.append(key)
        
#             # 处理 O2 的双线参数
#             if key.startswith("O2_params-"):
#                 base = key.split("-", 1)[1]
        
#                 # 特殊规则：v_0 → v_0_2
#                 if base == "v_0" and "O2_params-v_0_2" in flattened:
#                     flattened_linekeys.append("O2_params-v_0_2")
#                     skip_next.add("O2_params-v_0_2")
        
#                 # 规则：I01_specN → I02_specN
#                 if base.startswith("I01_spec"):
#                     i02 = key.replace("I01_", "I02_")
#                     if i02 in flattened:
#                         flattened_linekeys.append(i02)
#                         skip_next.add(i02)
        
#         # Append shared_params. Assign back latex and prior values
#         flattened_keys = flattened_shared_keys + flattened_linekeys
#         sorted_par = {}
#         for key in flattened_keys: 
#             sorted_par[key] = fitting_par[key.split('-')[0]][key.split('-')[1]]
            
#         fitting_par = sorted_par
    
#     return fitting_par



slit_num  = 3
iter_nums = [ [ 1,  2,  3,  4,  5],  
              [11, 12, 13, 14, 15], 
              [21, 22, 23, 24, 25],  
              [31, 32, 33, 34, 35], 
              [41, 42, 43, 44, 45],  
              [51, 52, 53, 54, 55],
              ]

# sini_list  = np.sqrt(1 - sini_list**2)
cosi_list  = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
cosis      = np.tile(cosi_list,  
                     (len(iter_nums), 1)) # expand along axis=0
thetas    = np.tile(np.linspace(0, np.pi, 6, endpoint=False)[:, np.newaxis],  
                    (1, len(iter_nums[0])))
spec_snrs = np.tile(np.array(40),                         
                    (len(iter_nums), len(iter_nums[0])))
n_cosi  =  cosis.shape[1]
n_theta = thetas.shape[0]
n_spec_snr = 1

# runs_folder       = '../../../../../RSCH3/kl_github/mock/slit_003_runs/Slit_{slit_num:03d}_{case_num:03d}/post.txt'
runs_folder_noisy = '../../../../../RSCH3/kl_github/mock/slit_003_runs/'
config_filename = "config/binospec_fitting_params.yaml"

want_to_see     = ['shared_params-g1', 'shared_params-g2']
want_to_see_ref = [ 0.05, 0.05]
n_param_to_see  = len(want_to_see)
# dic_all_iter_noise_yes = make_a_dic_for_slits(iter_nums, runs_folder, if_add_noise=True)
# plot_diagnostics(dic_all_iter_noise_yes, runs_folder, if_add_noise=True)

# dic_all_iter_noise_no  = make_a_dic_for_slits(iter_nums, 
#                                               runs_folder, if_add_noise=False, 
#                                               cosis=cosis, thetas=thetas)
dic_all_iter_noisy     = make_a_dic_for_slits(slit_num, iter_nums, 
                                              runs_folder_noisy, if_add_noise=True, 
                                              config_filename=config_filename,
                                              cosis=cosis, thetas=thetas)

plot_diagnostics(dic_all_iter_noisy, n_spec_snr, 
                 first_runs_are_changing_param_A=cosis, 
                 param_B_maybe_not_periodic_in_your_runs=thetas,
                 average_over_param_B=True,
                 runs_folder=runs_folder_noisy, 
                 # compare=dic_all_iter_noise_no, 
                 # comparename='No noise, SNR=40', 
                 labelname='Noisy',
                 add_covert_cosi=True)
