import json
import yaml
import joblib
import galsim

from klm.nautilus_sampler import NautilusSampler
from klm.transformations import Transformations
from klm.parameters import Parameters
from core.post_fitting import plot_obs_fit_res
from core.make_config_dic import make_config_dic
from core.fitting_result_utils import complete_flattened_fit_params

slit_num = 1
pkl_folder  =  './'
slit_folder = f'./Slit_{slit_num:03d}_Nauti_g015/'
fiduci_yaml =  "../config/binospec_fid_params_Pranjal.yaml"
fittin_yaml =  "../config/binospec_fitting_params_Pranjal_.yaml"


# ------------- 1. Load observation data or mock ---------------- #
with open(f'{pkl_folder}pkl/slit_{slit_num:03d}_converted.pkl', "rb") as f:
    data_info = joblib.load(f)

# Recover wcs(galsim.wcs) from ap_wcs
ap_wcs  = data_info['image']['meta']['ap_wcs']
data_info['image']['meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)

# ------------- 2. Load configuration --------------------------- #
with open(fiduci_yaml, "r", encoding="utf-8") as file1:
    fid_params     = yaml.safe_load(file1)
with open(fittin_yaml, "r", encoding="utf-8") as file2:
    fitting_params = yaml.safe_load(file2)

config_dic = make_config_dic(
    [spec['meta']['line_species'] for spec in data_info['spec']], 
    fitting_params, fid_params, 
    log10_Mstar=data_info['galaxy']['log10_Mstar'], 
    log10_Mstar_err=data_info['galaxy']['log10_Mstar_err'],
    use_line_profile = 'extracted', # None, 'raw' 
    )

# ------------- 3. Combine best fit parameters ------------------ #
nautilus_sampler = NautilusSampler(data_info, config_dic)

with open(slit_folder + "best_fit.json", "r", encoding="utf-8") as f:
    best_dict = json.load(f)

fitting_params_flat = Parameters._flatten(fitting_params, level=1)
fitting_par = complete_flattened_fit_params(
    fitting_params_flat, 
    line_species=nautilus_sampler.config.galaxy_params.line_species
    )
best_fit_dict = nautilus_sampler.params.gen_param_dict(
    fitting_par.keys(), 
    best_dict['posterior_mode']['point'].values()
    )

# ------------- 4. Choose any spectrum to check ------------------ #
i = 1 # index of the spectrum to check

lines_all = [spec['meta']['line_species'] for spec in data_info['spec']]
line      = lines_all[i]
counts    = lines_all[: i + 1].count(line) # occurrences of this line
set_num   = counts

nautilus_sampler.spec_model[i]._init_observable(
    data_info['galaxy'],
    data_info['spec'][i]['meta'],
)

# Collapse per-set intensity keys (I01_specN → I01) for this row
best_one_level = {
    **best_fit_dict['shared_params'],
    **best_fit_dict[f'{line}_params'],
}
for k in ('I01', 'I02'):
    if k in best_one_level.keys():
        best_one_level[k] = best_one_level.get(f'{k}_spec{set_num}')

this_spec_model = nautilus_sampler.spec_model[i]

# 在观测参考系（单位为角秒）中的各坐标(obs_xx, obs_yy)对应的盘面的横坐标disk_xx的2D数组值、纵坐标disk_yy的2D数组值
disk_xx, disk_yy = Transformations.transform_frame(
    this_spec_model.obs_xx, this_spec_model.obs_yy, params=best_one_level, 
    start='obs', end='disk'
    )

vfields = this_spec_model.build_line_vfield(best_one_level, disk_xx, disk_yy)

import matplotlib.pyplot as plt
idx_vfield = 0
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3))
im = ax.imshow(vfields[idx_vfield], aspect='auto', cmap='viridis', origin='lower', 
               extent=[this_spec_model.obs_xx.min(), this_spec_model.obs_xx.max(), 
                       this_spec_model.obs_yy.min(), this_spec_model.obs_yy.max()])
ax.set_xlabel(f'obs_xx (arcsec), len = {this_spec_model.obs_xx.shape[1]}')
ax.set_ylabel(f'obs_yy (arcsec), len = {this_spec_model.obs_yy.shape[0]}')
plt.colorbar(im, ax=ax)
plt.title(f'Velocity field (km/s) of Spec #{i}, Line #{idx_vfield} \nby kl_measurement')
plt.tight_layout()
plt.savefig('test.jpg')









# Match keys with kl_pipe's keys definition
# v_0 --> v0
# vscale --> vel_rscale
best_one_level['v0'] = best_one_level['v_0']
best_one_level['vel_rscale'] = best_one_level['vscale']
best_one_level.pop('v_0')
best_one_level.pop('vscale')

# 接下来需要在KL-Roman-env下以同样参数运行kl_pipe并输出生成的速度场，然后比较。
import io
import os
import pickle
import subprocess
import numpy as np

def call_generate_vfield_spencer(best_one_level, obs_xx=None, obs_yy=None, roman_env_python=None):
    """
    跨环境调用 generate_vfield_Spencer(arg1, arg2, arg3)
    """
    arg1, arg2, arg3 = best_one_level, obs_xx, obs_yy

    if roman_env_python is None:
        # 请替换为你本地 KL-Roman-env 环境的实际 python 解释器绝对路径
        roman_env_python = "/home/u9/jiyundi/miniconda3/envs/KL-Roman-env/bin/python"

    script_path = os.path.abspath("../../roman/run_gen_vfield_Spencer_stream.py")

    # 1. 将 3 个参数打包序列化
    # 这里同时支持传递普通数值、字符串、列表、甚至是 NumPy 2D/3D 数组
    args_to_send = ((arg1, arg2, arg3), {})
    input_bytes = pickle.dumps(args_to_send)

    # 2. 启动子进程，将参数写入 stdin，并捕获 stdout 的二进制输出
    result_ = subprocess.run(
        [roman_env_python, script_path],
        input=input_bytes,       # 通过 stdin 将参数传给子进程
        capture_output=True,     # 捕获 stdout 和 stderr
        check=False
    )

    # 3. 错误检查：如果子进程报错，抛出详细信息
    if result_.returncode != 0:
        err_msg = result_.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"KL-Roman-env 子进程运行报错:\n{err_msg}")

    # 4. 解包文件数据
    result = pickle.loads(result_.stdout)

    return result

# 跨环境调用函数并获取结果字典
vfield_Roman = call_generate_vfield_spencer(best_one_level, this_spec_model.obs_xx, this_spec_model.obs_yy)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,3))
im = ax.imshow(vfield_Roman, aspect='auto', cmap='viridis', origin='lower', 
               extent=[this_spec_model.obs_xx.min(), this_spec_model.obs_xx.max(), 
                       this_spec_model.obs_yy.min(), this_spec_model.obs_yy.max()])
ax.set_xlabel(f'obs_xx (arcsec), len = {this_spec_model.obs_xx.shape[1]}')
ax.set_ylabel(f'obs_yy (arcsec), len = {this_spec_model.obs_yy.shape[0]}')
plt.colorbar(im, ax=ax)
plt.title(f'Velocity field (km/s) of Spec #{i}, Line #{idx_vfield} \nby kl_roman_pipe')
plt.tight_layout()
plt.savefig('test2.jpg')

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(4,6))
im0 = axs[0].imshow(vfield_Roman, aspect='auto', cmap='viridis', origin='lower', 
                    extent=[this_spec_model.obs_xx.min(), this_spec_model.obs_xx.max(), 
                            this_spec_model.obs_yy.min(), this_spec_model.obs_yy.max()])
im1 = axs[1].imshow(vfields[idx_vfield] - vfield_Roman, aspect='auto', cmap='coolwarm', origin='lower', 
                    extent=[this_spec_model.obs_xx.min(), this_spec_model.obs_xx.max(), 
                            this_spec_model.obs_yy.min(), this_spec_model.obs_yy.max()],
                    vmin=np.min(vfields[idx_vfield]), 
                    vmax=np.max(vfields[idx_vfield]))
plt.colorbar(im0, ax=axs[0])
plt.colorbar(im1, ax=axs[1])
axs[0].set_xlabel(f'obs_xx (arcsec), len = {this_spec_model.obs_xx.shape[1]}')
axs[0].set_ylabel(f'obs_yy (arcsec), len = {this_spec_model.obs_yy.shape[0]}')
axs[1].set_xlabel(f'obs_xx (arcsec), len = {this_spec_model.obs_xx.shape[1]}')
axs[1].set_ylabel(f'obs_yy (arcsec), len = {this_spec_model.obs_yy.shape[0]}')
axs[0].set_title(f'Velocity field (km/s) of Spec #{i}, Line #{idx_vfield} \nby kl_roman_pipe')
axs[1].set_title(f'Difference = kl_measurement - kl_roman_pipe')
plt.tight_layout()
plt.savefig('test2.jpg')

print("Done.")