# import joblib

# def another_load_mock(pkl_folder='mock/', slit_num=95):
#     with open(f'{pkl_folder}pkl/slit_{slit_num:03d}_converted.pkl', "rb") as f:
#         data_info = joblib.load(f)
#     return data_info

# data_info = another_load_mock(
#             pkl_folder='../scripts_Pranjal/', 
#             slit_num=2)

# print(data_info['galaxy']['log10_Mstar'])
# print(data_info['galaxy']['log10_Mstar_err'])
# a = 1.718
# b = 3.869
# sigmaTF_intr = 0.058
# log10_Mstar  = data_info['galaxy']['log10_Mstar']
# exponent = (log10_Mstar - a) / b
# TF_vcirc = 10 ** (exponent)
# TF_vcirc_up = 10 ** (exponent + sigmaTF_intr)
# TF_vcirc_dn = 10 ** (exponent - sigmaTF_intr)
# print(TF_vcirc)
# print(TF_vcirc_up - TF_vcirc)
# print(TF_vcirc_dn - TF_vcirc)




import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = "Slit_002/run0.01/chain.hdf5"

with h5py.File(filename, "r") as f:

    points = []
    logl = []

    sampler = f["sampler"]

    # 读取所有 batch
    i = 0
    while f"points_{i}" in sampler:

        p = sampler[f"points_{i}"][:]
        l = sampler[f"log_l_{i}"][:]

        if len(p) > 0:
            points.append(p)
            logl.append(l)

        i += 1

    # 最后一批
    if "points_t" in sampler:
        points.append(sampler["points_t"][:])
        logl.append(sampler["log_l_t"][:])

points = np.vstack(points)
logl = np.concatenate(logl)

print(points.shape)
print(logl.shape)

nparams = points.shape[1]

labels = [
    f'Param {i+1}' for i in range(nparams)
]

labels = [
    'g1 [-0.15, 0.15]', 
    'g2 [-0.15, 0.15]',
    'theta_int [-3.14, 3.14]',
    'vcirc',
    'cosi [0, 1]',
    'r_hl_disk [0.2, 3.0]',
    'flux [3, 6]',
    'dx_disk [-1, 1]',
    'dy_disk [-1, 1]',
    'vscale [0.01, 4]',
    'O2: v_0 [-180, 100]',
    'O2: v_0_2 [-180, 100]',
    'O2: I01_spec1 [0, 1000]',
    'O2: I02_spec1 [0, 1000]',
    'Hb: v_0 [-180, 100]',
    'Hb: I01_spec1 [0, 1000]',
    'O3b: v_0 [-180, 100]',
    'O3b: I01_spec1 [0, 1000]',
]

fig, axes = plt.subplots(18,1,figsize=(12,32),sharex=True)

x = np.arange(len(points))

for i in range(18):

    axes[i].plot(
        x,
        points[:,i],
        '.',
        ms=1,
        alpha=0.25
    )

    axes[i].set_ylabel(labels[i])

plt.xlabel("Sample index")
plt.tight_layout()
plt.savefig('test.jpg', dpi=100)