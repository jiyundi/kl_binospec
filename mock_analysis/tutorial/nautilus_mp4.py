import json
import numpy as np
# import corner
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from binospec_main_fitting_nautilus import load_mock, make_config_dic
from post_fitting import complete_flattened_fit_params
from klm.parameters import Parameters
from klm.nautilus_sampler import NautilusSampler

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Inter",
    "font.serif": "Inter",
})

def init_nautilus_sampler():
    pkl_folder  =  'about_binospec_mock_analysis/kl_tutorial/kl_tutorial_pkl/'
    Ms_folder   =  '../../bagpipes-KL/'

    # ------------- 1. Load observation data or mock ---------------- #
    data_info = load_mock(pkl_folder, Ms_folder, slit_name)

    linespecies = []
    for spec in data_info['spec']:
        linespecies.append(spec['par_meta']['line_species'])

    config_dic = make_config_dic(
        linespecies, fitting_params, fid_params, 
        log10_Mstar=data_info['galaxy']['log10_Mstar'], 
        log10_Mstar_err=data_info['galaxy']['log10_Mstar_err'],
        use_line_profile = None, #  'meta' 
        )

    nautilus_sampler = NautilusSampler(data_info, config_dic)
    fitting_params_flat = Parameters._flatten(fitting_params, level=1)
    fitting_par = complete_flattened_fit_params(
        fitting_params_flat, 
        line_species=nautilus_sampler.config.galaxy_params.line_species
        )

    return nautilus_sampler, fitting_par


def generate_image_model(par_vector, fitting_par, nautilus_sampler):
    best_fit_dict  = nautilus_sampler.params.gen_param_dict(fitting_par.keys(), 
                                                            par_vector)
    image_fit = nautilus_sampler.image_model.get_image(best_fit_dict['shared_params'])
    
    return image_fit


def best_fit_mp4():
    # Sort by likelihood?
    # sort   = np.argsort(log_l)
    # points = points[sort]
    # log_l  = log_l[sort]
    
    # Final frame - find best fit from samples:
    # (also you can use maximum_likelihood key from jsonf)
    Lmax_idx   = np.argmax(log_l)
    idxs[  -1] = len(points) - 1
    points[-1] = points[Lmax_idx]
    log_l[ -1] = log_l[ Lmax_idx]
    
    nautilus_sampler, fitting_par = init_nautilus_sampler()
    
    fig = plt.figure(figsize=(18, 6))  # (length, height)
    gs = fig.add_gridspec(nrows=1, ncols=3, height_ratios=[1], width_ratios=[1,1,1])
    plt.subplots_adjust(hspace=0.05, wspace=0.25) # h=height
    ax1 = fig.add_subplot(gs[0, 0], projection=nautilus_sampler.meta_image['ap_wcs'])
    ax2 = fig.add_subplot(gs[0, 1], projection=nautilus_sampler.meta_image['ap_wcs'])
    ax3 = fig.add_subplot(gs[0, 2], projection=nautilus_sampler.meta_image['ap_wcs'])

    # image observed
    image_obs = np.flip(nautilus_sampler.data_image, axis=1)
    noise = np.std(image_obs)
    imshow1 = ax1.imshow(image_obs, origin='lower', 
                         cmap='gray', aspect='equal', 
                         vmin=0, vmax=0 + 5*noise)
    ax1.coords['ra' ].set_major_formatter('dd:mm:ss')
    ax1.coords['dec'].set_major_formatter('dd:mm:ss')
    ax1.set_xlabel('RA')
    ax1.set_ylabel('Dec', labelpad=-1)
    ax1.set_title('Observation', fontsize=20)
    fig.colorbar(imshow1, ax=ax1)

    # image fit
    image_fit = np.flip(generate_image_model(par_vector, fitting_par, nautilus_sampler), axis=1)
    imshow2 = ax2.imshow(image_fit, origin='lower', 
                         cmap='gray', aspect='equal', 
                         vmin=0, vmax=0 + 5*noise)
    ax2.coords['ra' ].set_major_formatter('dd:mm:ss')
    ax2.coords['dec'].set_major_formatter('dd:mm:ss')
    ax2.set_xlabel('RA')
    ax2.set_ylabel('Dec', labelpad=-1)
    ax2.set_title('Model'+'\n'+f'(Sample #{idxs[0]+1}/{len(points)})', fontsize=20)
    fig.colorbar(imshow2, ax=ax2)

    # image residual
    imshow3 = ax3.imshow(np.flip(image_obs - image_fit, axis=1), 
                         origin='lower', 
                         cmap='twilight', aspect='equal', 
                         vmin=0 - 5*noise, vmax=0 + 5*noise)
    ax3.coords['ra' ].set_major_formatter('dd:mm:ss')
    ax3.coords['dec'].set_major_formatter('dd:mm:ss')
    ax3.set_xlabel('RA')
    ax3.set_ylabel('Dec', labelpad=-1)
    ax3.set_title('Residual'+'\n'+f'logL = {log_l[idxs[0]]:.0f}'+'\n'+f'(Sample #{idxs[0]+1}/{len(points)})', fontsize=20)
    fig.colorbar(imshow3, ax=ax3)
    
    def update(i):
        par_vector = points[idxs[i]]
        image_fit  = np.flip(generate_image_model(par_vector, fitting_par, nautilus_sampler), axis=1)
        image_res  = np.flip(image_obs - image_fit, axis=1)
        
        imshow2.set_data(image_fit)
        imshow3.set_data(image_res)
        ax2.set_title('Model'+'\n'+f'(Sample #{idxs[i]+1}/{len(points)})', fontsize=20)
        ax3.set_title('Residual'+'\n'+f'logL = {log_l[idxs[i]]:.0f}'+'\n'+f'(Sample #{idxs[i]+1}/{len(points)})', fontsize=20)
        
    ani = FuncAnimation(fig, update, frames=list(range(len(idxs)))+[len(idxs)-1]*180)

    ani.save(f"about_binospec_mock_analysis/kl_tutorial/nautilus_fit_evolution_{slit_name}_{nframes}.mp4", fps=30)

    return


def corner_mp4():
    fig = plt.figure(figsize=(10, 10))  # (length, height)
    plt.subplots_adjust(hspace=0.25, wspace=0.4) # h=height
    gs  = fig.add_gridspec(nrows=len(pars_to_show), 
                           ncols=len(pars_to_show), 
                           height_ratios=[1]*len(pars_to_show), 
                           width_ratios=[1]*len(pars_to_show))

    ndim = len(pars_to_show)
    axes_corner = np.empty((ndim, ndim), dtype=object)
    for i in range(ndim):
        # histogram title - param name
        axes_corner[i,i] = fig.add_subplot(gs[i,i])
        axes_corner[i,i].set_title(pars_to_label[i], fontsize=16)
    
        for j in range(ndim):
            if i <= j:
                continue
            axes_corner[i,j] = fig.add_subplot(gs[i,j])
            
            # bottom row
            if i == ndim - 1:
                axes_corner[i,j].set_xlabel(pars_to_label[j], fontsize=16)
            
            # left column
            if j == 0:
                axes_corner[i,j].set_ylabel(pars_to_label[i], fontsize=16)
        
    # Fixed x/y-axis range limits
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes_corner[yi,xi]
            ax.set_xlim(points[:,i_cols[xi]].min(), points[:,i_cols[xi]].max())
            ax.set_ylim(points[:,i_cols[yi]].min(), points[:,i_cols[yi]].max())
    
    # Corner plot - init
    scatter_objs = {}
    hist_objs = {}
    
    # histograms
    for i in range(ndim):
        ax = axes_corner[i,i]
        hist_objs[i] = ax.hist([], bins=30, color='white')[2]
    
    # corners
    for i in range(ndim):
        for j in range(i):
            ax = axes_corner[i,j]
            scat = ax.scatter([], [], s=3, color='lightgray', alpha=0.2)
            scatter_objs[(i,j)] = scat
                
            # truth lines
            ax.axvline(true_to_plot[j], color='gray', linestyle=':')
            ax.axhline(true_to_plot[i], color='gray', linestyle=':')
            
    def update(i):
        # corners
        par_vector_plot = points[idxs[:i+1]][:, i_cols]
        log_w_to_plot   = log_w[ idxs[:i+1]]
        log_l_to_plot   = log_l[ idxs[:i+1]]
        log_w_threshold = np.median(np.exp(log_w_to_plot))
        for yi in range(ndim):
            for xi in range(yi):
                scat = scatter_objs[(yi,xi)]
                ax   = axes_corner[yi,xi]
                scat.set_offsets(
                    np.c_[par_vector_plot[:,xi], par_vector_plot[:,yi]]
                )
                scat.set_alpha(np.exp(log_l_to_plot/corner_dens))
                mask = (np.exp(log_w_to_plot) > log_w_threshold)
                if sum(mask) > 2:
                    scat.set_color('steelblue')
                
        # histograms
        for k in range(ndim):
            ax = axes_corner[k,k]
            ax.cla()
            arr_hist = par_vector_plot[:,k]
            mask = (np.exp(log_w_to_plot) > log_w_threshold)
            if sum(mask) > 2:
                ax.hist(arr_hist[mask], bins=30, color='#52a5e7')
                ax.set_xlim(arr_hist[mask].min(), arr_hist[mask].max())
            else:
                ax.hist(arr_hist, bins=30, color='lightgray')
                ax.set_xlim(points[:,i_cols[k]].min(), points[:,i_cols[k]].max())
            ax.set_title(f'{np.median(arr_hist):.2f} '+r'$\pm$'+f' {np.std(arr_hist):.2f}\n'+pars_to_label[k], fontsize=16)
            ax.vlines(true_to_plot[k], color='black', linestyle='--', 
                      ymin=0, ymax=1, transform=ax.get_xaxis_transform())
            
        axes_corner[ndim-1,ndim-1].text(
            0.75, 0.70, f'Sample \n#{idxs[i]+1}/{len(points)}', fontsize=16, 
            va='bottom', ha='right', transform=fig.transFigure, color='black')
        
    ani = FuncAnimation(fig, update, frames=list(range(len(idxs)))+[len(idxs)-1]*10)
    
    print('Saving mp4...')
    ani.save(
        f"about_binospec_mock_analysis/kl_tutorial/nautilus_corner_evolution_{slit_name}_{nframes}.mp4", 
        fps=60, progress_callback=lambda i, n: print(f"{i}/{n}")
        )

    return





corner_dens = 100
nframes     = 1000
slit_name   = 111
slit_folder = f'about_binospec_mock_analysis/kl_tutorial/Slit_{slit_name:03d}_runs/'

with open(f'{slit_folder}post.txt', 'r') as f: 
    post_header = f.readline().split(' ')
    
pars_to_show = [
    'shared_params-g1',
    'shared_params-g2',
    'shared_params-theta_int',
    'shared_params-vcirc',
    'shared_params-cosi'
    ]
pars_to_label = [r'$g_1$', r'$g_2$', r'$\theta_{\mathrm{int}}$', 
                 r'$v_{\rm circ}$', r'$\cos{(i)}$']
true_to_plot  = [0, 0, 0, 138.23079553462523, 0.5]
i_cols = [post_header.index(p)-2 for p in pars_to_show]

with open(slit_folder + "best_fit.json", "r", encoding="utf-8") as f:
    jsonf = json.load(f)
fitting_params = jsonf['fitting_params']
fid_params     = jsonf['fid_params']

points_saved = np.loadtxt(f'{slit_folder}post.txt', skiprows=1)
log_w  = points_saved[:, 1 ]
log_l  = points_saved[:, 1 ]
points = points_saved[:, 2:]

# choose video frames to resample
# idxs = np.logspace(0, np.log(len(points)), nframes, base=np.exp(1)).astype(int) - 1
# par_vector = points[idxs[0]]
# best_fit_mp4()

idxs = np.linspace(0, len(points)-1, nframes).astype(int)
par_vector = points[idxs[0]]
corner_mp4()

