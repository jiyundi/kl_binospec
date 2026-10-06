import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def make_exam_plot_GIF(mock_data_info_list, slit_name, iter_num, frames):
    n_sets = len(mock_data_info_list[0]['spec'])
    
    fig = plt.figure(figsize=(11, 5*n_sets))  # (length, height)
    plt.subplots_adjust(hspace=0.4, wspace=0.0) # h=height
    gs = fig.add_gridspec(nrows=1*n_sets, ncols=2, 
                          height_ratios=[1]*n_sets, 
                          width_ratios=[1, 2])
    ims, axes, texts = [], [], []
    
    spec_data_list   = []
    spec_noise_list  = []
    extent_list      = []
    slit_ylim_list   = []
    
    image_data_list  = [mock_data_info['image']['data'].T for mock_data_info in mock_data_info_list]
    image_noise_list = [np.std(image_data)                for image_data     in image_data_list    ]
    for mock_data_info in mock_data_info_list:
        spec_data_list.append( [       mock_data_info['spec'][i]['data']  for i in range(n_sets)])
        spec_noise_list.append([np.std(mock_data_info['spec'][i]['data']) for i in range(n_sets)])
        extent_list.append(    [[ mock_data_info['spec'][i]['par_meta']['lambda_grid'][0, 0].value, # x_left
                                  mock_data_info['spec'][i]['par_meta']['lambda_grid'][0,-1].value, # x_right
                                 -mock_data_info['spec'][i]['par_meta']['pixScale']*len(mock_data_info['spec'][i]['data'])/2,  # y_bottom
                                 +mock_data_info['spec'][i]['par_meta']['pixScale']*len(mock_data_info['spec'][i]['data'])/2]  # y_top
                                for i in range(n_sets)])
        slit_ylim_list.append( [[-mock_data_info['spec'][i]['par_meta']['pixScale']*len(mock_data_info['spec'][i]['data'])/2,
                                 +mock_data_info['spec'][i]['par_meta']['pixScale']*len(mock_data_info['spec'][i]['data'])/2]
                                for i in range(n_sets)])
    

    slitRA_list  = [mock_data_info['spec'][i]['par_meta']['slitRA'].value  for i in range(n_sets)]
    slitDec_list = [mock_data_info['spec'][i]['par_meta']['slitDec'].value for i in range(n_sets)]
    ap_wcs       = mock_data_info['image']['par_meta']['ap_wcs']
    x0slit_list  = [ap_wcs.wcs_world2pix([[slitRA_list[i], slitDec_list[i]]], 0)[0][0] for i in range(n_sets)]
    y0slit_list  = [ap_wcs.wcs_world2pix([[slitRA_list[i], slitDec_list[i]]], 0)[0][1] for i in range(n_sets)]
    ax1txts_list, ax2txts_list = [], []
    for m in range(frames):
        mock_data_info = mock_data_info_list[m]
        ax1txts_li, ax2txts_li = [], []
        for i in range(n_sets):
            ax1txts, ax2txts = '', ''
            for key, arr in mock_data_info['par_fit'].items():
                if key[:14] == 'shared_params-':
                    ax2txts += (key[14:] + ' = ' + '{:.3g}'.format(arr) + '\n')
                elif key[:11] == 'OII_params-':
                    ax1txts += (key[11:] + ' = ' + '{:.0f}'.format(arr) + '\n')
            ax1txts_li.append(ax1txts)
            ax2txts_li.append(ax2txts[:82])
        ax1txts_list.append(ax1txts_li)
        ax2txts_list.append(ax2txts_li)
    
    # for i in range(n_sets):
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[0, 1], projection=mock_data_info['image']['par_meta']['ap_wcs'])
    ax4 = fig.add_subplot(gs[1, 1], projection=mock_data_info['image']['par_meta']['ap_wcs'])
    ax6 = fig.add_subplot(gs[2, 1], projection=mock_data_info['image']['par_meta']['ap_wcs'])
    
    # 初始造图
    im1spec = ax1.imshow(spec_data_list[0][0], origin='lower', 
                         extent=extent_list[0][0],
                         cmap='viridis', aspect='auto', 
                         vmin=0-spec_noise_list[0][0], vmax=0 + 5*spec_noise_list[0][0], animated=True)
    im3spec = ax3.imshow(spec_data_list[0][1], origin='lower', 
                         extent=extent_list[0][1],
                         cmap='viridis', aspect='auto', 
                         vmin=0-spec_noise_list[0][1], vmax=0 + 5*spec_noise_list[0][1], animated=True)
    im5spec = ax5.imshow(spec_data_list[0][2], origin='lower', 
                         extent=extent_list[0][2],
                         cmap='viridis', aspect='auto', 
                         vmin=0-spec_noise_list[0][2], vmax=0 + 5*spec_noise_list[0][2], animated=True)
    fig.colorbar(im1spec, ax=ax1)
    fig.colorbar(im3spec, ax=ax3)
    fig.colorbar(im5spec, ax=ax5)
    ax1.set_ylim(slit_ylim_list[0][i])
    ax3.set_ylim(slit_ylim_list[0][i])
    ax5.set_ylim(slit_ylim_list[0][i])
    ax1.xaxis.get_major_formatter().set_useOffset(False)
    ax3.xaxis.get_major_formatter().set_useOffset(False)
    ax5.xaxis.get_major_formatter().set_useOffset(False)
    ax1.set_xlabel(r'Observed Wavelength $\lambda$ ($\AA$)')
    ax3.set_xlabel(r'Observed Wavelength $\lambda$ ($\AA$)')
    ax5.set_xlabel(r'Observed Wavelength $\lambda$ ($\AA$)')
    ax1.set_ylabel(r'Spatial Position (arcsec)')
    ax3.set_ylabel(r'Spatial Position (arcsec)')
    ax5.set_ylabel(r'Spatial Position (arcsec)')
    ax1.grid(linestyle=':', color='white', alpha=0.5)
    ax3.grid(linestyle=':', color='white', alpha=0.5)
    ax5.grid(linestyle=':', color='white', alpha=0.5)
    ax1.set_title('Mock MMT/Binospec 2D Spec')
    ax3.set_title('Mock MMT/Binospec 2D Spec')
    ax5.set_title('Mock MMT/Binospec 2D Spec')
    
    image_data = mock_data_info['image']['data'].T
    objRA      = mock_data_info['image']['par_meta']['RA']
    objDec     = mock_data_info['image']['par_meta']['Dec']
    pixScale   = mock_data_info['image']['par_meta']['pixScale']
    slitLen    = mock_data_info['spec'][i]['par_meta']['slitLen'] 
    slitWidth  = mock_data_info['spec'][i]['par_meta']['slitWidth'] 
    slit_LPA   = mock_data_info['spec'][i]['par_meta']['slitLPA'].value
    x0obj,  y0obj  = ap_wcs.wcs_world2pix([[objRA,          objDec         ]], 0)[0]  
    x0slit, y0slit = ap_wcs.wcs_world2pix([[slitRA_list[0], slitDec_list[0]]], 0)[0]  
    
    im2imag = ax2.imshow(image_data_list[0], origin='lower', 
                         cmap='viridis', aspect='equal', 
                         vmin=0-image_noise_list[0], vmax=0 + 5*image_noise_list[0], animated=True)
    im4imag = ax4.imshow(image_data_list[0], origin='lower', 
                         cmap='viridis', aspect='equal', 
                         vmin=0-image_noise_list[0], vmax=0 + 5*image_noise_list[0], animated=True)
    im6imag = ax6.imshow(image_data_list[0], origin='lower', 
                         cmap='viridis', aspect='equal', 
                         vmin=0-image_noise_list[0], vmax=0 + 5*image_noise_list[0], animated=True)
    fig.colorbar(im2imag, ax=ax2)
    fig.colorbar(im4imag, ax=ax4)
    fig.colorbar(im6imag, ax=ax6)
    ax2.scatter(x0obj, y0obj, marker='x', s=360, color='black', zorder=1)
    ax4.scatter(x0obj, y0obj, marker='x', s=360, color='black', zorder=1)
    ax6.scatter(x0obj, y0obj, marker='x', s=360, color='black', zorder=1)
    
    x01, y01 = x0slit_list[0], y0slit_list[0], 
    x02, y02 = x0slit_list[1], y0slit_list[1], 
    x03, y03 = x0slit_list[2], y0slit_list[2], 
    def rect(x0, y0):
        dx, dy = slitWidth/pixScale, slitLen/pixScale, 
        rotation = (slit_LPA)/57.3
        xUL = x0 + (-dx/2)*np.cos(rotation) - (+dy/2)*np.sin(rotation)
        yUL = y0 + (-dx/2)*np.sin(rotation) + (+dy/2)*np.cos(rotation)
        xUR = x0 + (+dx/2)*np.cos(rotation) - (+dy/2)*np.sin(rotation)
        yUR = y0 + (+dx/2)*np.sin(rotation) + (+dy/2)*np.cos(rotation)
        xLL = x0 + (-dx/2)*np.cos(rotation) - (-dy/2)*np.sin(rotation)
        yLL = y0 + (-dx/2)*np.sin(rotation) + (-dy/2)*np.cos(rotation)
        xLR = x0 + (+dx/2)*np.cos(rotation) - (-dy/2)*np.sin(rotation)
        yLR = y0 + (+dx/2)*np.sin(rotation) + (-dy/2)*np.cos(rotation)
        return [xUL, xUR, xLR, xLL, xUL], [yUL, yUR, yLR, yLL, yUL]
    xpoints1, ypoints1 = rect(x01, y01)
    xpoints2, ypoints2 = rect(x02, y02)
    xpoints3, ypoints3 = rect(x03, y03)
    ax2.plot(xpoints1, ypoints1, 
                        color='white', linestyle='-', animated=True)
    ax4.plot(xpoints2, ypoints2, 
                        color='white', linestyle='-', animated=True)
    ax6.plot(xpoints3, ypoints3, 
                        color='white', linestyle='-', animated=True)
    
    ax2.set_xlim(left=0, right=image_data.shape[1]-1)
    ax4.set_xlim(left=0, right=image_data.shape[1]-1)
    ax6.set_xlim(left=0, right=image_data.shape[1]-1)
    ax2.set_ylim(bottom=0, top=image_data.shape[0]-1)
    ax4.set_ylim(bottom=0, top=image_data.shape[0]-1)
    ax6.set_ylim(bottom=0, top=image_data.shape[0]-1)
    ax2.coords['ra' ].set_major_formatter('dd:mm:ss')
    ax2.coords['dec'].set_major_formatter('dd:mm:ss')
    ax4.coords['ra' ].set_major_formatter('dd:mm:ss')
    ax4.coords['dec'].set_major_formatter('dd:mm:ss')
    ax6.coords['ra' ].set_major_formatter('dd:mm:ss')
    ax6.coords['dec'].set_major_formatter('dd:mm:ss')
    ax2.set_xlabel('RA')
    ax4.set_xlabel('RA')
    ax6.set_xlabel('RA')
    ax2.set_ylabel('Dec', labelpad=-1)
    ax4.set_ylabel('Dec', labelpad=-1)
    ax6.set_ylabel('Dec', labelpad=-1)
    ax2.grid(linestyle=':', color='white', alpha=0.5)
    ax4.grid(linestyle=':', color='white', alpha=0.5)
    ax6.grid(linestyle=':', color='white', alpha=0.5)
    ax2.set_title('Mock Subaru Imaging')
    ax4.set_title('Mock Subaru Imaging')
    ax6.set_title('Mock Subaru Imaging')
    
    text1 = ax1.text(1, 1, ax1txts_list[0][0], fontsize=14, color='white', ha='right', va='top', 
                     transform=ax1.transAxes, animated=True)
    text2 = ax2.text(1, 1, ax2txts_list[0][0], fontsize=18, color='white', ha='right', va='top', 
                     transform=ax2.transAxes, animated=True)
    text3 = ax3.text(1, 1, ax1txts_list[0][1], fontsize=14, color='white', ha='right', va='top', 
                     transform=ax3.transAxes, animated=True)
    text4 = ax4.text(1, 1, ax2txts_list[0][1], fontsize=18, color='white', ha='right', va='top', 
                     transform=ax4.transAxes, animated=True)
    text5 = ax5.text(1, 1, ax1txts_list[0][2], fontsize=14, color='white', ha='right', va='top', 
                     transform=ax5.transAxes, animated=True)
    text6 = ax6.text(1, 1, ax2txts_list[0][2], fontsize=18, color='white', ha='right', va='top', 
                     transform=ax6.transAxes, animated=True)
    
    # 存储对象
    axes.append((ax1,    ax2,     ax3,     ax4,     ax5,     ax6))
    ims.append((im1spec, im2imag, im3spec, im4imag, im5spec, im6imag))
    texts.append((text1, text2,   text3,   text4,   text5,   text6))
    
    # 更新函数
    def update(frame):
        # 更新 imshow 数据
        im1spec.set_array(spec_data_list[frame][0])
        im3spec.set_array(spec_data_list[frame][1])
        im5spec.set_array(spec_data_list[frame][2])
        im2imag.set_array(image_data_list[frame])
        im4imag.set_array(image_data_list[frame])
        im6imag.set_array(image_data_list[frame])
        
        # 更新 extent
        im1spec.set_extent(extent_list[frame][0])
        im3spec.set_extent(extent_list[frame][1])
        im5spec.set_extent(extent_list[frame][2])
        
        # 更新 vmin / vmax
        noise1 = np.std(image_data_list[frame])
        noise2 = np.std(image_data_list[frame])
        noise3 = np.std(image_data_list[frame])
        vmin1, vmax1 = -noise1, 5*noise1
        vmin2, vmax2 = -noise2, 5*noise2
        vmin3, vmax3 = -noise3, 5*noise3
        im2imag.set_clim(vmin1, vmax1)
        im4imag.set_clim(vmin2, vmax2)
        im6imag.set_clim(vmin3, vmax3)
        
        # 更新 ylim
        ax1.set_ylim(slit_ylim_list[frame][0])
        ax3.set_ylim(slit_ylim_list[frame][1])
        ax5.set_ylim(slit_ylim_list[frame][2])

        # 更新文本
        text1.set_text(ax1txts_list[frame][0])
        text2.set_text(ax2txts_list[frame][0])
        text3.set_text(ax1txts_list[frame][1])
        text4.set_text(ax2txts_list[frame][1])
        text5.set_text(ax1txts_list[frame][2])
        text6.set_text(ax2txts_list[frame][2])
    
        return [obj for pair in ims for obj in pair] + [txt for pair in texts for txt in pair]# + [s for s in slit_lines]

    
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, blit=True)
    
    # 保存动画
    ani.save("animation.gif", writer="pillow", fps=1)
    return

