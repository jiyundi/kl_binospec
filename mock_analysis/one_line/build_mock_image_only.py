import joblib
# import pickle
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import galsim
from astropy import wcs
# from astropy.io import fits
# from astropy.coordinates import SkyCoord
# from scipy.stats import qmc

from parameters  import Parameters
from spec_model  import SlitModel
from image_model import ImageModel
from mock import Mock

# Now get object data from object catalog
mask      = 'C'
slit_name = '095'
RA_obj     = 42.*u.deg
Dec_obj    = -3.*u.deg
redshift   =  0.94
emi_line_lst = ['OII']
meta_gal = {}
meta_gal = {
            'redshift': redshift,
            'RA':      RA_obj,
            'Dec':     Dec_obj,
            'beta':    0*u.deg,
            # 'log10_Mstar':    10.0,
            # 'log10_Mstar_err': 0.2,
            }

iter_num      = 203

mock_params = {
    'shared_params-image_snr': 80,
    'shared_params-spec_snr':  20,
    'shared_params-gamma_t':    0.0,
    'shared_params-cosi':       0.2,
    'shared_params-theta_int':  0.0,
    'shared_params-vscale':     0.2,
    'shared_params-r_hl_disk':  1.0,
    'shared_params-r_hl_bulge': 0.8,
    
    'shared_params-vcirc':    142,
    'shared_params-v_0':        0,
    'shared_params-dx_disk':    0,
    'shared_params-dy_disk':    0,
    'shared_params-dx_bulge':   0,
    'shared_params-dy_bulge':   0,
    'shared_params-flux':       3.0, # = log(F)
    'shared_params-flux_bulge': 2.0, # = log(F_bulge)
    
    'OII_params-dx_vel':      0,
    'OII_params-dx_vel_2':    0,
    'OII_params-I01':        10,
    'OII_params-I02':        10,
    'OII_params-bkg_level':   1
}

slit_len   =  8
slit_width =  1
slit_LPA   = 25*u.deg
slit_LPA   = 90*u.deg - slit_LPA  # degrees east of north. Want w.r.t. east.


slit_RA    = RA_obj  + slit_width*u.arcsec * np.sin(slit_LPA)
slit_Dec   = Dec_obj - slit_width*u.arcsec * np.cos(slit_LPA)


image_shape     = (3, 3)
image_pix_scale = 0.2     # arcsec/pix: 0.2 for HSC image
psfFWHM         = 0.6       # arcsec
ap_wcs           = wcs.WCS(naxis=2) # Create WCS
ap_wcs.wcs.crpix = np.array([image_shape[0]/2+0.5,
                             image_shape[0]/2+0.5]) # Cntrl ref pix (0.5-based)
# ap_wcs.wcs.crpix = ap_wcs.wcs.crpix + 1 # 1-based --> 2-based just for imshow
ap_wcs.wcs.crval = [RA_obj.value, Dec_obj.value] # RA/Dec (deg) central pixel
ap_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN'] 
ap_wcs.wcs.cdelt = [-image_pix_scale/3600, image_pix_scale/3600] # deg/px
galsim_wcs       = galsim.AstropyWCS(wcs=ap_wcs)
meta_image = {
    'ngrid':    image_shape,
    'pixScale': image_pix_scale,
    'psfFWHM':  psfFWHM,
    'wcs':      galsim_wcs,
    'ap_wcs':   ap_wcs,
    'RA':       RA_obj.value,
    'Dec':      Dec_obj.value}





# spec_data_info = {
#     # 'data':       mock_data['spec_data'],
#     # 'var':        mock_data['spec_var'],
#     # 'cont_model': mock_data['cont_model_spec'],
#     # 'par_meta':   mock_data['meta_spec']
#     }

# image_data_info = {
#     'data':     mock_data['image_data'],
#     'var':      mock_data['image_var'],
#     'par_meta': meta_image
#     }

# mock_data_info = {
#     'spec':    [spec_data_info],
#     'image':   image_data_info,
#     'galaxy':  meta_gal,
#     'par_fit': mock_params}


fig = plt.figure(figsize=(13, 5))  # (length, height)
plt.subplots_adjust(hspace=0.4, wspace=0.0) # h=height
gs = fig.add_gridspec(nrows=1, ncols=2, 
                      height_ratios=[1], 
                      width_ratios=[1, 2])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], 
                      projection=ap_wcs)

ax1.set_xlabel(r'Observed Wavelength $\lambda$ ($\AA$)')
ax1.set_ylabel(r'Spatial Position (arcsec)')
ax1.grid(linestyle=':', color='white', alpha=0.5)
ax1.set_title('Mock MMT/Binospec 2D Spec')

def rot_rectangle(ax, x0, y0, dx, dy, rotation, color, ls):
    xUL = x0 + (-dx/2)*np.cos(rotation) - (+dy/2)*np.sin(rotation)
    yUL = y0 + (-dx/2)*np.sin(rotation) + (+dy/2)*np.cos(rotation)
    xUR = x0 + (+dx/2)*np.cos(rotation) - (+dy/2)*np.sin(rotation)
    yUR = y0 + (+dx/2)*np.sin(rotation) + (+dy/2)*np.cos(rotation)
    xLL = x0 + (-dx/2)*np.cos(rotation) - (-dy/2)*np.sin(rotation)
    yLL = y0 + (-dx/2)*np.sin(rotation) + (-dy/2)*np.cos(rotation)
    xLR = x0 + (+dx/2)*np.cos(rotation) - (-dy/2)*np.sin(rotation)
    yLR = y0 + (+dx/2)*np.sin(rotation) + (-dy/2)*np.cos(rotation)
    ax.plot([xUL, xUR, xLR, xLL, xUL], 
            [yUL, yUR, yLR, yLL, yUL], color=color, linestyle=ls)
    return 


params       = Parameters(line_species=emi_line_lst)
updated_dict = params.gen_param_dict(mock_params.keys(), 
                                     mock_params.values())
image_model   = ImageModel(meta_image=meta_image)
image_data    = image_model.get_image(updated_dict['shared_params'])

objRA      = RA_obj.value
objDec     = Dec_obj.value
pixScale   = image_pix_scale
slitRA     = slit_RA.value
slitDec    = slit_Dec.value
slitLen    = slit_len
slitWidth  = slit_width
slit_LPA   = slit_LPA.value
im_imag = ax2.imshow(
    # ax2.matshow(
    image_data, 
    # origin='lower'), 
                     cmap='viridis', aspect='equal')
fig.colorbar(im_imag, ax=ax2)
x0obj,  y0obj  = ap_wcs.wcs_world2pix([[objRA,  objDec ]], 0)[0]  
x0slit, y0slit = ap_wcs.wcs_world2pix([[slitRA, slitDec]], 0)[0]  
ax2.scatter(x0obj, y0obj, marker='x', s=360, color='black', zorder=1)
rot_rectangle(ax2, x0slit, y0slit, slitWidth/pixScale, slitLen/pixScale, 
              (90-slit_LPA)/57.3, 'white', '-')
ax2.set_xlim(left=0, right=image_data.shape[1]-1)
ax2.set_ylim(bottom=0, top=image_data.shape[0]-1)
ax2.coords['ra' ].set_major_formatter('dd:mm:ss')
ax2.coords['dec'].set_major_formatter('dd:mm:ss')
ax2.set_xlabel('RA')
ax2.set_ylabel('Dec')
ax2.grid(linestyle=':', color='white', alpha=0.5)
ax2.set_title('Mock Subaru Imaging')

plt.savefig(f'delete.png', 
            dpi=150, bbox_inches='tight')
plt.show()
plt.close()