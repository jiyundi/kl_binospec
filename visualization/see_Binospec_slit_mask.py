import numpy as np
from   astropy.io  import fits
from   astropy.wcs import WCS
from   astropy.visualization import make_lupton_rgb
from   adjustText import adjust_text
import matplotlib.pyplot as plt
from   matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from klm.safe_plot import setup; setup() # must before plt
plt.style.use('default')
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})
# plt.style.use('dark_background')


def read_subaru_img_wcs(image_science_filename, 
                        image_weights_filename=None):
    """
    Read Subaru imaging science and weights data in units of 
    e/s and (e/s)**-2.
    
    Returns
    ----------
    {science_data, science_wcs, weights_data, weights_wcs}
    """
    science_hdul = fits.open(image_science_filename)
    
    # Pixel data
    science_raw = science_hdul[0].data
    
    # WCS mapping
    science_wcs = WCS(image_science_filename)
    science_hdr0_raw = science_hdul[0].header
    sci_GAIN = science_hdr0_raw['GAIN']
    
    # Unit conversion
    # Science = science_raw (ADU/s)     * sci_GAIN    (e/ADU)
    # Weights = weights_raw (s^2/ADU^2) * wei_GAIN^-2 (e/ADU)^-2
    science_data = science_raw * sci_GAIN
    
    weights_data, weights_wcs = None, None
    if image_weights_filename is not None: 
        weights_hdul = fits.open(image_weights_filename)
        weights_raw  = weights_hdul[0].data
        weights_wcs  = WCS(image_weights_filename)
        weights_hdr0_raw = weights_hdul[0].header
        wei_GAIN     = weights_hdr0_raw['GAIN']
        weights_data = weights_raw * wei_GAIN**(-2)
    
    return {'science_data': science_data, 
            'science_wcs':  science_wcs, 
            'weights_data': weights_data, 
            'weights_wcs':  weights_wcs}


def cutoutimg(arrimg, wcs_loaded, objRA, objDEC, 
              img_width=30, img_height=30, outputWCS=False):
    img_pix_scale = 0.2 # arcsec/pix: 0.2 for HSC image
    img_RAwidth   = img_width  * img_pix_scale
    img_DECheight = img_height * img_pix_scale
    RAmin  = objRA  - img_RAwidth  /2/3600  
    RAmax  = objRA  + img_RAwidth  /2/3600
    DECmin = objDEC - img_DECheight/2/3600
    DECmax = objDEC + img_DECheight/2/3600
    eastIndex, southIndex = wcs_loaded.wcs_world2pix(RAmax, DECmin, 0)
    westIndex, northIndex = wcs_loaded.wcs_world2pix(RAmin, DECmax, 0)
    arrimgcutout =       arrimg[int(southIndex):int(northIndex)+1, :]
    arrimgcutout = arrimgcutout[:,   int(eastIndex):int(westIndex)+1]
    if outputWCS:
        wcs_cutout = wcs_loaded[int(southIndex):int(northIndex)+1,
                                int( eastIndex):int( westIndex)+1]
        return arrimgcutout, [RAmin, RAmax], [DECmin, DECmax], \
               wcs_cutout
    else:
        return arrimgcutout, [RAmin, RAmax], [DECmin, DECmax]


def binospec_fov(ax, ctr_pt, rotatn, color='#2176ff'):
    ra0, de0 = ctr_pt[0], ctr_pt[1]
    cnrA1r = ra0 + ((-480-96)/3600)*np.cos(rotatn) - ((+450)/3600)*np.sin(rotatn)
    cnrA1d = de0 + ((-480-96)/3600)*np.sin(rotatn) + ((+450)/3600)*np.cos(rotatn)
    cnrA2r = ra0 + ((-000-96)/3600)*np.cos(rotatn) - ((+450)/3600)*np.sin(rotatn)
    cnrA2d = de0 + ((-000-96)/3600)*np.sin(rotatn) + ((+450)/3600)*np.cos(rotatn)
    cnrA3r = ra0 + ((-000-96)/3600)*np.cos(rotatn) - ((-450)/3600)*np.sin(rotatn)
    cnrA3d = de0 + ((-000-96)/3600)*np.sin(rotatn) + ((-450)/3600)*np.cos(rotatn)
    cnrA4r = ra0 + ((-480-96)/3600)*np.cos(rotatn) - ((-450)/3600)*np.sin(rotatn)
    cnrA4d = de0 + ((-480-96)/3600)*np.sin(rotatn) + ((-450)/3600)*np.cos(rotatn)
    Ax_values = [cnrA1r, cnrA2r, cnrA3r, cnrA4r, cnrA1r]
    Ay_values = [cnrA1d, cnrA2d, cnrA3d, cnrA4d, cnrA1d]
    cnrB1r = ra0 + ((+480+96)/3600)*np.cos(rotatn) - ((+450)/3600)*np.sin(rotatn)
    cnrB1d = de0 + ((+480+96)/3600)*np.sin(rotatn) + ((+450)/3600)*np.cos(rotatn)
    cnrB2r = ra0 + ((+000+96)/3600)*np.cos(rotatn) - ((+450)/3600)*np.sin(rotatn)
    cnrB2d = de0 + ((+000+96)/3600)*np.sin(rotatn) + ((+450)/3600)*np.cos(rotatn)
    cnrB3r = ra0 + ((+000+96)/3600)*np.cos(rotatn) - ((-450)/3600)*np.sin(rotatn)
    cnrB3d = de0 + ((+000+96)/3600)*np.sin(rotatn) + ((-450)/3600)*np.cos(rotatn)
    cnrB4r = ra0 + ((+480+96)/3600)*np.cos(rotatn) - ((-450)/3600)*np.sin(rotatn)
    cnrB4d = de0 + ((+480+96)/3600)*np.sin(rotatn) + ((-450)/3600)*np.cos(rotatn)
    Bx_values = [cnrB1r, cnrB2r, cnrB3r, cnrB4r, cnrB1r]
    By_values = [cnrB1d, cnrB2d, cnrB3d, cnrB4d, cnrB1d]
    ax.plot(Ax_values, Ay_values, color=color, # BinoSpec FoV-1A
            transform=ax.get_transform('world'), zorder=2)
    ax.plot(Bx_values, By_values, color=color, # BinoSpec FoV-1B
            transform=ax.get_transform('world'), zorder=2)
    return 


def scale_calc(z, H0, Omega_M, Omega_Lam):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0, Om0=Omega_M) # a selected cosmological model
    d_A = cosmo.kpc_proper_per_arcmin(z)
    d_A = d_A.value/60 # kpc/arcmin --> kpc/arcsec
    return d_A


def draw_r500_r200(plt, ax, r_500, r_200, color='greenyellow'):
    circ_500 = plt.Circle(( (( 2)+(48)/60+( 3.3)/3600)*15,
                           -(( 3)+(31)/60+(46.4)/3600)   ), 
                          r_500/3600, 
                          color=color, fill=False, linestyle='--', # 'dashdot'
                          linewidth=1.2, zorder=1, 
                          transform=ax.get_transform('world'))
    circ_200 = plt.Circle(( (( 2)+(48)/60+( 3.3)/3600)*15,
                           -(( 3)+(31)/60+(46.4)/3600)   ), 
                          r_200/3600, 
                          color=color, fill=False, linestyle=':', 
                          zorder=1, 
                          transform=ax.get_transform('world'))
    ax.text( (( 2)+(48)/60+( 3.3)/3600)*15,
             -(( 3)+(31)/60+(46.4)/3600) - r_500/3600,
             r'$r_{\rm 500}$', color=color, size=18, 
             verticalalignment='top', 
             transform=ax.get_transform('world'))
    ax.text( (( 2)+(48)/60+( 3.3)/3600)*15,
             -(( 3)+(31)/60+(46.4)/3600) - r_200/3600,
             r'$r_{\rm 200}$', color=color, size=18, 
             verticalalignment='top', 
             transform=ax.get_transform('world'))
    ax.add_patch(circ_500)
    ax.add_patch(circ_200)
    return circ_500, circ_200










cluster_centerRA  =  ( ( 2)+(48)/60+( 3.3)/3600 )*15
cluster_centerDEC = -( ( 3)+(31)/60+(46.4)/3600 )

# Read slit info: C (2023.1019)
hdulist01     = fits.open('../../../RSCH3/UAO-S156-23B-A383/psf/231019/obj_abs_slits_extr.fits')
slitsInfo01   = np.append(hdulist01[4].data, hdulist01[5].data, axis=0)
n_slits_sideA = len(hdulist01[4].data)
slitsInfo01['SLIT'][n_slits_sideA:] = slitsInfo01['SLIT'][n_slits_sideA:] + n_slits_sideA
maskCenterRA, maskCenterDEC = slitsInfo01['MASK_RA'][0], slitsInfo01['MASK_DEC'][0]

# Read imaging and WCS
img_dir  = '../../../RSCH3/HSC_img_A383/'
sci_fn_R = 'hlsp_clash_subaru_suprimecam_a383_z_2010-2002-v20110405_drz.fits'
sci_fn_G = 'hlsp_clash_subaru_suprimecam_a383_ip_2005-v20110422_drz.fits'
sci_fn_B = 'hlsp_clash_subaru_suprimecam_a383_v_2002-2005-2008-v20110405_drz.fits'
img_data_R = read_subaru_img_wcs(img_dir+sci_fn_R, None)
img_data_G = read_subaru_img_wcs(img_dir+sci_fn_G, None)
img_data_B = read_subaru_img_wcs(img_dir+sci_fn_B, None)
imgR, wcsR = img_data_R['science_data'], img_data_R['science_wcs' ]
imgG, wcsG = img_data_G['science_data'], img_data_G['science_wcs' ]
imgB, wcsB = img_data_B['science_data'], img_data_B['science_wcs' ]
scale = 0.2 # arcsec/pix: 0.2 for image
image_size = 25 # arcmin
img_cutoutR, RAlim, DEClim, wcs_cutoutR = cutoutimg(
    imgR, wcsR, maskCenterRA, maskCenterDEC, 
    img_width=(image_size)*60/scale, 
    img_height=(image_size)*60/scale, 
    outputWCS=True
    )
img_cutoutG, RAlim, DEClim, wcs_cutoutG = cutoutimg(
    imgG, wcsG, maskCenterRA, maskCenterDEC, 
    img_width=(image_size)*60/scale, 
    img_height=(image_size)*60/scale, 
    outputWCS=True
    )
img_cutoutB, RAlim, DEClim, wcs_cutoutB = cutoutimg(
    imgB, wcsB, maskCenterRA, maskCenterDEC, 
    img_width=(image_size)*60/scale, 
    img_height=(image_size)*60/scale, 
    outputWCS=True
    )

# spec1d2dfolder02 = '../../RSCH3/UAO-S156-23B-A383/psf/231019/1d2dspecfiles/'
# hdulist01 = fits.open(spec1d2dfolder02+'../obj_abs_slits_extr.fits')
# slitsInfo01 = np.append(hdulist01[4].data, hdulist01[5].data, axis=0)
# slitsInfo01['SLIT'][n_slits_sideA:] = slitsInfo01['SLIT'][n_slits_sideA:] + n_slits_sideA
# maskCenterRA, maskCenterDEC = slitsInfo01['MASK_RA'][0], slitsInfo01['MASK_DEC'][0]


fig = plt.figure(figsize=(8, 8), dpi=300)
plt.subplots_adjust(hspace=0.2, wspace=0) # h=height
gs = fig.add_gridspec(1, 1,
                      height_ratios=[1],
                      width_ratios=[1])
ax1 = fig.add_subplot(gs[0, 0], projection=wcs_cutoutG)

# Annotate slits & Avoid overlapping
texts = []
for i in range(len(slitsInfo01['SLIT'])):
    ax1.scatter(slitsInfo01['RA' ][i], 
                slitsInfo01['DEC'][i], 
                marker='o',
                s=16, facecolors='none', edgecolors='royalblue',
                transform=ax1.get_transform('world'), zorder=4)
        
    texts.append(
        ax1.text(slitsInfo01['RA'  ][i]-5/3600, 
                 slitsInfo01['DEC' ][i]+5/3600, 
                 slitsInfo01['SLIT'][i], 
                 ha='left', va='bottom', 
                 fontsize=6, color='royalblue', zorder=5, alpha=0.75, 
                 transform=ax1.get_transform('world'))
        )
    
    # Add fake texts to hold scatter points' postitions
    texts.append(
        ax1.text(slitsInfo01['RA'  ][i], 
                 slitsInfo01['DEC' ][i], 
                 '  ', 
                 ha='center', va='center', 
                 fontsize=4, color='royalblue', zorder=5, alpha=0, 
                 transform=ax1.get_transform('world'))
        )

adjust_text(texts, ax=ax1, arrowprops=None)

# Binospec FOV
binospec_fov(ax1, 
             [slitsInfo01['MASK_RA' ][0], slitsInfo01['MASK_DEC'][0]], 
             np.radians(-slitsInfo01['MASK_PA'][0]),
             color='orangered')

# R_500, R_200(=R_500/0.65)
R_500 = 944 # kpc, Vikhlinin et al. (2006) arXiv:astro-ph/0507092, Table 4
d_A   = scale_calc(0.1883, 70, 0.3, 0.7) # kpc/arcsec
r_500 = R_500 / d_A
r_200 = r_500 / 0.65
c500, c200 = draw_r500_r200(plt, ax1, r_500, r_200, color='green')
    
# Astropy RGB solution
# ax1.imshow(make_lupton_rgb(img_cutoutR*0.01, 
#                            img_cutoutG*0.01, 
#                            img_cutoutB*0.004, 
#                            # stretch=0.2, Q=10
#                            ), 
#            origin='lower', zorder=-1, alpha=1)

rgb = make_lupton_rgb(img_cutoutR*0.010,
                      img_cutoutG*0.010,
                      img_cutoutB*0.004
                     ).astype(float) / 255.
rgb_inv    = 1 - rgb # Step 1: 反色
hsv        = rgb_to_hsv(rgb_inv)
hsv[...,0] = (hsv[...,0] + 0.5) % 1.0 # Step 2: HSV空间 Hue+180°
rgb_final  = hsv_to_rgb(hsv)
ax1.imshow(rgb_final, origin='lower', zorder=-1, alpha=1)

# Add RA/DEC float (deg) top and right
lon = ax1.coords[0]
lat = ax1.coords[1]
lon.set_major_formatter('hh:mm:ss')
lat.set_major_formatter('dd:mm')
overlay = ax1.get_coords_overlay('fk5')
overlay[0].set_axislabel('RA (deg)')
overlay[1].set_axislabel('DEC (deg)')
overlay[0].set_major_formatter('d.d')
overlay[1].set_major_formatter('d.d')
overlay[0].set_ticks_position('t')
overlay[1].set_ticks_position('r')
overlay[0].set_ticklabel_position('t')
overlay[1].set_ticklabel_position('r')

ax1.set_xlabel('RA (deg)', fontsize=15)
ax1.set_ylabel('DEC (deg)', fontsize=15)
ax1.minorticks_on()
ax1.grid(linestyle=':', color='green', alpha=0.5, zorder=0)
# ax1.legend(prop={'size': 12})
ax1.set_aspect(1)

plt.savefig("binospec_mask_slits_.jpg", dpi=300, bbox_inches='tight')



