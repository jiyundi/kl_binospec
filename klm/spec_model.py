from abc import abstractmethod
# from matplotlib import axis
from matplotlib.patches import Polygon
import numpy as np
# import scipy.signal

# import galsim
import astropy.units as u

from klm.utils           import build_map_grid
from klm.parameters      import Parameters
from klm.transformations import Transformations
from klm.velocity        import VelocityModel
from klm.emission_line   import EmissionLine


class KinematicModel():
    def __init__(self) -> None:
        pass

    # Each subclass must have a method that initializes the observable
    @abstractmethod
    def _init_observable():
        '''
        Initializes the observable for the spec model
        Setup grid and necessary class attributes here
        '''
        pass

    # Each subclass must have a method that returns the observable
    @abstractmethod
    def get_observable():
        '''
        Method for returning the observable Spectrum/velocity field
        '''
        pass


# class IFUModel(KinematicModel):
#     pass


class SlitModel(KinematicModel):
    '''
    Class for creating model spectrum
    '''
    def __init__(self, obj_param=None, meta_param=None, 
                 line_profile=None, redo_data=None, 
                 rc_type='arctan'):
        self.velocity_model = VelocityModel(rc_type)
        self._init_observable(obj_param, meta_param)

        self.sigma_intr = 0.01*u.nm
        self.line_name, self.line_wav = Parameters._get_species(
            meta_param['line_species'], obj_param['redshift']
            )

        # sigma1 & sigma2 are the line widths of the two half-Gaussians
        if line_profile == 'redo':
            self.emission_line_model = EmissionLine(
                use_analytic=False, 
                line_species=meta_param['line_species'], 
                line_profile={'data': redo_data})
        
        elif line_profile is not None:
            self.emission_line_model = EmissionLine(
                use_analytic=False, 
                line_species=meta_param['line_species'], 
                line_profile=meta_param['line_profile'])

        elif 'rhl' in meta_param.keys():
            self.emission_line_model = EmissionLine(
                use_analytic=True, 
                line_species=meta_param['line_species'], 
                rhl=meta_param['rhl'], 
                slit_grid=self.slit_x)

        else:
            raise ValueError('Either need line profile from obs or rhl to instantiate analytic line profile')

        self.profile_sigma1, self.profile_sigma2 = self.emission_line_model.line_profile.get_linewidth_profile(self.ndim)
        self.profile_one_over_sigma1 = [1/p for p in self.profile_sigma1]
        self.profile_one_over_sigma2 = [1/p for p in self.profile_sigma2]
        self.profile_A = self.emission_line_model.intensity.get_intensity_profile(self.ndim)

    def _init_observable(self, obj_param, meta_param):
        '''Initialize grid and slit mask for spec calculation
        Assumes ngrid[0] corresponds to the spatial axis

        Parameters
        ----------
        obj_param : dict
            Object parameters including RA, Dec
        meta_param : dict
            Spectrum meta parameters
        '''
        # convert degrees to arcsec
        RA_obj   = obj_param['RA' ].to(u.arcsec)
        Dec_obj  = obj_param['Dec'].to(u.arcsec)
        RA_slit  = meta_param['slitRA' ].to(u.arcsec)
        Dec_slit = meta_param['slitDec'].to(u.arcsec)
        
        # lambda should run along rows & spatial position along columns
        self.lambda_grid = meta_param['lambda_grid'].to(u.nm).value  

        # Shape of 3D data-cube
        # Indexed as (x, y, lambda)
        self.ndim = (
            meta_param['ngrid'][0], 
            meta_param['ngrid'][0]+10, 
            meta_param['ngrid'][1])

        slit_xx, slit_yy = build_map_grid(Nx=self.ndim[0], 
                                          Ny=self.ndim[1], 
                                          pix_scale=meta_param['pixScale'])
        
        # Special note of slit_xx:  axis 0 = REAL spectrum spatial axis
        #                           axis 1 = spectrum spatial axis (expanded to include 10 extra pixels)
        #                           slit_xx:
        #                           62 *****************************
        #                           61 =============================
        #                           60 -----------------------------
        #                           ..
        #                            0 .............................
        #                              0 1 2 3 4 5 6 7 ... 62 ... 72

        # All columns (NOT ROWS) are the same, slit_xx[:,0] = slit_xx[:,1] = ...
        self.slit_xx = slit_xx 
        self.slit_yy = slit_yy
        self.slitLPA = meta_param['slitLPA']

        # This is spectrum spatial axis, slit_xx[0, 0] = slit bottom
        self.slit_x  = slit_xx[:,0] 

        self.slit_patch = self._get_sky_xaxis_aligned_slit_patch(
            RA_slit.value-RA_obj.value, Dec_slit.value-Dec_obj.value, 
            meta_param['slitLen'], meta_param['slitWidth'], 
            meta_param['slitLPA'], meta_param['slitWPA']
            )
        self.slit_mask = self._get_slit_mask(self.slit_patch, 
                                             slit_xx, slit_yy)
        self.obs_xx, self.obs_yy = Transformations.transform_frame(
            self.slit_xx, self.slit_yy, start='slit', end='obs', 
            params={'slitLPA': meta_param['slitLPA']}
            )


    def get_observable(self, params, return_type='spec', wave_only=False):
        if return_type == 'rotation_curve':
            return self.get_rotation_curve_model(params, wave_only)
        elif return_type == 'spec':
            return self._build_spectrum(params)
        elif return_type == 'spec+rotation_curve':
            return [self._build_spectrum(params), self.get_rotation_curve_model(params)]
        else:
            raise ValueError(f"Invalid return_type: {return_type}. Must be 'rotation_curve' or 'spec'.")


    def get_rotation_curve_model(self, params, wave_only=False):
        """
        Generate the predicted emission-line center k_model(y), see math below.

        Returns
        -------
        k_model : ndarray, shape (n_spatial,)
            Model of the emission-line center in wavelength-pixel coordinates (k) 
            as a function of the spatial coordinate (y).
        """
        vfields = self._build_slit_vfield(params)

        if self.line_name not in ['O2a', 'O2']:
            # 提取沿狭缝方向的一维旋转曲线 v_model(y)
            v_model = self._extract_slit_rotation_curve(vfields[0])

            # 把视向速度转换为光谱列坐标 k
            k_model = self._velocity_to_k(v_model, self.line_wav, 
                                          need_lambda_model=wave_only)

        elif self.line_name in ['O2a', 'O2']:
            raise ValueError('spec_model.py: We have not supported the O2 yet.')

        else:
            raise ValueError('No idea what line you want to use.')

        return k_model


    def _extract_slit_rotation_curve(self, vfields):
        """
        Extract v_model(y) at the transverse center of the slit.
        """
        vfields = np.asarray(vfields)

        if vfields.shape != self.slit_mask.shape:
            raise ValueError(
                "vfields and slit_mask have inconsistent shapes: "
                f"{vfields.shape} versus {self.slit_mask.shape}"
            )

        n_spatial = vfields.shape[0]
        v_model = np.full(n_spatial, np.nan)

        for y in range(n_spatial):

            valid = ~self.slit_mask[y]

            if not np.any(valid):
                continue

            valid_indices = np.flatnonzero(valid)

            # 在狭缝内部选择最接近 slit_xx = 0 的像素
            local_index = np.argmin(
                np.abs(self.slit_xx[y, valid_indices])
            )

            center_index = valid_indices[local_index]

            v_model[y] = vfields[y, center_index]

        return v_model


    def _velocity_to_k(self, v_model, lambda_systemic, need_lambda_model=False):
        """
        Convert line-of-sight velocity into spectral-pixel position.
        (I.e. velocity --> wavelength --> pixel position)
        """
        c_kms = 299792.458

        # 非相对论 Doppler 近似
        lambda_model = lambda_systemic.to(u.nm).value * (
            1.0 + v_model / c_kms
        )
        if need_lambda_model:
            return lambda_systemic * (
                1.0 + v_model / c_kms
            )

        lambda_grid = self.lambda_grid # values, in nm

        n_spatial, n_wave = lambda_grid.shape

        if len(v_model) != n_spatial:
            raise ValueError(
                "v_model and lambda_grid have inconsistent spatial sizes: "
                f"{len(v_model)} versus {n_spatial}"
            )

        pixel_grid = np.arange(n_wave, dtype=float)
        k_model = np.full(n_spatial, np.nan)

        for y in range(n_spatial):

            if not np.isfinite(lambda_model[y]):
                continue

            # Wavelength axis at this spatial position
            wave_y = lambda_grid[y] 

            # mask for finite values
            finite = np.isfinite(wave_y) 

            # Finite values check
            if np.count_nonzero(finite) < 2:
                continue

            wave_valid = wave_y[finite]
            pixel_valid = pixel_grid[finite]

            # np.interp 要求横坐标递增
            if wave_valid[0] > wave_valid[-1]:
                wave_valid = wave_valid[::-1]
                pixel_valid = pixel_valid[::-1]

            # 超出模型波长范围时返回 nan
            if (
                lambda_model[y] < wave_valid[0]
                or lambda_model[y] > wave_valid[-1]
            ):
                continue

            k_model[y] = np.interp(
                lambda_model[y],
                wave_valid,
                pixel_valid,
            )

        return k_model


    def _build_spectrum(self, params):
        """
        Generate the full 2D spectrum.
        """
        vfields = self._build_slit_vfield(params)

        self.vfield = vfields

        galaxy_emission = self._add_emission_line(params, vfields)
        
        # Size and axis: (slit_px, slit_px+10, wavelength)
        spectrum3D = galaxy_emission * (~self.slit_mask)[:, :, np.newaxis]
        
        spectrum2D = np.sum(spectrum3D, axis=1)
        
        return spectrum2D + params['bkg_level']
    
    
    def _build_slit_vfield(self, params):
        ''' Evaluates the velocity field on a 2D grid. Computes two separate rotation curves
        if the emission line is a doublet.
        Actual computation is done in a lower level function.
        Args:
            params (_type_): _description_
            Xgrid (_type_): _description_
            Ygrid (_type_): _description_

        Returns:
            _type_: _description_
        '''
        disk_xx, disk_yy = Transformations.transform_frame(
                    self.obs_xx, self.obs_yy, params=params, 
                    start='obs', end='disk'
                    )

        Vfield1, Vfield2 = 0., 0.

        params_vfield = {'v_0':     params['v_0'],
                         'vcirc':   params['vcirc'],
                         'cosi':    params['cosi'],
                         'vscale':  params['vscale'],
                         'v_outer': params['v_outer'],
                         'dx_vel':  params['dx_vel'],
                         'dy_vel':  params['dy_vel'],
                         'r_hl_disk': params['r_hl_disk']}
        Vfield1 = self.velocity_model.build_vfield(params_vfield, disk_xx, disk_yy)

        if self.line_name in ['O2a', 'O2']:
            params_vfield2 = {'v_0':     params['v_0_2'],
                              'vcirc':   params['vcirc'],
                              'cosi':    params['cosi'],
                              'vscale':  params['vscale'],
                              'v_outer': params['v_outer'],
                              'dx_vel':  params['dx_vel_2'],
                              'dy_vel':  params['dy_vel_2'],
                              'r_hl_disk': params['r_hl_disk']}
            Vfield2 = self.velocity_model.build_vfield(params_vfield2, disk_xx, disk_yy)

        return Vfield1, Vfield2

    def _add_emission_line(self, params, velocity_fields):
        '''Returns a delta wavelength grid
        '''
        c = 299792.45  # Speed of light in km/s
        cube = 0.0
        
        # For each line in this singlet/doublet line
        for i, this_line in enumerate(self.line_wav):
            vfield = velocity_fields[i]
            line_center = (1 + vfield/c) * (this_line.to(u.nm).value)
            # if i == 0:
            #     # for left/right sigma of line 1
            #     profile_one_over_sigma = [1 / (ss * 1) 
            #                               for ss in self.profile_sigma1]
            # else:
            #     # for left/right sigma of line 2
            #     profile_one_over_sigma = [1 / (ss * 1) 
            #                               for ss in self.profile_sigma2]
            
            delta_wavelength = self.lambda_grid[:, np.newaxis, :] - line_center[:, :, np.newaxis]
            
            self.delta_wav = delta_wavelength

            exponent = np.zeros_like(delta_wavelength)
            idx = delta_wavelength < 0
            
            exponent[ idx] = -0.5 * (
                delta_wavelength[ idx] * 
                self.profile_one_over_sigma1[i][ idx] * 
                params[f'f{i+1}_1']
                ) ** 2
            exponent[~idx] = -0.5 * (
                delta_wavelength[~idx] * 
                self.profile_one_over_sigma2[i][~idx] * 
                params[f'f{i+1}_2']
                ) ** 2
            
            cube += params[f'I0{i+1}'] * np.exp(exponent) \
                * self.profile_A[i][:, :, np.newaxis] # / np.max(self.profile_A[i])

        return cube

    def _get_slit_patch(self, RA, Dec, LEN, WID, LPA, WPA):
        '''_summary_

        Parameters
        ----------
        RA : float
            RA (in arcsec)
        Dec : float
            Dec (in arcsec)
        LEN : float
            Slit length (in arcsec)
        WID : float
            slit width (in arcsec)
        LPA : float
            Position angle of the long side of the slit (in degrees)
        WPA : float
            Position angle of the short side of the slit (in degrees)

        Returns
        -------
        _type_
            _description_
        '''
        LPA, WPA = LPA.to(u.radian).value, WPA.to(u.radian).value
        v_center = np.array([RA, Dec])

        # vectors from center to mid-points of the four sides
        vec_N = WID/2*np.array([np.cos(WPA), np.sin(WPA)])  # North
        vec_S = WID/2*np.array([np.cos(np.pi+WPA), np.sin(np.pi+WPA)])  # South
        vec_E = LEN/2*np.array([np.cos(LPA), np.sin(LPA)])  # East
        vec_W = LEN/2*np.array([np.cos(np.pi+LPA), np.sin(np.pi+LPA)])  # West

        # The four corners of the slit mask
        vec_NW = v_center + vec_N + vec_W
        vec_NE = v_center + vec_N + vec_E
        vec_SE = v_center + vec_S + vec_E
        vec_SW = v_center + vec_S + vec_W

        vec_slit = [vec_NW, vec_NE, vec_SE, vec_SW]
        self._assert_slit_PA(vec_slit, LPA, WPA)
        # print('Creating slit mask...')
        slit_patch = Polygon(vec_slit)
        slit_patch.set_closed(True)

        return slit_patch

    def _get_sky_xaxis_aligned_slit_patch(self, RA, Dec, LEN, WID, LPA, WPA):
        original_slit_patch = self._get_slit_patch(RA, Dec, LEN, WID, LPA, WPA)

        vertices = original_slit_patch.get_xy()
        xp, yp = Transformations._apply_rotation(vertices[:, 0], vertices[:, 1], -LPA)
        aligned_patch = Polygon(np.column_stack((xp, yp)))
        aligned_patch.set_closed(True)

        return aligned_patch


    def _assert_slit_PA(self, vectors, LPA, WPA):
        ''' Checks if the slit mask has the correct PA given coordinates of the
        four slit corners
        '''
        vec_NW, vec_NE, vec_SE, vec_SW = vectors
        unit_x = np.array([1, 0])

        vec_left = (vec_NW-vec_SW)/np.linalg.norm(vec_NW-vec_SW)
        vec_right = (vec_NE-vec_SE)/np.linalg.norm(vec_NE-vec_SE)
        vec_top = (vec_NE-vec_NW)/np.linalg.norm(vec_NE-vec_NW)
        vec_bot = (vec_SE-vec_SW)/np.linalg.norm(vec_SE-vec_SW)

        angle_left = np.arccos(np.dot(vec_left, unit_x))
        angle_right = np.arccos(np.dot(vec_right, unit_x))
        angle_top = np.arccos(np.dot(vec_top, unit_x))
        angle_bot = np.arccos(np.dot(vec_bot, unit_x))

        assert np.isclose(np.cos(angle_left), np.cos(WPA)), f'Angle of left segement is {angle_left*180/np.pi:.2f} but PA is {WPA*180/np.pi}!'
        assert np.isclose(np.cos(angle_right), np.cos(WPA)), f'Angle of right segement is {angle_right*180/np.pi:.2f} but PA is {WPA*180/np.pi}!'
        assert np.isclose(np.cos(angle_top), np.cos(LPA)), f'Angle of top segement is {angle_top*180/np.pi:.2f} but PA is {LPA*180/np.pi}!'
        assert np.isclose(np.cos(angle_bot), np.cos(LPA)), f'Angle of bot segement is {angle_bot*180/np.pi:.2f} but PA is {LPA*180/np.pi}!'


    def _get_slit_mask(self, slit_patch, x_grid, y_grid):
        '''Checks if each pair of (x, y) is in the slit patch and returns
        a mask for masking the x, y grids
        The points where the mask is True will be masked out
        '''
        mask = np.ones(x_grid.shape, dtype=bool)
        for i, (these_x, these_y) in enumerate(zip(x_grid, y_grid)):
            mask[i] = ~slit_patch.contains_points(np.column_stack((these_x, these_y))) # ,
                                                  # radius=0) # This radius added by JD

        return mask

    def _fourier_shift_2d(self, image, Dy, Dx=0, pad=True, pad_width=None):
        """
            image: 2D numpy array (Ny, Nx)
            dy: shift in pixels along axis 0 (positive -> image *up* visually)
            dx: shift in pixels along axis 1 (optional)
            pad: whether to pad to reduce wrap-around
            pad_width: int or tuple ((top,bottom),(left,right)) or None
        """
        Ny, Nx = image.shape
        dx, dy = Dx, Dy
    
        # optional padding to reduce wrap-around effects
        if pad:
            if pad_width is None:
                # pad by at least abs(round(shift))+10
                pad_y = abs(int(np.ceil(dy))) + 10
                pad_x = abs(int(np.ceil(dx))) + 10
                padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0.0)
            else:
                padded = np.pad(image, pad_width, mode='constant', constant_values=0.0)
        else:
            padded = image
    
        PNy, PNx = padded.shape
    
        # Fourier transform
        F = np.fft.fftn(padded)
    
        # frequency grids (cycles per pixel)
        ky = np.fft.fftfreq(PNy)[:, None]  # shape (PNy,1)
        kx = np.fft.fftfreq(PNx)[None, :]  # shape (1,PNx)
    
        # phase ramp: exp(-2pi i (kx*dx + ky*dy))
        phase = np.exp(-2j * np.pi * (ky * dy + kx * dx))
    
        F_shifted = F * phase
        shifted = np.fft.ifftn(F_shifted).real
    
        # crop back to original size
        if pad:
            pad_y = (PNy - Ny) // 2
            pad_x = (PNx - Nx) // 2
            cropped = shifted[pad_y:pad_y+Ny, pad_x:pad_x+Nx]
        else:
            cropped = shifted
    
        return cropped
