## To install, clone and change your working directory to this repo level. Run
```
pip install -e .
```

## To run, find this code
Go to `scripts_beta` > `binospec_main_fitting_nautilus.py`. This is where I start the fitting. 

If running other Python scripts, it may require extra metadata (e.g., raw imaging or spectrum data), and these files are over 1 GB, which exceeds my GitHub storage limit.

## Copyright hints
The folder `klm` was originally from Pranjal's `kl_measurement` [GitHub repo](https://github.com/emhuff/kl_measurement/tree/manga/klm) (ask Eric for access), and some code was modified for my preferences. Other codes are mine. Happy to take questions!

continuum.py
├── [独立函数 (Functions)]
│    ├── another_load_mock()
│    ├── build_2d_continuum()
│    ├── estimate_sigma_from_data()
│    ├── fit_continuum()
│    ├── gauss()

cut_emis_center.py
├── [独立函数 (Functions)]
│    ├── curve_fit()
├── [类与方法 (Classes & Methods)]
│    ├── Class: EmissionLineFitter
│    │    ├── _double_gaussian()
│    │    ├── _gaussian()
│    │    ├── fit_1d()
│    │    ├── fit_2d_double()
│    │    ├── fit_2d_single()
│    ├── Class: EmissionProcessor
│    │    ├── process()
│    ├── Class: SpectrumCropper
│    │    ├── crop()

get_r_hl_image.py
├── [独立函数 (Functions)]
│    └── (无独立函数)
├── [类与方法 (Classes & Methods)]
│    └── (无定义类)

image_utils.py
├── [独立函数 (Functions)]
│    ├── Meta_image()
│    ├── cutoffimg()
│    ├── half_light_radius_exp()
│    ├── mask_neighbor_gal()

line_processing.py
├── [独立函数 (Functions)]
│    ├── check_x0y0_in_cropped()
│    ├── measure_v0()
│    ├── process_single_line()
│    ├── redshift_update()
│    ├── resolve_bad_lines()
其中已imported: 
from   cut_emis_center import EmissionProcessor
from   manual_fix      import manual_correct
from   mask_utils      import mask_out_pixels, remove_bad_pixels
from   read_save_utils import real_data_pack
from   spec_utils      import cutoffspec
from   continuum       import build_2d_continuum

main.py
├── [独立函数 (Functions)]
│    ├── main_cut_off()
其中已imported: 
from   image_utils     import cutoffimg, Meta_image, half_light_radius_exp
from   line_processing import process_single_line
from   meta_utils      import meta_spec_ABC
from   plot            import make_exam_plots
from   read_save_utils import save_dic_and_pkl, read_spec2d, readinfodat
from   spec_utils      import stack_spec2d

manual_fix.py
├── [独立函数 (Functions)]
│    ├── manual_correct()

mask_utils.py
├── [独立函数 (Functions)]
│    ├── mask_out_pixels()
│    ├── remove_bad_pixels()

meta_utils.py
├── [独立函数 (Functions)]
│    ├── meta_spec_ABC()

plot.py
├── [独立函数 (Functions)]
│    ├── make_exam_plots()
│    ├── rot_rectangle()
│    ├── solve_snr()

post_processing.py
├── [独立函数 (Functions)]
│    ├── check_changed_dict()
│    ├── reload_raw_imaging()
├── [类与方法 (Classes & Methods)]
│    ├── Class: ImagingProcessing
│    │    ├── Gauss_2d()
│    │    ├── mask_neighbor_star()
│    ├── Class: RedoImaging
│    │    ├── redo_img_data()
│    │    ├── run_redo_img_data()
│    ├── Class: SpecProcessing
│    │    ├── add_M_stellar_and_r_hl_disk()
│    │    ├── clear_cont()
其中已imported: 
from read_save_utils import cont, readinfodat
from image_utils     import cutoffimg, Meta_image
from plot import make_exam_plots

read_save_utils.py
├── [独立函数 (Functions)]
│    ├── cont()
│    ├── read_hsc_img_wcs()
│    ├── read_spec2d()
│    ├── readinfodat()
│    ├── real_data_pack()
│    ├── save_dic_and_pkl()

sky_line_bands.py
├── [独立函数 (Functions)]
│    ├── merge_intervals()
│    ├── plot_sky()

spec_utils.py
├── [独立函数 (Functions)]
│    ├── cutoffspec()
│    ├── stack_spec2d()
