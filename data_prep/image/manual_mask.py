"""Notebook editor for neighboring-source image masks."""

from pathlib import Path

import numpy as np

from .masking import (mask_neighbor_star, read_mask_plan, save_mask_plan,
                      target_pixel)
from .read_and_cutout import cutoutimg, read_subaru_img_wcs
from spec.read_and_cutout import readinfodat


def review_image_mask(slit_num, image_path, info_path, width=40, height=40,
                      plan_path=None):
    """Open an interactive review for one slit; save with the button."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from IPython.display import display

    if plan_path is None:
        plan_path = (Path(__file__).resolve().parents[1] / 'image_mask_plans' /
                     f'slit_{slit_num:03d}_image_masks.json')
    info = readinfodat(info_path)
    ra, dec = info['RA'], info['DEC']
    imaging = read_subaru_img_wcs(str(image_path))
    image, _, _, wcs = cutoutimg(imaging['science_data'],
                                 imaging['science_wcs'], ra, dec,
                                 width, height, outputWCS=True)
    target = target_pixel(wcs, ra, dec)
    ellipses = [dict(item) for item in read_mask_plan(plan_path, slit_num,
                                                      allow_missing=True)]

    with plt.ioff():
        fig, axes = plt.subplots(1, 2, figsize=(4, 2), layout='constrained')
    selector = widgets.Dropdown(description='Ellipse', options=[], layout=widgets.Layout(width='220px'))
    fields = {
        'offset_x': widgets.FloatText(description='Offset x', step=0.5),
        'offset_y': widgets.FloatText(description='Offset y', step=0.5),
        'radius_x': widgets.FloatText(description='Radius x', step=0.5),
        'radius_y': widgets.FloatText(description='Radius y', step=0.5),
        'angle_deg': widgets.FloatText(description='Angle (deg)', step=5),
    }
    for field in fields.values():
        field.layout.width = '220px'
        field.style.description_width = '85px'
    add = widgets.Button(description='Add ellipse')
    remove = widgets.Button(description='Remove selected')
    save = widgets.Button(description='Save plan', button_style='success')
    status = widgets.HTML()
    updating = False
    preview_colorbar = None

    def sync_selector(index=None):
        nonlocal updating
        updating = True
        selector.options = [(f'{i + 1}', i) for i in range(len(ellipses))]
        selector.value = index if ellipses else None
        for key, field in fields.items():
            field.disabled = not ellipses
            field.value = ellipses[selector.value][key] if ellipses else 0
        remove.disabled = not ellipses
        updating = False
        draw()

    def draw():
        nonlocal preview_colorbar
        for ax in axes:
            ax.clear()
        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, [2, 99.5]) if finite.size else (0, 1)
        masked = mask_neighbor_star(image, np.ones(image.shape, bool), ellipses,
                                    target)
        visible = image[masked & np.isfinite(image)]
        preview_low, preview_high = (np.percentile(visible, [2, 99.5])
                                     if visible.size else (low, high))
        for ax, pixels, title, limits in zip(
                axes, (image, np.where(masked, image, np.nan)),
                ('Select ellipses here', 'Mask preview'),
                ((low, high), (preview_low, preview_high))):
            plotted = ax.imshow(pixels, origin='lower',
                                vmin=limits[0], vmax=limits[1])
            if ax is axes[1]:
                if preview_colorbar is None:
                    preview_colorbar = fig.colorbar(plotted, ax=ax)
                else:
                    preview_colorbar.update_normal(plotted)
            ax.plot(*target, marker='+', color='cyan', markersize=10)
            ax.set_title(title)
            ax.set_xlim(-0.5, image.shape[1] - 0.5)
            ax.set_ylim(-0.5, image.shape[0] - 0.5)
        for i, ellipse in enumerate(ellipses):
            axes[0].add_patch(Ellipse(
                (target[0] + ellipse['offset_x'], target[1] + ellipse['offset_y']),
                2 * ellipse['radius_x'], 2 * ellipse['radius_y'],
                angle=ellipse['angle_deg'], fill=False,
                edgecolor='red' if i == selector.value else 'yellow', linewidth=2))
        fig.canvas.draw_idle()

    def on_select(change):
        if not updating and change['name'] == 'value' and change['new'] is not None:
            sync_selector(change['new'])

    def on_field(change):
        nonlocal updating
        if updating or selector.value is None:
            return
        key = next(key for key, field in fields.items() if field is change['owner'])
        if key.startswith('radius_') and change['new'] <= 0:
            status.value = '<span style="color:red">Radius must be greater than 0</span>'
            updating = True
            fields[key].value = ellipses[selector.value][key]
            updating = False
            return
        ellipses[selector.value][key] = change['new']
        status.value = 'Unsaved changes'
        draw()

    def on_click(event):
        if event.inaxes is not axes[0] or event.xdata is None or selector.value is None:
            return
        fields['offset_x'].value = round(event.xdata - target[0], 2)
        fields['offset_y'].value = round(event.ydata - target[1], 2)

    def on_add(_):
        ellipses.append({'offset_x': 0., 'offset_y': 0., 'radius_x': 3.,
                         'radius_y': 3., 'angle_deg': 0.})
        sync_selector(len(ellipses) - 1)
        status.value = '⏳ Unsaved changes'

    def on_remove(_):
        index = selector.value
        if index is not None:
            ellipses.pop(index)
            sync_selector(min(index, len(ellipses) - 1) if ellipses else None)
            status.value = '⏳ Unsaved changes'

    def on_save(_):
        if any(item['radius_x'] <= 0 or item['radius_y'] <= 0 for item in ellipses):
            status.value = '<span style="color:red">Radius must be greater than 0</span>'
            return
        save_mask_plan(plan_path, slit_num, ellipses)
        status.value = f'✅ Saved: {Path(plan_path).resolve()}'

    selector.observe(on_select, names='value')
    for field in fields.values():
        field.observe(on_field, names='value')
    fig.canvas.mpl_connect('button_press_event', on_click)
    add.on_click(on_add)
    remove.on_click(on_remove)
    save.on_click(on_save)
    sync_selector(0 if ellipses else None)
    display(widgets.VBox([
        widgets.HTML(
            f'<h1> Crop for Slit {slit_num}</h1>'+
            '<b> [Action needed] </b> Please add an ellipse, click the left '+
            'image to set its center, then adjust its radii and angle. '
            'The right image previews the mask; the cyan cross marks the '
            'target. If no mask is needed, you will still have to click <b>"Save plan"</b>'
        ),
        fig.canvas,
        widgets.VBox([
            widgets.HBox([selector, fields['angle_deg']]),
            widgets.HBox([fields['offset_x'], fields['offset_y']]),
            widgets.HBox([fields['radius_x'], fields['radius_y']]),
        ]),
        widgets.HBox([add, remove, save]), status,
    ]))
    return ellipses
