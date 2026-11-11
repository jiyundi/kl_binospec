"""Notebook crop review on Windows or a remote HPC Jupyter kernel.

Only NumPy, Astropy, SciPy, Matplotlib and ipywidgets/ipympl are needed.
"""

from types import SimpleNamespace

import numpy as np

from .crop_plan import (SETS, make_plan, prepare_rough, redshift_from_crop,
                        save_plan)
from .line_center_crop import EmissionProcessor
from .read_and_cutout import read_spec2d, stack_spec2d


def load_sources(paths):
    """Replicate main.py's A stack and bottom-to-top ordering.

    paths has keys A1, A2, A3, B and C, each pointing to one spec2d FITS.
    """
    a1, a2, a3 = (read_spec2d(paths[key]) for key in ('A1', 'A2', 'A3'))
    sources = {
        'A': {
            'wave': np.tile(a1['wave'][0],
                            (max(len(a1['flux']), len(a2['flux']), len(a3['flux'])), 1)),
            'flux': stack_spec2d(a1['flux'], a2['flux'], a3['flux']),
        },
        'C': read_spec2d(paths['C']),
        'B': read_spec2d(paths['B']),
    }
    for source in sources.values():
        for key in ('flux', 'wave'):
            source[key] = np.flip(source[key], axis=0)
        source['mask'] = np.ones(source['flux'].shape, dtype=bool)
    return SimpleNamespace(specA=sources['A'], specC=sources['C'],
                           specB=sources['B'])


class NotebookCropReview:
    """One line at a time. Call show() in an ipympl-enabled notebook."""

    def __init__(self, sources, name, line_waves, redshift,
                 width, height, plan_path):
        self.sources = [sources.specA, sources.specC, sources.specB]
        self.name = name
        self.line_waves = list(line_waves)
        self.redshift = float(redshift)
        self.width = int(width)
        self.height = int(height)
        self.plan_path = plan_path
        self.cuts, self.columns = prepare_rough(self.sources, self.line_waves,
                                                 self.redshift)
        self.processor = EmissionProcessor(*self.cuts, self.redshift)
        self.auto_result = None
        try:
            self.auto_result = self.processor.process(
                self.name, self.line_waves, self.width, self.height)
        except (IndexError, RuntimeError, ValueError) as exc:
            self.auto_error = str(exc)

    def _save(self, cropped, how_cut, clicks=None):
        corrected = redshift_from_crop(cropped, how_cut, self.line_waves)
        plan = make_plan(self.name, self.line_waves, self.redshift,
                         self.width, self.height, self.sources, self.columns,
                         how_cut, corrected, clicks=clicks)
        save_plan(self.plan_path, plan)
        return plan

    def show(self):
        """Render click selection, numeric controls and save buttons."""
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from IPython.display import display

        fig, axes = plt.subplots(3, 1, figsize=(9, 8), layout='constrained')
        positions = []
        rectangles = []
        for i, (ax, cut) in enumerate(zip(axes, self.cuts)):
            flux = np.where(cut['mask'], cut['flux'], np.nan)
            sigma = np.nanstd(flux)
            ax.imshow(flux, origin='lower', aspect='auto',
                      vmin=-sigma, vmax=5 * sigma)
            if self.auto_result is None:
                x = (flux.shape[1] - 1) / 2
                y = (flux.shape[0] - 1) / 2
            else:
                info = self.auto_result[1][f'Set{i}']
                x = np.mean([info[f'line{j+1}']['x0'] + info['LF']
                             for j in range(len(self.line_waves))])
                y = info['line1']['y0'] + info['UP']
            x_input = widgets.FloatText(value=float(x), description=f'{SETS[i]} x')
            y_input = widgets.FloatText(value=float(y), description=f'{SETS[i]} y')
            positions.append((x_input, y_input))
            box = Rectangle((x - self.width / 2, y - self.height / 2),
                            self.width, self.height,
                            fill=False, edgecolor='red', linewidth=1.5)
            ax.add_patch(box)
            rectangles.append(box)
            ax.set_title(f'{self.name}: Set {SETS[i]}')
            ax.set_xlabel('rough-cut wavelength pixel x')
            ax.set_ylabel('flipped spatial pixel y')

            def update(change, index=i):
                xx, yy = positions[index]
                rectangles[index].set_xy((xx.value - self.width / 2,
                                          yy.value - self.height / 2))
                fig.canvas.draw_idle()
            x_input.observe(update, names='value')
            y_input.observe(update, names='value')

        def click(event):
            for i, ax in enumerate(axes):
                if event.inaxes is ax and event.button == 1:
                    positions[i][0].value = float(event.xdata)
                    positions[i][1].value = float(event.ydata)
                    break
        fig.canvas.mpl_connect('button_press_event', click)

        output = widgets.Output()
        accept_auto = widgets.Button(description='Accept automatic fit',
                                     disabled=self.auto_result is None)
        refit = widgets.Button(description='Refit from selected centers')

        def accept(_):
            with output:
                output.clear_output()
                try:
                    self._save(*self.auto_result)
                    print(f'Saved {self.name} to {self.plan_path}')
                except (ValueError, OSError) as exc:
                    print(f'Could not save: {exc}')
        accept_auto.on_click(accept)

        def fit_and_save(_):
            with output:
                output.clear_output()
                clicks = [[x.value, y.value] for x, y in positions]
                try:
                    cropped, how_cut = self.processor.process(
                        self.name, self.line_waves, self.width, self.height,
                        idx_xs=[xy[0] for xy in clicks],
                        idx_ys=[xy[1] for xy in clicks])
                    self._save(cropped, how_cut, clicks=clicks)
                    print(f'Saved fitted crop for {self.name} to {self.plan_path}')
                except (IndexError, RuntimeError, ValueError, OSError) as exc:
                    print(f'Fit/save failed: {exc}')
        refit.on_click(fit_and_save)

        display(fig.canvas)
        display(widgets.VBox([widgets.HBox(pair) for pair in positions]))
        display(widgets.HBox([accept_auto, refit]), output)
        if self.auto_result is None:
            with output:
                print('Automatic fit unavailable:', self.auto_error)
        return self
