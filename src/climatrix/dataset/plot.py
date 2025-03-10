import threading
import time
import webbrowser

import hvplot.xarray  # Ensure hvplot and panel are installed: pip install hvplot panel
import pandas as pd
import panel as pn
import xarray as xr

from climatrix.dataset.base import BaseDataset

pn.extension("plotly")


class InteractivePlotter:
    def __init__(self, dataset: BaseDataset):
        if not isinstance(dataset, BaseDataset):
            raise TypeError(
                "Dataset can be created only based on "
                "BaseDataset object, "
                f"but provided {type(dataset).__name__}"
            )
        self.dataset = dataset

        self.var_selector = pn.widgets.Select(
            name="Variable", options=list(self.dataset.fields_names)
        )
        time_labels = [str(t) for t in self.dataset.time.values]
        self.time_slider = pn.widgets.DiscreteSlider(
            name="Time",
            options=dict(zip(time_labels, self.dataset.time.values)),
        )
        self.plot_pane = pn.pane.HoloViews()
        self.spinner = pn.indicators.LoadingSpinner(size=20, value=False)

        self.layout = pn.Column(
            pn.Row(
                self.var_selector,
                self.time_slider,
                sizing_mode="stretch_width",
            ),
            pn.Row(self.spinner, self.plot_pane),
            sizing_mode="stretch_both",
        )

        self.var_selector.param.watch(self._update_plot, "value")
        self.time_slider.param.watch(self._update_plot, "value")

        self._update_plot()

    def _update_plot(self, event=None):
        self.spinner.value = True
        self._delayed_plot_update()

    def _delayed_plot_update(self):
        var = self.var_selector.value
        t = self.time_slider.value
        da = self.dataset.dset[var].sel(**{self.dataset._def.time_name: t})
        plot = da.hvplot().opts(
            title=f"Field: {var} at {str(t)}", height=600, width=900
        )
        self.plot_pane.object = plot
        self.spinner.value = False

    def show(self):
        return self.layout.show()
