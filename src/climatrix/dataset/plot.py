import cartopy.crs as ccrs
import geoviews.feature as gf
import hvplot.xarray
import panel as pn

from climatrix.dataset.base import BaseDataset

pn.extension("plotly")


class InteractiveDensePlotter:
    def __init__(self, dataset: BaseDataset):
        if not isinstance(dataset, BaseDataset):
            raise TypeError(
                "Dataset can be created only based on "
                "BaseDataset object, "
                f"but provided {type(dataset).__name__}"
            )
        self.dataset = dataset

        time_labels = [str(t) for t in self.dataset.time.values]
        self.time_slider = pn.widgets.DiscreteSlider(
            name="Time",
            options=dict(zip(time_labels, self.dataset.time.values)),
        )
        self.plot_pane = pn.pane.HoloViews()
        self.spinner = pn.indicators.LoadingSpinner(size=20, value=False)

        self.layout = pn.Column(
            pn.Row(
                self.time_slider,
                sizing_mode="stretch_width",
            ),
            pn.Row(self.spinner, self.plot_pane),
            # sizing_mode="stretch_both",
        )

        self.time_slider.param.watch(self._update_plot, "value")

        self._update_plot()

    def _update_plot(self, event=None):
        self.spinner.value = True
        self._delayed_plot_update()

    def _delayed_plot_update(self):
        t = self.time_slider.value
        da = self.dataset.da.sel(**{self.dataset._def.time_name: t})
        proj = ccrs.PlateCarree()
        plot = da.hvplot().opts(
            title=f"Field: {self.dataset._def.name} at {str(t)}",
            height=700,
            width=700,
        )
        plot = (
            plot
            * gf.borders().opts(projection=proj)
            * gf.coastline().opts(projection=proj)
        )
        self.plot_pane.object = plot
        self.spinner.value = False

    def show(self):
        return self.layout.show()


class InteractiveScatterPlotter:

    def __init__(self, dataset: BaseDataset):
        if not isinstance(dataset, BaseDataset):
            raise TypeError(
                "Dataset can be created only based on "
                "BaseDataset object, "
                f"but provided {type(dataset).__name__}"
            )
        self.dataset = dataset

        # self.var_selector = pn.widgets.Select(
        #     name="Variable", options=list(self.dataset.fields_names)
        # )
        time_labels = [str(t) for t in self.dataset.time.values]
        self.time_slider = pn.widgets.DiscreteSlider(
            name="Time",
            options=dict(zip(time_labels, self.dataset.time.values)),
        )
        self.plot_pane = pn.pane.HoloViews()
        self.spinner = pn.indicators.LoadingSpinner(size=20, value=False)

        self.layout = pn.Column(
            pn.Row(
                # self.var_selector,
                self.time_slider,
                sizing_mode="stretch_width",
            ),
            pn.Row(self.spinner, self.plot_pane),
            sizing_mode="stretch_both",
        )

        # self.var_selector.param.watch(self._update_plot, "value")
        self.time_slider.param.watch(self._update_plot, "value")

        self._update_plot()

    def _update_plot(self, event=None):
        self.spinner.value = True
        self._delayed_plot_update()

    def _delayed_plot_update(self):
        # var = self.var_selector.value
        t = self.time_slider.value
        da = self.dataset.da.sel(**{self.dataset._def.time_name: t})
        da = da.assign_coords(
            {
                self.dataset._def.longitude_name: (
                    ((da[self.dataset._def.longitude_name] + 180) % 360) - 180
                )
            }
        )
        proj = ccrs.PlateCarree()
        plot = da.to_dataset(name="values").hvplot.scatter(
            x=self.dataset._def.longitude_name,
            y=self.dataset._def.latitude_name,
            c="values",
            geo=True,
            projection=proj,
            title=f"Sparse dataset for {self.dataset._def.name}",
            height=600,
            width=900,
            marker="x",
        )
        plot = (
            plot
            * gf.borders().opts(projection=proj)
            * gf.coastline().opts(projection=proj)
        )
        self.plot_pane.object = plot
        self.spinner.value = False

    def show(self):
        return self.layout.show()
