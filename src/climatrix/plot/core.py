import importlib
import importlib.resources
import logging
import os
import threading
import time
import webbrowser

import numpy as np
from flask import (
    Flask,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
    url_for,
)

from climatrix.dataset.base import BaseClimatrixDataset
from climatrix.dataset.domain import AxisType

LOD_THRESHOLD: int = int(os.environ.get("LOD_THRESHOLD", 10_000))


class Plot:
    def __init__(self, dataset: BaseClimatrixDataset):
        self.dataset = dataset.to_signed_longitude()
        self.global_min = self.dataset.da.min().item()
        self.global_max = self.dataset.da.max().item()
        self.app = Flask(__name__)
        self.app.logger.setLevel(logging.WARNING)
        self.setup_routes()
        self.app.static_folder = importlib.resources.files(
            "climatrix.resources"
        ).joinpath("static")

    def setup_routes(self):
        @self.app.route("/")
        def index():
            logo_url = url_for("serve_mylibrary_asset", filename="logo.svg")
            return render_template_string(
                self.get_html_template(), logo_url=logo_url
            )

        @self.app.route("/climatrix_assets/<path:filename>")
        def serve_mylibrary_asset(filename):
            return send_from_directory(self.app.static_folder, filename)

        @self.app.route("/api/data")
        def get_data():
            time_idx = request.args.get("time_idx", 0, type=int)
            vertical_idx = request.args.get("vertical_idx", 0, type=int)
            zoom_level = request.args.get("zoom", 1, type=float)

            lat_min = request.args.get("lat_min", type=float)
            lat_max = request.args.get("lat_max", type=float)
            lon_min = request.args.get("lon_min", type=float)
            lon_max = request.args.get("lon_max", type=float)

            data_load = self.prepare_data(
                time_idx,
                vertical_idx,
                zoom_level,
                lat_min,
                lat_max,
                lon_min,
                lon_max,
            )
            return jsonify(data_load)

        @self.app.route("/api/metadata")
        def get_metadata():
            return jsonify(self.get_metadata())

    def get_metadata(self):
        metadata = {
            "has_time": self.dataset.domain.has_axis(AxisType.TIME),
            "has_vertical": self.dataset.domain.has_axis(AxisType.VERTICAL),
            "is_sparse": self.dataset.domain.is_sparse,
        }

        if metadata["has_time"]:
            metadata["time_values"] = [
                str(t) for t in self.dataset.domain.time.values
            ]
            metadata["time_count"] = len(self.dataset.domain.time.values)

        if metadata["has_vertical"]:
            metadata["vertical_values"] = (
                self.dataset.domain.vertical.values.tolist()
            )
            metadata["vertical_count"] = len(
                self.dataset.domain.vertical.values
            )
            metadata["vertical_name"] = self.dataset.domain.vertical.name

        return metadata

    def prepare_data(
        self,
        time_idx=0,
        vertical_idx=0,
        zoom_level=1,
        lat_min=None,
        lat_max=None,
        lon_min=None,
        lon_max=None,
    ):

        if self.dataset.domain.is_sparse:
            return self.prepare_sparse_data(time_idx, vertical_idx)
        else:
            return self.prepare_dense_data(
                time_idx,
                vertical_idx,
                zoom_level,
                lat_min,
                lat_max,
                lon_min,
                lon_max,
            )

    def prepare_sparse_data(self, time_idx, vertical_idx):
        lats = self.dataset.domain.latitude.values
        lons = self.dataset.domain.longitude.values

        data_slice = self.dataset.da
        if self.dataset.domain.has_axis(AxisType.TIME):
            data_slice = data_slice.isel(
                {self.dataset.domain.time.name: time_idx}
            )
        if self.dataset.domain.has_axis(AxisType.VERTICAL):
            if len(data_slice.shape) > 1:
                data_slice = data_slice.isel(
                    {self.dataset.domain.vertical.name: vertical_idx}
                )

        valid_idx = np.isfinite(data_slice.values)
        lats = lats[valid_idx]
        lons = lons[valid_idx]
        data_slice = data_slice[valid_idx]

        return {
            "type": "scatter",
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "values": data_slice.values.flatten().tolist(),
            "min_val": self.global_min,
            "max_val": self.global_max,
        }

    def prepare_dense_data(
        self,
        time_idx,
        vertical_idx,
        zoom_level=1,
        lat_min=None,
        lat_max=None,
        lon_min=None,
        lon_max=None,
    ):
        lats = self.dataset.domain.latitude.values
        lons = self.dataset.domain.longitude.values

        data_slice = self.dataset.da
        if self.dataset.domain.has_axis(AxisType.TIME):
            data_slice = data_slice.isel(
                {self.dataset.domain.time.name: time_idx}
            )
        if self.dataset.domain.has_axis(AxisType.VERTICAL):
            if len(data_slice.shape) > 2:
                data_slice = data_slice.isel(
                    {self.dataset.domain.vertical.name: vertical_idx}
                )

        values = data_slice.values

        lats_optimized, lons_optimized, values_optimized = (
            self._apply_lod_optimization(
                lats,
                lons,
                values,
                zoom_level,
                lat_min,
                lat_max,
                lon_min,
                lon_max,
            )
        )
        values_cleaned = np.where(
            np.isnan(values_optimized), None, values_optimized
        ).tolist()

        return {
            "type": "mesh",
            "lats": lats_optimized.tolist(),
            "lons": lons_optimized.tolist(),
            "values": values_cleaned,
            "min_val": self.global_min,
            "max_val": self.global_max,
        }

    def _apply_lod_optimization(
        self,
        lats,
        lons,
        values,
        zoom_level,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    ):

        if len(values.shape) == 2:
            total_points = values.shape[0] * values.shape[1]
        else:
            total_points = len(values)

        if total_points <= LOD_THRESHOLD:
            return lats, lons, values

        if zoom_level >= 8:
            step = 1
        elif zoom_level >= 5:
            step = 2
        elif zoom_level >= 3:
            step = 4
        elif zoom_level >= 1:
            step = 8
        else:
            step = 16

        if all(
            param is not None for param in [lat_min, lat_max, lon_min, lon_max]
        ):
            lat_mask = (lats >= lat_min) & (lats <= lat_max)
            lon_mask = (lons >= lon_min) & (lons <= lon_max)

            if len(values.shape) == 2:
                lat_indices = np.where(lat_mask)[0]
                lon_indices = np.where(lon_mask)[0]

                if len(lat_indices) == 0 or len(lon_indices) == 0:
                    return (
                        lats[:: step * 4],
                        lons[:: step * 4],
                        values[:: step * 4, :: step * 4],
                    )

                lat_start = lat_indices[0]
                lat_end = lat_indices[-1] + 1
                lon_start = lon_indices[0]
                lon_end = lon_indices[-1] + 1

                lat_roi_indices = slice(lat_start, lat_end, step)
                lon_roi_indices = slice(lon_start, lon_end, step)

                return (
                    lats[lat_roi_indices],
                    lons[lon_roi_indices],
                    values[lat_roi_indices, lon_roi_indices],
                )
            else:
                combined_mask = lat_mask & lon_mask
                masked_indices = np.where(combined_mask)[0]
                if len(masked_indices) == 0:
                    return (
                        lats[:: step * 4],
                        lons[:: step * 4],
                        values[:: step * 4],
                    )

                stepped_indices = masked_indices[::step]
                return (
                    lats[stepped_indices],
                    lons[stepped_indices],
                    values[stepped_indices],
                )

        if len(values.shape) == 2:
            return lats[::step], lons[::step], values[::step, ::step]
        else:
            return lats[::step], lons[::step], values[::step]

    def get_html_template(self):
        return (
            importlib.resources.files("climatrix.resources")
            .joinpath("static", "plot_template.html")
            .read_text(encoding="utf8")
        )

    def show(self, port=5000, debug=False):

        def run_server():
            self.app.run(
                host="localhost", port=port, debug=debug, use_reloader=False
            )

        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()

        time.sleep(1)
        webbrowser.open(f"http://localhost:{port}")

        print(
            f"Climate data visualization server started at http://localhost:{port}"
        )
        print("Press Ctrl+C to stop the server")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")
