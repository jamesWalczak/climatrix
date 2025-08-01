import importlib
import importlib.resources
import logging
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

from climatrix.dataset.domain import AxisType


class Plot:
    def __init__(self, dataset):
        self.dataset = dataset
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
            data_load = self.prepare_data(time_idx, vertical_idx)
            return jsonify(data_load)

        @self.app.route("/api/metadata")
        def get_metadata():
            return jsonify(self.get_metadata())

    def get_metadata(self):
        """Get dataset metadata for UI controls"""
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

    def prepare_data(self, time_idx=0, vertical_idx=0):
        """Prepare data for visualization"""
        if self.dataset.domain.is_sparse:
            return self.prepare_sparse_data(time_idx, vertical_idx)
        else:
            return self.prepare_dense_data(time_idx, vertical_idx)

    def prepare_sparse_data(self, time_idx, vertical_idx):
        """Prepare sparse data as scatter points"""
        lats = self.dataset.domain.latitude.values
        lons = self.dataset.domain.longitude.values

        # Get data values for specific time/vertical slice if applicable
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

        return {
            "type": "scatter",
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "values": data_slice.values.flatten().tolist(),
            "min_val": float(np.min(data_slice)),
            "max_val": float(np.max(data_slice)),
        }

    def prepare_dense_data(self, time_idx, vertical_idx):
        """Prepare dense data as mesh/contour"""
        lats = self.dataset.domain.latitude.values
        lons = self.dataset.domain.longitude.values

        # Get data values for specific time/vertical slice
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
        return {
            "type": "mesh",
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "values": data_slice.values.tolist(),
            "min_val": float(np.min(data_slice)),
            "max_val": float(np.max(data_slice)),
        }

    def get_html_template(self):
        return (
            importlib.resources.files("climatrix.resources")
            .joinpath("static", "plot_template.html")
            .read_text()
        )

    def show(self, port=5000, debug=False):
        """Start the web server and open the visualization"""

        def run_server():
            self.app.run(
                host="localhost", port=port, debug=debug, use_reloader=False
            )

        # Start server in a separate thread
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()

        # Wait a moment for server to start, then open browser
        time.sleep(1)
        webbrowser.open(f"http://localhost:{port}")

        print(
            f"Climate data visualization server started at http://localhost:{port}"
        )
        print("Press Ctrl+C to stop the server")

        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")
