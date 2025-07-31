"""Core plotting functionality for climatrix datasets."""

from __future__ import annotations

import logging
import webbrowser
from typing import TYPE_CHECKING, Any, Optional

from climatrix.exceptions import MissingDependencyError

if TYPE_CHECKING:
    from climatrix.dataset.base import BaseClimatrixDataset

log = logging.getLogger(__name__)


def _check_plotting_dependencies() -> None:
    """Check if required plotting dependencies are available."""
    try:
        import plotly  # noqa: F401
        import dash  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            "Interactive plotting requires plotly and dash. "
            "Install with: pip install plotly dash"
        ) from e


class Plot:
    """
    Interactive plotting utility for climatrix datasets.
    
    This class provides an extensive interactive plotting interface using
    Plotly and Dash to create web-based visualizations with material design
    styling and advanced interactive features.
    
    Parameters
    ----------
    dataset : BaseClimatrixDataset
        The climatrix dataset to visualize
    port : int, optional
        Port for the Dash server (default: 8050)
    host : str, optional
        Host for the Dash server (default: '127.0.0.1')
    debug : bool, optional
        Enable debug mode (default: False)
    auto_open : bool, optional
        Automatically open browser (default: True)
    **kwargs
        Additional configuration options
        
    Attributes
    ----------
    dataset : BaseClimatrixDataset
        The underlying dataset
    app : dash.Dash
        The Dash application instance
    is_sparse : bool
        Whether the dataset is sparse (scatter points) or dense
    has_time : bool
        Whether the dataset has a time dimension
    has_vertical : bool
        Whether the dataset has a vertical dimension
        
    Examples
    --------
    >>> import climatrix as cm
    >>> import xarray as xr
    >>> ds = xr.open_dataset("example.nc")
    >>> cm_ds = ds.cm
    >>> plot = cm.plot.Plot(cm_ds)
    >>> plot.show()  # Opens interactive plot in browser
    """
    
    def __init__(
        self,
        dataset: "BaseClimatrixDataset",
        port: int = 8050,
        host: str = "127.0.0.1",
        debug: bool = False,
        auto_open: bool = True,
        **kwargs: Any,
    ):
        # Check dependencies first
        _check_plotting_dependencies()
        
        # Import here after dependency check
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output, State
        import plotly.graph_objects as go
        import plotly.express as px
        
        self.dataset = dataset
        self.port = port
        self.host = host
        self.debug = debug
        self.auto_open = auto_open
        self.config = kwargs
        
        # Analyze dataset properties
        self._analyze_dataset()
        
        # Initialize Dash app with material design
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
    def _analyze_dataset(self) -> None:
        """Analyze dataset properties to determine visualization approach."""
        from climatrix.dataset.axis import AxisType
        
        # Check if dataset is sparse or dense
        self.is_sparse = self._is_sparse_dataset()
        
        # Check for time dimension
        self.has_time = AxisType.TIME in self.dataset.dims
        
        # Check for vertical dimension
        self.has_vertical = AxisType.VERTICAL in self.dataset.dims
        
        # Get coordinate information
        self._get_coordinate_info()
        
    def _is_sparse_dataset(self) -> bool:
        """Determine if dataset should be treated as sparse."""
        # Simple heuristic: if data is significantly smaller than
        # a full grid, treat as sparse
        from climatrix.dataset.axis import AxisType
        
        if AxisType.LATITUDE in self.dataset.dims and AxisType.LONGITUDE in self.dataset.dims:
            lat_size = self.dataset.da.sizes.get(
                self.dataset.domain.axes[AxisType.LATITUDE].name, 0
            )
            lon_size = self.dataset.da.sizes.get(
                self.dataset.domain.axes[AxisType.LONGITUDE].name, 0
            )
            total_points = lat_size * lon_size
            
            # If we have fewer than 1000 points, treat as sparse
            return total_points < 1000
        
        return True  # Default to sparse if uncertain
        
    def _get_coordinate_info(self) -> None:
        """Extract coordinate information from dataset."""
        from climatrix.dataset.axis import AxisType
        
        self.coords = {}
        
        # Get latitude/longitude ranges
        if AxisType.LATITUDE in self.dataset.dims:
            lat_axis = self.dataset.domain.axes[AxisType.LATITUDE]
            self.coords['lat'] = {
                'name': lat_axis.name,
                'values': self.dataset.da[lat_axis.name].values,
                'range': [
                    float(self.dataset.da[lat_axis.name].min()),
                    float(self.dataset.da[lat_axis.name].max())
                ]
            }
            
        if AxisType.LONGITUDE in self.dataset.dims:
            lon_axis = self.dataset.domain.axes[AxisType.LONGITUDE]
            self.coords['lon'] = {
                'name': lon_axis.name,
                'values': self.dataset.da[lon_axis.name].values,
                'range': [
                    float(self.dataset.da[lon_axis.name].min()),
                    float(self.dataset.da[lon_axis.name].max())
                ]
            }
            
        # Get time information
        if self.has_time:
            time_axis = self.dataset.domain.axes[AxisType.TIME]
            time_values = self.dataset.da[time_axis.name].values
            self.coords['time'] = {
                'name': time_axis.name,
                'values': time_values,
                'range': [0, len(time_values) - 1],
                'labels': [str(t) for t in time_values]
            }
            
        # Get vertical dimension information
        if self.has_vertical:
            vert_axis = self.dataset.domain.axes[AxisType.VERTICAL]
            vert_values = self.dataset.da[vert_axis.name].values
            self.coords['vertical'] = {
                'name': vert_axis.name,
                'values': vert_values,
                'range': [0, len(vert_values) - 1],
                'labels': [str(v) for v in vert_values]
            }
    
    def _setup_layout(self) -> None:
        """Setup the Dash app layout with material design."""
        from dash import dcc, html
        
        # Material design colors
        colors = {
            'primary': '#1976d2',
            'secondary': '#424242',
            'surface': '#ffffff',
            'background': '#fafafa',
            'text': '#212121',
            'text_secondary': '#757575'
        }
        
        # Build control components
        controls = []
        
        # Time slider if time dimension exists
        if self.has_time:
            controls.append(
                html.Div([
                    html.Label(
                        f"Time ({self.coords['time']['name']})",
                        style={'color': colors['text'], 'fontSize': '14px', 'fontWeight': '500'}
                    ),
                    dcc.Slider(
                        id='time-slider',
                        min=self.coords['time']['range'][0],
                        max=self.coords['time']['range'][1],
                        value=self.coords['time']['range'][0],
                        marks={
                            i: {'label': label, 'style': {'fontSize': '10px'}}
                            for i, label in enumerate(self.coords['time']['labels'][::max(1, len(self.coords['time']['labels'])//10)])
                        },
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], style={'margin': '20px 0'})
            )
            
        # Vertical slider if vertical dimension exists
        if self.has_vertical:
            controls.append(
                html.Div([
                    html.Label(
                        f"Vertical Level ({self.coords['vertical']['name']})",
                        style={'color': colors['text'], 'fontSize': '14px', 'fontWeight': '500'}
                    ),
                    dcc.Slider(
                        id='vertical-slider',
                        min=self.coords['vertical']['range'][0],
                        max=self.coords['vertical']['range'][1],
                        value=self.coords['vertical']['range'][0],
                        marks={
                            i: {'label': label, 'style': {'fontSize': '10px'}}
                            for i, label in enumerate(self.coords['vertical']['labels'][::max(1, len(self.coords['vertical']['labels'])//5)])
                        },
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], style={'margin': '20px 0'})
            )
        
        # Projection dropdown (only if we have geographic coordinates)
        if 'lat' in self.coords and 'lon' in self.coords:
            controls.append(
                html.Div([
                    html.Label(
                        "Map Projection",
                        style={'color': colors['text'], 'fontSize': '14px', 'fontWeight': '500'}
                    ),
                    dcc.Dropdown(
                        id='projection-dropdown',
                        options=[
                            {'label': 'Plate Carrée (Equirectangular)', 'value': 'equirectangular'},
                            {'label': 'Natural Earth', 'value': 'natural earth'},
                            {'label': 'Mercator', 'value': 'mercator'},
                            {'label': 'Robinson', 'value': 'robinson'},
                            {'label': 'Mollweide', 'value': 'mollweide'}
                        ],
                        value='equirectangular',  # Default to Plate Carrée
                        style={'color': colors['text']},
                        clearable=False
                    )
                ], style={'margin': '20px 0'})
            )
        
        # App layout
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1(
                    "Climatrix Interactive Plot",
                    style={
                        'color': colors['surface'],
                        'margin': '0',
                        'fontSize': '24px',
                        'fontWeight': '400'
                    }
                )
            ], style={
                'background': colors['primary'],
                'padding': '20px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),
            
            # Main content
            html.Div([
                # Controls panel
                html.Div([
                    html.H3(
                        "Controls",
                        style={'color': colors['text'], 'marginTop': '0'}
                    ),
                    *controls
                ], style={
                    'width': '25%',
                    'float': 'left',
                    'padding': '20px',
                    'background': colors['surface'],
                    'minHeight': '80vh',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'
                }),
                
                # Plot area
                html.Div([
                    dcc.Loading(
                        id="loading",
                        type="circle",
                        color=colors['primary'],
                        children=[
                            dcc.Graph(
                                id='main-plot',
                                style={'height': '80vh'},
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToAdd': ['select2d', 'lasso2d']
                                }
                            )
                        ]
                    )
                ], style={
                    'width': '75%',
                    'float': 'right',
                    'padding': '20px',
                    'background': colors['background']
                })
            ]),
            
            # Error display
            html.Div(id='error-display', style={'display': 'none'})
            
        ], style={
            'fontFamily': 'Roboto, Arial, sans-serif',
            'margin': '0',
            'padding': '0'
        })
    
    def _setup_callbacks(self) -> None:
        """Setup Dash callbacks for interactivity."""
        from dash.dependencies import Input, Output, State
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Determine inputs based on available dimensions
        inputs = []
        if 'lat' in self.coords and 'lon' in self.coords:
            inputs.append(Input('projection-dropdown', 'value'))
        if self.has_time:
            inputs.append(Input('time-slider', 'value'))
        if self.has_vertical:
            inputs.append(Input('vertical-slider', 'value'))
            
        # If no geographic coordinates, use a simple callback without projection
        if not inputs:
            inputs = [Input('main-plot', 'id')]  # Dummy input
            
        @self.app.callback(
            Output('main-plot', 'figure'),
            inputs
        )
        def update_plot(*args):
            try:
                arg_idx = 0
                projection = None
                if 'lat' in self.coords and 'lon' in self.coords:
                    projection = args[arg_idx]
                    arg_idx += 1
                    
                time_idx = args[arg_idx] if self.has_time and arg_idx < len(args) else None
                if self.has_time:
                    arg_idx += 1
                    
                vert_idx = args[arg_idx] if self.has_vertical and arg_idx < len(args) else None
                
                return self._generate_plot(projection, time_idx, vert_idx)
                
            except Exception as e:
                log.error(f"Error updating plot: {e}")
                # Return empty figure with error message
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="red")
                )
                return fig
    
    def _generate_plot(self, projection: Optional[str] = None, time_idx: Optional[int] = None, vert_idx: Optional[int] = None):
        """Generate the plot based on current settings."""
        import plotly.graph_objects as go
        import plotly.express as px
        import numpy as np
        
        # Get data subset based on current selection
        data = self._get_data_subset(time_idx, vert_idx)
        
        # Always create 2D plot, with projection if geographic data is available
        return self._create_2d_plot(data, projection)
    
    def _get_data_subset(self, time_idx: Optional[int] = None, vert_idx: Optional[int] = None):
        """Get data subset based on current slider positions."""
        data = self.dataset.da
        
        # Select time slice if applicable
        if self.has_time and time_idx is not None:
            time_name = self.coords['time']['name']
            data = data.isel({time_name: time_idx})
            
        # Select vertical level if applicable
        if self.has_vertical and vert_idx is not None:
            vert_name = self.coords['vertical']['name']
            data = data.isel({vert_name: vert_idx})
            
        return data
    
    def _create_2d_plot(self, data, projection: Optional[str] = None):
        """Create enhanced 2D plot with projection support and geographic features."""
        import plotly.graph_objects as go
        import plotly.express as px
        import numpy as np
        
        if 'lat' not in self.coords or 'lon' not in self.coords:
            # If no spatial coordinates, create a simple plot
            fig = go.Figure()
            
            # Handle different data shapes
            if len(data.dims) == 0:
                # Scalar data
                fig.add_trace(go.Scatter(
                    x=[0],
                    y=[float(data.values)],
                    mode='markers',
                    name=data.name or 'Data',
                    marker=dict(size=12, color='#1976d2')
                ))
            else:
                # Multi-dimensional data - flatten
                values = data.values.flatten()
                fig.add_trace(go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='lines+markers',
                    name=data.name or 'Data',
                    line=dict(color='#1976d2', width=2),
                    marker=dict(size=6, color='#1976d2')
                ))
                
            fig.update_layout(
                title=dict(
                    text=f"{data.name or 'Data'} - Data Plot",
                    font=dict(size=18, color='#212121')
                ),
                xaxis_title="Index",
                yaxis_title="Value",
                height=600,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Roboto, Arial, sans-serif")
            )
            return fig
        
        lat_name = self.coords['lat']['name']
        lon_name = self.coords['lon']['name']
        
        # Default to equirectangular projection (Plate Carrée)
        if projection is None:
            projection = 'equirectangular'
        
        if self.is_sparse:
            # Scatter plot for sparse data on geographic projection
            lats = []
            lons = []
            values = []
            
            try:
                if data.dims == (lat_name, lon_name):
                    # 2D grid case
                    for i in range(data.sizes[lat_name]):
                        for j in range(data.sizes[lon_name]):
                            val = data.isel({lat_name: i, lon_name: j}).values
                            if not np.isnan(val):
                                lats.append(float(data[lat_name].isel({lat_name: i}).values))
                                lons.append(float(data[lon_name].isel({lon_name: j}).values))
                                values.append(float(val))
                elif len(data.dims) == 1:
                    # 1D case - station data
                    dim_name = data.dims[0]
                    for i in range(data.sizes[dim_name]):
                        val = data.isel({dim_name: i}).values
                        if not np.isnan(val):
                            lat_val = data[lat_name].isel({dim_name: i}).values
                            lon_val = data[lon_name].isel({dim_name: i}).values
                            lats.append(float(lat_val))
                            lons.append(float(lon_val))
                            values.append(float(val))
                else:
                    # Fallback - use coordinate arrays directly
                    lats = [float(x) for x in data[lat_name].values.flatten()]
                    lons = [float(x) for x in data[lon_name].values.flatten()]
                    values = [float(x) for x in data.values.flatten()]
                    
                # Filter out invalid coordinates
                valid_data = [(lat, lon, val) for lat, lon, val in zip(lats, lons, values) 
                             if -90 <= lat <= 90 and -180 <= lon <= 180 and not np.isnan(val)]
                
                if not valid_data:
                    # No valid data points
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No valid data points to display",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=16, color='#757575')
                    )
                    return fig
                    
                lats, lons, values = zip(*valid_data)
                
            except Exception as e:
                log.warning(f"Error processing sparse data: {e}")
                # Fallback to simple plot
                return self._create_fallback_plot(data)
            
            # Create scatter plot on geographic projection
            fig = go.Figure()
            
            fig.add_trace(go.Scattergeo(
                lat=lats,
                lon=lons,
                mode='markers',
                marker=dict(
                    size=10,
                    color=values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=data.name or 'Value', font=dict(size=14)),
                        tickfont=dict(size=12)
                    ),
                    cmin=min(values) if values else 0,
                    cmax=max(values) if values else 1,
                    line=dict(width=1, color='white')
                ),
                text=[f"Value: {v:.3f}" for v in values],
                hovertemplate="<b>Lat:</b> %{lat:.3f}<br><b>Lon:</b> %{lon:.3f}<br>%{text}<extra></extra>"
            ))
            
        else:
            # Heatmap for dense data on geographic projection
            fig = go.Figure()
            
            try:
                # Ensure data is 2D
                if len(data.dims) != 2 or data.dims != (lat_name, lon_name):
                    # Try to reshape or select appropriate slice
                    if lat_name in data.dims and lon_name in data.dims:
                        # Select first slice of other dimensions
                        data_subset = data
                        for dim in data.dims:
                            if dim not in [lat_name, lon_name]:
                                data_subset = data_subset.isel({dim: 0})
                        data = data_subset
                
                # For dense geographic data, use scattergeo with interpolated grid
                lats = data[lat_name].values
                lons = data[lon_name].values
                
                # Create mesh grid for visualization
                lon_mesh, lat_mesh = np.meshgrid(lons, lats)
                
                fig.add_trace(go.Scattergeo(
                    lat=lat_mesh.flatten(),
                    lon=lon_mesh.flatten(),
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=data.values.flatten(),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title=dict(text=data.name or 'Value', font=dict(size=14)),
                            tickfont=dict(size=12)
                        ),
                        opacity=0.8
                    ),
                    hovertemplate="<b>Lat:</b> %{lat:.3f}<br><b>Lon:</b> %{lon:.3f}<br><b>Value:</b> %{marker.color:.3f}<extra></extra>"
                ))
                
            except Exception as e:
                log.warning(f"Error creating geographic heatmap: {e}")
                return self._create_fallback_plot(data)
        
        # Configure geographic layout with coastlines and country borders
        fig.update_geos(
            projection_type=projection,
            showcoastlines=True,
            coastlinecolor="RebeccaPurple",
            coastlinewidth=1,
            showland=True,
            landcolor="LightGray",
            showocean=True,
            oceancolor="LightBlue",
            showcountries=True,
            countrycolor="DarkGray",
            countrywidth=0.5,
            showlakes=True,
            lakecolor="LightBlue"
        )
        
        # Enhanced layout styling
        fig.update_layout(
            title=dict(
                text=f"{data.name or 'Data'} - Geographic Visualization",
                font=dict(size=18, color='#212121'),
                x=0.5,
                xanchor='center'
            ),
            height=700,
            font=dict(family="Roboto, Arial, sans-serif"),
            paper_bgcolor='white',
            margin=dict(l=0, r=0, t=50, b=0)
        )
            
        return fig
    
    def _create_fallback_plot(self, data):
        """Create a fallback plot when normal plotting fails."""
        import plotly.graph_objects as go
        import numpy as np
        
        fig = go.Figure()
        
        # Try to create a simple line/scatter plot
        values = data.values
        if values.ndim > 1:
            values = values.flatten()
            
        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode='lines+markers',
            name=data.name or 'Data'
        ))
        
        fig.update_layout(
            title=f"{data.name or 'Data'} - Fallback Plot",
            xaxis_title="Index",
            yaxis_title="Value",
            height=600
        )
        
        return fig
    
    
    def show(self) -> None:
        """
        Launch the interactive plot in a web browser.
        
        This starts the Dash server and opens the plot in the default browser.
        The server will continue running until manually stopped.
        """
        url = f"http://{self.host}:{self.port}"
        
        if self.auto_open:
            def open_browser():
                import time
                time.sleep(1)  # Give server time to start
                webbrowser.open(url)
                
            import threading
            threading.Thread(target=open_browser).start()
        
        log.info(f"Starting Climatrix interactive plot at {url}")
        print(f"Climatrix interactive plot available at: {url}")
        print("Press Ctrl+C to stop the server")
        
        try:
            self.app.run_server(
                host=self.host,
                port=self.port,
                debug=self.debug
            )
        except KeyboardInterrupt:
            print("\nServer stopped by user")
        except Exception as e:
            log.error(f"Error running server: {e}")
            raise
    
    def save_html(self, filename: str) -> None:
        """
        Save the current plot as a static HTML file.
        
        Parameters
        ----------
        filename : str
            Path where to save the HTML file
        """
        # Generate current plot
        fig = self._generate_plot('2d')  # Default to 2D view
        
        # Save as HTML
        fig.write_html(filename)
        log.info(f"Plot saved as {filename}")
        print(f"Static plot saved as: {filename}")