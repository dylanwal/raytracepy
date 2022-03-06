"""
Calculates hit on a flat surface, then uses this data to generate angular and radial distribution functions.

"""

import numpy as np
import plotly.graph_objs as go
import pandas as pd
import datashader as ds

import raytracepy.theory as raypy_t
import raytracepy.utils.analysis_func as raypy_a
import examples.plot_format as plot_format


n = 5_000_000  # number of points
h = 5  # height of light
x_dim = 10  # x size of plane the light hits
y_dim = x_dim  # y size of plane the light hits

hits = raypy_t.hits_on_flat_surface(n, x_dim, y_dim, h)

hit_x = hits[:, 0]
hit_y = hits[:, 1]

# histogram
res = 300
x = np.linspace(-x_dim, x_dim, res)
y = np.linspace(-y_dim, y_dim, res)
df = pd.DataFrame(np.column_stack((hit_x, hit_y)), columns=["x", "y"])
canvas = ds.Canvas(plot_width=res, plot_height=res)
agg = canvas.points(df, 'x', 'y')
fig = go.Figure()
fig.add_trace(go.Heatmap(x=x, y=y, z=agg, colorbar=dict(title="count")))
fig.update_xaxes(title="<b>x (cm)</b>")
fig.update_yaxes(title="<b>y (cm)</b>")
fig.update_layout(height=700, width=800)
fig.write_html("histogram.html", auto_open=True, include_plotlyjs='cdn')

# angular distribution function [-pi, pi]
x, hist = raypy_a.adf(np.column_stack((hit_x, hit_y)), bins=20, normalize=True)
fig = go.Figure(go.Scatter(x=x, y=hist, mode="lines"))
plot_format.add_plot_format(fig, x_axis="radian", y_axis="hits")
fig.write_html("angular_dist.html", auto_open=True, include_plotlyjs='cdn')

# radial distribution function (normalized) [0, x_dim]
x, hist = raypy_a.rdf(np.column_stack((hit_x, hit_y)), bins=20, normalize=True)
fig = go.Figure(go.Scatter(x=x, y=hist/np.max(hist), mode="lines"))

x_theory = np.linspace(0, x_dim, 50)
y_theory = raypy_t.intensity_on_flat_surface(x_theory, 0, h)
fig.add_trace(go.Scatter(x=x_theory, y=y_theory/np.max(y_theory), mode="lines"))

plot_format.add_plot_format(fig, x_axis="distance (cm)", y_axis="normalized intensity")
fig.write_html("radial_dist.html", auto_open=True, include_plotlyjs='cdn')

