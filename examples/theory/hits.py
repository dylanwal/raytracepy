"""
Calculates hit on a flat surface, then uses this data to generate angular and radial distribution functions.

"""

import numpy as np
import plotly.graph_objs as go

import raytracepy.theory as raypy_t
import raytracepy.utils.analysis_func as raypy_a

n = 1_000_000  # number of points
h = 5  # height of light
x_dim = 10
y_dim = x_dim

hits = raypy_t.hits_on_flat_surface(n, x_dim, y_dim, h)

hit_x = hits[:, 0]
hit_y = hits[:, 1]

# histogram
fig = go.Figure(go.Histogram2d(x=hit_x, y=hit_y, nbinsx=20, nbinsy=20))
fig.write_html("temp.html", auto_open=True)

# angular distribution function [-pi, pi]
x, hist = raypy_a.adf(np.column_stack((hit_x, hit_y)), bins=20, normalize=True)
fig = go.Figure(go.Scatter(x=x, y=hist, mode="lines"))
fig.write_html("temp2.html", auto_open=True)

# radial distribution function (normalized) [0, x_dim]
x, hist = raypy_a.rdf(np.column_stack((hit_x, hit_y)), bins=20, normalize=True)
fig = go.Figure(go.Scatter(x=x, y=hist, mode="lines"))
x_theory = np.linspace(0, x_dim, 50)
y_theory = raypy_t.intensity_on_flat_surface(x_theory, 0, h)
fig.add_trace(go.Scatter(x=x_theory, y=y_theory/np.max(y_theory), mode="lines"))
fig.write_html("temp3.html", auto_open=True)

