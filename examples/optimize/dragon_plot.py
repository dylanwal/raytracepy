import pickle

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
import scipy.spatial

from examples import results_html

colors = {
    "circle": "red",
    "ogrid": "blue",
    "grid": "green",
    "spiral": "purple"
}

# search
df_grid = pd.read_csv("grid_search_mirror_49.csv", index_col=0)
# add color column
new_col = []
for grid_type in df_grid["grid_type"]:
    new_col.append(colors[grid_type])
df_grid['colors'] = pd.Series(new_col, index=df_grid.index)
# add std_fix
new_col = []
for std in df_grid["std"]:
    new_col.append(100-std)
df_grid['std_fix'] = pd.Series(new_col, index=df_grid.index)

df_grid["mean"][df_grid.grid_type=="ogrid"] = df_grid["mean"][df_grid.grid_type=="ogrid"] * 25/23

hull_points = df_grid[["mean", "std_fix"]].to_numpy()
hull = scipy.spatial.ConvexHull(hull_points)
hull_xy = np.array([hull_points[hull.vertices, 0], hull_points[hull.vertices, 1]])

# Load data
with open("result_2022_04_08-10_04_07_PM.pickle", 'rb') as file:
    results = pickle.load(file)


def results_to_df(points, vals):
    out = []
    for p, v in zip(points, vals):
        out.append([p[0][0], p[0][1], p[1][0], v[0], v[1]])

    return out


pareto_points = results.curr_true_pareto_points[-1]
pareto_vals = np.array(results.curr_true_pareto_vals[-1])
pareto_vals_sort = np.copy(pareto_vals)
pareto_vals_sort = pareto_vals_sort[pareto_vals_sort[:, 0].argsort()]

all_points = results.query_points
all_vals = np.array(results.query_vals)

df_all = pd.DataFrame(results_to_df(all_points, all_vals), columns=["light_height", "light_width", "grid_type", "mean", "std"])

new_col = []
for grid_type in df_all["grid_type"]:
    new_col.append(colors[grid_type])
df_all['colors'] = pd.Series(new_col, index=df_all.index)

#########################################################
#########################################################
fig_control = go.Figure(px.scatter(df_grid, x="mean", y="std_fix", color="grid_type",
                                   hover_data=["light_height", "light_width", "grid_type", "mean", "std_fix"]))
fig_control.add_trace(go.Scatter(x=hull_xy[0, :], y=hull_xy[1, :], mode="lines", name="control_hull"))

layout = {
    "autosize": False,
    "width": 900,
    "height": 790,
    "showlegend": True,
    "font": dict(family="Arial", size=18, color="black"),
    "plot_bgcolor": "white"
}
xaxis = {
    "title": "<b>normalized mean intensity<b>",
    "tickprefix": "<b>",
    "ticksuffix": "</b>",
    "showline": True,
    "linewidth": 5,
    "mirror": True,
    "linecolor": 'black',
    "ticks": "outside",
    "tickwidth": 4,
    "showgrid": False,
    "gridwidth": 1,
    "gridcolor": 'lightgray'
}
yaxis = {
    "title": "<b>normalized std of intensity<b>",
    "tickprefix": "<b>",
    "ticksuffix": "</b>",
    "showline": True,
    "linewidth": 5,
    "mirror": True,
    "linecolor": 'black',
    "ticks": "outside",
    "tickwidth": 4,
    "showgrid": False,
    "gridwidth": 1,
    "gridcolor": 'lightgray'
}
fig_control.update_layout(layout)
fig_control.update_xaxes(xaxis)
fig_control.update_yaxes(yaxis)

# add plot formatting
fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                  plot_bgcolor="white", showlegend=True, legend=dict(x=.05, y=.95))
fig.update_xaxes(title="<b>mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                 linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                 gridwidth=1, gridcolor="lightgray")
fig.update_yaxes(title="<b>std irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                 linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                 gridwidth=1, gridcolor="lightgray")

#########################################################
#########################################################
#########################################################
#########################################################
fig = go.Figure(px.scatter(df_all, x="mean", y="std", color="grid_type",
                           hover_data=["light_height", "light_width", "grid_type", "mean", "std"]))
fig.add_trace(go.Scatter(x=pareto_vals_sort[:, 0], y=pareto_vals_sort[:, 1], mode="lines", showlegend=False))
fig.add_trace(go.Scatter(x=df_grid["mean"], y=df_grid["std"], mode="markers", name="control"))
fig.add_trace(go.Scatter(x=hull_xy[0, :], y=hull_xy[1, :], mode="lines", name="control_hull"))

layout = {
    "autosize": False,
    "width": 900,
    "height": 790,
    "showlegend": True,
    "font": dict(family="Arial", size=18, color="black"),
    "plot_bgcolor": "white"
}
xaxis = {
    "title": "<b>normalized mean intensity<b>",
    "tickprefix": "<b>",
    "ticksuffix": "</b>",
    "showline": True,
    "linewidth": 5,
    "mirror": True,
    "linecolor": 'black',
    "ticks": "outside",
    "tickwidth": 4,
    "showgrid": False,
    "gridwidth": 1,
    "gridcolor": 'lightgray'
}
yaxis = {
    "title": "<b>normalized std of intensity<b>",
    "tickprefix": "<b>",
    "ticksuffix": "</b>",
    "showline": True,
    "linewidth": 5,
    "mirror": True,
    "linecolor": 'black',
    "ticks": "outside",
    "tickwidth": 4,
    "showgrid": False,
    "gridwidth": 1,
    "gridcolor": 'lightgray'
}
fig.update_layout(layout)
fig.update_xaxes(xaxis)
fig.update_yaxes(yaxis)

#########################################################
#########################################################

fig_grid = ff.create_scatterplotmatrix(df_all, diag='box', index='colors',
                                       colormap_type='cat',
                                       height=1200, width=1200
                                       )

#######################################################
#######################################################
#########################################################
#########################################################

fig_control_grid = ff.create_scatterplotmatrix(df_grid, diag='box', index='colors',
                                               colormap_type='cat',
                                               height=1200, width=1200
                                               )

#######################################################
#######################################################
html = df_all.to_html()
results_html.merge_html_figs([fig_control, fig_control_grid, fig, fig_grid, html], "results.html", auto_open=True)
