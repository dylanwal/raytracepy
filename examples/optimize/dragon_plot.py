import pickle

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff


# Load data
with open("result_2022_03_17-08_26_41_PM.pickle", 'rb') as file:
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

df_all = pd.DataFrame(results_to_df(all_points, all_vals), columns=["height", "width", "pattern", "mean", "std"])

fig = go.Figure()
fig.add_trace(go.Scatter(x=all_vals[:, 0], y=all_vals[:, 1], mode="markers"))
fig.add_trace(go.Scatter(x=pareto_vals_sort[:, 0], y=pareto_vals_sort[:, 1], mode="lines"))

layout = {
        "autosize": False,
        "width": 900,
        "height": 790,
        "showlegend": False,
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

fig.write_html("pareto.html", auto_open=True)


#########################################################

colors = {
    "circle": "red",
    "ogrid": "blue",
    "grid": "green",
    "spiral": "purple"
}

new_col = []
for pattern in df_all["pattern"]:
    new_col.append(colors[pattern])
df_all['Colors'] = pd.Series(new_col, index=df_all.index)

fig = ff.create_scatterplotmatrix(df_all, diag='box', index='Colors',
                                  colormap_type='cat',
                                  height=1200, width=1200)
layout = {
        "autosize": False,
        "width": 900,
        "height": 790,
        "showlegend": False,
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
# fig.update_layout(layout)
# fig.update_xaxes(xaxis)
# fig.update_yaxes(yaxis)

fig.write_html("matrix.html", auto_open=True)
