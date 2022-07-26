

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


from examples import results_html, plot_format


def plot_height(df: pd.DataFrame) -> go.Figure:
    # plot data
    colors = plot_format.get_plot_color(3)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["height"] * 10, y=df["mean"], mode="lines+markers", name="<b>mean</b>",
                             marker=dict(color=colors[0], size=10), line=dict(color=colors[0])))
    fig.add_trace(go.Scatter(x=df["height"] * 10, y=df["std"], mode="lines+markers", name="<b>std</b>",
                             marker=dict(color=colors[1], size=10), line=dict(color=colors[1]), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["height"] * 10, y=df["std"]/df["mean"], mode="lines+markers", name="<b>std/mean</b>",
                             marker=dict(color=colors[2], size=10), line=dict(color=colors[2]), yaxis="y3"))

    # add plot formatting
    fig.update_layout(autosize=False, width=int(800*0.7), height=int(600*.7), font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.2, y=1, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title="<b>height of lights (mm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", domain=[0.0, 0.8], range=[0, 160])
    fig.update_yaxes(title="<b>mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_layout(yaxis2=dict(title="<b>std irradiance<b>", range=[0, 40], anchor="x",
                                 overlaying="y", side="right"))
    fig.update_layout(yaxis3=dict(title="<b>std/mean<b>", tickprefix="<b>", ticksuffix="</b>", range=[0, 0.25],
                                  anchor="free", overlaying="y", side="right", position=1,
                                  linecolor="black", linewidth=5
                                  ))

    return fig


def plot_height2(df: pd.DataFrame, fig: go.Figure, marker = None) -> go.Figure:
    if marker == "open":
        marker_symbol = "circle-open"
        line_dash = "dash"
        legend_show = False
    else:
        marker_symbol = "circle"
        line_dash = None
        legend_show = True

    # plot data
    colors = plot_format.get_plot_color(3)

    fig.add_trace(go.Scatter(x=df["height"] * 10, y=df["mean"], mode="lines+markers", name="<b>mean</b>", showlegend=legend_show,
                             marker=dict(color=colors[0], size=10, symbol=marker_symbol), line=dict(color=colors[0], dash=line_dash)))
    fig.add_trace(go.Scatter(x=df["height"] * 10, y=df["std"], mode="lines+markers", name="<b>std</b>", showlegend=legend_show,
                             marker=dict(color=colors[1], size=10, symbol=marker_symbol), line=dict(color=colors[1], dash=line_dash), yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["height"] * 10, y=df["std"]/df["mean"], mode="lines+markers", name="<b>std/mean</b>", showlegend=legend_show,
                             marker=dict(color=colors[2], size=10, symbol=marker_symbol), line=dict(color=colors[2], dash=line_dash), yaxis="y3"))

    # add plot formatting
    fig.update_layout(autosize=False, width=int(800), height=int(600), font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.4, y=0.8, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(title="<b>height of lights (mm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", domain=[0.0, 0.88], range=[0, 160])
    fig.update_yaxes(title="<b>mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_layout(yaxis2=dict(title="<b>std irradiance<b>", range=[0, 40], anchor="x",
                                 overlaying="y", side="right"))
    fig.update_layout(yaxis3=dict(title="<b>std/mean<b>", tickprefix="<b>", ticksuffix="</b>", range=[0, 0.25],
                                  anchor="free", overlaying="y", side="right", position=1,
                                  linecolor="black", linewidth=5
                                  ))

    return fig


def main():
    # load data
    df = pd.read_csv("mirror_data.csv", index_col=0)

    # select data
    fig_height = plot_height(df.loc[(df["reflect"] == 4)])
    fig_height_perfect = plot_height(df.loc[(df["reflect"] == 7)])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig = plot_height2(df.loc[(df["reflect"] == 4)], fig)
    fig = plot_height2(df.loc[(df["reflect"] == 7)], fig, "open")

    results_html.merge_html_figs([fig_height, fig_height_perfect, fig], "results.html", auto_open=True)


if __name__ == "__main__":
    main()
