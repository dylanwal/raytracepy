import numpy as np
import plotly.graph_objs as go

from raytracepy.utils.sig_figs import sig_figs


def calc(heatmap: np.ndarray):
    his_array = np.reshape(heatmap,
                           (heatmap.shape[0] * heatmap.shape[1],))
    mean_ = sig_figs(np.mean(his_array))
    std = sig_figs(np.std(his_array)/mean_)
    p10 = sig_figs(np.percentile(his_array, 10)/mean_)
    p30 = sig_figs(np.percentile(his_array, 30)/mean_)
    p50 = sig_figs(np.percentile(his_array, 50)/mean_)
    p70 = sig_figs(np.percentile(his_array, 70)/mean_)
    p90 = sig_figs(np.percentile(his_array, 90)/mean_)
    min_ = sig_figs(np.min(his_array)/mean_)
    max_ = sig_figs(np.max(his_array)/mean_)
    print(f"mean: {mean_}, std: {std} || min: {min_}  p10: {p10}, p30: {p30}, p50: {p50}, p70: {p70}, "
          f"p90: {p90}, max: {max_}")


def plot(heatmap: np.ndarray):
    fig = go.Figure()
    x = np.linspace(-5, 5, heatmap.shape[0])
    y = np.linspace(-5, 5, heatmap.shape[0])
    fig.add_trace(go.Heatmap(x=x, y=y, z=heatmap/np.mean(np.mean(heatmap))))

    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.4, y=.95))
    fig.update_xaxes(title="<b>x (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", domain=[0.03, 0.88])
    fig.update_yaxes(title="<b>y (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    return fig


def plot_swatch(data: list[np.ndarray]):
    fig = go.Figure()
    x = np.linspace(-5, 5, data[0].shape[0])
    for dat in data:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.mean(dat[22:27, :]/np.mean(np.mean(dat)), axis=0),
                mode="lines"
            )
        )
    return fig


def main():
    n = 50
    with open(f"heatmap_raytrace.csv", "r") as file:
        data_ray = np.loadtxt(file, delimiter=",")

    with open(f"heatmap_valid.csv", "r") as file:
        data_valid = np.loadtxt(file, delimiter=",")

    calc(data_ray)
    calc(data_valid)

    fig = plot(data_ray)
    # fig.write_html("raytace.html", include_plotlyjs='cdn')
    fig2 = plot(data_valid)
    # fig2.write_html("valid.html", include_plotlyjs='cdn')

    fig3 = plot_swatch([data_ray, data_valid])
    fig3.show()


if __name__ == "__main__":
    main()
