import numpy as np
import plotly.graph_objs as go
from numba import njit, config

import raytracepy as rpy

config.DISABLE_JIT = False


def get_light_xy(grid_type, light_width, number_lights):
    if number_lights == 1:
        return np.array([[0, 0]], dtype="float64")

    if grid_type == "circle":
        grid = rpy.CirclePattern(
            center=np.array([0, 0]),
            outer_radius=light_width / 2,
            layers=3,
            num_points=number_lights)
    elif grid_type == "ogrid":
        grid = rpy.OffsetGridPattern(
            center=np.array([0, 0]),
            x_length=light_width,
            y_length=light_width,
            num_points=number_lights)
    elif grid_type == "grid":
        grid = rpy.GridPattern(
            center=np.array([0, 0]),
            x_length=light_width,
            y_length=light_width,
            num_points=number_lights)
    elif grid_type == "spiral":
        grid = rpy.SpiralPattern(
            center=np.array([0, 0]),
            radius=light_width / 2,
            radius_start=.5,
            velocity=0.2,
            a_velocity=1,
            num_points=number_lights)
    else:
        raise ValueError(f"{grid_type} invalid choice.")

    return grid.xy_points


@njit
def abs_e3(light_height, x, y, xmin, xmax, ymin, ymax, mirror_eff, xy_led, num_reflect):
    if num_reflect == 0:
        return abs_e2(light_height, xy_led, 0, 0)

    dx = xmax - xmin
    dy = ymax - ymin

    n = num_reflect * 2
    x_M = np.empty(n)
    y_M = np.empty(n)
    for i in range(n):
        x_M[i] = xmin + (i - num_reflect) * dx + (i % 2) * (x - xmin) + (xmax - x) - (i % 2) * (xmax - x)
        y_M[i] = ymin + (i - num_reflect) * dy + (i % 2) * (y - ymin) + (ymax - y) - (i % 2) * (ymax - y)
    # i = np.linspace(1, n, n)
    # x_M = xmin + (i - num_reflect) * dx + i % 2 * (x - xmin) + (xmax - x) - i % 2 * (xmax - x)
    # y_M = ymin + (i - num_reflect) * dy + i % 2 * (y - ymin) + (ymax - y) - i % 2 * (ymax - y)
    

    E = 0
    for i in range(n):
        for ii in range(n):
            N_ref = np.abs(i - num_reflect) + np.abs(ii - num_reflect)
            E += abs_e2(light_height, xy_led, x_M[i], y_M[ii]) * mirror_eff ** N_ref

    return E


@njit
def abs_e2(light_height, xy_led, x, y):
    p = [0.014587371165815, - 0.001132696169541, - 0.100665969423576, 0.006192872725832,
         0.243105330641826, - 0.011949844019489, - 0.285411492348233,
         0.009356382808296, - 0.042818342694045, - 0.000100700328521, 1.000556027166679]

    e = 0
    for x_led, y_led in xy_led:
        delta = np.sqrt((x_led - x) ** 2 + (y_led - y) ** 2)
        gamma = np.arctan(delta / light_height)
        z = np.sqrt((x_led - x) ** 2 + (y_led - y) ** 2 + light_height ** 2)
        ang_p = gamma
        F_L = p[0] * ang_p ** 10 + p[1] * ang_p ** 9 + p[2] * ang_p ** 8 + p[3] * ang_p ** 7 + p[4] * ang_p ** 6 + p[
            5] * ang_p ** 5 + p[6] * ang_p ** 4 + p[7] * ang_p ** 3 + p[8] * ang_p ** 2 + p[9] * ang_p + p[10]

        e += F_L * np.cos(gamma) / z ** 2

    return e


@njit
def main_loop(light_height, surface_width, surface_length, xy_led, mirror_eff, resolution, num_reflect):
    xmin = -surface_width/2
    xmax = -1 * xmin
    ymin = -surface_length/2
    ymax = -1 * ymin
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)

    e_p2t = np.empty((resolution, resolution))
    for i in range(resolution):
        for ii in range(resolution):
            e_p2t[i, ii] = abs_e3(light_height, x[i], y[ii], xmin, xmax, ymin, ymax, mirror_eff, xy_led, num_reflect)

    return e_p2t


def run(light_height: float = 10, light_width: float = 10, number_lights: int = 1, grid_type: str = "ogrid",
        resolution: int = 40, mirror_eff: float = 0.85, surface_width: float = 10, surface_length: float = 10,
        num_reflect: int = 50):
    xy_led = get_light_xy(grid_type, light_width, number_lights)
    e_p2t = main_loop(light_height, surface_width, surface_length, xy_led, mirror_eff, resolution, num_reflect)
    return e_p2t # / np.mean(np.mean(e_p2t))


def main():
    box = 50
    light_height = box/2
    light_width = 0
    number_lights = 1
    grid_type = "ogrid"
    resolution = 100
    mirror_eff = 0.85
    surface_width = box
    surface_length = box
    num_reflect = 10

    heatmap = run(light_height, light_width, number_lights, grid_type, resolution, mirror_eff, surface_width,
                  surface_length, num_reflect)

    np.savetxt("heatmap_single.csv", heatmap, delimiter=",")

    fig = go.Figure()
    x = np.linspace(-5, 5, resolution)
    y = np.linspace(-5, 5, resolution)
    fig.add_trace(go.Heatmap(x=x, y=y, z=heatmap/np.mean(np.mean(heatmap)), zsmooth="fast"))

    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=True, legend=dict(x=.4, y=.95))
    fig.update_xaxes(title="<b>x (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray", domain=[0.03, 0.88])
    fig.update_yaxes(title="<b>y (cm)</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")

    fig.write_html("temp.html", auto_open=True)
    calc(heatmap)
    print("hi")


def calc(heatmap = None):
    # heatmap = np.loadtxt(open("heatmap_single.csv", "rb"), delimiter=",")
    his_array = np.reshape(heatmap,
                           (heatmap.shape[0] * heatmap.shape[1],))
    mean_ = np.mean(his_array)
    std = np.std(his_array)
    p10 = np.percentile(his_array, 10)
    p90 = np.percentile(his_array, 90)
    print(f"mean: {mean_}, std: {std},  p10: {p10}, p90: {p90}, max: {np.max(his_array)}, min: {np.min(his_array)}")
    print(f"mean: {mean_}, std: {std/mean_},  p10: {p10/mean_}, p90: {p90/mean_}, max: {np.max(his_array)/mean_}, "
          f"min: {np.min(his_array)/mean_}")


if __name__ == "__main__":
    main()
    # calc()
