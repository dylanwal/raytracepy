"""
Single Light over a horizontal plane
"""
import raytracepy as rpy

import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit


def main():
    # file_name = r"C:\Users\nicep\Desktop\Reseach_Post\Case_studies\raytracepy\examples\single_30K.pickle"
    # sim_30K = rpy.RayTrace.load_data(file_name)
    file_name = r"C:\Users\nicep\Desktop\Reseach_Post\Case_studies\raytracepy\examples\single_3M.pickle"
    sim_3M = rpy.RayTrace.load_data(file_name)
    # grid = rpy.GridPattern(
    #     center=np.array([0, 0]),
    #     x_length=17,
    #     y_length=11.5,
    #     x_count=10, y_count=10)
    # sim.planes["ground"].plot_sensor(grid.xy_points, 1, normalize=True)
    # sim.planes["ground"].plot_heat_map()
    # sim.planes["ground"].plot_heat_map(zsmooth="best")
    # sim.plot_traces()

    # def func(data, a, b, c):
    #     x = data[0]
    #     y = data[1]
    #     return (a * x) + (y * b) + c
    #
    # hist = sim_3M.planes["ground"].hist
    # x = np.linspace(0, hist.shape[0]-1, hist.shape[0])
    # y = np.tile(x, hist.shape[1])
    # x = np.repeat(x, hist.shape[1])
    # hist = hist.reshape(hist.size)

    # parameters, covariance = curve_fit(func, [x, y], hist)
    # print(parameters)

    # fig = go.Figure()
    # points = go.Scatter3d(x=x, y=y, z=hist, mode="markers")
    # # fig.add_trace(points)
    #
    # surf = go.Surface(x=x, y=y, z=hist)
    # fig.add_trace(surf)
    #
    # fig.write_html("temp.html", auto_open=True)

    # sim_3M.planes["ground"].plot_rdf(bins=200)
    x, y = sim_3M.planes["ground"].rdf(bins=200)
    for i in range(x.size):
        print(f"{x[i]},{y[i]}")

    # print("hi")


if __name__ == "__main__":
    main()
