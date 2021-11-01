"""
Single Light over a horizontal plane
"""
import raytracepy as rpy

import numpy as np


def main():
    file_name = r"C:\Users\nicep\Desktop\Reseach_Post\Case_studies\raytracepy\examples\data_array_10high_20wide.pickle"
    sim = rpy.RayTrace.load_data(file_name)
    grid = rpy.GridPattern(
        center=np.array([0, 0]),
        x_length=17,
        y_length=11.5,
        x_count=10, y_count=10)
    sim.planes["ground"].plot_sensor(grid.xy_points, 1, normalize=True)
    sim.planes["ground"].plot_heat_map()
    sim.planes["ground"].plot_heat_map( zsmooth="best")

    print(sim)


if __name__ == "__main__":
    main()
