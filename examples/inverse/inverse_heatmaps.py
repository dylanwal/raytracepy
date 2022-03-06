"""
Heatmaps for Inverse Square Law

"""

import glob

import raytracepy as rpy
from examples.plotting_tools import heatmap_array


def main():
    # load pickle files
    _dir = r".\*.pickle"
    sims = [rpy.RayTrace.load_data(file) for file in sorted(glob.glob(_dir))]
    print("data loaded")

    # heatmap plot
    res = 300
    plot_titles = ["h = 1 cm", "h = 4 cm", "h = 7 cm", "h = 10 cm", "h = 13 cm", "h = 16 cm"]
    x_range = (-10, 10)
    y_range = (-10, 10)
    title = "Heatmaps for Inverse Square Law"
    file_name: str = "inverse_heatmap.html"

    data = [sim.planes[0].hits[:, :2] for sim in sims]

    heatmap_array(data, plot_titles, x_range, y_range, res, title, file_name)
    print("done")


if __name__ == '__main__':
    main()
