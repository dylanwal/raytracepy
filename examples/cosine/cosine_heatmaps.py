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
    plot_titles = [" 0 radian", " 0.11 radian", "0.22 radian", "0.34 radian", "0.46 radian", "0.59 radian",
                   "0.73 radian", "0.89 radian", "1.10 radian", "1.57 radian"]
    x_range = (-10, 10)
    y_range = (-10, 10)
    title = "Heatmaps for Cosine Law"
    file_name: str = "cosine_heatmap.html"

    data = [sim.planes[0].hits[:, :2] for sim in sims]

    heatmap_array(data, plot_titles, x_range, y_range, res, title, file_name)
    print("done")


if __name__ == '__main__':
    main()
