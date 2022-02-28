"""

"""
import glob

import numpy as np
import plotly.graph_objs as go

import raytracepy as rpy
import raytracepy.utils.analysis_func as raypy_a
import examples.plot_format as plot_format


def generate_figures(sim: rpy.RayTrace):
    sim.plot_traces(plane_hits="ground")
    ground = sim.planes["ground"]
    ground.plot_heat_map()
    ground.plot_rdf(bins=40, normalize=True)


def main():
    # load pickle files
    _dir = r".\*.pickle"
    sims = [rpy.RayTrace.load_data(file) for file in sorted(glob.glob(_dir))]
    print("data loaded")
    for sim in sims:
        generate_figures(sim)
        print(f"sim: {sim} done")


if __name__ == '__main__':
    main()
