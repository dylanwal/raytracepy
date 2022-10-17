"""
After running 'inverse_sim.py' this file takes the '*.pickle' data files and processes them.
It calculates radial distribution functions.

"""


import glob

import numpy as np

import raytracepy as rpy

# load pickle files
_dir = r".\*.pickle"
sims = [rpy.RayTrace.load_data(file) for file in sorted(glob.glob(_dir))]

# calculate radial distribution functions
rdf = None
count = 0
for sim in sims:
    x, y = sim.planes["ground"].rdf(bins=30)
    if rdf is None:
        rdf = np.zeros((x.size, len(sims)+1))
        rdf[:, 0] = x
        count += 1
    rdf[:, count] = y
    count += 1
np.savetxt("rdf_inverse2.csv", rdf, delimiter=",")  # save result as csv

# calculate hits within circle
for sim in sims:
    print(f"{sim.lights[0].position[2]},{sim.planes[0].hits_in_circle(np.array([0,0]), 0.1)}")
