
import glob

import numpy as np
import plotly.graph_objs as go

import raytracepy as rpy


_dir = r"C:\Users\nicep\Desktop\Reseach_Post\Case_studies\raytracepy\examples\inverse\*"

sims = [rpy.RayTrace.load_data(file) for file in sorted(glob.glob(_dir))]

rdf = None
count = 0
for sim in sims:
    x, y = sim.planes["ground"].rdf(bins=100)
    if rdf is None:
        rdf = np.zeros((x.size, len(sims)+1))
        rdf[:, 0] = x
        count += 1
    rdf[:, count] = y
    count += 1

np.savetxt("rdf.csv", rdf, delimiter=",")


for sim in sims:
    print(f"{sim.lights[0].position[2]},{sim.planes[0].hits_in_circle(np.array([0,0]), 0.2)}")

print("hi")
