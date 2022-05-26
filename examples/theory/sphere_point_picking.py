
import numpy as np
import plotly.graph_objs as go
from scipy.interpolate import interp1d

from raytracepy.core import spherical_to_cartesian, get_phi
from raytracepy.light_plane_funcs import sphere_correction, theta_func_selector


def map_number(old_value, old_min, old_max, new_min, new_max):
    return interp1d([old_min, old_max], [new_min, new_max])(old_value)


def main():
    n = 10000
    r = 1
    phi = np.random.uniform(low=0, high=2*np.pi, size=(n,))
    theta = np.random.uniform(low=0, high=1, size=(n,))
    theta_corr = sphere_correction(theta)
    theta = map_number(theta, 0, 1, -np.pi, np.pi)
    theta_corr = map_number(theta_corr, 0, 1, 0, np.pi/2)

    xyz = spherical_to_cartesian(theta, phi, r)
    xyz_corr = spherical_to_cartesian(theta_corr, phi, r)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xyz[:,0], y=xyz[:, 1], z=xyz[:,2], mode="markers", marker=dict(size=1, color="black")))
    fig.write_html("temp.html", auto_open=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter3d(x=xyz_corr[:, 0], y=xyz_corr[:, 1], z=xyz_corr[:,2], mode="markers", marker=dict(size=1, color="black")))
    fig2.write_html("temp2.html", auto_open=True)


if __name__ == "__main__":
    main()
