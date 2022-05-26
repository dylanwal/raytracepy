import numpy as np
import plotly.graph_objs as go
from scipy.interpolate import interp1d

from raytracepy.core import spherical_to_cartesian
from raytracepy.light_plane_funcs import sphere_correction


def map_number(old_value, old_min, old_max, new_min, new_max):
    return interp1d([old_min, old_max], [new_min, new_max])(old_value)


def circle_vertical(center: list | tuple | np.ndarray = (0, 0, 0), num_points: int = 100, radius: float = 1,
                    angle: float = 0):
    """

    Parameters
    ----------
    center: list|tuple|np.ndarray
        center of circle
    num_points: int
        number of points
    radius: float
        radius of circle
    angle: float
        angle of circle

    Returns
    -------
    points

    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_ = np.ones_like(theta) * center[0]
    y_ = radius * np.cos(theta) + center[1]
    x = x_ * np.cos(angle) - y_ * np.sin(angle)
    y = y_ * np.cos(angle) + x_ * np.sin(angle)
    z = radius * np.sin(theta) + center[2]

    return x, y, z


def circle_horizontal(center: list | tuple | np.ndarray = (0, 0, 0), num_points: int = 100, radius: float = 1):
    """

    Parameters
    ----------
    center: list|tuple|np.ndarray
        center of circle
        * center of sphere

    num_points: int
        number of points
    radius: float
        radius of circle
        *radius of sphere

    Returns
    -------
    points

    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    z = np.ones_like(x) * center[2]
    return x, y, z


def sphere_lines(fig, num_hor: int = 5, num_ver: int = 4, num_points: int = 100, radius: float = 1,
                 center: list | tuple | np.ndarray = (0, 0, 0)):
    offset = 0.1
    z_pos = np.linspace(-radius + offset, radius - offset, num_hor) + center[2]
    for i in range(num_hor):
        r = np.sqrt(radius ** 2 - (z_pos[i] - center[1]) ** 2) + center[0]
        x, y, z = circle_horizontal((center[0], center[1], z_pos[i]), num_points, r)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", connectgaps=True,
                                   line=dict(color="red", width=4), name="latitudes", legendgroup="latitudes"))

    angles = np.linspace(0, np.pi, num_ver + 1)[:-1]  # the last point will duplicate angle=0, so remove it
    for i in range(num_ver):
        x, y, z = circle_vertical(center, num_points, radius, angles[i])
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", connectgaps=True,
                                   line=dict(color="blue", width=4), name="longitudes", legendgroup="longitudes"))


def sphere_lines_2d(fig, num_hor: int = 5, num_ver: int = 4, num_points: int = 100, radius: float = 1,
                    center: list | tuple | np.ndarray = (0, 0, 0)):
    offset = 0.1
    z_pos = np.linspace(-radius + offset, radius - offset, num_hor) + center[2]
    for i in range(num_hor):
        r = np.sqrt(radius ** 2 - (z_pos[i] - center[1]) ** 2) + center[0]
        x, y, z = circle_horizontal((center[0], center[1], z_pos[i]), num_points, r)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", connectgaps=True,
                                 line=dict(color="red", width=2), name="latitudes", legendgroup="latitudes"))

    angles = np.linspace(0, np.pi, num_ver + 1)[:-1]  # the last point will duplicate angle=0, so remove it
    for i in range(num_ver):
        x, y, z = circle_vertical(center, num_points, radius, angles[i])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", connectgaps=True,
                                 line=dict(color="blue", width=2), name="longitudes", legendgroup="longitudes"))


def main():
    n = 10000
    r = 1
    phi = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
    theta = np.random.uniform(low=0, high=1, size=(n,))
    theta_corr = sphere_correction(theta)
    theta = map_number(theta, 0, 1, 0, np.pi)
    theta_corr = map_number(theta_corr, 0, 1, 0, np.pi / 2)

    xyz = spherical_to_cartesian(theta, phi, r=r * 0.99)
    xyz_corr = spherical_to_cartesian(theta_corr, phi, r=r * 0.99)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", marker=dict(size=1,
                                                                                                  color="black")))
    sphere_lines(fig)

    # add plot formatting
    fig.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=False, legend=dict(x=.05, y=.95))
    fig.update_xaxes(title="<b>normalized mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title="<b>std irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.write_html("temp.html", auto_open=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter3d(x=xyz_corr[:, 0], y=xyz_corr[:, 1], z=xyz_corr[:, 2], mode="markers", marker=dict(
        size=1, color="black")))
    sphere_lines(fig2)

    # add plot formatting
    fig2.update_layout(autosize=False, width=800, height=600, font=dict(family="Arial", size=18, color="black"),
                       plot_bgcolor="white", showlegend=False, legend=dict(x=.05, y=.95))
    fig2.update_xaxes(title="<b>normalized mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                      linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                      gridwidth=1, gridcolor="lightgray")
    fig2.update_yaxes(title="<b>std irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                      linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                      gridwidth=1, gridcolor="lightgray")
    fig2.write_html("temp2.html", auto_open=True)


def main2d():
    n = 15000
    r = 1
    phi = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
    theta = np.random.uniform(low=0, high=1, size=(n,))
    theta_corr = sphere_correction(theta)
    theta = map_number(theta, 0, 1, 0, np.pi)
    theta_corr = map_number(theta_corr, 0, 1, 0, np.pi / 2)

    xyz = spherical_to_cartesian(theta, phi, r=r * 0.99)
    xyz_corr = spherical_to_cartesian(theta_corr, phi, r=r * 0.99)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xyz[:, 0], y=xyz[:, 1], mode="markers", marker=dict(size=3, color="black")))
    sphere_lines_2d(fig)

    # add plot formatting
    fig.update_layout(autosize=False, width=800, height=800, font=dict(family="Arial", size=18, color="black"),
                      plot_bgcolor="white", showlegend=False)
    fig.update_xaxes(title="<b>normalized mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(title="<b>std irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                     linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                     gridwidth=1, gridcolor="lightgray")
    fig.write_html("temp.html", auto_open=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xyz_corr[:, 0], y=xyz_corr[:, 1], mode="markers", marker=dict(size=3, color="black")))
    sphere_lines_2d(fig2)

    # add plot formatting
    fig2.update_layout(autosize=False, width=800, height=800, font=dict(family="Arial", size=18, color="black"),
                       plot_bgcolor="white", showlegend=False)
    fig2.update_xaxes(title="<b>normalized mean irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                      linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                      gridwidth=1, gridcolor="lightgray")
    fig2.update_yaxes(title="<b>std irradiance</b>", tickprefix="<b>", ticksuffix="</b>", showline=True,
                      linewidth=5, mirror=True, linecolor='black', ticks="outside", tickwidth=4, showgrid=False,
                      gridwidth=1, gridcolor="lightgray")
    fig2.write_html("temp2.html", auto_open=True)


if __name__ == "__main__":
    main2d()
