"""
Single Light over a horizontal plane
"""
import numpy as np

import raytracepy as rpy


def main():
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=20,
        width=20,
    )

    # define lights
    light = rpy.Light(
        position=np.array([0, 0, 5], dtype='float64'),
        direction=np.array([0, 0, -1], dtype='float64'),
        num_traces=100,
        theta_func=0
    )

    # Create sim and run it
    sim = rpy.RayTrace(
        planes=ground,
        lights=light,
        total_num_rays=3000000
    )
    sim.run()

    # Analyze/plot output
    ground.plot_heat_map()
    sim.print_stats()
    ground.print_hit_stats()
    ground.print_hit_stats(True)
    light.plot_sphere_dist()
    # sim.plot_traces()

    def func(x, c: float = 1, r: float = 1):
        return c / (x**2 + r**2)

    import plotly.graph_objs as go
    # from scipy.optimize import curve_fit
    # popt, pcov = curve_fit(func, _x, hist)
    # print(popt)

    fig = go.Figure()
    ground.plot_add_rdf(fig, bins=20, normalize=True)
    ground.plot_add_hits_x(fig, bins=20, normalize=True)
    x = np.linspace(0, 10, 11)
    fig.add_trace(go.Scatter(x=x, y=func(x, 1, 5)/0.04, mode="lines"))
    fig.write_html(f'temp_s.html', auto_open=True)

    # sim.save_data(file_name="single_3M")
    print("hi")


if __name__ == "__main__":
    main()
