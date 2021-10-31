"""
Single Light over a horizontal plane
"""
import raytracepy as rpy

import numpy as np


def main():
    # define planes
    ground = rpy.Plane(
        name="ground",
        position=np.array([0, 0, 0], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=10,
        width=10)
    diffuser = rpy.Plane(
        name="ground",
        trans_type="transmit",
        position=np.array([0, 0, 4], dtype='float64'),
        normal=np.array([0, 0, 1], dtype='float64'),
        length=20,
        width=20)

    planes = [diffuser, ground]

    # define lights
    grid = rpy.OffsetGridPattern(
        center=np.array([0, 0]),
        x_length=12.5,
        y_length=12.5,
        num_points=50)

    height = 5
    lights = []
    for xy_pos in grid.xy_points:
        lights.append(
            rpy.Light(
                position=np.array(xy_pos + [height], dtype='float64'),
                direction=np.array([0, 0, -1], dtype='float64'),
                emit_light_fun_id=1
            ))

    # Create ref_data class
    data = rpy.RayTraceData(
        planes=ground,
        lights=lights,
        num_rays=3_000_000
    )

    # pass setup/ref_data object to the simulation; then run the simulation
    sim = rpy.RayTrace(data)
    sim.run()

    # 5) Analyze/plot output
    data.percentile_table()
    data.percentile_table(normalized=True)
    data.plot_hist()


if __name__ == "__main__":
    main()
