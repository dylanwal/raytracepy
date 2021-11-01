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
    box_dim = 15
    mirror_left = rpy.Plane(
        name="mirror_left",
        trans_type="reflect",
        reflect_fun_id=4,
        position=np.array([-box_dim / 2, 0, box_dim / 2], dtype='float64'),
        normal=np.array([1, 0, 0], dtype='float64'),
        length=box_dim,
        width=box_dim)
    mirror_right = rpy.Plane(
        name="mirror_right",
        trans_type="reflect",
        reflect_fun_id=4,
        position=np.array([box_dim / 2, 0, box_dim / 2], dtype='float64'),
        normal=np.array([-1, 0, 0], dtype='float64'),
        length=box_dim,
        width=box_dim)
    mirror_front = rpy.Plane(
        name="mirror_front",
        trans_type="reflect",
        reflect_fun_id=4,
        position=np.array([0, -box_dim / 2, box_dim / 2], dtype='float64'),
        normal=np.array([0, 1, 0], dtype='float64'),
        length=box_dim,
        width=box_dim)
    mirror_back = rpy.Plane(
        name="mirror_back",
        trans_type="reflect",
        reflect_fun_id=4,
        position=np.array([0, box_dim / 2, box_dim / 2], dtype='float64'),
        normal=np.array([0, -1, 0], dtype='float64'),
        length=box_dim,
        width=box_dim)
    top = rpy.Plane(
        name="top",
        trans_type="reflect",
        reflect_fun_id=6,
        position=np.array([0, 0, box_dim], dtype='float64'),
        normal=np.array([0, 0, -1], dtype='float64'),
        length=box_dim,
        width=box_dim)

    planes = [mirror_left, mirror_right, mirror_front, mirror_back, top, ground]

    # define lights
    light = rpy.Light(
        position=np.array([0, 0, 5], dtype='float64'),
        direction=np.array([0, 0, -1], dtype='float64'),
        emit_light_fun_id=1
    )

    # Create ref_data class
    data = rpy.RayTraceData(
        planes=planes,
        lights=light,
        # num_rays=3_000_000
        bounces=4
    )

    # Create sim and run it
    sim = rpy.RayTrace(
        planes=planes,
        lights=light,
        total_num_rays=10_000
    )
    sim.run()

    # Analyze/plot output
    ground.plot_heat_map()
    sim.print_stats()
    ground.print_hit_stats()
    ground.print_hit_stats(True)
    print("hi")


if __name__ == "__main__":
    main()
