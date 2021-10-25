"""
This is the enter point into the simulation.

There are several presets defined that the users can try.
Just uncomment the option of interest at the bottom of this python file.

Alternatively you can make a custom simulation by following these steps:
Steps to use the code:
1) define planes (ground, mirror, diffuser)
2) generate light pattern
3) define lights
4) create data class (to let simulation know what data you want)
5) pass planes and lights into the simulation
6) Analyze/plot output


Notes:
    numba is used in the physics engine, and requires data all arrays to be numpy arrays with dtype of "float64".
"""

import numpy as np

import physics_engine
import light_layouts


def main_single_light():
    # 1) define planes
    ground = physics_engine.Plane(position=np.array([0, 0, 0], dtype='float64'),
                                  normal=np.array([0, 0, 1], dtype='float64'),
                                  length=12, width=12)

    # 2) generate light pattern and 3) define lights
    light = physics_engine.Light(position=np.array([0, 0, 5], dtype='float64'),
                                 direction=np.array([0, 0, -1], dtype='float64'),
                                 emit_light_fun_id=1
                                 )

    # 4) Create data class
    data = physics_engine.SimData(data_planes=ground)

    # 4) pass planes and lights into the simulation
    physics_engine.main_simulation_loop(planes=ground, lights=light, data=data, num_rays=3_000_000)

    # 5) Analyze/plot output
    data.percentile_table()
    data.percentile_table(normalized=True)
    data.plot_hist()


def main_single_light_with_diffuser():
    # 1) define planes
    ground = physics_engine.Plane(position=np.array([0, 0, 0], dtype='float64'),
                                  normal=np.array([0, 0, 1], dtype='float64'),
                                  length=12, width=12)
    diffuser = physics_engine.Plane(trans_type="transmit",
                                    position=np.array([0, 0, 4], dtype='float64'),
                                    normal=np.array([0, 0, 1], dtype='float64'),
                                    length=15, width=15)
    planes = [diffuser, ground]

    # 2) generate light pattern and 3) define lights
    light = physics_engine.Light(position=np.array([0, 0, 5], dtype='float64'),
                                 direction=np.array([0, 0, -1], dtype='float64'),
                                 emit_light_fun_id=1
                                 )

    # 4) Create data class
    data = physics_engine.SimData(data_planes=ground)

    # 4) pass planes and lights into the simulation
    physics_engine.main_simulation_loop(planes=planes, lights=light, data=data, num_rays=3_000_000, bounces=1)

    # 5) Analyze/plot output
    data.percentile_table()
    data.percentile_table(normalized=True)
    data.plot_hist()


def main_single_light_mirror():
    # 1) define planes
    ground = physics_engine.Plane(position=np.array([0, 0, 0], dtype='float64'),
                                  normal=np.array([0, 0, 1], dtype='float64'),
                                  length=12, width=12)
    cube_dim = 9
    mirror_left = physics_engine.Plane(trans_type="reflect",
                                       reflect_fun_id=4,
                                       position=np.array([-cube_dim / 2, 0, cube_dim / 2], dtype='float64'),
                                       normal=np.array([1, 0, 0], dtype='float64'),
                                       length=cube_dim, width=cube_dim)
    mirror_right = physics_engine.Plane(trans_type="reflect",
                                        reflect_fun_id=4,
                                        position=np.array([cube_dim / 2, 0, cube_dim / 2], dtype='float64'),
                                        normal=np.array([-1, 0, 0], dtype='float64'),
                                        length=cube_dim, width=cube_dim)
    mirror_front = physics_engine.Plane(trans_type="reflect",
                                        reflect_fun_id=4,
                                        position=np.array([0, -cube_dim / 2, cube_dim / 2], dtype='float64'),
                                        normal=np.array([0, 1, 0], dtype='float64'),
                                        length=cube_dim, width=cube_dim)
    mirror_back = physics_engine.Plane(trans_type="reflect",
                                       reflect_fun_id=4,
                                       position=np.array([0, cube_dim / 2, cube_dim / 2], dtype='float64'),
                                       normal=np.array([0, -1, 0], dtype='float64'),
                                       length=cube_dim, width=cube_dim)
    pcb = physics_engine.Plane(trans_type="reflect",
                               reflect_fun_id=6,
                               position=np.array([0, 0, cube_dim], dtype='float64'),
                               normal=np.array([0, 0, -1], dtype='float64'),
                               length=cube_dim, width=cube_dim)

    planes = [mirror_left, mirror_right, mirror_front, mirror_back, pcb, ground]

    # 2) generate light pattern and 3) define lights
    light = physics_engine.Light(position=np.array([0, 0, 5], dtype='float64'),
                                 direction=np.array([0, 0, -1], dtype='float64'),
                                 emit_light_fun_id=1
                                 )

    # 4) Create data class
    data = physics_engine.SimData(data_planes=ground)

    # 4) pass planes and lights into the simulation
    physics_engine.main_simulation_loop(planes=planes, lights=light, data=data, num_rays=500_000, bounces=4)

    # 5) Analyze/plot output
    data.hit_stats()
    data.percentile_table(normalized=True)
    data.plot_hist()


def main_light_array():
    # 1) define planes
    ground = physics_engine.Plane(position=np.array([0, 0, 0], dtype='float64'),
                                  normal=np.array([0, 0, 1], dtype='float64'),
                                  length=10, width=10)

    # 2) generate light pattern and 3) define lights
    light_positions = light_layouts.point_in_rows(center=np.array([0, 0]), height=12.5, width=12.5, num_points=50,
                                                  offset_op=True)
    # light_positions = light_layouts.points_around_circle(center=np.array([0, 0]), outer_radius=6,
    #                                                      num_points=50, layers=3, center_point=True)
    height = 2.5
    light = []
    for position in light_positions:
        light.append(physics_engine.Light(position=np.hstack([position, height]),
                                          direction=np.array([0, 0, -1], dtype='float64'),
                                          emit_light_fun_id=1
                                          ))

    # 4) Create data class
    data = physics_engine.SimData(data_planes=ground)

    # 4) pass planes and lights into the simulation
    physics_engine.main_simulation_loop(planes=ground, lights=light, data=data, num_rays=3_000_000)

    # 5) Analyze/plot output
    data.percentile_table()
    data.percentile_table(normalized=True)
    data.plot_hist()


def main_light_array_mirror():
    # 1) define planes
    ground = physics_engine.Plane(position=np.array([0, 0, 0], dtype='float64'),
                                  normal=np.array([0, 0, 1], dtype='float64'),
                                  length=12, width=12)
    cube_dim = 12
    mirror_left = physics_engine.Plane(trans_type="reflect",
                                       reflect_fun_id=4,
                                       position=np.array([-cube_dim / 2, 0, cube_dim / 2], dtype='float64'),
                                       normal=np.array([1, 0, 0], dtype='float64'),
                                       length=cube_dim, width=cube_dim)
    mirror_right = physics_engine.Plane(trans_type="reflect",
                                        reflect_fun_id=4,
                                        position=np.array([cube_dim / 2, 0, cube_dim / 2], dtype='float64'),
                                        normal=np.array([-1, 0, 0], dtype='float64'),
                                        length=cube_dim, width=cube_dim)
    mirror_front = physics_engine.Plane(trans_type="reflect",
                                        reflect_fun_id=4,
                                        position=np.array([0, -cube_dim / 2, cube_dim / 2], dtype='float64'),
                                        normal=np.array([0, 1, 0], dtype='float64'),
                                        length=cube_dim, width=cube_dim)
    mirror_back = physics_engine.Plane(trans_type="reflect",
                                       reflect_fun_id=4,
                                       position=np.array([0, cube_dim / 2, cube_dim / 2], dtype='float64'),
                                       normal=np.array([0, -1, 0], dtype='float64'),
                                       length=cube_dim, width=cube_dim)
    pcb = physics_engine.Plane(trans_type="reflect",
                               reflect_fun_id=6,
                               position=np.array([0, 0, 15], dtype='float64'),
                               normal=np.array([0, 0, -1], dtype='float64'),
                               length=cube_dim, width=cube_dim)

    planes = [mirror_left, mirror_right, mirror_front, mirror_back, pcb, ground]

    # 2) generate light pattern and 3) define lights
    light_positions = light_layouts.point_in_rows(center=np.array([0, 0]), height=12.5, width=12.5, num_points=50,
                                                  offset_op=True)
    # light_positions = light_layouts.points_around_circle(center=np.array([0, 0]), outer_radius=6,
    #                                                      num_points=50, layers=3, center_point=True)
    height = 15
    lights = []
    for position in light_positions:
        lights.append(physics_engine.Light(position=np.hstack([position, height]),
                                           direction=np.array([0, 0, -1], dtype='float64'),
                                           emit_light_fun_id=1
                                           ))

    # 4) Create data class
    data = physics_engine.SimData(data_planes=ground)

    # 4) pass planes and lights into the simulation
    physics_engine.main_simulation_loop(planes=planes, lights=lights, data=data, num_rays=10_000_000, bounces=5)

    # 5) Analyze/plot output
    data.percentile_table()
    data.percentile_table(normalized=True)
    data.plot_traces()
    data.plot_hist()
    data.save_data()


if __name__ == "__main__":
    # import time
    # start = time.time()

    # main_single_light()
    # main_single_light_with_diffuser()
    main_single_light_mirror()
    # main_light_array()
    # main_light_array_diffuser()
    # main_light_array_mirror()

    # print(time.time()-start)
