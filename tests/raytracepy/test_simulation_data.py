if __name__ == "__main__":
    import physics_engine, light_layouts

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
                               position=np.array([0, 0, 15], dtype='float64'),
                               normal=np.array([0, 0, -1], dtype='float64'),
                               length=cube_dim, width=cube_dim)

    planes = [mirror_left, mirror_right, mirror_front, mirror_back, pcb, ground]

    # 2) generate light pattern and 3) define lights
    light_positions = light_layouts.point_in_rows(center=np.array([0, 0]), height=12.5, width=12.5, num_points=50,
                                                  offset_op=True)
    height = 15
    lights = []
    for position in light_positions:
        lights.append(physics_engine.Light(position=np.hstack([position, height]),
                                           direction=np.array([0, 0, -1], dtype='float64'),
                                           emit_light_fun_id=1
                                           ))

    traces = [
        np.array([
            [0, 0, 15],
            [1, 1, 5],
            [2, 4, 0]
        ]),
        np.array([
            [0, 0, 15],
            [-1, -1, 5],
            [2, -4, 0]
        ]),
        np.array([
            [0, 0, 15],
            [1, 6, 5],
            [2, 7, 0]
        ])
    ]
    plot_traces(planes=planes, lights=lights, traces=traces)