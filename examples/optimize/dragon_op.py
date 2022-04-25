import pickle
import datetime

import numpy as np
import pandas as pd

from argparse import Namespace
from dragonfly import load_config
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.exd.experiment_caller import CPMultiFunctionCaller
from dragonfly.opt.multiobjective_gp_bandit import CPMultiObjectiveGPBandit

import raytracepy as rpy


def define_planes(light_height: float = 5, mirrors: bool = False, length: float = 10, width: float = 10, **kwargs) \
        -> list[rpy.Plane]:
    if mirrors:
        ground = rpy.Plane(
            name="ground",
            position=np.array([0, 0, 0], dtype='float64'),
            normal=np.array([0, 0, 1], dtype='float64'),
            length=length,
            width=width,
            transmit_type="absorb",
            bins=(100, 100)
        )

        if 'light_width' in kwargs and "box_offset" in kwargs:
            # mirror box can't be smaller than the area of interest
            box_dim = max([width, kwargs["light_width"] + kwargs["box_offset"]])
        else:
            box_dim = width

        box_height = light_height + 0.25
        mirror_left = rpy.Plane(
            name="mirror_left",
            position=np.array([-box_dim / 2, 0, box_height / 2], dtype='float64'),
            normal=np.array([1, 0, 0], dtype='float64'),
            length=box_dim,
            width=box_height,
            transmit_type="reflect",
            reflect_func=4,
        )
        mirror_right = rpy.Plane(
            name="mirror_right",
            position=np.array([box_dim / 2, 0, box_height / 2], dtype='float64'),
            normal=np.array([-1, 0, 0], dtype='float64'),
            length=box_dim,
            width=box_height,
            transmit_type="reflect",
            reflect_func=4,
        )
        mirror_front = rpy.Plane(
            name="mirror_front",
            position=np.array([0, -box_dim / 2, box_height / 2], dtype='float64'),
            normal=np.array([0, 1, 0], dtype='float64'),
            length=box_dim,
            width=box_height,
            transmit_type="reflect",
            reflect_func=4,
        )
        mirror_back = rpy.Plane(
            name="mirror_back",
            position=np.array([0, box_dim / 2, box_height / 2], dtype='float64'),
            normal=np.array([0, -1, 0], dtype='float64'),
            length=box_dim,
            width=box_height,
            transmit_type="reflect",
            reflect_func=4,
        )
        top = rpy.Plane(
            name="top",
            position=np.array([0, 0, box_height], dtype='float64'),
            normal=np.array([0, 0, -1], dtype='float64'),
            length=box_dim,
            width=box_dim,
            transmit_type="reflect",
            reflect_func=6,
        )
        planes = [top, mirror_left, mirror_right, mirror_front, mirror_back, ground]

    else:
        ground = rpy.Plane(
            name="ground",
            position=np.array([0, 0, 0], dtype='float64'),
            normal=np.array([0, 0, 1], dtype='float64'),
            length=10,
            width=10,
            transmit_type="absorb",
            bins=(100, 100)
        )
        planes = [ground]

    return planes


def define_lights(grid_type: str = "ogrid", light_height: float = 5, light_width: float = 10, number_lights: int = 25,
                  **kwargs) -> list[rpy.Light]:
    if grid_type == "circle":
        grid = rpy.CirclePattern(
            center=np.array([0, 0]),
            outer_radius=light_width / 2,
            layers=3,
            num_points=number_lights)
    elif grid_type == "ogrid":
        grid = rpy.OffsetGridPattern(
            center=np.array([0, 0]),
            x_length=light_width,
            y_length=light_width,
            num_points=number_lights)
    elif grid_type == "grid":
        grid = rpy.GridPattern(
            center=np.array([0, 0]),
            x_length=light_width,
            y_length=light_width,
            num_points=number_lights)
    elif grid_type == "spiral":
        grid = rpy.SpiralPattern(
            center=np.array([0, 0]),
            radius=light_width / 2,
            radius_start=.5,
            velocity=0.2,
            a_velocity=1,
            num_points=number_lights)
    else:
        raise ValueError(f"{grid_type} invalid choice.")

    # define lights
    lights = []
    for xy_pos in grid.xy_points:
        lights.append(
            rpy.Light(
                position=np.insert(xy_pos, 2, light_height),
                direction=np.array([0, 0, -1], dtype='float64'),
                num_traces=5,
                theta_func=1,
                num_rays=100_000,
            ))
    return lights


def simulation(params, domain_vars: list[dict], **kwargs):
    # parse params
    params = {name["name"]: value for value, name in zip(params, domain_vars)}
    params = params | kwargs

    # run simulation
    lights = define_lights(**params)
    planes = define_planes(**params)
    sim = rpy.RayTrace(
        planes=planes,
        lights=lights,
        bounce_max=5,
    )
    sim.run()

    # calculate dependent parameters
    histogram = sim.planes["ground"].histogram
    his_array = np.reshape(histogram.values,
                           (histogram.values.shape[0] * histogram.values.shape[1],))
    mean_ = np.mean(his_array)  # /(sim.total_num_rays / sim.planes["ground"].bins[0] ** 2)
    std = 100 - np.std(his_array)  # /(sim.total_num_rays/ sim.planes["ground"].bins[0] ** 2)

    return mean_, std


def calc_pareto_area(opt):
    xy = np.array([[row[0], row[1]] for row in opt.curr_pareto_vals])
    xy = xy[xy[:, 0].argsort()]
    return np.trapz(x=xy[:, 0], y=xy[:, 1])


def multi_objective(func: callable, config, init_expts: int = 8, max_expts: int = 30, args: dict = None):
    if args is None:
        args = {}

    # Customizable algorithm settings
    options = Namespace(
        # batch size (number of new experiments you want to query at each iteration)
        build_new_model_every=1,
        # number of initialization experiments (-1 is included since Dragonfly generates n+1 expts)
        init_capital=init_expts - 1,
        # Criterion for tuning GP hyperparameters. Options: 'ml' (works well for smooth surfaces), 'post_sampling',
        # or 'ml-post_sampling' (algorithm default).
        gpb_hp_tune_criterion='ml',
        # Scalarization approach for multi-objective opt. Options: 'tchebychev or 'linear' (works well)
        moors_scalarisation='linear',
    )

    # Create optimizer object
    func_caller = CPMultiFunctionCaller(None, config.domain, domain_orderings=config.domain_orderings)
    func_caller.num_funcs = 2  # must specify how many funcs are being optimized

    # Ask-tell interface for multi-objective opt hasn't been implemented yet by Dragonfly developers
    # This is a hack that works
    wm = SyntheticWorkerManager(1)
    opt = CPMultiObjectiveGPBandit(func_caller, wm, options=options)
    opt.ask_tell_mode = True
    opt.worker_manager = None
    opt._set_up()
    opt.initialise()  # this generates initialization points

    # Initialization phase (initial LHS space-filling design)
    for j in range(init_expts):
        x = opt.ask()  # get point to evaluate
        y = func(x, **args)  # simulate reaction
        opt.step_idx += 1  # increment experiment number
        print("iter:", opt.step_idx, ", x:", x, ", y:", y)
        opt.tell([(x, y)])  # return result to algorithm

    # Refinement phase (algorithm proposes conditions that try to maximize objective)
    refine_expts = max_expts - init_expts
    pareto_area = np.zeros(refine_expts)
    for j in range(refine_expts):
        opt._build_new_model()  # key line! update model using prior results
        opt._set_next_gp()  # key line! set next GP
        x = opt.ask()
        y = func(x, **args)
        opt.step_idx += 1
        opt.tell([(x, y)])

        area = calc_pareto_area(opt)
        print("iter:", opt.step_idx, "area", area, "x:", x, ", y:", y)
        # area_cutoff = pareto_area[pareto_area.argsort()][-3]
        # if (area - area_cutoff) / area_cutoff <= 0.02:
        #     print("area not changing")
        #     break
        pareto_area[j] = area

    return opt.history


def save_data(data, file_name: str = "data", _dir: str = None):
    """ Save data in pickle format. """
    _date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    file_name = f"{file_name}_{_date}"
    if _dir is not None:
        file_name = _dir + "\\" + file_name

    with open(file_name + '.pickle', 'wb+') as file:
        pickle.dump(data, file)


def run():
    # Define variables
    domain = [
        # continuous
        # {"name": "number_lights", "type": "int", "min": 4, "max": 81},
        {"name": "light_height", "type": "float", "min": 1, "max": 15},
        {"name": "light_width", "type": "float", "min": 5, "max": 15},
        {"name": "mirror_offset", "type": "float", "min": 0.1, "max": 10},
        # discrete (always put last)
        # {'name': 'grid_type', 'type': 'discrete', 'items': ['circle', 'ogrid', 'grid', 'spiral']},
    ]
    args = {"mirrors": True, "number_lights": 49, "grid_type": "ogrid"}
    init_expts = 6
    max_expts = 30

    # Create domain
    config = load_config({'domain': domain})


    # Optimize
    result = multi_objective(func=simulation, config=config, init_expts=init_expts, max_expts=max_expts,
                             args={**args, "domain_vars": domain})

    # save results
    result.args = args
    result.init_expts = init_expts
    result.max_expts = max_expts
    result.domain = domain
    result.domain_ordering = config.domain.raw_name_ordering
    result.value_ordering = ["mean", "std"]
    save_data(result, "results")

    print("done")


if __name__ == "__main__":
    run()
