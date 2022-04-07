import itertools
import pickle
import datetime

import numpy as np
import pandas as pd

# Modules required for Dragonfly
from argparse import Namespace
from dragonfly import load_config
from dragonfly.exd.worker_manager import SyntheticWorkerManager
from dragonfly.exd.experiment_caller import CPFunctionCaller, CPMultiFunctionCaller
from dragonfly.opt.multiobjective_gp_bandit import CPMultiObjectiveGPBandit  # for multi-objective

import raytracepy as rpy


def run_single(lights, mirrors: bool = False, height: float = 10, width: float = 10, box_offset: float = 0.5, **kwargs):
    # define planes
    if mirrors:
        ground = rpy.Plane(
            name="ground",
            position=np.array([0, 0, 0], dtype='float64'),
            normal=np.array([0, 0, 1], dtype='float64'),
            length=10,
            width=10,
            transmit_type="absorb",
            bins=(100, 100)
        )

        box_dim = width + box_offset
        box_height = height + 0.25
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

    # Create ref_data class
    sim = rpy.RayTrace(
        planes=planes,
        lights=lights,
        total_num_rays=500_000,
        bounce_max=5,
    )
    sim.run()
    return sim


def define_lights(grid_type: str = "ogrid", height: float = 5, width: float = 10, num_lights: int = 25, **kwargs) \
        -> list[rpy.Light]:
    if grid_type == "circle":
        grid = rpy.CirclePattern(
            center=np.array([0, 0]),
            outer_radius=width / 2,
            layers=3,
            num_points=num_lights)
    elif grid_type == "ogrid":
        grid = rpy.OffsetGridPattern(
            center=np.array([0, 0]),
            x_length=width,
            y_length=width,
            num_points=num_lights)
    elif grid_type == "grid":
        grid = rpy.GridPattern(
            center=np.array([0, 0]),
            x_length=width,
            y_length=width,
            num_points=num_lights)
    elif grid_type == "spiral":
        grid = rpy.SpiralPattern(
            center=np.array([0, 0]),
            radius=width / 2,
            radius_start=.5,
            velocity=0.2,
            a_velocity=1,
            num_points=num_lights)
    else:
        raise ValueError(f"{grid_type} invalid choice.")

    # define lights
    lights = []
    for xy_pos in grid.xy_points:
        lights.append(
            rpy.Light(
                position=np.insert(xy_pos, 2, height),
                direction=np.array([0, 0, -1], dtype='float64'),
                num_traces=5,
                theta_func=1,
            ))
    return lights


def simulation(params, domain_vars: list[dict], num_lights: int = 25, mirrors: bool = False, **kwargs):
    # parse params
    params = {name["name"]: value for value, name in zip(params, domain_vars)}
    params = params | kwargs

    # run simulation
    lights = define_lights(num_lights=num_lights, **params)
    sim = run_single(lights, mirrors, **params)

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
        area_cutoff = pareto_area[pareto_area.argsort()][-3]
        if (area-area_cutoff)/area_cutoff <= 0.02:
            print("area not changing")
            break
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
    domain_vars = [
        {'name': 'grid_type', 'type': 'discrete', 'items': ['circle', 'ogrid', 'grid', 'spiral']},
        {"name": "light_height", "type": "float", "min": 0.5, "max": 10},
        {"name": "light_width", "type": "float", "min": 5, "max": 20},
        # {"name": "box_offset", "type": "float", "min": 0.1, "max": 5}
    ]
    # Create domain
    config = load_config({'domain': domain_vars})

    # Simulate
    result = multi_objective(func=simulation, config=config, init_expts=6, max_expts=30,
                    args={"mirrors": False, "num_lights": 25, "objective": "multi", "domain_vars": domain_vars})

    # save results
    save_data(result, "result")

    print("done")


def grid_search():
    n = 5
    domain_vars = [
        {'name': 'grid_type', 'type': 'discrete', 'items': ['circle', 'ogrid', 'grid', 'spiral']},
        {"name": "light_height", "type": "float", "min": 0.5, "max": 10},
        {"name": "light_width", "type": "float", "min": 5, "max": 20},
        {"name": "box_offset", "type": "float", "min": 0.1, "max": 5}
    ]

    combinations = []
    for var_ in domain_vars:
        if var_["type"] == "discrete":
            combinations.append(var_["items"])
        else:
            combinations.append(list(np.linspace(var_["min"], var_["max"], n)))

    all_combinations = list(itertools.product(*combinations))

    temp = np.zeros((len(all_combinations), 2))
    df_output = pd.DataFrame(temp, columns=["mean", "std"])
    df = pd.DataFrame(all_combinations, columns=[var_["name"] for var_ in domain_vars])

    for i, row in df.iterrows():
        mean, std = simulation(list(row.values), domain_vars=domain_vars, mirrors=True)
        df_output.iloc[i] = [mean, std]
        print(i, "out of", len(all_combinations))

    # from multiprocessing import Pool
    # kwargs = {"domain_vars": domain_vars, "mirrors": True}
    # from functools import partial
    # with Pool(3) as p:
    #     output = p.map(partial(simulation, **kwargs), all_combinations)

    df = pd.concat([df, df_output], axis=1)
    print(df.head())
    df.to_csv("grid_search_mirror.csv")


if __name__ == "__main__":
    # run()
    grid_search()

