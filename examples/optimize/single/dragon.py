from argparse import Namespace

import numpy as np
import pandas as pd

from setup_problem import RayTraceProblem


def calc_pareto_area(opt):
    xy = np.array([[row[0], row[1]] for row in opt.curr_pareto_vals])
    xy = xy[xy[:, 0].argsort()]
    return np.trapz(x=xy[:, 0], y=xy[:, 1])


def dragonfly_algorithm(problem: RayTraceProblem, init_expts: int = 8, max_expts: int = 30):

    # Customizable algorithm settings
    options = Namespace(
        # batch size (number of new experiments you want to query at each iteration)
        build_new_model_every=1,
        # number of initialization experiments (-1 is included since Dragonfly generates n+1 expts)
        init_capital=init_expts - 1,
        # Criterion for tuning GP hyperparameters. Options: 'ml' (works well for smooth surfaces), 'post_sampling',
        # or 'ml-post_sampling' (algorithm default).
        gpb_hp_tune_criterion='ml-post_sampling'
    )

    # Create optimizer object
    from dragonfly.exd.experiment_caller import CPFunctionCaller
    from dragonfly.opt.gp_bandit import CPGPBandit
    from dragonfly import load_config
    config = load_config({'domain': problem.indep_args})
    func_caller = CPFunctionCaller(None, config.domain, domain_orderings=config.domain_orderings)
    opt = CPGPBandit(func_caller, 'default', ask_tell_mode=True, options=options)
    opt.initialise()  # this generates initialization points

    # Initialization phase (initial LHS space-filling design)
    for j in range(init_expts):
        x = opt.ask()  # get point to evaluate
        y = -1 * problem._evaluate(x)  # simulate reaction
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
        y = -1 * problem._evaluate(x)
        opt.step_idx += 1
        opt.tell([(x, y)])
        print("iter:", opt.step_idx, "x:", x, ", y:", y)

    return opt.history


def main():
    # Define variables
    domain = [
        # continuous
        {"name": "num_rays", "type": "int", "value": 100_000},
        {"name": "number_lights", "type": "int", "min": 16, "max": 16},
        {"name": "mirrors", "type": "bool", "value": True},
        {"name": "light_height", "type": "float", "min": 1, "max": 15},
        {"name": "light_width", "type": "float", "min": 5, "max": 15},
        {"name": "mirror_offset", "type": "float", "min": 0.1, "max": 10},
        # discrete (always put last)
        {'name': 'grid_type', 'type': 'discrete', 'items': 'ogrid'},  # ['circle', 'ogrid', 'grid', 'spiral']
    ]

    problem = RayTraceProblem(domain=domain)

    # Optimize
    result = dragonfly_algorithm(problem, init_expts=10, max_expts=20)

    problem.df.to_csv("dragon.csv")
    print("hi")


if __name__ == "__main__":
    main()
