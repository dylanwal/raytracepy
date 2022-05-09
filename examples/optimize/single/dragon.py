from argparse import Namespace

import numpy as np
import pandas as pd
import dragonfly

import raytracepy as rpy

from problem import simulation


class RayTraceProblem:
    def __init__(self, domain: list[dict], n_obj: int = 1, n_constr: int = 0):
        """

        Parameters
        ----------
        domain
        n_obj
        n_constr
        """
        n_var, xl, xu, indep_args, set_args = self.parse_domain(domain)
        self.xl = xl
        self.xu = xu
        self.indep_args = indep_args
        self.indep_args_names = [var["name"] for var in self.indep_args]
        self.set_args = set_args
        self.function = simulation
        self.df = pd.DataFrame(columns=self.indep_args_names + ["mean", "std", "step"])
        self.step = 0

    def __call__(self, *args, **kwargs):
        return self._evaluate(*args, **kwargs)

    @staticmethod
    def parse_domain(domain: list[dict]):
        """
        kwargs_plane = ["light_height", "mirrors", "length", "width", "box_offset"]
        kwargs_light = ["grid_type", "light_height", "light_width", "number_lights", "num_rays"]
        """
        n_var = 0
        xl = []
        xu = []
        set_args = {}
        indep_args = []
        for var in domain:
            if "value" in var:
                set_args[var["name"]] = var["value"]
                continue

            if var["type"] == "float":
                if var["min"] == var["max"]:
                    set_args[var["name"]] = var["min"]
                else:
                    n_var += 1
                    xl.append(var["min"])
                    xu.append(var["max"])
                    indep_args.append(var)
            elif var["type"] == "int":
                set_args[var["name"]] = var["min"]
            elif var["type"] == "discrete":
                set_args[var["name"]] = var["items"]

        return n_var, xl, xu, indep_args, set_args

    def _evaluate(self, x: np.ndarray, *args, **kwargs):
        sim = self.function({**self._array_to_kwargs(x), **self.set_args})
        result = self.metric(sim)
        self._add_data_to_df(x, result)
        self.step += 1
        return result

    def _array_to_kwargs(self, x: np.ndarray) -> dict:
        return {k: v for k, v in zip(self.indep_args_names, x)}

    def _add_data_to_df(self, x: np.ndarray, results):
        self.df.loc[self.step] = [i for i in x] + [i for i in results] + [self.step]

    @staticmethod
    def metric(sim: rpy.RayTrace):
        # calculate dependent parameters
        histogram = sim.planes["ground"].histogram
        his_array = np.reshape(histogram.values,
                               (histogram.values.shape[0] * histogram.values.shape[1],))

        mean_ = np.mean(his_array)  # /(sim.total_num_rays / sim.planes["ground"].bins[0] ** 2)
        std = np.std(his_array)  # 100-np.std(his_array)
        return -mean_, std


def calc_pareto_area(opt):
    xy = np.array([[row[0], row[1]] for row in opt.curr_pareto_vals])
    xy = xy[xy[:, 0].argsort()]
    return np.trapz(x=xy[:, 0], y=xy[:, 1])


def dragonfly_algorithm(problem, init_expts: int = 8, max_expts: int = 30):

    # Customizable algorithm settings
    options = Namespace(
        # batch size (number of new experiments you want to query at each iteration)
        build_new_model_every=1,
        # number of initialization experiments (-1 is included since Dragonfly generates n+1 expts)
        init_capital=init_expts - 1,
        # Criterion for tuning GP hyperparameters. Options: 'ml' (works well for smooth surfaces), 'post_sampling',
        # or 'ml-post_sampling' (algorithm default).
        gpb_hp_tune_criterion='ml-post_sampling',
        # Scalarization approach for multi-objective opt. Options: 'tchebychev or 'linear' (works well)
        moors_scalarisation='tchebychev',
    )

    # Create optimizer object
    from dragonfly.exd.worker_manager import SyntheticWorkerManager
    from dragonfly.exd.experiment_caller import CPMultiFunctionCaller
    from dragonfly.opt.multiobjective_gp_bandit import CPMultiObjectiveGPBandit
    from dragonfly import load_config
    config = load_config({'domain': problem.indep_args})
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
        y = problem._evaluate(x)  # simulate reaction
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
        y = problem._evaluate(x)
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
    result = dragonfly_algorithm(problem, init_expts=8, max_expts=25)

    problem.df.to_csv("dragon3.csv")
    print("hi")


if __name__ == "__main__":
    main()
