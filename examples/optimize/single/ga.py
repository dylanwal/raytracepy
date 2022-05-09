import os
import multiprocessing

import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

import raytracepy as rpy

from problem import simulation


class RayTraceProblem(Problem):
    def __init__(self, domain: list[dict], n_obj: int = 1, n_constr: int = 0, multiprocessing: bool = False):
        """

        Parameters
        ----------
        domain
        n_obj
        n_constr
        """
        n_var, xl, xu, indep_args, set_args = self.parse_domain(domain)
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.indep_args = indep_args
        self.indep_args_names = [var["name"] for var in self.indep_args]
        self.set_args = set_args
        self.function = simulation
        self.multiprocessing = multiprocessing
        self.df: pd.DataFrame = None
        self.step = 0

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

    def _evaluate(self, x: np.ndarray, out, *args, **kwargs):
        if self.multiprocessing:
            with multiprocessing.Pool(os.cpu_count() - 1) as p:
                results = p.map(self._single_evaluate, x)
        else:
            results = self._single_evaluate(x)

        results = np.array(results)
        self._add_data_to_df(x, results)
        self.step += 1
        out["F"] = results[:, 1] - results[:, 0]

        # out["G"] =

    def _single_evaluate(self, x):
        results = []
        for row in x:
            sim = self.function({**self._array_to_kwargs(row), **self.set_args})
            results.append(self.metric(sim))

        return results

    def _add_data_to_df(self, x: np.ndarray, results: np.ndarray):
        step = np.ones(x.shape[0]) * self.step
        df = pd.DataFrame(np.column_stack((x, results, step)),
                          columns=self.indep_args_names + ["mean", "std", "step"])
        if self.df is None:
            self.df = df
        else:
            self.df = self.df.append(df)

    def _array_to_kwargs(self, x: np.ndarray) -> dict:
        return {k: v for k, v in zip(self.indep_args_names, x)}

    @staticmethod
    def metric(sim: rpy.RayTrace):
        # calculate dependent parameters
        histogram = sim.planes["ground"].histogram
        his_array = np.reshape(histogram.values,
                               (histogram.values.shape[0] * histogram.values.shape[1],))

        mean_ = np.mean(his_array)  # /(sim.total_num_rays / sim.planes["ground"].bins[0] ** 2)
        std = np.std(his_array)  # 100-np.std(his_array)
        return mean_, std


def main():
    # Define variables
    domain = [
        # continuous
        {"name": "num_rays", "type": "int", "value": 100_000},
        {"name": "number_lights", "type": "int", "value": 16},
        {"name": "mirrors", "type": "bool", "value": True},
        {"name": "light_height", "type": "float", "min": 1, "max": 15},
        {"name": "light_width", "type": "float", "min": 5, "max": 15},
        {"name": "mirror_offset", "type": "float", "min": 0.1, "max": 10},
        # discrete (always put last)
        {'name': 'grid_type', 'type': 'discrete', 'items': 'ogrid'},  # ['circle', 'ogrid', 'grid', 'spiral']
    ]

    problem = RayTraceProblem(domain, n_obj=2)
    algorithm = GA(pop_size=5, eliminate_duplicates=True)

    res = minimize(problem, algorithm,
                   termination=SingleObjectiveDefaultTermination(n_max_evals=25),
                   seed=1,
                   verbose=True)

    problem.df.to_csv("ga_data.csv")
    print("hi")


if __name__ == "__main__":
    main()
