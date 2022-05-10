import os
import multiprocessing

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem

import raytracepy as rpy

from setup_simulation import simulation


class RayTraceProblem(Problem):
    def __init__(self, domain: list[dict], n_obj: int = 1, n_constr: int = 0, eval_value: str = "value"):
        """

        Parameters
        ----------
        domain
        n_obj
        n_constr
        eval_value
        """
        n_var, xl, xu, indep_args, set_args, min_max = self.parse_domain(domain)
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.min_max = min_max
        self.indep_args = indep_args
        self.indep_args_names = [var["name"] for var in self.indep_args]
        self.set_args = set_args
        self.function = simulation
        self.multiprocessing = multiprocessing
        self.df = pd.DataFrame(columns=self.indep_args_names + ["mean", "std", "step"])
        self.step = 0
        self.return_value = eval_value

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
        min_max = []
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
                    min_max.append([var["min"], var["max"]])
                    indep_args.append(var)
            elif var["type"] == "int":
                set_args[var["name"]] = var["min"]
            elif var["type"] == "discrete":
                set_args[var["name"]] = var["items"]

        return n_var, xl, xu, indep_args, set_args, min_max

    def _evaluate(self, x: (list, np.ndarray), *args, **kwargs):
        # do calculations
        if (isinstance(x, np.ndarray) and len(x.shape) == 1) or (isinstance(x, list) and not isinstance(x[0], list)):
            results = self._single_evaluate(x)
            results = np.array(results)
            self._add_data_to_df(x, results)
        else:
            raise NotImplemented

        self.step += 1
        # return value
        if self.return_value == "value":
            return self.objective(results)
        elif self.return_value == "pymoo":
            args["F"] = self.objective(results)
            # out["G"] =

    def _single_evaluate(self, x):
        sim = self.function({**self._array_to_kwargs(x), **self.set_args})
        return self.metric(sim)

    def _add_data_to_df(self, x, results):
        self.df.loc[self.step] = [i for i in x] + [i for i in results] + [self.step]

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

    def objective(self, results):
        return results[1] - results[0]
