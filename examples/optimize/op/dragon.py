
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
            if var["type"] == "float":
                if var["min"] == var["max"]:
                    set_args["name"] = var["min"]
                else:
                    n_var += 1
                    xl.append(var["min"])
                    xu.append(var["max"])
                    indep_args.append(var)
            elif var["type"] == "int":
                set_args["name"] = var["min"]
            elif var["type"] == "discrete":
                set_args["name"] = var["items"]

        return n_var, xl, xu, indep_args, set_args

    def _evaluate(self, x: np.ndarray, *args, **kwargs):
        if len(x.shape) == 1:
            sim = self.function({**self._array_to_kwargs(x), **self.set_args})
            return self.metric(sim)
        else:
            results = []
            for row in x:
                sim = self.function({**self._array_to_kwargs(row), **self.set_args})
                results.append(self.metric(sim))
            return np.array(results)

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
        return -mean_, std


def main():
    # Define variables
    domain = [
        # continuous
        {"name": "number_lights", "type": "int", "min": 25, "max": 25},
        {"name": "light_height", "type": "float", "min": 1, "max": 15},
        {"name": "light_width", "type": "float", "min": 5, "max": 15},
        {"name": "mirror_offset", "type": "float", "min": 0.1, "max": 10},
        # discrete (always put last)
        {'name': 'grid_type', 'type': 'discrete', 'items': 'ogrid'},  # ['circle', 'ogrid', 'grid', 'spiral']
    ]

    problem = RayTraceProblem(domain=domain)

    # Optimize
    result = dragonfly.multiobjective_minimize_functions(
        funcs=(problem, 2),
        domain=[[lb, ub] for lb, ub in zip(problem.xl, problem.xu)],
        max_capital=25
    )

    print("hi")


if __name__ == "__main__":
    main()
