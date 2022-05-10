
import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

import raytracepy as rpy

from problem import simulation


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
