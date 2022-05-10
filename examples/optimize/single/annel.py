
from scipy import optimize

from setup_problem import RayTraceProblem


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

    problem = RayTraceProblem(domain)

    results = optimize.dual_annealing(problem._evaluate, problem.min_max, maxiter=15)

    problem.df.to_csv("anneal_data.csv")
    print("hi")


if __name__ == "__main__":
    main()
