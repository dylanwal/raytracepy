
from scipy import optimize

import problem


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

    prob = problem.RayTraceProblem(domain, n_obj=2)

    results = optimize.shgo(prob._evaluate, [[1, 15], [5, 15], [0.1, 10]], options=dict(maxfev=10))

    prob.df.to_csv("scipy_data.csv")
    print("hi")


if __name__ == "__main__":
    main()
