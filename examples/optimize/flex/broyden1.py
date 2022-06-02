from scipy import optimize
from setup_simulation import simulation


def func(args):
    param = dict(
                light_height=args[0],
        light_width=args[1],
        mirror_offset=args[2],
    )
    kwargs = dict(

        num_rays=100_000,
        number_lights=16,
        mirrors=False,
        grid_type='ogrid',
        param_names=["light_height", "light_width", "mirror_offset"])

    mean_, std, p10, p90 = simulation(args, **kwargs)
    print(f"{args}  ->  {[mean_,std,std/mean_]}")
    return std / mean_


sol = optimize.broyden1(func, [8, 10, 5], maxiter=27)
print(sol)
print("hi")
