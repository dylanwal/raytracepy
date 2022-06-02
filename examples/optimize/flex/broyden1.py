from scipy import optimize
from setup_simulation import simulation

counter = 0
def func(args):
    global counter
    kwargs = dict(

        num_rays=100_000,
        number_lights=16,
        mirrors=False,
        grid_type='ogrid',
        param_names=["light_height", "light_width"])

    mean_, std, p10, p90 = simulation(args, **kwargs)
    metric = std / mean_
    print(f"{counter}{args}  ->  {[mean_,std,metric]}")
    counter += 1
    return [metric]

def fun(x):
    global counter
    print(counter)
    return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
            0.5 * (x[1] - x[0])**3 + x[1]]

sol = optimize.shgo(func, [[1, 15], [5, 15]], iters=2, n=2, options=dict(maxfev=27,  local_iter=2,
                                                                         minimize_every_iter=True))
print(sol)
print("hi")
