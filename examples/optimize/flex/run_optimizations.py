import copy

import flex_optimization as fo

import raytracepy as rpy
from setup_simulation import simulation

rpy.config.single_warning = True


class Batch:

    def __init__(self, method, num_repeat: int):
        self.method = method
        self.num_repeat = num_repeat

    def run(self, multiprocess: bool = False):
        if multiprocess:
            import multiprocessing
            with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
                pool.map(self.run_single, range(self.num_repeat))
        else:
            for i in range(self.num_repeat):
                self.run_single(i)

    def run_single(self, index_: int = 0):
        method = copy.deepcopy(self.method)
        if hasattr(method, "seed"):
            method.seed = index_
        method.run()
        method.recorder.save(f"{type(method).__name__}_no_mirror_{index_}")


def metric(args):
    mean_ = args[0]
    std = args[1]
    print("metric")
    return std/mean_


def main():
    problem = fo.Problem(
        func=simulation,
        variables=[
            fo.ContinuousVariable(name="light_height", min_=1, max_=15),
            fo.ContinuousVariable(name="light_width", min_=5, max_=15),
            fo.ContinuousVariable(name="mirror_offset", min_=0.1, max_=10),
            # fo.DiscreteVariable(name='grid_type', items=['circle', 'ogrid', 'grid', 'spiral'])
        ],
        kwargs=dict(
            num_rays=100_000,
            number_lights=16,
            mirrors=False,
            grid_type='ogrid',
            param_names=["light_height", "light_width", "mirror_offset"]
        ),
        metric=metric,
        pass_kwargs=False
    )

    method = fo.methods.MethodBFGS(
        problem=problem,
        x0 = [8, 10, 5],
        stop_criterion=fo.stop_criteria.StopFunctionEvaluation(27))

    # run single
    method.run()
    method.recorder.df.to_csv("BFGS_no_mirror_1.csv")
    vis = fo.VizOptimization(method.recorder)
    fig = vis.plot_4d_vis()
    fig.write_html("temp.html", auto_open=True)

    # run multiple
    # batch = Batch(method, num_repeat=10)
    # batch.run(multiprocess=True)


if __name__ == '__main__':
    main()
