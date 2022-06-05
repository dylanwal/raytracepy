
import flex_optimization as fo

import raytracepy as rpy
from setup_simulation import simulation

rpy.config.single_warning = True


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
            fo.ContinuousVariable(name="number_lights", min_=10, max_=100),
            # fo.DiscreteVariable(name='grid_type', items=['circle', 'ogrid', 'grid', 'spiral'])
        ],
        kwargs=dict(
            num_rays=100_000,
            # number_lights=16,
            mirrors=False,
            grid_type='spiral',
            param_names=["light_height", "light_width", "number_lights"]
        ),
        metric=metric,
        pass_kwargs=False
    )

    method = fo.methods.MethodFactorial(
        problem=problem,
        levels = 10, multiprocess=True)
        # x0 = [10, 10],
        # stop_criterion=fo.stop_criteria.StopFunctionEvaluation(27))

    # run single
    # method.run()
    method.recorder.df.to_csv("Factorial_mirror.csv")
    # vis = fo.VizOptimization(method.recorder)
    # fig = vis.plot_4d_vis()
    # fig.write_html("temp.html", auto_open=True)

    # run multiple
    # batch = Batch(method, num_repeat=10)
    # batch.run()


if __name__ == '__main__':
    main()
