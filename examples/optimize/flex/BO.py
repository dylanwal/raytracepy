import numpy as np
import flex_optimization as fo
import flex_optimization.methods as fo_m
import flex_optimization.stop_criteria as fo_s

import raytracepy as rpy
from setup_simulation import simulation

fo.logger.setLevel(fo.logger.MONITOR)
rpy.config.single_warning = True


def metric(sim: rpy.RayTrace):

    return std - mean_


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
            mirrors=True,
            grid_type='ogrid',
            param_names=["light_height", "light_width", "mirror_offset"]
        ),
        metric=metric
    )

    # method = fo_m.MethodBODragon(problem=problem, stop_criteria=fo_s.StopFunctionEvaluation(27))
    # method = fo_m.MethodFactorial(problem=problem, levels=3)
    for i in range(9):
        method = fo_m.MethodSobol(problem=problem, stop_criteria=fo_s.StopFunctionEvaluation(27))
        method.run()
        method.data.to_csv(f"Sobol{i}.csv")
        print(i)

    vis = fo.OptimizationVis(problem, method.data)
    fig = vis.plot_4d_vis()
    fig.write_html("temp.html", auto_open=True)


if __name__ == '__main__':
    main()
