import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit


# def func(h, c):
#     return c/h * np.arctan(0.2/h)
#
#
# height = np.array([
#     15,
#     10,
#     7,
#     5,
#     3.16227766,
#     2,
#     1.414213562,
#     1.154700538,
#     1,
# ]
# )
#
# hits = np.array([
#     7557,
#     11448,
#     16168,
#     22769,
#     36597,
#     57146,
#     81026,
#     99178,
#     114890
#
# ]
# )
#
#
# popt, pcov = curve_fit(func, height, hits)
# print(popt)
# print(pcov)
# print(np.sqrt(np.diag(pcov)))
#
# scatter = go.Scatter(x=height, y=hits, mode="markers")
# fig = go.Figure(scatter)
#
# fit = go.Scatter(x=height, y=func(height, *popt), mode="lines")
# fig.add_trace(fit)
#
#
# fig.write_html(f'temp.html', auto_open=True)


scipy.integrate.dblquad()

