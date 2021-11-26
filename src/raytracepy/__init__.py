dtype = "float64"


from numba import config


config.DISABLE_JIT = False


from functools import wraps
from time import time


def time_it(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()
        print(f'func:{func.__name__}  took: {te-ts} s to run.')
        return result
    return wrap


object_counter: int = -1


def get_object_uid() -> int:
    """ Gives uid to objects. """
    global object_counter
    object_counter += 1
    return object_counter


_figure_counter: int = 0


def default_plot_layout(fig, **kwargs):
    global _figure_counter
    layout = {
        "autosize": False,
        "width": 900,
        "height": 790,
        "showlegend": False,
        "font": dict(family="Arial", size=18, color="black"),
        "plot_bgcolor": "white"
    }
    xaxis = {
        "title": "<b>X<b>",
        "tickprefix": "<b>",
        "ticksuffix": "</b>",
        "showline": True,
        "linewidth": 5,
        "mirror": True,
        "linecolor": 'black',
        "ticks": "outside",
        "tickwidth": 4,
        "showgrid": False,
        "gridwidth": 1,
        "gridcolor": 'lightgray'
    }
    yaxis = {
        "title": "<b>Y<b>",
        "tickprefix": "<b>",
        "ticksuffix": "</b>",
        "showline": True,
        "linewidth": 5,
        "mirror": True,
        "linecolor": 'black',
        "ticks": "outside",
        "tickwidth": 4,
        "showgrid": False,
        "gridwidth": 1,
        "gridcolor": 'lightgray'
    }
    fig.update_layout(layout)
    fig.update_xaxes(xaxis)
    fig.update_yaxes(yaxis)
    fig.write_html(f'temp{_figure_counter}.html', auto_open=True)  # fig.show()
    _figure_counter += 1


from .light_layouts import CirclePattern, GridPattern, OffsetGridPattern
from .light import Light
from .plane import Plane
from .raytrace import RayTrace

__all__ = ["CirclePattern", "GridPattern", "OffsetGridPattern", "Light", "Plane", "RayTrace"]
