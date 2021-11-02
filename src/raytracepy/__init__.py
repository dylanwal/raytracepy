from typing import Dict

dtype = "float64"


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


def default_plot_layout(fig,
                        layout_kwargs: Dict = None,
                        xaxis_kwargs: Dict = None,
                        yaxis_kwargs: Dict = None):
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
    if layout_kwargs:
        layout = layout | layout_kwargs
    if xaxis_kwargs:
        xaxis = xaxis | xaxis_kwargs
    if yaxis_kwargs:
        yaxis = yaxis | yaxis_kwargs

    fig.update_layout(layout)
    fig.update_xaxes(xaxis)
    fig.update_yaxes(yaxis)
    fig.write_html('temp.html', auto_open=True)  # fig.show()


from .light_layouts import CirclePattern, GridPattern, OffsetGridPattern
from .light import Light
from .plane import Plane
from .raytrace import RayTrace

__all__ = ["CirclePattern", "GridPattern", "OffsetGridPattern", "Light", "Plane", "RayTrace"]
