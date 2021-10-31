number_type = "float64"


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


def default_plot_layout(fig):
    fig.update_layout(autosize=False, width=900, height=790, showlegend=False,
                      font=dict(family="Arial", size=18, color="black"), plot_bgcolor="white")
    fig.update_xaxes(title="<b>X<b>", tickprefix="<b>", ticksuffix="</b>", showline=True, linewidth=0, mirror=True,
                     linecolor='black', showgrid=False, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title="<b>Y<b>", tickprefix="<b>", ticksuffix="</b>", showline=True, mirror=True, linewidth=0,
                     linecolor='black', showgrid=False, gridwidth=1, gridcolor='lightgray')
    fig.write_html('temp.html', auto_open=True)  # fig.show()


from .light_layouts import CirclePattern, GridPattern, OffsetGridPattern
from .light import Light
from .plane import Plane
from .raytrace import RayTrace

__all__ = ["CirclePattern", "GridPattern", "OffsetGridPattern", "Light", "Plane", "RayTrace"]
