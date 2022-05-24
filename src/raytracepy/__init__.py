from raytracepy.config import config

import os
dtype = "float64"

import numba  # import all numba stuff here so it can be toggled on/off
njit = numba.njit
numba.config.DISABLE_JIT = False

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
def default_plot_layout(fig, save_open: bool = True):
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

    if save_open:
        fig.write_html(f'temp{_figure_counter}.html', auto_open=True, include_plotlyjs='cdn')  # fig.show()
        _figure_counter += 1


def merge_html_figs(figs, filename: str = "merge.html", auto_open: bool = True):
    """
    Merges plotly figures.

    Parameters
    ----------
    figs: list[go.Figure, str]
        list of figures to append together
    filename:str
        file name
    auto_open: bool
        open html in browser after creating

    """
    if filename[-5:] != ".html":
        filename += ".html"

    with open(filename, 'w') as file:
        file.write(f"<html><head><title>{filename[:-5]}</title><h1>{filename[:-5]}</h1></head><body>" + "\n")
        for fig in figs:
            if isinstance(fig, str):
                file.write(fig)
                continue

            inner_html = fig.to_html(include_plotlyjs="cdn").split('<body>')[1].split('</body>')[0]
            file.write(inner_html)

        file.write("</body></html>" + "\n")

    if auto_open:
        os.system(fr"start {filename}")


from .light_layouts import CirclePattern, GridPattern, OffsetGridPattern, SpiralPattern
from .light import Light
from .plane import Plane
from .raytrace import RayTrace

__all__ = ["CirclePattern", "GridPattern", "OffsetGridPattern", "SpiralPattern", "Light", "Plane", "RayTrace"]
