
import os
import sys
import pathlib

import inspect
from inspect import getframeinfo
import io
from contextlib import redirect_stdout

from IPython.display import display, clear_output

import ipywidgets
from ipywidgets import Text, HTML

from .image import *

from . import utils

from . import plot
from .plot import scatter_image




def table(data):
    display(HTML(
    '<table><tr>{}</tr></table>'.format(
        '</tr><tr>'.join(
            '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
        )
    ))



def progress(iterator, length=None, info=None):
    if info is not None:
        print(info)

    if length is None:
        try:
            length = len(iterator)
        except:
            print("Failed determine length of iterator, progress bar failed to display. Please provide the 'length' argument.")
            for i in iterator:
                yield i
            return

    f = ipywidgets.IntProgress(min=0, max=length, step=1, value=0) # instantiate the bar
    display(f)

    for i in iterator:
        yield i
        f.value += 1

def local_import(level=0):
    """ 
        Allow importing local .py files.
    """
    
    if not level:
        path = pathlib.Path(".").absolute()
    else:
        path = pathlib.Path(level * "../").absolute() 
    # absolute doesnt seem to get rid of ../ ? 
    module_path = str(path)
    if module_path not in sys.path:
        sys.path.append(module_path)

def cell_variables():
    """ 
    Get all of the (global) variables in the current (or previous) Jupyter Notebook cell.

    Returns:
        dict: all global variables in the cell.
    """
    ipy = get_ipython()
    out = io.StringIO()
    
    with redirect_stdout(out): #get all cell inputs
        ipy.magic("history {0}".format(ipy.execution_count))
    cell_inputs = out.getvalue()

    #get caller globals ---- LOL HACKz
    frame = inspect.stack()[1][0]
    c_line = getframeinfo(frame).lineno
    g = frame.f_globals
    if not "_" in g:
        raise ValueError("The function \"cell_variables\" must be called from within a Jupyter Notebook.")
    
    IGNORE = "#ignore"
    #process each line...
    x = cell_inputs.replace(" ", "").split("\n")
    x.pop(c_line - 1) #lines are 1 indexed, remove the calling line 
    x = [a.split("=")[0] for a in x if "=" in a and IGNORE not in a] #all of the variables in the cell
    result = {k:g[k] for k in x if k in g}

    return result

