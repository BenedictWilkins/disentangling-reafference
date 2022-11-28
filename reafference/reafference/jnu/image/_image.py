

import numpy as np

from IPython.display import display
from ipycanvas import Canvas

from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

import ipywidgets
import skimage
import skimage.transform

from .. import utils

from . import transform as T

__all__ = ('resize', 'hgallery', 'image', 'images')

# used to convert images to the correct display format...
FORMAT_AXES = "CHW"
FORMAT_COLOUR = "RGB"
FORMAT_DTYPE = np.float32
FORMAT_RANGE = (0,1)

@utils.as_numpy
def image(image, scale=1, show=True):
    image_widget = Image(image, scale=scale)
    if show:
        image_widget.display()
    return image_widget

@utils.as_numpy
def images(images, scale=1, on_interact=lambda x: None, step=0, value=0, show=True):
    image_widget = Image(images[step], scale=scale)
    
    # make it easy to display list meta data
    if hasattr(on_interact, '__getitem__'):
        l = on_interact
        def list_on_interact(z):
            print("value:", l[z]) #this only works with later version of ipython?
        on_interact = list_on_interact

    def slide(x):
        image_widget.update(images[x])
        on_interact(x)

    ipywidgets.interact(slide, x=ipywidgets.IntSlider(min=0, max=len(images)-1, step=step + 1, value=value, layout=dict(width='99%'))) #width 100% makes a scroll bar appear...?
    if show:
        image_widget.display()
    return image_widget

@utils.as_numpy
def resize(image, size):
    return skimage.transform.resize(image, size, order=0, preserve_range=True)

@utils.as_numpy
def hgallery(x, n=10):
    # must be in HWC format


    if n is None:
        n = x.shape[0]
    m,h,w,c = x.shape
    n = min(m, n) #if n is larger, just use m
    if m % n != 0:
        pad = ((0, n - (m % n)),*([(0,0)]*(len(x.shape)-1)))
        x = np.pad(x, pad)
        m,h,w,c = x.shape
    return x.swapaxes(1,2).reshape(m//n, w * n, h, c).swapaxes(1,2)


class Image:

    def __init__(self, image, scale=1):
        self.scale = scale
        self.image = self.format(image)
    
        self.canvas = Canvas(width=self.image.shape[1], height=self.image.shape[0], scale=1)
        self.canvas.put_image_data(self.image, 0, 0) 

    @utils.as_numpy
    def format(self, image):
        # convert the image to HWC [0-255] format
        if len(image.shape) == 2:
            image = image[np.newaxis,...]

        assert len(image.shape) == 3

        if image.shape[0] in [3,1] and not image.shape[-1] in [3,1]: 
            # in CHW format, otherwise we might be in HWC format...
            image = T.transpose(image, "CHW", "HWC")

        assert image.shape[-1] in [1,3] # TODO alpha blend if 4? 

        if T.is_float(image): # convert to [0-255], ipycanvas has a weird image format
            image = image * 255. # assume
        image = image.astype(np.float32)

        if self.scale != 1: # match image scale
            shape = list(image.shape)
            shape[0] *= self.scale
            shape[1] *= self.scale
            image = skimage.transform.resize(image, shape, order=0, preserve_range=True)

        if image.shape[-1] == 1: # convert to 3 channel image
            image = np.repeat(image, 3, axis=-1)

        return image

    def update(self, image):
        image = self.format(image)
        self.canvas.put_image_data(image, 0, 0) 

    def display(self): # TODO update
        box_layout = ipywidgets.Layout(display='flex',flex_flow='row',align_items='center',width='100%')
        display(ipywidgets.HBox([self.canvas], layout=box_layout))


