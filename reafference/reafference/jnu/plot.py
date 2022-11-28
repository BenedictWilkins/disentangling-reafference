
from IPython.display import display
from IPython.display import HTML

from matplotlib import pyplot as plt
from scipy.spatial import KDTree
import numpy as np

from . import image as J_image
    
def scatter_image(data, images, image_scale=1, **kwargs):
    assert len(data.shape) == 2
    assert data.shape[0] == images.shape[0]
    assert data.shape[1] == 2
   
    fig = plt.figure()
    plt.scatter(*data.T, **kwargs)
    
    image_widget = J_image(images[0], scale=image_scale)
    tree = KDTree(data)

    def on_hover(event):
        _, i = tree.query(np.array([event.xdata, event.ydata]))
        image_widget.update(images[i])
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    plt.show()
    return fig, image_widget