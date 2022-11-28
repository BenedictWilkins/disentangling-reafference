

import numpy as np
import skimage.transform

from types import SimpleNamespace

#interpolation = SimpleNamespace(nearest=0, bilinear=1, biquadratic=2, bicubic=3, biquartic=4, biquintic=5)
#mode = SimpleNamespace(HWC="HWC", CHW="CHW")

__all__ = ("transpose", "is_integer", "is_float", "to_float", "to_integer")


def transpose(x, f1, f2):
    """ Tranpose x to a new axes format.

    Args:
        x (np.ndarray): array to transpose
        f1 (str): axes format of x
        f2 (str): desired axes format

    Returns:
        np.ndarray: x tranposed


    Example: 
        x = np.zeros((2,4))
        y = _transpose(x, "WH", "CHW")
        y.shape
        >> (1,4,2)
    """
    f1, f2 = list(f1.lower()), list(f2.lower())
    assert len(f1) == len(x.shape) # f1 does not describe x axes
    assert all([f in f2 for f in f1]) # axis label miss-match
    if len(f1) != len(f2): # fill in missing axes
        mask = np.array([f in f1 for f in f2])
        f1_expand = np.array(f2.copy())
        f1_expand[mask] = f1
        f1 = f1_expand.tolist()
        x = np.expand_dims(x, np.arange(len(f1))[np.logical_not(mask)])
    indx = np.array([f1.index(f) for f in f2])
    return x.transpose(indx)

def is_integer(image):
    return issubclass(image.dtype.type, np.integer)

def is_float(image):
    return issubclass(image.dtype.type, np.floating)

def to_float(image):
    if is_float(image):
        return image.astype(np.float32)
    elif is_integer(image):
        return image.astype(np.float32) / 255.
    else:
        return TypeError("Invalid array type: {0} for float32 conversion.".format(image.dtype))

def to_integer(image):
    if is_integer(image):
        return image.astype(np.uint8)
    elif is_float(image):
        return (image * 255.).astype(np.uint8) 
    else:
        return TypeError("Invalid array type: {0} for uint8 conversion.".format(image.dtype))