""" Utils for ndarray wrangling """

import numpy as np


def _binarize_arrays(v, method: str, threshold: float) -> np.ndarray:
    """ Binarize 1 or more array-like object/s of floats to have values :math:`\in \{0, 1\}`.

    :param v: (1d array-like or 2d array-like of shape (n_features, n_samples)).
    :param method: (str) binarization method.
        - 'abs': features with absolute <= :code:`threshold` will be set to 0, 1 otherwise.
        - 'thres': features <= :code:`threshold` will be set to 0, 1 otherwise.
    :param threshold: (float) Threshold value.
    :return: (1d ndarray of shape (n_features,) or 2d ndarray of shape (n_features, n_samples)). """

    if not isinstance(v, (list, tuple, np.ndarray)):
        raise TypeError('Array format not supported.')

    if method == 'abs' and threshold < 0:
        raise ValueError("The threshold value must be >= 0 if method == 'abs'")

    if method == 'abs':
        if isinstance(v, np.ndarray):
            return (np.abs(v) > threshold).astype(np.int32)
        elif isinstance(v, (list, tuple)):
            return (np.abs(np.array(v)) > threshold).astype(np.int32)
    elif method == 'thres':
        if isinstance(v, np.ndarray):
            return (v > threshold).astype(np.int32)
        elif isinstance(v, (list, tuple)):
            return (np.array(v) > threshold).astype(np.int32)
    else:
        raise ValueError("Thresholding method not supported. Use one in ['abs', 'thres']")


def _normalize_binary_mask(mask: np.array) -> np.array:
    """
    Normalizes a binary array (NOT 0 mean, 1 std).
    :param mask: Binary np.array.
    :return: np.array with values in {0, 1}.
    """
    if not _array_is_binary(mask):
        raise Exception('Array is not binary.')
    else:
        return _minmax_normalize_array(mask)


def _array_is_binary(array: np.array) -> bool:
    """
    Checks wether an array contains two or 1 unique values.
    :param array: ndarray
    :return: Bool. True if array is binary.
    """
    uniqueVals = len(np.unique(array))
    return True if uniqueVals == 2 or uniqueVals == 1 else False


def _minmax_normalize_array(array: np.array) -> np.array:
    """
    Normalizes an array via min-max.
    :param array: np.array to normalize
    :return: np.array with values in [0, 1]
    """
    arrayMin = np.min(array)
    arrayMax = np.max(array)
    if arrayMin == arrayMax:
        return array
    else:
        return (array - arrayMin) / (arrayMax - arrayMin)


def _is_rgb(img):
    """ img (ndarray) """
    return True if len(img.shape) == 3 and img.shape[2] == 3 else False
