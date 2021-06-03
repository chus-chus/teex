""" Utils for ndarray wrangling """

import numpy as np


def _binarize_arrays(v, threshold):
    """ Binarize 1 or more array-like object/s of floats to have values :math:`\in \{0, 1\}`.

    :param v: (1d array-like or 2d array-like of shape (n_features, n_samples)).
    :return: (1d ndarray of shape (n_features,) or 2d ndarray of shape (n_features, n_samples)). """

    assert 0 < threshold < 1, 'threshold should be a value between 0 and 1.'
    if isinstance(v, np.ndarray):
        return (v > threshold).astype(np.int32)
    elif isinstance(v, (list, tuple)):
        return (np.array(v) > threshold).astype(np.int32)
    else:
        raise ValueError('Format not supported.')


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
    return (array - arrayMin) / (np.max(array) - arrayMin)


def _is_rgb(img):
    """ img (ndarray) """
    return True if len(img.shape) == 3 and img.shape[2] == 3 else False
