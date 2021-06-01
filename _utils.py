""" Private utility methods module. """

import os
import cv2
import numpy as np

from string import ascii_letters

from cv2 import COLOR_RGB2GRAY, cvtColor, COLOR_BGR2RGB, imread, findContours, drawContours, THRESH_OTSU, RETR_CCOMP, \
    CHAIN_APPROX_SIMPLE, threshold


def _list_images(pathName, returnType='list'):
    """ Returns {filename: image, ...} """
    _check_dir(pathName)
    if returnType == 'list':
        images = []
    elif returnType == 'dict':
        images = {}
    else:
        raise ValueError('returnType not supported.')
    for filename in os.scandir(pathName):
        if filename.is_file():
            # todo non RGB
            if returnType == 'list':
                images.append(cv2.cvtColor(cv2.imread(filename.path), cv2.COLOR_BGR2RGB).astype('float32'))
            elif returnType == 'dict':
                images[filename.name] = cv2.cvtColor(cv2.imread(filename.path), cv2.COLOR_BGR2RGB).astype('float32')
        else:
            raise Exception('Some images could not be read.')
    if returnType == 'list':
        return images
    elif returnType == 'dict':
        return images


def _rgb_to_grayscale(img):
    """ Transforms a 3 channel RGB image into a grayscale image (1 channel) """
    return cvtColor(img.astype('float32'), COLOR_RGB2GRAY)


def _binarize_rgb_mask(img, bgValue='high') -> np.array:
    """
    Binarizes a RGB binary mask, letting the background (negative class) be 0. Use this function when the
    image to binarize has a very defined background.

    :param img: (ndarray), RGB mask to binarize.
    :param bgValue: (str) Intensity of the negative class of the image to binarize: {'light', 'dark'}
    :return: (ndarray) a binary mask.
    """
    if bgValue not in {'high', 'low'}:
        raise ValueError(f"bgColor should ve {['high', 'low']}")

    imgmod = _rgb_to_grayscale(img)
    maxVal = np.max(imgmod)
    minVal = np.min(imgmod)
    if bgValue == 'high':
        # assign darker pixels the positive class
        imgmod[imgmod < maxVal] = 1
        imgmod[imgmod == maxVal] = 0
    elif bgValue == 'low':
        # assign lighter pixels the positive class
        imgmod[imgmod > minVal] = 1
        imgmod[imgmod == minVal] = 0
    return imgmod


def _binarize_rgb_img(img, fillHoles=True) -> np.array:
    """
    Binarizes a RGB image with automatic OTSU thresholding, minimizing intra-class variance. Use this function when the
    image to binarize does not have a very defined background.

    :param img: ndarray, RGB image to binarize.
    :param fillHoles: (bool, default True)Should holes created from the binarization be filled?
    :return: (ndarray) a grayscale binarized mask.
    """
    imgmod = _rgb_to_grayscale(img)
    _, imgmod = threshold(imgmod, 0, 255, THRESH_OTSU)
    if fillHoles:
        contours, _ = findContours(imgmod, RETR_CCOMP, CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            drawContours(imgmod, [cnt], 0, 255, -1)
    imgmod = _normalize_binary_mask(imgmod)
    return imgmod


def _read_rgb_img(imgPath: str) -> np.array:
    """
    Read RGB image from path.
    :param imgPath: relative (from w.d.) or absolute path of the file.
    :return: np.array as RGB image.
    """
    return cvtColor(imread(imgPath), COLOR_BGR2RGB)


def _normalize_binary_mask(mask: np.array) -> np.array:
    """
    Normalizes a binary array (NOT 0 mean, 1 std).
    :param mask: Binary np.array.
    :return: np.array with values in {0, 1}.
    """
    if not _array_is_binary(mask):
        raise Exception('Array is not binary.')
    else:
        return _normalize_array(mask)


def _array_is_binary(array: np.array) -> bool:
    """
    Checks wether an array contains two or 1 unique values.
    :param array: ndarray
    :return: Bool. True if array is binary.
    """
    uniqueVals = len(np.unique(array))
    return True if uniqueVals == 2 or uniqueVals == 1 else False


def _normalize_array(array: np.array) -> np.array:
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


def _generate_feature_names(nFeatures):
    """ Generates a list of length *nFeatures* with combinatinos of the abecedary. """

    if nFeatures > len(ascii_letters):
        featureNames = list()
        name, i, j = 0, 0, -1
        while name < nFeatures:
            if j == -1:
                fName = ascii_letters[i]
            else:
                fName = featureNames[j] + ascii_letters[i]
            featureNames.append(fName)
            i += 1
            if i % len(ascii_letters) == 0:
                j += 1
                i = 0
            name += 1
    else:
        featureNames = [ascii_letters[i] for i in range(nFeatures)]
    return featureNames


def _check_dir(pathName):
    """ Checks if a directory exists and, if it does, if it is empty. """

    if not os.path.isdir(pathName):
        raise Exception('Data path not valid: {}'.format(pathName))
    if len(os.listdir(pathName)) == 0:
        raise Exception('Empty directory: {}'.format(pathName))


def _check_and_create_dir(pathName):
    """ Checks if a directory exists and, if it does not, it creates it. Does nothing if the directory exists. """

    if not os.path.exists(pathName):
        os.mkdir(pathName)


def _check_file(pathName):
    """ Checks if a file exists. """

    if not os.path.isfile(pathName):
        raise Exception('File not valid: {}'.format(pathName))
