import numpy as np

from cv2 import COLOR_RGB2GRAY, threshold, findContours, RETR_CCOMP, CHAIN_APPROX_SIMPLE, drawContours, cvtColor, \
    THRESH_OTSU, COLOR_BGR2RGB, imread


def binarize_rgb_img(img: np.array, bgColor: str = 'white', fillHoles: bool = True, normalize: bool = True) -> np.array:
    # todo add binarization threshold?
    """
    Binarizes a RGB mask image, letting the background be black.
    :param img: array, image to binarize.
    :param bgColor: Background color of the image to binarize.
    :param fillHoles: Should holes created from the binarization be filled?
    :param normalize: Should the output values be in {0, 1}?
    :return: a grayscale binarized mask.
    """
    imgmod = cvtColor(img, COLOR_RGB2GRAY)
    if bgColor == 'white':
        # background to black
        imgmod = np.where(imgmod == 255, 0, imgmod)
    _, imgmod = threshold(imgmod, 0, 255, THRESH_OTSU)
    if fillHoles:
        contours, _ = findContours(imgmod, RETR_CCOMP, CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            drawContours(imgmod, [cnt], 0, 255, -1)
    if normalize:
        imgmod = normalize_binary_mask(imgmod)
    return imgmod


def read_rgb_img(imgPath: str) -> np.array:
    """
    Read RGB image from path.
    :param imgPath: relative (from w.d.) or absolute path of the file.
    :return: np.array as RGB image.
    """
    return cvtColor(imread(imgPath), COLOR_BGR2RGB)


def normalize_binary_mask(mask: np.array) -> np.array:
    """
    Normalizes a binary np.array.
    :param mask: Binary np.array.
    :return: np.array with values in [0, 1].
    """
    if not array_is_binary(mask):
        raise Exception('Array is not binary.')
    else:
        return mask / np.max(mask)


def array_is_binary(array: np.array) -> bool:
    """
    Checks wether an array only contains two unique values.
    :param array: Input array-like.
    :return: Bool. True if array is binary.
    """
    arrayLength = len(np.unique(array))
    if arrayLength < 2 or arrayLength > 2:
        return False
    else:
        return True
