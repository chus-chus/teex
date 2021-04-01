import numpy as np

from cv2 import COLOR_RGB2GRAY, cvtColor, COLOR_BGR2RGB, imread, findContours, drawContours, THRESH_OTSU, RETR_CCOMP, \
    CHAIN_APPROX_SIMPLE, threshold


def binarize_rgb_mask(img: np.array, bgColor: str = 'white') -> np.array:
    """
    Binarizes a RGB binary mask, letting the background (negative class) be 0.
    :param img: array, RGB mask to binarize.
    :param bgColor: Color of the negative class of the image to binarize: {'white', 'black'}
    :return: a binary mask.
    """
    imgmod = cvtColor(img, COLOR_RGB2GRAY)
    if bgColor == 'white':
        # assign non-white the positive class
        imgmod[imgmod < 255] = 1
        imgmod[imgmod == 255] = 0
    elif bgColor == 'black':
        # assign non-black pixels the positive class
        imgmod[imgmod > 0] = 1
        imgmod[imgmod == 0] = 0
    return imgmod


def binarize_rgb_img(img: np.array, fillHoles: bool = True) -> np.array:
    """
    Binarizes a RGB image with automatic OTSU thresholding, minimizing intra-class variance.
    :param img: array, RGB image to binarize.
    :param fillHoles: Should holes created from the binarization be filled?
    :return: a grayscale binarized mask.
    """
    imgmod = cvtColor(img, COLOR_RGB2GRAY)
    _, imgmod = threshold(imgmod, 0, 255, THRESH_OTSU)
    if fillHoles:
        contours, _ = findContours(imgmod, RETR_CCOMP, CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            drawContours(imgmod, [cnt], 0, 255, -1)
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
    Normalizes a binary array (NOT 0 mean, 1 std).
    :param mask: Binary np.array.
    :return: np.array with values in {0, 1}.
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


def normalize_array(array: np.array) -> np.array:
    """
    Normalizes an array (NOT 0 mean, 1 std).
    :param array: np.array to normalize
    :return: np.array with values in {0, 1}
    """
    return array / np.max(array)
