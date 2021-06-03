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
