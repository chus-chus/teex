""" Utils for dealing with paths and directories """

import cv2
import os


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
