""" Utils for dataset handling """
from cv2 import imread, COLOR_BGR2RGB, cvtColor
from teex._utils._paths import _check_dir

from os import scandir


def _list_images(pathName, returnType='list'):
    """ Returns {filename: image, ...} """
    _check_dir(pathName)
    if returnType == 'list':
        images = []
    elif returnType == 'dict':
        images = {}
    else:
        raise ValueError('returnType not supported.')
    for filename in scandir(pathName):
        if filename.is_file():
            # only RGB for now
            if returnType == 'list':
                images.append(cvtColor(imread(filename.path), COLOR_BGR2RGB).astype('float32'))
            elif returnType == 'dict':
                images[filename.name] = cvtColor(imread(filename.path), COLOR_BGR2RGB).astype('float32')
        else:
            raise Exception('Some images could not be read.')
    if returnType == 'list':
        return images
    elif returnType == 'dict':
        return images
