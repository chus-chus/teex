""" Utils for dealing with paths and directories """

import os

from pathlib import Path


def _check_pathlib_dir(path: Path) -> bool:
    """ Returns true if directory exists and is not empty. """
    return path.exists() and any(path.iterdir())


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
