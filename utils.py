""" Utility functions module. """

import os

from string import ascii_letters


def _generate_feature_names(nFeatures):
    """ Generates a list of length *nFeatures* with combinatinos of the abecedary. """

    if nFeatures > len(ascii_letters):
        featureNames = []
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
