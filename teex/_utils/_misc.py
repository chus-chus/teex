""" Miscellaneous utils """
import os
import shutil
import urllib
import urllib.request
import urllib.error
from zipfile import ZipFile

from tqdm import tqdm

from string import ascii_letters
import warnings
import functools

from teex._utils._paths import _check_pathlib_dir


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def _generate_feature_names(nFeatures: int) -> list:
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


def _url_retrieve(url: str, filePath: str, chunkSize: int = 1024) -> None:
    """ From `TorchVision <https://github.com/pytorch/vision/blob/ef711591a5db69d36f904ab5c39dec13627a58ad/torchvision/
    datasets/utils.py#L30>`_ """

    with open(filePath, "wb") as fh:
        with urllib.request.urlopen(urllib.request.Request(url)) as response:
            with tqdm(total=response.length) as pBar:
                for chunk in iter(lambda: response.read(chunkSize), ""):
                    if not chunk:
                        break
                    pBar.update(chunkSize)
                    fh.write(chunk)


def _download_file(url: str, root: str, filename: str) -> None:
    """ From `TorchVision <https://github.com/pytorch/vision/blob/ef711591a5db69d36f904ab5c39dec13627a58ad/torchvision/
    datasets/utils.py#L30>`_ """

    fpath = os.path.join(root, filename)
    try:
        print('Downloading ' + url + ' to ' + fpath)
        _url_retrieve(url, fpath)
    except (urllib.error.URLError, IOError) as exc:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                  ' Downloading ' + url + ' to ' + fpath)
            _url_retrieve(url, fpath)
        else:
            raise exc


def _download_extract_zip(filePath, fileUrl, fileName):
    """
    1. checks if the path exists and:
        1.1 Deletes the contents if it exists
        1.2 Creates the path
    2. Downloads the zip file into the path
    3. Extracts the contents and deletes the zip file

    :param Path filePath: file path
    :param str fileUrl: file url
    :param str fileName: file name """

    if _check_pathlib_dir(filePath):
        shutil.rmtree(filePath)

    os.makedirs(filePath)

    _download_file(fileUrl, filePath, fileName)

    with ZipFile(filePath / fileName, 'r') as zipObj:
        for elem in zipObj.namelist():
            zipObj.extract(elem, filePath)

    os.remove(filePath / fileName)
