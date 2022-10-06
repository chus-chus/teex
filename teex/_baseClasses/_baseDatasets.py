""" Skeletons for datasets. """

import abc
import warnings

from pathlib import Path
from shutil import rmtree

from teex._utils._data import query_yes_no

class _ClassificationDataset(object):

    """ The base class for real classification datasets. One must override :code:`__getitem__`, :code:`__len__`,
    :code:`_get_class_map`, :code:`_check_integrity`

    :param path: (pathlib.Path) Root directory of the dataset. """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._isDownloaded = False
        self.classMap = {}

    @abc.abstractmethod
    def __getitem__(self, item):
        """ Returns image, target and explanation """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def _check_integrity(self) -> bool:
        """ Returns True if the dataset is already downloaded and integral """
        pass

    @abc.abstractmethod
    def _download(self) -> bool:
        """ Overwrites previously existing data and downloads and saves the data in self.path.
        Returns True if the download was succesful """
        pass

    @abc.abstractmethod
    def _get_class_map(self) -> dict:
        """ Returns single level dict. that maps class numbers to a relevant str. representation, if any. """
        pass
    
    def delete_data(self):
        """ Deletes downloaded data from disk. """
        
        warnings.warn("This will remove all downloaded data associated "
                      "with this class from disk and invalidate this Dataset "
                      "instance. Procede?")
        
        if query_yes_no(""):
            try:
                rmtree(self._path)
                self._isDownloaded = False
            except FileNotFoundError:
                warnings.warn("There is no data downloaded.")
    
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self._isDownloaded:
            return f'Dataset {self.__class__.__name__} @ {self._path}. \n{self.__len__()} observations.'
        else:
            return f'Dataset missing from disk. Re-instantiate to download.'


class _SyntheticDataset(object):
    """ The base class for synthetic datasets. One must override :code:`__getitem__`, :code:`__len__` """

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        """ Returns image, target and explanation """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'Synthetic dataset {self.__class__.__name__}.'
