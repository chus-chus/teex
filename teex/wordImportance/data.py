""" Module for real datasets with available ground truth word importance explanations. Also contains
methods and classes for word importance data manipulation. """

import json

from teex._utils._misc import _download_extract_zip
from teex._utils._paths import _check_pathlib_dir
from teex._datasets.info.newsgroup import _newsgroupRoot, _newsgroupLabels, _newsgroupNEntries, _newsgroupAll, \
    _newsgroup_url
from teex._baseClasses._baseDatasets import _ClassificationDataset


class Newsgroup(_ClassificationDataset):
    """ 20 Newsgroup dataset from https://github.com/SinaMohseni/ML-Interpretability-Evaluation-Benchmark

    Contains 188 human annotaded newsgroup texts belonging to two categories.

    :Example:

    >>> nDataset = Newsgroup()
    >>> obs, label, exp = nDataset[1]

    where :code:`obs` is a str, :code:`label` is an int and :code:`exp` is a dict. containing a score for each important
    word in :code:`obs`. When a slice is performed, obs, label and exp are lists of the objects described above. """

    def __init__(self):
        super(Newsgroup, self).__init__(path=_newsgroupRoot)

        if self._check_integrity() is False:
            print('Files do not exist or are corrupted:')
            self._download()

        self.classMap = self._get_class_map()

    def __getitem__(self, item):
        if isinstance(item, slice):
            obs, label, exp = [], [], []
            fileNames = _newsgroupAll[item]
            labels = _newsgroupLabels[item]
            for name, classLabel in zip(fileNames, labels):
                with open(str(self._path / ('data/' + name)), 'rb') as t:
                    obs.append(t.read())
                label.append(classLabel)
                with open(str(self._path / ('expl/' + name + '.json')), 'rb') as t:
                    exp.append(json.load(t)['words'])
        elif isinstance(item, int):
            with open(str(self._path / ('data/' + _newsgroupAll[item])), 'rb') as t:
                obs = t.read()
            label = _newsgroupLabels[item]
            with open(str(self._path / ('expl/' + _newsgroupAll[item] + '.json')), 'rb') as t:
                exp = json.load(t)['words']
        else:
            raise TypeError('Invalid argument type.')

        return obs, label, exp

    def __len__(self) -> int:
        return _newsgroupNEntries

    def _check_integrity(self) -> bool:
        return (_check_pathlib_dir(self._path / 'expl') and
                _check_pathlib_dir(self._path / 'data'))

    def _download(self) -> None:
        _download_extract_zip(self._path, _newsgroup_url, 'rawNewsgroup.zip')

    def _get_class_map(self) -> dict:
        return {0: 'electronics', 1: 'medicine'}

