""" Available dataset classes """
import json

from PIL import Image

# noinspection PyProtectedMember
from TAIAOexp.utils._paths import _check_pathlib_dir
# noinspection PyProtectedMember
from TAIAOexp._baseClasses._baseDatasets import _ClassificationDataset
# noinspection PyProtectedMember
from TAIAOexp._datasets.info.kahikatea import _kahikateaIndexes, _kahikateaLabels, _kahikateaNEntries, \
    _kahikatea_root, _kahikatea_url
# noinspection PyProtectedMember
from TAIAOexp._datasets.info.newsgroup import _newgroupRoot, _newsgroupIndexes, _newsgroupLabels, _newsgroupNEntries


class Kahikatea(_ClassificationDataset):
    """ Binary classification dataset from URL. If an image belongs to the negative class, None is provided as an
    explanation. An example of a returned observation when indexing:

    .. code::

        kDataset = Kahikatea()
        img, label, exp = kDataset[1]

    where :code:`img` is a PIL Image, :code:`label` is an int and :code:`exp` is a PIL Image.

    """

    def __init__(self):

        super(Kahikatea, self).__init__(path=_kahikatea_root)

        if self._check_integrity() is False:
            self._download()

        self.classMap = self._get_class_map()

    def __getitem__(self, item: int):
        if isinstance(item, slice):
            img, label, exp = [], [], []
            indexes = list(range(_kahikateaNEntries))[item]
            for index in indexes:
                i, l, e = self._read_items(str(self._path / ('data/' + _kahikateaIndexes[index])),
                                           str(self._path / ('exps/' + _kahikateaIndexes[index])),
                                           index)
                img.append(i)
                label.append(l)
                exp.append(e)
        elif isinstance(item, int):
            img, label, exp = self._read_items(str(self._path / ('data/' + _kahikateaIndexes[item])),
                                               str(self._path / ('exps/' + _kahikateaIndexes[item])),
                                               item)
        else:
            raise TypeError('Invalid argument type.')

        return img, label, exp

    def __len__(self) -> int:
        return _kahikateaNEntries

    def _check_integrity(self) -> bool:
        return (_check_pathlib_dir(self._path / 'exps') and
                _check_pathlib_dir(self._path / 'data'))

    def _download(self) -> bool:
        # todo
        url = _kahikatea_url
        pass

    def _get_class_map(self) -> dict:
        return {0: 'Not in image', 1: 'In image'}

    @staticmethod
    def _read_items(obsPath: str, expPath: str, index: int):
        img = Image.open(obsPath).convert('RGB')
        label = _kahikateaLabels[index]
        if label == 0:
            exp = None
        else:
            exp = Image.open(expPath).convert('RGB')
        return img, label, exp


class Newsgroup(_ClassificationDataset):
    """ 20 Newsgroup dataset from  https://github.com/SinaMohseni/ML-Interpretability-Evaluation-Benchmark

    The returned items when indexing are:

    .. code::

        nDataset = Newsgroup()
        obs, label, exp = nDataset[1]

    where :code:`obs` is a str, :code:`label` is an int and :code:`exp` is a dict. containing a score for each important
    word in :code:`obs`. """

    def __init__(self):
        super(Newsgroup, self).__init__(path=_newgroupRoot)

        if self._check_integrity() is False:
            self._download()

        self.classMap = self._get_class_map()
        self._parse_explanations()

    def __getitem__(self, item):
        if isinstance(item, slice):
            indexes = (i for i in range(slice.start, slice.stop, slice.step))
            pass
        elif isinstance(item, int):
            with open(str(self._path / ('data/' + _newsgroupIndexes[item])), 'r') as t:
                obs = t.readlines()
            label = _newsgroupLabels[item]
            with open(str(self._path / ('exps/' + _newsgroupIndexes[item])), 'r') as t:
                exp = json.load(t)['words']
        else:
            raise TypeError('Invalid argument type.')

        return obs, label, exp

    def __len__(self) -> int:
        return _newsgroupNEntries

    def _check_integrity(self) -> bool:
        return (_check_pathlib_dir(self._path / 'exps') and
                _check_pathlib_dir(self._path / 'data'))

    def _download(self) -> bool:
        # todo
        pass

    def _get_class_map(self) -> dict:
        return {0: 'electronics', 1: 'medicine'}

    def _parse_explanations(self):
        # todo for each explanation, transform it s.t. it is a dict with words as keys and importances as values.
        pass


def _datasets_main():
    import matplotlib.pyplot as plt
    kData = Kahikatea()
    im, label, exp = kData[:3]

    print('Images: ', im)
    print('Labels: ', label)
    print('Exps: ', exp)

    plt.imshow(im[0])
    plt.show()

    plt.imshow(exp[0])
    plt.show()


if __name__ == '__main__':
    _datasets_main()
