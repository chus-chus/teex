""" Available dataset classes """

from PIL import Image

# noinspection PyProtectedMember
from TAIAOexp.utils._paths import _check_pathlib_dir
# noinspection PyProtectedMember
from TAIAOexp._baseClasses._baseDatasets import _BinaryClassificationDataset
# noinspection PyProtectedMember
from TAIAOexp._datasets.info.kahikatea import _kahikateaIndexes, _kahikateaLabels, _kahikateaNEntries, \
    _kahikatea_root, _kahikatea_url


class Kahikatea(_BinaryClassificationDataset):
    """ Binary classification dataset from URL. If an image belongs to the negative class, None is provided as an
    explanation. """

    def __init__(self):

        super(Kahikatea, self).__init__(path=_kahikatea_root)

        if self._check_integrity() is False:
            self._download()

        self.classMap = self._get_class_map()

    def __getitem__(self, item: int):
        img = Image.open(str(self._path / 'data' + _kahikateaIndexes[item]))
        label = _kahikateaLabels[item]
        if label == 0:
            exp = None
        else:
            exp = Image.open(str(self._path / 'exps' + _kahikateaIndexes[item]))
        return img, label, exp

    def __len__(self) -> int:
        return _kahikateaNEntries

    def _check_integrity(self) -> bool:
        return (_check_pathlib_dir(self._path / 'exps') and
                _check_pathlib_dir(self._path / 'data'))

    def _download(self) -> bool:
        # todo implement download
        url = _kahikatea_url
        pass

    def _get_class_map(self) -> dict:
        return {0: 'Not in image', 1: 'In image'}
