""" Module for synthetic and real datasets with available ground truth saliency map explanations. Also contains
methods and classes for saliency map data manipulation.

All of the datasets must be instanced first. Then, when sliced, they all return the observations, labels and ground
truth explanations, respectively. Note that all real-world datasets implement the 
:code:`delete_data` method, which allows to delete all of their downloaded internal data. In this module, images
and explanations are represented by the :code:`PIL.Image.Image` class. """

import os
import random
import shutil
import warnings
import numpy as np

from PIL import Image
from pathlib import Path

from teex._utils._misc import _download_extract_file
from teex._utils._paths import _check_pathlib_dir
from teex._utils._data import query_yes_no

from teex._baseClasses._baseDatasets import _ClassificationDataset, \
    _SyntheticDataset
from teex._baseClasses._baseClassifier import _BaseClassifier

from teex._datasets.info.kahikatea import _kahikateaLabels, _kahikateaNEntries, \
    _kahikatea_root, _kahikatea_url, _kahikateaAll
from teex._datasets.info.CUB200 import _cub_root, _cub_segmentations_url, _cub_url, \
    _cub_length
from teex._datasets.info.OxfordIIT_Pet import _oxford_iit_root, _oxford_iit_url, \
    _oxford_iit_masks_url, _oxford_iit_length
# Datasets

class TransparentImageClassifier(_BaseClassifier):
    """ Used on the higher level data generation class :class:`SenecaSM` (**use that and get it from there
    preferably**).

    Transparent, pixel-based classifier with pixel (features) importances as explanations. Predicts the
    class of the images based on whether they contain a certain specified pattern or not. Class 1 if they contain
    the pattern, 0 otherwise. To be trained only a pattern needs to be fed. Follows the sklean API. Presented in
    [Evaluating local explanation methods on ground truth, Riccardo Guidotti, 2021]. """

    def __init__(self):
        super().__init__()
        self.pattern = None
        self.pattH = None
        self.pattW = None
        self.cellH = None
        self.cellW = None
        self._binaryPattern = None
        self._patternMask = None
        self._maskedPattern = None

    def fit(self, pattern: np.ndarray, cellH: int = 1, cellW: int = 1) -> None:
        self.pattern = pattern.astype(np.float32)
        self.pattH, self.pattW = pattern.shape[0], pattern.shape[1]
        self.cellH, self.cellW = cellH, cellW
        self._binaryPattern = np.squeeze(np.where(np.delete(pattern, (0, 1), 2) != 0, 1, 0))
        # noinspection PyArgumentList
        self._patternMask = self._binaryPattern.astype(bool).reshape(self.pattH, self.pattW, 1)
        self._patternMask = np.concatenate((self._patternMask, self._patternMask, self._patternMask), axis=2)
        self._maskedPattern = self.pattern[self._patternMask]

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """ Predicts the class for each observation.

        :param np.ndarray obs: array of n images as ndarrays of np.float32 type.
        :return: array of n predicted labels. """
        ret = []
        for image in obs:
            ret.append(1) if self._has_pattern(image) else ret.append(0)
        return ret

    def predict_proba(self, obs: np.ndarray) -> np.ndarray:
        """ Predicts probability that each observation belongs to class 1 or 0. Probability of class 1 will be 1 if
        the image contains the pattern and 0 otherwise.

        :param np.ndarray obs: array of n images as ndarrays.
        :return: array of n probability tuples of length 2. """
        ret = []
        for image in obs:
            if self._has_pattern(image):
                ret.append([0., 1.])
            else:
                ret.append([1., 0.])
        return ret

    def explain(self, obs: np.ndarray) -> np.ndarray:
        """ Explain observations' predictions with binary masks (pixel importance arrays).

        :param np.ndarray obs: array of n images as ndarrays.
        :return: list with n binary masks as explanations. """
        exps = []
        for image in obs:
            hasPat, indices = self._has_pattern(image, retIndices=True)
            exp = np.zeros((image.shape[0], image.shape[1]))
            if hasPat:
                exp[indices[0]:(indices[0] + self.pattH), indices[1]:(indices[1] + self.pattW)] = self._binaryPattern
            exps.append(exp)
        return exps

    def _has_pattern(self, image: np.ndarray, retIndices: bool = False) -> bool:
        """ Searches for the pattern in the image and returns whether it contains it or not and its upper left indices
        if specified. The pattern is contained within an image if the distribution and color of the cells != 0 coincide.
        """

        hasPat = False
        indices = (0, 0)
        for row in range(0, len(image[0]) - self.pattH, self.cellH):
            for col in range(0, len(image[1]) - self.pattW, self.cellW):
                if (image[row:(row + self.pattH), col:(col + self.pattW)][
                        self._patternMask] == self._maskedPattern).all():
                    hasPat = True
                    indices = (row, col)
                    break
            if hasPat:
                break
        if retIndices:
            return hasPat, indices
        else:
            return hasPat


class SenecaSM(_SyntheticDataset):
    """ Synthetic dataset with available saliency map explanations.

    Images and g.t. explanations generated following the procedure presented in [Evaluating local
    explanation methods on ground truth, Riccardo Guidotti, 2021]. The g.t. explanations are binary ndarray masks
    of shape (imageH, imageW) that indicate the position of the pattern in an image (zero array if the pattern is
    not present) and are generated  The generated RGB images belong to one class if they contain a certain
    generated pattern and to the other if not. The images are composed of homogeneous cells of size
    (cellH, cellW), which in turn compose a certain pattern of shape (patternH, patternW) that is inserted on
    some of the generated images.

    From this class one can also obtain a trained transparent model (instance of :class:`TransparentImageClassifier`).

    When sliced, this object will return

        - X (ndarray) of shape (nSamples, imageH, imageW, 3) or (imageH, imageW, 3). Generated image data.
        - y (ndarray) of shape (nSamples,) or int. Image labels. 1 if an image contains the pattern and 0 otherwise.
        - explanations (ndarray) of shape (nSamples, imageH, imageW) or (imageH, imageW). Ground truth explanations.

    :param int nSamples: number of images to generate.
    :param int imageH: height in pixels of the images. Must be multiple of :code:`cellH`.
    :param int imageW: width in pixels of the images. Must be multiple of :code:`cellW`.
    :param int patternH: height in pixels of the pattern. Must be <= :code:`imageH` and multiple of :code:`cellH`.
    :param int patternW: width in pixels of the pattern. Must be <= :code:`imageW` and multiple of :code:`cellW`.
    :param int cellH: height in pixels of each cell.
    :param int cellW: width in pixels of each cell.
    :param float patternProp: ([0, 1]) percentage of appearance of the pattern in the dataset.
    :param float fillPct: ([0, 1]) percentage of cells filled (not black) in each image.
    :param float colorDev: ([0, 0.5])
        maximum val summed to 0 valued channels and minimum val substracted to 1 valued channels of filled cells.
        If 0, each cell will be completely red, green or blue. If > 0, colors may be a mix of the three channels
        (one ~1, the other two ~0).
    :param int randomState: random seed. """

    def __init__(self, nSamples=1000, imageH=32, imageW=32, patternH=16, patternW=16, cellH=4,
                 cellW=4, patternProp=0.5, fillPct=0.4, colorDev=0.1, randomState=888) -> None:
        self.nSamples = nSamples
        self.imageH = imageH
        self.imageW = imageW
        self.patternH = patternH
        self.patternW = patternW
        self.cellH = cellH
        self.cellW = cellW
        self.patternProp = patternProp
        self.fillPct = fillPct
        self.colorDev = colorDev
        self.randomState = randomState

        random.seed(randomState)
        self._rng = np.random.default_rng(randomState)

        self.pattern = None

        # self.transparentModel is the model instance trained to recognize the pattern in the data
        self.X, self.y, self.exp, self.pattern, self.transparentModel = self._gen_seneca_dataset_sm()

    def __getitem__(self, item):
        if isinstance(item, (slice, int)):
            return self.X[item], self.y[item], self.exp[item]
        else:
            raise TypeError('Argument is not a slice nor an index.')

    def __len__(self) -> int:
        return len(self.y)

    def _gen_seneca_dataset_sm(self):

        if self.imageH % self.patternH != 0 or self.imageW % self.patternW != 0 or self.imageH % self.cellH != 0 or \
                self.imageW % self.cellW != 0:
            raise ValueError('Image height and width not multiple of cell or pattern dimensions.')
        if self.imageH < self.patternH or self.imageH < self.cellH or self.imageW < self.patternW or \
                self.imageW < self.cellW or self.patternH < self.cellH or self.patternW < self.cellW:
            raise ValueError('Cells should be smaller than patterns and patterns than image size.')

        nWithPattern = round(self.nSamples * self.patternProp)
        pattern = self._generate_image_seneca(imageH=self.patternH, imageW=self.patternW, cellH=self.cellH,
                                              cellW=self.cellW, pattern=None)

        # transform pattern into a 2d array and then set the channel to 1 if the pixel had any intensity at all (if a
        # cell was part of the pattern it will have at least some intensity). Squeeze it so it has shape pattH x pattW.
        binaryPattern = np.squeeze(np.where(np.delete(pattern, (0, 1), 2) != 0, 1, 0))

        data = []
        for _ in range(nWithPattern):
            image, explanation = self._generate_image_seneca(imageH=self.imageH, imageW=self.imageW, cellH=self.cellH,
                                                             cellW=self.cellW, pattern=pattern,
                                                             binaryPattern=binaryPattern)
            data.append((image, explanation, 1))
        for _ in range(self.nSamples - nWithPattern):
            image = self._generate_image_seneca(imageH=self.imageH, imageW=self.imageW, cellH=self.cellH,
                                                cellW=self.cellW, pattern=None)
            # blank explanation
            explanation = np.zeros((self.imageH, self.imageW))
            data.append((image, explanation, 0))

        random.shuffle(data)
        imgs, exps, labels = zip(*data)

        mod = TransparentImageClassifier()
        mod.fit(pattern, cellH=self.cellH, cellW=self.cellW)
        return np.array(imgs, dtype=np.float32), np.array(labels, dtype=int), np.array(exps, dtype=int), pattern, mod

    def _generate_image_seneca(self, imageH, imageW, cellH, cellW, pattern=None, binaryPattern=None) -> np.ndarray:
        """ Generates RGB image as ndarray of shape (imageH, imageW, 3) pixels and uniform cells of cellH * cellW
        pixels. fillPct% [0, 1] of the cells are != 0. If 'pattern' is an ndarray of shape (patternH, patternW, 3),
        inserts it into the generated image in a random position and also returns a binary feature importance ndarray
        of shape (imageH, imageW) as ground truth explanation for the generated image. """

        totalCells = (imageH / cellH) * (imageW / cellW)
        filledCells = round(totalCells * self.fillPct)

        # starting pixels (upper left) of each cell
        startingPixelsRow = np.arange(0, imageH, cellH)
        startingPixelsCol = np.arange(0, imageW, cellW)

        # choose random cells to fill
        cellIndexes = zip(self._rng.choice(startingPixelsRow, size=filledCells),
                          self._rng.choice(startingPixelsCol, size=filledCells))

        img = np.zeros((imageH, imageW, 3))
        for cellRow, cellCol in cellIndexes:
            # set reg, green and blue for each chosen cell
            img[cellRow:(cellRow + cellH), cellCol:(cellCol + cellW)] = self._generate_rgb()

        if pattern is not None:
            # choose where the pattern goes (upper left corner) and overwrite the image
            patternRow = self._rng.choice(np.arange(0, imageH - pattern.shape[0], cellH))
            patternCol = self._rng.choice(np.arange(0, imageW - pattern.shape[1], cellW))
            img[patternRow:(patternRow + pattern.shape[0]), patternCol:(patternCol + pattern.shape[1])] = pattern

            exp = np.zeros((imageH, imageW))
            exp[patternRow:(patternRow + pattern.shape[0]), patternCol:(patternCol + pattern.shape[1])] = binaryPattern
            return img, exp
        else:
            return img

    def _generate_rgb(self) -> np.ndarray:
        """ Generates a ndarray of shape (3,)representing  RGB color with one of the channels turned to, at least,
        1 - colorDev and the other channels valued at, at most, self.colorDev. 'self.colorDev' must be between 0 and 1.
        """

        # first array position will be turned on, last two turned off
        order = self._rng.choice(3, size=3, replace=False)
        colors = np.zeros(3)
        colors[order[0]] = 1 - self._rng.uniform(0, self.colorDev)
        colors[order[1]] = self._rng.uniform(0, self.colorDev)
        colors[order[2]] = self._rng.uniform(0, self.colorDev)
        return colors


class Kahikatea(_ClassificationDataset):
    """ Binary classification dataset from [Y. Jia et al. (2021) Studying and Exploiting the Relationship Between Model
    Accuracy and Explanation Quality, ECML-PKDD 2021].

    This dataset contains images for Kahikatea (an endemic tree in New Zealand) classification. Positive examples
    (in which Kahikatea trees can be identified) are annotated with true explanations such that the Kahikatea trees are
    highlighted. If an image belongs to the negative class, None is provided as an explanation.

    :Example:

    >>> kDataset = Kahikatea()
    >>> img, label, exp = kDataset[1]

    where :code:`img` is a PIL Image, :code:`label` is an int and :code:`exp` is a PIL Image.
    When a slice is performed, obs, label and exp are lists of the objects described above.

    """

    def __init__(self):

        super(Kahikatea, self).__init__(path=_kahikatea_root)

        if self._check_integrity() is False:
            print('Files do not exist or are corrupted. Building dataset...')
            self._download()
        else:
            self._isDownloaded = True

        self.classMap = self._get_class_map()

    def __getitem__(self, item):
        if isinstance(item, slice):
            img, label, exp = [], [], []
            imgNames = _kahikateaAll[item]
            imgLabels = _kahikateaLabels[item]
            for imgName, imgLabel in zip(imgNames, imgLabels):
                i, e = self._read_items(str(self._path / ('data/' + imgName)),
                                        str(self._path / ('expl/' + imgName)),
                                        imgLabel)
                img.append(i)
                label.append(imgLabel)
                exp.append(e)
        elif isinstance(item, int):
            imgName = _kahikateaAll[item]
            label = _kahikateaLabels[item]
            img, exp = self._read_items(str(self._path / ('data/' + imgName)),
                                        str(self._path / ('expl/' + imgName)),
                                        label)
        else:
            raise TypeError('Invalid argument type.')

        return img, label, exp
    
    def get_class_observations(self, classId: int) -> list:
        """Get observations of a particular class.

        :param classId: Class ID. See attribute :code:`classMap`.
        :type classId: int
        :return: Observations of the specified type.
        :rtype: list
        """
        if classId not in self.classMap:
            raise ValueError("Class ID not valid.")
        
        imgs = []
        exps = []

        classPath = "ve_positive" if classId == 1 else "ve_negative"
        
        for filename in os.scandir(self._path / "data" / classPath):
            if filename.is_file() and filename.name[0] != ".":
                imgs.append(Image.open(self._path / "data" / classPath / filename).convert("RGB"))
                
        for filename in os.scandir(self._path / "expl" / classPath):
            if filename.is_file() and filename.name[0] != ".":
                exps.append(Image.open(self._path / "expl" / classPath / filename).convert("RGB"))
            
        labels = [classId for _ in range(len(imgs))]
        
        return imgs, labels, exps

    def __len__(self) -> int:
        return _kahikateaNEntries

    def _check_integrity(self) -> bool:
        return (_check_pathlib_dir(self._path / 'expl/ve_positive') and
                _check_pathlib_dir(self._path / 'data/ve_positive') and
                _check_pathlib_dir(self._path / 'data/ve_negative'))

    def _download(self) -> bool: # pragma: no cover
        try:
            _download_extract_file(self._path, _kahikatea_url, 'rawKahikatea.zip')
            self._isDownloaded = True
        except:
            warnings.warn("Download interruped.")
            shutil.rmtree(self._path)

    def _get_class_map(self) -> dict:
        return {0: 'Not in image', 1: 'In image'}

    @staticmethod
    def _read_items(obsPath: str, expPath: str, obsLabel: int):
        img = Image.open(obsPath).convert('RGB')
        if obsLabel == 0:
            exp = None
        else:
            exp = Image.open(expPath).convert('RGB')
        return img, exp


class CUB200(_ClassificationDataset):
    """ The CUB-200-2011 Classification Dataset. 11788 observations with 
    200 different classes. From 
    
    Wah, Branson, Welinder, Perona, & Belongie. (2022). CUB-200-2011 (1.0) [Data set]. 
    CaltechDATA. https://doi.org/10.22002/D1.20098
    """
    
    def __init__(self):

        super(CUB200, self).__init__(path=_cub_root)

        if self._check_integrity() is False:
            print('Files do not exist or are corrupted. Building dataset...')
            if self._download():
                self._isDownloaded = True
                self._organize()
        else:
            self._isDownloaded = True

        self.classMap = self._get_class_map()
        self._imNames = self._read_image_names()
        self._expNames = [name[:-3] + "png" for name in self._imNames]
        self._imLabels = self._read_image_labels()
        
    def get_class_observations(self, classId: int):
        """Get all observations from a particular class given its index.

        Args:
            classId (int): Class index. It can be consulted from the attribute 
             :attr:`.CUB200.classMap`

        Returns:
            imgs (list): Images pertaining to the specified class.
            labels (list): Int labels pertaining to the specified class.
            exps (list): Explanations pertaining to the specified class.
        """
        
        if classId not in self.classMap:
            raise ValueError("Invalid class id (see self.classMap).")
        
        imgs = []
        exps = []
        
        for lab, img, exp in zip(self._imLabels, self._imNames, self._expNames):
            if lab == classId:
                imgs.append(Image.open(self._path / "images" / img).convert("RGB"))
                exps.append(Image.open(self._path / "segmentations" / exp))

        labels = [classId for _ in range(len(imgs))]
        
        return imgs, labels, exps

                
    def __getitem__(self, item):
        if isinstance(item, slice):
            img, exp = [], []
            
            for imName, expName in zip(self._imNames[item], self._expNames[item]):
                img.append(Image.open(self._path / "images" / imName).convert('RGB'))
                exp.append(Image.open(self._path / "segmentations" / expName))
                
        elif isinstance(item, int):
            img = Image.open(self._path / "images" / self._imNames[item]).convert('RGB')
            exp = Image.open(self._path / "segmentations" / self._expNames[item])
        else:
            raise TypeError('Invalid argument type.')
        
        label = self._imLabels[item]

        return img, label, exp
    
    def _read_image_names(self) -> list:
        
        with open(self._path / "images.txt", "r") as file:
            lines = file.readlines()
            
        imNames = []
        for line in lines:
            imNames.append(line.split(" ")[1].rstrip())
        
        return imNames
    
    def _read_image_labels(self) -> list:
        
        with open(self._path / "image_class_labels.txt", "r") as file:
            lines = file.readlines()
            
        imLabels = []
        for line in lines:
            imLabels.append(int(line.split(" ")[1].rstrip()) - 1)
        
        return imLabels
        

    def __len__(self) -> int:
        return _cub_length
        
    def _download(self) -> bool: # pragma: no cover
        queryResponse = query_yes_no('This download will take ~1.2GB of disk. Procede?')
        try:
            if queryResponse:
                _download_extract_file(self._path.parent.absolute(), _cub_url, 'cubImages.tgz', 'tar')
                _download_extract_file(self._path, _cub_segmentations_url, 'cubExpl.tgz', 'tar',
                                       deletePrevDir = False)
                return True
            else:
                shutil.rmtree(self._path)
                warnings.warn("Dataset download aborted.")
                return False
        except:
            warnings.warn("Download interruped.")
            shutil.rmtree(self._path)
            
        
    def _check_integrity(self) -> bool:
        return (_check_pathlib_dir(self._path / 'images') and
                _check_pathlib_dir(self._path / 'segmentations'))
        
    def _organize(self) -> None: # pragma: no cover
        """ Given the extracted files, deletes folders that 
        are not used and tidies up the directory.
        """
        os.remove(self._path.parent.absolute() / 'attributes.txt')
        shutil.rmtree(self._path / "attributes")
        shutil.rmtree(self._path / "parts")
        os.remove(self._path / 'bounding_boxes.txt')
        os.remove(self._path / 'README')
        os.remove(self._path / 'train_test_split.txt')
        return
            
    def _get_class_map(self) -> dict:
        with open(self._path / "classes.txt", "r") as file:
            lines = file.readlines()
            
        classMap = {}
        for c in lines:
            r = c.split(" ")
            classMap[int(r[0]) - 1] = r[1].rstrip()
            
        return classMap
            
        
        
class OxfordIIIT(_ClassificationDataset):
    """ The Oxford-IIIT Pet Dataset. 7347 images from 37 categories with 
    approximately 200 images per class. From
    
    O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar, "Cats and dogs," 
    2012 IEEE Conference on Computer Vision and Pattern Recognition, 2012, 
    pp. 3498-3505, doi: 10.1109/CVPR.2012.6248092.
    """
    
    def __init__(self):

        super(OxfordIIIT, self).__init__(path=_oxford_iit_root)

        if self._check_integrity() is False:
            print('Files do not exist or are corrupted. Building dataset...')
            if self._download():
                self._isDownloaded = True
                self._organize()
        else:
            self._isDownloaded = True

        self.classMap, self._labelMap = self._get_class_map()
        self._imNames, self._imLabels, self._expNames = self._read_image_info()
        
    def _read_image_info(self):
        imgNames = []
        expNames = []
        for filename in os.scandir(self._path / "images"):
            # missing explanations for these 2 files
            if filename.is_file() and filename.name[0] != "." \
                and filename.name[:-4] != "Egyptian_Mau_165" \
                and filename.name[:-4] != "Egyptian_Mau_167" \
                and filename.name[-3:] != "mat":
                imgNames.append(filename.name)
                expNames.append(filename.name[:-4] + ".png")
        
        # get labels from labelMap
        imgLabels = []
        for name in imgNames:
            imgLabels.append(self._labelMap[name[:-4].rsplit("_", 1)[0]])
        
        return imgNames, imgLabels, expNames
        
    def get_class_observations(self, classId: int) -> list:
        """Get all observations from a particular class given its index.

        Args:
            classId (int): Class index. It can be consulted from the attribute 
             :attr:`classMap`.

        Returns:
            imgs (np.ndarray): Images pertaining to the specified class.
            labels (np.ndarray): Int labels pertaining to the specified class.
            exps (np.ndarray): Explanations pertaining to the specified class.
        """
        
        if classId not in self.classMap:
            raise ValueError("Invalid class id (see self.classMap).")
        
        imgs = []
        exps = []
        
        for lab, img, exp in zip(self._imLabels, self._imNames, self._expNames):
            if lab == classId:
                imgs.append(Image.open(self._path / "images" / img).convert("RGB"))
                exps.append(Image.open(self._path / "annotations" / exp))
            
        labels = [classId for _ in range(len(imgs))]
        
        return imgs, labels, exps

    def __getitem__(self, item):
        if isinstance(item, slice):
            img, label, exp = [], [], []
            
            for imName, expName in zip(self._imNames[item], self._expNames[item]):
                img.append(Image.open(self._path / "images" / imName).convert('RGB'))
                exp.append(Image.open(self._path / "annotations" / expName))
                
        elif isinstance(item, int):
            img = Image.open(self._path / "images" / self._imNames[item]).convert('RGB')
            exp = Image.open(self._path / "annotations" / self._expNames[item])
        else:
            raise TypeError('Invalid argument type.')
        
        label = self._imLabels[item]

        return img, label, exp

    def __len__(self) -> int:
        return _oxford_iit_length
        
    def _download(self) -> bool: # pragma: no cover
        queryResponse = query_yes_no('This download will take ~800MB of disk. Procede?')
        try:
            if queryResponse:
                _download_extract_file(self._path, _oxford_iit_url, 'images.tar.gz', 'tar')
                _download_extract_file(self._path, _oxford_iit_masks_url, 'annotations.tar.gz', 'tar',
                                       deletePrevDir = False)
                return True
            else:
                shutil.rmtree(self._path)
                warnings.warn("Dataset download aborted.")
                return False
        except:
            warnings.warn("Download interruped.")
            shutil.rmtree(self._path)
        
    def _check_integrity(self) -> bool:
        return (_check_pathlib_dir(self._path / 'images') and
                _check_pathlib_dir(self._path / 'annotations'))
        
    def _organize(self) -> None: # pragma: no cover
        """ Given the extracted files, deletes folders that 
        are not used and performs other preparation instructions.
        """
        os.remove(self._path / 'annotations/README')
        os.remove(self._path / 'annotations/test.txt')
        os.remove(self._path / 'annotations/trainval.txt')
        os.remove(self._path / 'annotations/._trimaps')
        os.rename(self._path / "annotations/list.txt", 
                  self._path / "list.txt")
        
        shutil.rmtree(self._path / "annotations/xmls")
        
        # convert annotations to ..255 images, where 0 is background
        # not annotated and 255 is foreground
        for filename in os.scandir(self._path / "annotations/trimaps"):
            if filename.is_file() and filename.name[0] != ".":
                arr = np.array(Image.open(self._path / "annotations/trimaps" / filename))
                corrImg = Image.fromarray(np.uint8(np.where(arr == 1, 255, 0)))
                corrImg.save(self._path / "annotations" / filename.name)
                os.remove(self._path / "annotations/trimaps" / filename)
        shutil.rmtree(self._path / "annotations/trimaps")
        return
            
    def _get_class_map(self) -> dict:
        with open(self._path / "list.txt", "r") as file:
            lines = file.readlines()[6:]
            
        classMap = {}
        for c in lines:
            r = c.split(" ")
            species = "cat" if r[0][0].isupper() else "dog"
            index = r[0].rfind("_")
            className = f"{species}_{r[0][:index]}"
            classId = int(r[1]) - 1
            if not classId in classMap:
                classMap[classId] = className
            
        labelMap = {v[4:]: k for k, v in classMap.items()}
        return classMap, labelMap


# Data utils

def delete_sm_data() -> None:
    """Removes from internal storage all downloaded Saliency Map datasets. 
    See the :code:`delete_data` method of all Saliency Map datasets to delete only 
    their corresponding data.
    """
    smPath = Path(__file__).parent.parent.absolute() / "_datasets/saliencyMap/"
    try:
        shutil.rmtree(smPath)
    except FileNotFoundError:
        warnings.warn("There is no data downloaded.")
        raise FileNotFoundError

def rgb_to_grayscale(img):
    """ Transforms a 3 channel RGB image into a grayscale image (1 channel).

     :param np.ndarray img: of shape (imageH, imageW, 3)
     :return np.ndarray: of shape (imageH, imageW) """
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

def binarize_rgb_mask(img, bgValue='high') -> np.array:
    """ Binarizes a RGB binary mask, letting the background (negative class) be 0. Use this function when the image to
    binarize has a very defined background.

    :param img: (ndarray) of shape (imageH, imageW, 3), RGB mask to binarize.
    :param bgValue: (str) Intensity of the negative class of the image to binarize: {'high', 'low'}
    :return: (ndarray) a binary mask. """

    if bgValue not in {'high', 'low'}:
        raise ValueError(f"bgColor should ve {['high', 'low']}")

    imgmod = rgb_to_grayscale(img)
    maxVal = np.max(imgmod)
    minVal = np.min(imgmod)
    res = np.zeros((img.shape[0], img.shape[1]))
    if bgValue == 'high':
        # assign darker pixels the positive class
        res[imgmod < maxVal] = 1
    elif bgValue == 'low':
        # assign lighter pixels the positive class
        res[imgmod > minVal] = 1
    return res