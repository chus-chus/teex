import unittest
import warnings

from teex._utils import _data
from teex._utils import _misc
from teex._utils import _paths
from teex._utils import _arrays
from teex._utils import _errors

import string
import os
import pathlib
import shutil
import numpy as np

from teex.__version__ import __version__


class TestVersion(unittest.TestCase):
    
    def test_ver(self):
        self.assertIsNotNone(__version__)

class TestDataUtils(unittest.TestCase):
    
    def test_query_yes_no_true(self):
        res = _data.query_yes_no("", default="yes", _expectInput=False)
        self.assertTrue(res)
        
    def test_query_yes_no_false(self):
        res = _data.query_yes_no("", default="no", _expectInput=False)
        self.assertFalse(res)
        
    def test_query_yes_no_exc_1(self):
        self.assertRaises(ValueError, _data.query_yes_no, "", "invalidDefault!", False)
        
    def test_query_yes_no_exc_2(self):
        self.assertRaises(ValueError, _data.query_yes_no, "", None, False)
        
class TestMiscUtils(unittest.TestCase):
    
    def test_gen_feature_names_small(self):
        f = _misc._generate_feature_names(3)
        self.assertEqual(f, ['a', 'b', 'c'])
        
    def test_gen_feature_names_big(self):
        f = _misc._generate_feature_names(len(string.ascii_letters) + 1)
        self.assertEqual(string.ascii_letters + "aa", "".join(f))
        
    def test_download_extract(self):
        url = "https://zenodo.org/record/6670800/files/daviddiazvico/scikit-datasets-v0.2.1.zip?download=1"
        path = pathlib.Path().absolute() / "tests/zipTest"
        r = _misc._download_extract_file(path, url, "zipFile")
        self.assertTrue(r)
        shutil.rmtree(path)

class TestPathUtils(unittest.TestCase):
    
    def setUp(self) -> None:
        self.path = pathlib.Path().absolute() / "tests"
    
    def test_check_dir(self):
        self.assertIsNone(_paths._check_dir(self.path))
        
    def test_check_dir_exc_1(self):
        self.assertRaises(Exception, _paths._check_dir, self.path / "notExistant")
        
    def test_check_dir_exc_2(self):
        os.mkdir(self.path / "emptyDir")
        self.assertRaises(Exception, _paths._check_dir, self.path / "emptyDir")
        shutil.rmtree(self.path / "emptyDir")
        
    def test_check_create_dir(self):
        self.assertIsNone(_paths._check_and_create_dir(self.path / "testInstance"))
        os.rmdir(self.path / "testInstance")
        
    def test_check_file(self):
        # create file
        with open(self.path / 'testFile', 'a'):
            os.utime(self.path / 'testFile', None)
        self.assertIsNone(_paths._check_file(self.path / 'testFile'))
        os.remove(self.path / 'testFile')

class TestArrays(unittest.TestCase):
    
    def test_bin_array_wtype(self):
        self.assertRaises(TypeError, _arrays._binarize_arrays, None, 'abs', .5)
    
    def test_bin_array_wthres(self):
        self.assertRaises(ValueError, _arrays._binarize_arrays, [1], 'abs', -1)
        
    def test_bin_array_wmethod(self):
        self.assertRaises(ValueError, _arrays._binarize_arrays, [], 'wrong!', 0)
    
    def test_bin_array_list(self):
        r = _arrays._binarize_arrays([1, 1, 1], "abs", 1)
        self.assertTrue((np.array([0, 0, 0]) == r).all())
        
    def test_bin_array_thres_np(self):
        r = _arrays._binarize_arrays(np.array([1, 1, 1]), "thres", 1)
        self.assertTrue((np.array([0, 0, 0]) == r).all())
        
    def test_bin_array_thres_list(self):
        r = _arrays._binarize_arrays([1, 1, 1], "thres", 1)
        self.assertTrue((np.array([0, 0, 0]) == r).all())
    
    def test_norm_bin_mask(self):
        r = _arrays._normalize_binary_mask(np.array([1, 0, 1]))
        self.assertTrue((r == np.array([1., 0., 1.])).all())
        
    def test_arr_is_rgb(self):
        r = _arrays._is_rgb(np.array([[[1, 1, 1]]]))
        self.assertTrue(r)
    
    def test_scale_arr_case_1(self):
        r, _ = _arrays._scale_array(np.array([2, 2]))
        self.assertTrue((r == np.array([1, 1])).all())
    
    def test_scale_arr_case_2(self):
        r, _ = _arrays._scale_array(np.array([-2, 0]))
        self.assertTrue((r == np.array([-1, 1])).all())
    
    def test_scale_arr_case_3(self):
        r, _ = _arrays._scale_array(np.array([-2, 2]), verbose=True)
        self.assertTrue((r == np.array([-1, 1])).all())
        
class TestErrors(unittest.TestCase):
    
    def raiser(self, error, *kwargs):
        raise error(kwargs)
    
    def test_metric_na(self):
        self.assertRaises(_errors.MetricNotAvailableError, 
                          self.raiser, 
                          _errors.MetricNotAvailableError, 
                          ["sampleMetric", "Message"])
    
    def test_inc_GT(self):
        self.assertRaises(_errors.IncompatibleGTAndPredError, 
                          self.raiser, 
                          _errors.IncompatibleGTAndPredError, 
                          "Message")

    def test_failed_DS_DW(self):
        self.assertRaises(_errors.FailedDataSetDownloadError, 
                          self.raiser, 
                          _errors.FailedDataSetDownloadError, 
                          "Message")
    
    def test_failed_DS_extraction(self):
        self.assertRaises(_errors.FailedDataSetExtractionError, self.raiser, _errors.FailedDataSetExtractionError, "Message")