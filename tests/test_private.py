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
        

class TestMiscUtils(unittest.TestCase):
    
    def test_gen_feature_names_small(self):
        f = _misc._generate_feature_names(3)
        self.assertEqual(f, ['a', 'b', 'c'])
        
    def test_gen_feature_names_big(self):
        f = _misc._generate_feature_names(len(string.ascii_letters) + 1)
        self.assertEqual(string.ascii_letters + "aa", "".join(f))
        
    def test_download_extract(self):
        pass # TODO implement
        # url = ""
        # r = _misc._download_extract_file()

class TestPathUtils(unittest.TestCase):
    
    def setUp(self) -> None:
        self.path = pathlib.Path().absolute() / "tests"
    
    def test_check_dir(self):
        self.assertIsNone(_paths._check_dir(self.path))
        
    def test_check_create_dir(self):
        self.assertIsNone(_paths._check_and_create_dir(self.path / "testInstance"))
        os.rmdir(self.path / "testInstance")
        
    def test_check_file(self):
        # create file
        with open(self.path / 'testFile', 'a'):
            os.utime(self.path / 'testFile', None)
        self.assertIsNone(_paths._check_file(self.path / 'testFile'))
        os.remove(self.path / 'testFile')
