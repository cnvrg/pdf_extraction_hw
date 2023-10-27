import unittest
from batch_predict import get_all_documents
import yaml
from extractor_hw import extraction
from yaml.loader import SafeLoader
import os
import cv2
YAML_ARG_TO_TEST = "test_arguments"

class Test_batch(unittest.TestCase):
    """Defining the sample data and files to carry out the testing"""

    def setUp(self):
        cfg_path = os.path.dirname(os.path.abspath(__file__))
        cfg_file = cfg_path + "/" + "test_config.yaml"
        self.test_cfg = {}
        with open(cfg_file) as c_info_file:
            self.test_cfg = yaml.load(c_info_file, Loader=SafeLoader)
        self.test_cfg = self.test_cfg[YAML_ARG_TO_TEST]
        self.pdf = self.test_cfg['pdf_path']
        self.img_path = self.test_cfg['img_path']
        self.img_content = self.test_cfg['img_content']
        self.extractor = extraction()


class Test_extractor(Test_batch):
    """Testing the extractor code used to extract text from pdfs"""

    def test_pdf(self):
        result = self.extractor._process_file_pdf(self.pdf)
        self.assertIsInstance(result, dict)
        for pagenumber,content in self.test_cfg['digital_pages'].items():
            self.assertEqual(result[pagenumber],content)

    def test_pdf_to_img(self):
        result = self.extractor.pdf_to_img(self.pdf)
        self.assertIsInstance(result, list)

    def test_do_ocr(self):
        img = cv2.imread(self.img_path)
        result = self.extractor.do_ocr(img)
        self.assertEqual(result, self.img_content)

    def test_extract_pdf(self):
        result = self.extractor.extract_pdf(self.pdf)
        self.assertIsInstance(result, dict)

class Test_batch_predict(Test_batch):
    """Testing batch_predict functions"""

    def test_get_all_documents(self):
        result = []
        result = get_all_documents(self.test_cfg['directory_path'],result)
        self.assertIsInstance(result, list)
        result.sort()
        given = self.test_cfg['files_in_directory']
        given.sort()
        self.assertListEqual(result, given)


