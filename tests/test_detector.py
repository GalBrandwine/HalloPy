import os

import cv2
import numpy as np
from HalloPy.hallopy.detector import Detector
from HalloPy.util.image_comp_tool import ImageTestTool
from HalloPy.util.files import get_full_path


class TestDetector:
    def test_set_frame(self):
        """Test if set_frame perform prepossessing correctly.  """

        # setup
        test_path = get_full_path('docs/testing_img.jpg')
        test_image = cv2.imread(test_path)
        expected = test_image.copy()

        expected = cv2.bilateralFilter(expected, 5, 50, 100)  # smoothing filter
        expected = cv2.flip(expected, 1)

        detector = Detector()
        detector.set_frame(test_image)

        # run
        # range [-1, 1] with a value of one being a “perfect match”.
        ssim = ImageTestTool.compare_imaged(detector.input_frame, expected)
        assert ssim >= 0.9

    def test_cover_faces(self):
        """Test if cover_faces cover detected faces with black rec's correctly.  """

        # setup
        test_path = get_full_path('docs/testing_img.jpg')
        test_image = cv2.imread(test_path)
        expected = test_image.copy()

        expected = cv2.bilateralFilter(expected, 5, 50, 100)  # smoothing filter
        expected = cv2.flip(expected, 1)

        detector = Detector()
        detector.set_frame(test_image)

        # run
        # range [-1, 1] with a value of one being a “perfect match”.
        ssim = ImageTestTool.compare_imaged(detector.input_frame, expected)
        assert ssim >= 0.9
