import os

import cv2
import numpy as np
from HalloPy.hallopy.detector import Detector
from HalloPy.util.image_comp_tool import ImageTestTool
from HalloPy.util import files


class TestDetector:
    def test_set_frame(self):
        """Test if set_frame perform prepossessing correctly.  """

        # setup
        test_path = files.get_full_path('docs/face_and_hand.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)
        expected = test_image.copy()
        expected = cv2.bilateralFilter(expected, 5, 50, 100)  # smoothing filter
        expected = cv2.flip(expected, 1)

        detector = Detector()
        detector.set_frame(test_image)

        # run
        # range [-1, 1] with a value of one being a “perfect match”.
        ssim = ImageTestTool.compare_imaged(detector.input_frame, expected)
        # print("SSIM: {}".format(ssim))
        assert ssim >= 0.95

    def test_cover_faces(self):
        """Test if cover_faces cover detected faces with black rec's correctly.  """

        # setup
        test_path = files.get_full_path('docs/testing_img.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)
        expected = test_image.copy()
        expected = cv2.bilateralFilter(expected, 5, 50, 100)  # smoothing filter
        expected = cv2.flip(expected, 1)
        faces = ImageTestTool.detect_faces(expected)
        ImageTestTool.draw_black_recs(expected, faces)

        detector = Detector()
        detector.set_frame(test_image)


        # run
        # range [-1, 1] with a value of one being a “perfect match”.
        ssim = ImageTestTool.compare_imaged(detector.out_put_frame, expected)
        # print("SSIM: {}".format(ssim))
        assert ssim >= 0.95

    def test_remove_back_ground(self):
        """Test if back ground removed correctly.  """

        # setup
        test_path = files.get_full_path('docs/face_and_hand_2.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)
        expected = test_image.copy()
        expected = cv2.bilateralFilter(expected, 5, 50, 100)  # smoothing filter
        expected = cv2.flip(expected, 1)
        faces = ImageTestTool.detect_faces(expected)
        ImageTestTool.draw_black_recs(expected, faces)

        detector = Detector()
        detector.set_frame(test_image)

        cv2.imshow('detected', detector.detected)
        cv2.waitKey()
