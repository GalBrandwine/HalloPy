import os

import cv2
import numpy as np
from HalloPy.hallopy.detector import Detector
from HalloPy.util.image_comp_tool import ImageTestTool
from HalloPy.util import files


class TestDetector:
    """Unittests for a Detector object.  """

    def test_set_frame(self):
        """Test if set_frame perform prepossessing correctly.  """
        # setup
        test_path = files.get_full_path('docs/face_and_hand.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from web-cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)
        expected = test_image.copy()
        expected = cv2.bilateralFilter(expected, 5, 50, 100)  # smoothing filter
        expected = cv2.flip(expected, 1)

        # Create detector
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
        # Because image loaded from local, and not received from web-cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)
        expected = test_image.copy()
        expected = cv2.bilateralFilter(expected, 5, 50, 100)  # smoothing filter
        expected = cv2.flip(expected, 1)
        faces = ImageTestTool.detect_faces(expected)
        ImageTestTool.draw_black_recs(expected, faces)

        # Create detector
        detector = Detector()
        detector.set_frame(test_image)

        # run
        # range [-1, 1] with a value of one being a “perfect match”.
        ssim = ImageTestTool.compare_imaged(detector.out_put_frame, expected)
        assert ssim >= 0.95

    def test_find_largest_contours(self):
        """Test if largest contours is found.  """
        # setup
        test_path = files.get_full_path('docs/hand_contour.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from web-cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)
        test_image = cv2.bitwise_not(test_image)
        # Get the contours.
        expected_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(expected_gray, (41, 41), 0)
        _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the biggest area
        max_area_contour = max(contours, key=cv2.contourArea)
        expected_area = cv2.contourArea(max_area_contour)
        # Create detector
        detector = Detector()
        detector.find_largest_contour(test_image)

        # run
        result_area = cv2.contourArea(detector.max_area_contour)
        assert result_area == expected_area

    def test_draw_axes(self):
        """Test if detected_out_put_center calculated properly.  """
        # setup
        test_path = files.get_full_path('docs/hand_contour.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from web-cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)
        expected = test_image.copy()
        roi = {'cap_region_y_end': 0.6, 'cap_region_x_begin': 0.6}
        expected = ImageTestTool.clip_roi(expected, roi)
        # Create detector
        detector = Detector()
        expected_detected_out_put_center = (
        int(expected.shape[1] / 2), int(expected.shape[0] / 2) + detector.horiz_axe_offset)

        # run
        detector.set_frame(test_image)
        assert expected_detected_out_put_center == detector.detected_out_put_center
