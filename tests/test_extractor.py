import cv2

from hallopy import utils
from hallopy.controller import FlagsHandler, Detector, Extractor
from util.image_comp_tool import ImageTestTool


class TestExtractor:
    """Test extractor functionality.  """

    def test_extract_center_of_mass(self):
        """Test if extract find center of mass.  """
        # setup
        test_path = utils.get_full_path('docs/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)
        expected_path = utils.get_full_path('docs/back_ground_removed_and_center_of_mass_discovered.jpg')
        expected_image = cv2.imread(expected_path)
        # Because image loaded from local, and not received from web-cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)

        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)
        # todo: use mockito here to stub detector
        detector.input_frame_for_feature_extraction = test_image

        # run
        extractor.extract = detector
        result_image = test_image.copy()
        cv2.circle(result_image, extractor.palm_center_point, 5, (255, 0, 0), thickness=5)
        # cv2.imshow('exepcted', expected_image)
        # cv2.imshow('result', result_image)
        # cv2.waitKey()
        ssim = ImageTestTool.compare_imaged(result_image, expected_image)
        # print("SSIM: {}".format(ssim))
        assert ssim >= 0.95

    def test_find_middle_finger_edge(self):
        """Test if middle finger edge was found correctly.  """
        # setup
        test_path = utils.get_full_path('docs/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)

        max_area_contour = ImageTestTool.get_max_area_contour(test_image)
        # expected_area = ImageTestTool.get_contour_area(max_area_contour)
        expected_middle_finger_edge_coord = ImageTestTool.get_middle_finger_edge_coord(max_area_contour)

        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)
        # todo: use mockito here to stub detector
        detector.input_frame_for_feature_extraction = test_image

        # run
        extractor.extract = detector
        result_image = test_image.copy()
        cv2.circle(result_image, extractor.middle_finger_edge, 5, (0, 255, 0), thickness=-1)
        # cv2.imshow('exepcted', extractor._detected_hand)
        cv2.imshow('result', result_image)
        cv2.waitKey()
