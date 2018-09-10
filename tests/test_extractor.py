import cv2

from hallopy import utils
from hallopy.controller import FlagsHandler, Detector, Extractor, BackGroundRemover, FrameHandler, FaceProcessor
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

    def test_get_contour_extreme_point(self):
        """Test if middle finger edge was found correctly.  """
        # setup
        test_path = utils.get_full_path('docs/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)

        max_area_contour = ImageTestTool.get_max_area_contour(test_image)
        # expected_area = ImageTestTool.get_contour_area(max_area_contour)
        expected_extLeft, expected_extRight, expected_extTop, expected_extBot = ImageTestTool.get_contour_extreme_points(
            max_area_contour)

        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)
        # todo: use mockito here to stub detector
        detector.input_frame_for_feature_extraction = test_image

        # run
        extractor.extract = detector

        assert expected_extLeft == extractor.extLeft
        assert expected_extRight == extractor.extRight
        assert expected_extTop == extractor.extTop
        assert expected_extBot == extractor.extBot

    def test_contour_extreme_point_tracking(self):
        """Test for tracking extreme_points withous optical flow.  """
        # setup
        test_path = utils.get_full_path('docs/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)

        flags_handler = FlagsHandler()
        frame_handler = FrameHandler()
        face_Processor = FaceProcessor()
        background_remover = BackGroundRemover(flags_handler)
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)
        # todo: use mockito here to stub detector
        detector.input_frame_for_feature_extraction = test_image

        cap = cv2.VideoCapture(0)

        while flags_handler.quit_flag is False:
            ret, frame = cap.read()
            # Processing pipe.
            frame_handler.input_frame = frame
            face_Processor.face_covered_frame = frame_handler.input_frame
            background_remover.detected_frame = face_Processor.face_covered_frame
            detector.input_frame_for_feature_extraction = background_remover.detected_frame
            extractor.extract = detector

            image = extractor.get_drawn_extreme_contour_points()
            cv2.imshow('test_contour_extreme_point_tracking', image)
            flags_handler.keyboard_input = cv2.waitKey(1)

    def test_palm_angle_calculation(self):
        # setup
        test_path = utils.get_full_path('docs/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)

        flags_handler = FlagsHandler()
        frame_handler = FrameHandler()
        face_Processor = FaceProcessor()
        background_remover = BackGroundRemover(flags_handler)
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)
        # todo: use mockito here to stub detector
        detector.input_frame_for_feature_extraction = test_image

        cap = cv2.VideoCapture(0)

        while flags_handler.quit_flag is False:
            ret, frame = cap.read()
            # Processing pipe.
            frame_handler.input_frame = frame
            face_Processor.face_covered_frame = frame_handler.input_frame
            background_remover.detected_frame = face_Processor.face_covered_frame
            detector.input_frame_for_feature_extraction = background_remover.detected_frame
            extractor.extract = detector

            image = extractor.get_drawn_extreme_contour_points()
            cv2.imshow('test_contour_extreme_point_tracking', image)
            print(extractor.palm_angle_in_degrees)
            flags_handler.keyboard_input = cv2.waitKey(1)

    def test_optical_flow(self):
        """Test if optical flow track correctly 5 points os interest.

        points for tracking:
            expected_extLeft
            expected_extRight
            expected_extTop
            expected_extBot
            palm_center_point
        """
