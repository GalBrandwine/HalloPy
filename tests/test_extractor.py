import cv2
import numpy as np
from hallopy import utils
from hallopy.controller import FlagsHandler, Detector, Extractor, BackGroundRemover, FrameHandler, FaceProcessor
from util.image_comp_tool import ImageTestTool


class TestExtractor:
    """Test extractor functionality.  """

    def test_extract_center_of_mass(self):
        """Test if extract find center of mass.  """
        # setup
        test_path = utils.get_full_path('docs/material_for_testing/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)
        expected_path = utils.get_full_path(
            'docs/material_for_testing/back_ground_removed_and_center_of_mass_discovered.jpg')
        expected_image = cv2.imread(expected_path)
        # Because image loaded from local, and not received from web-cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)

        # todo: use mockito here to mock detector
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)
        detector.input_frame_for_feature_extraction = test_image

        # run
        extractor.extract = detector
        result_image = test_image.copy()
        cv2.circle(result_image, extractor.palm_center_point, 5, (255, 0, 0), thickness=5)
        ssim = ImageTestTool.compare_imaged(result_image, expected_image)
        # print("SSIM: {}".format(ssim))
        assert ssim >= 0.95

    def test_get_contour_extreme_point(self):
        """Test if middle finger edge was found correctly.  """
        # setup
        test_path = utils.get_full_path('docs/material_for_testing/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)

        max_area_contour = ImageTestTool.get_max_area_contour(test_image)
        expected_extLeft, expected_extRight, expected_extTop, expected_extBot = ImageTestTool.get_contour_extreme_points(
            max_area_contour)

        # todo: use mockito here to mock detector
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)
        detector.input_frame_for_feature_extraction = test_image

        # run
        extractor.extract = detector

        assert expected_extLeft == extractor.ext_left
        assert expected_extRight == extractor.ext_right
        assert expected_extTop == extractor.ext_top
        assert expected_extBot == extractor.ext_bot

    def test_contour_extreme_point_tracking(self):
        """Test for tracking extreme_points without optical flow (e.g until calibrated).  """
        # setup
        test_path = utils.get_full_path('docs/material_for_testing/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)

        # todo: use mockito here to mock preprocessing elements
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)

        # Background model preparations.
        bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)

        cap = cv2.VideoCapture(0)
        while flags_handler.quit_flag is False:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Remove background from input frame.
            fgmask = bg_model.apply(frame, learningRate=0)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            res = cv2.bitwise_and(frame, frame, mask=fgmask)

            # Clip frames ROI.
            back_ground_removed_clipped = ImageTestTool.clip_roi(res,
                                                                 {'cap_region_x_begin': 0.6, 'cap_region_y_end': 0.6})

            if flags_handler.background_capture_required is True:
                bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
                flags_handler.background_capture_required = False

            detector.input_frame_for_feature_extraction = back_ground_removed_clipped
            extractor.extract = detector

            image = extractor.get_drawn_extreme_contour_points()
            cv2.imshow('test_contour_extreme_point_tracking', image)
            flags_handler.keyboard_input = cv2.waitKey(1)

    def test_palm_angle_calculation(self):
        """Test if angle is calculated correctly.

        Usage:
            1. press 'b': to calibrate back_ground_remover.
            2. insert hand into frame, so that middle_finger is aligned with the Y axe.
            3. rotate hand 15 degrees left. (degrees should go above 90).
            4. rotate hand 15 degrees right. (degrees should go below 90).
            5. press esc when done.
        """
        # setup
        # todo: use mockito here to mock preprocessing elements
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)

        # Background model preparations.
        bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
        cap = cv2.VideoCapture(0)

        while flags_handler.quit_flag is False:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Remove background from input frame.
            fgmask = bg_model.apply(frame, learningRate=0)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            res = cv2.bitwise_and(frame, frame, mask=fgmask)

            # Clip frames ROI.
            back_ground_removed_clipped = ImageTestTool.clip_roi(res,
                                                                 {'cap_region_x_begin': 0.6, 'cap_region_y_end': 0.6})

            if flags_handler.background_capture_required is True:
                bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
                flags_handler.background_capture_required = False

            detector.input_frame_for_feature_extraction = back_ground_removed_clipped
            extractor.extract = detector

            # run
            image = extractor.get_drawn_extreme_contour_points()
            cv2.imshow('test_contour_extreme_point_tracking', image)
            print(extractor.palm_angle_in_degrees)
            flags_handler.keyboard_input = cv2.waitKey(1)

    def test_5_second_calibration_time(self):
        """Test if 5 second calibration time works correctly according to flags_handler.

        Usage:
            1. press 'b': to calibrate back_ground_remover.
            2. insert hand into frame, center palms_center (white dot) with axes crossing.
            3. wait for #calibration_time (default 5 sec).
            4. press esc

        test: after calibration_time, center circle should be green.
        """
        # setup
        # todo: use mockito here to mock preprocessing elements
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)

        # Background model preparations.
        bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
        cap = cv2.VideoCapture(0)

        while flags_handler.quit_flag is False:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Remove background from input frame.
            fgmask = bg_model.apply(frame, learningRate=0)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            res = cv2.bitwise_and(frame, frame, mask=fgmask)

            # Clip frames ROI.
            back_ground_removed_clipped = ImageTestTool.clip_roi(res,
                                                                 {'cap_region_x_begin': 0.6, 'cap_region_y_end': 0.6})

            if flags_handler.background_capture_required is True:
                bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
                flags_handler.background_capture_required = False

            detector.input_frame_for_feature_extraction = back_ground_removed_clipped
            extractor.extract = detector

            # run
            image = extractor.get_drawn_extreme_contour_points()
            cv2.imshow('test_contour_extreme_point_tracking', image)
            flags_handler.keyboard_input = cv2.waitKey(1)

    def test_max_distance_between_top_ext_point_and_palm_center_point(self):
        """Test if max distance is found correctly. """
        # setup
        # todo: use mockito here to mock preprocessing elements
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)

        # Background model preparations.
        bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)

        cap = cv2.VideoCapture(0)
        while flags_handler.quit_flag is False:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Remove background from input frame.
            fgmask = bg_model.apply(frame, learningRate=0)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            res = cv2.bitwise_and(frame, frame, mask=fgmask)

            # Clip frames ROI.
            back_ground_removed_clipped = ImageTestTool.clip_roi(res,
                                                                 {'cap_region_x_begin': 0.6, 'cap_region_y_end': 0.6})

            if flags_handler.background_capture_required is True:
                bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
                flags_handler.background_capture_required = False

            detector.input_frame_for_feature_extraction = back_ground_removed_clipped
            extractor.extract = detector

            # run
            image = extractor.get_drawn_extreme_contour_points()
            cv2.line(image, extractor.palm_center_point, (extractor.ext_top[0], extractor.palm_center_point[
                1] - extractor.max_distance_from_ext_top_point_to_palm_center), (255, 255, 255), thickness=2)
            cv2.imshow('test_max_distance_between_top_ext_point_and_palm_center_point', image)
            flags_handler.keyboard_input = cv2.waitKey(1)

    def test_drawn_correctly(self):
        """Test if zero point is drawn correctly.

        zero point is the point that responsible for forward/backward commands extraction.
        """
        # setup
        # todo: use mockito here to mock preprocessing elements
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)

        # Background model preparations.
        bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)

        cap = cv2.VideoCapture(0)
        while flags_handler.quit_flag is False:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Remove background from input frame.
            fgmask = bg_model.apply(frame, learningRate=0)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            res = cv2.bitwise_and(frame, frame, mask=fgmask)

            # Clip frames ROI.
            back_ground_removed_clipped = ImageTestTool.clip_roi(res,
                                                                 {'cap_region_x_begin': 0.6, 'cap_region_y_end': 0.6})

            if flags_handler.background_capture_required is True:
                bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
                flags_handler.background_capture_required = False

            detector.input_frame_for_feature_extraction = back_ground_removed_clipped
            extractor.extract = detector

            # run
            image = extractor.get_drawn_extreme_contour_points()
            cv2.imshow('test_drawn_correctly', image)
            flags_handler.keyboard_input = cv2.waitKey(1)
