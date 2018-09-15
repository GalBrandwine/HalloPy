import cv2
from HalloPy.hallopy.controller import Detector, FlagsHandler, Extractor
from hallopy import utils
from util.image_comp_tool import ImageTestTool
import numpy as np


class TestDetector:
    """Unittests for a Detector object.  """

    def test_find_largest_contours(self):
        """Test if largest contours is found.  """
        # setup
        test_path = utils.get_full_path('docs/material_for_testing/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from web-cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)
        test_image = cv2.bitwise_not(test_image)

        max_area_contour = ImageTestTool.get_max_area_contour(test_image)
        expected_area = ImageTestTool.get_contour_area(max_area_contour)
        # Create detector
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)

        # run
        detector.input_frame_for_feature_extraction = test_image
        result_area = cv2.contourArea(detector.max_area_contour)

        assert result_area == expected_area

    def test_draw_axes(self):
        """Test if detected_out_put_center calculated properly.  """
        # setup

        test_path = utils.get_full_path('docs/material_for_testing/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from web-cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)
        expected = test_image.copy()
        # Create detector
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        expected_detected_out_put_center = (
            int(expected.shape[1] / 2), int(expected.shape[0] / 2) + detector.horiz_axe_offset)

        # run
        detector.input_frame_for_feature_extraction = test_image
        cv2.imshow('expected', expected)
        cv2.imshow('result', detector.input_frame_for_feature_extraction)
        cv2.waitKey()
        assert expected_detected_out_put_center == detector.detected_out_put_center

    def test_draw_contour(self):
        """Test is contour is being drawn accordingly to flags_handles.  """
        # setup
        # Input from camera.
        cv2.namedWindow('test_draw_contour')

        test_path = utils.get_full_path('docs/material_for_testing/back_ground_removed_frame.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from web-cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)
        expected = test_image.copy()
        flags_handler = FlagsHandler()
        # Set flags_handler in order to perform the test.
        flags_handler.lifted = True
        flags_handler.calibrated = True
        detector = Detector(flags_handler)

        # run
        while flags_handler.quit_flag is False:
            """
            Inside loop, update self._threshold according to flags_handler,
            
            Pressing 'c': in order to toggle control (suppose to change contour's color between green and red)
            Pressing 'l': to raise 'land' flag in flags_handler, in order to be able to break loop (with esc)
            Pressing esc: break loop.
            """
            detector.input_frame_for_feature_extraction = test_image
            cv2.imshow('test_draw_contour', detector.input_frame_for_feature_extraction)
            flags_handler.keyboard_input = cv2.waitKey(1)

        # teardown
        cv2.destroyAllWindows()

    def test_threshold_change(self):
        """Test if threshold is changed accordingly to flags_handler.  """
        # setup
        # Input from camera.
        cv2.namedWindow('test_threshold_change')
        cap = cv2.VideoCapture(0)
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)

        # run
        while flags_handler.quit_flag is False:
            """
            Inside loop, update self._threshold according to flags_handler,
            
            Pressing 'z': will make threshold thinner.
            Pressing 'x': will make threshold thicker.
            Pressing esc: break loop.
            """
            ret, frame = cap.read()
            if ret is True:
                detector.input_frame_for_feature_extraction = frame
                result = detector.input_frame_for_feature_extraction
                cv2.drawContours(result, [detector.max_area_contour], 0, (0, 0, 255), thickness=2)
                cv2.imshow('test_threshold_change', result)
                flags_handler.keyboard_input = cv2.waitKey(1)

        # teardown
        cap.release()
        cv2.destroyAllWindows()

    def test_detector_extract_and_track(self):
        """Test if Detector uses tracker object correctly.  """

        # setup
        # Input from camera.
        cv2.namedWindow('test_detector_extract_and_track')
        cap = cv2.VideoCapture(0)
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)
        extractor = Extractor(flags_handler)

        bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
        while flags_handler.quit_flag is False:
            """
            Inside loop, update self._threshold according to flags_handler,
            
            Pressing 'c': in order to toggle control (suppose to change contour's color between green and red)
            Pressing 'l': to raise 'land' flag in flags_handler, in order to be able to break loop (with esc)
            Pressing 'z': will make threshold thinner.
            Pressing 'x': will make threshold thicker.            
            Pressing esc: break loop.
            """
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Remove background from input frame.
            fgmask = bg_model.apply(frame, learningRate=0)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            res = cv2.bitwise_and(frame, frame, mask=fgmask)
            # Clip frames ROI.b
            roi = {'cap_region_x_begin': 0.6, 'cap_region_y_end': 0.6}
            back_ground_removed_clipped = ImageTestTool.clip_roi(res, roi)

            if flags_handler.background_capture_required is True:
                bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
                flags_handler.background_capture_required = False

            # Pipe:
            detector.input_frame_for_feature_extraction = back_ground_removed_clipped
            extractor.extract = detector

            cv2.imshow('test_detector_extract_and_track', extractor.get_drawn_extreme_contour_points())
            keyboard_input = cv2.waitKey(1)
            flags_handler.keyboard_input = keyboard_input

        # teardown
        cap.release()
        cv2.destroyAllWindows()
