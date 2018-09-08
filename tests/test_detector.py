import cv2
from HalloPy.hallopy.controller import Detector, FlagsHandler, BackGroundRemover
from HalloPy.util.image_comp_tool import ImageTestTool
from hallopy import utils


class TestDetector:
    """Unittests for a Detector object.  """

    def test_find_largest_contours(self):
        """Test if largest contours is found.  """
        # setup
        test_path = utils.get_full_path('docs/back_ground_removed_frame.jpg')
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
        flags_handler = FlagsHandler()
        detector = Detector(flags_handler)

        # run
        detector.input_frame_for_feature_extraction = test_image
        result_area = cv2.contourArea(detector.max_area_contour)
        # cv2.imshow('expected', expected_gray)
        # cv2.imshow('result', detector.input_frame_for_feature_extraction)
        # cv2.waitKey()

        assert result_area == expected_area

    def test_draw_axes(self):
        """Test if detected_out_put_center calculated properly.  """
        # setup
        test_path = utils.get_full_path('docs/back_ground_removed_frame.jpg')
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

    def test_threshold_change(self):
        """Test if threshold is changed accordingly to flags_handler.  """
        # setup
        # Input from camera.
        cv2.namedWindow('test')
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
                cv2.imshow('test', result)
                flags_handler.keyboard_input = cv2.waitKey(1)

        # teardown
        cap.release()
        cv2.destroyAllWindows()
