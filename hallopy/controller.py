"""Multi class incapsulation implementation.  """
import time

import av
import cv2
import logging
import numpy as np
from hallopy.icontroller import Icontroller
from hallopy import utils

# Create loggers.
flags_logger = logging.getLogger('flags_handler')
frame_logger = logging.getLogger('frame_handler')
face_processor_logger = logging.getLogger('face_processor_handler')
back_ground_remover_logger = logging.getLogger('back_ground_remover_handler')
detector_logger = logging.getLogger('detector_handler')
extractor_logger = logging.getLogger('extractor_handler')
controller_logger = logging.getLogger('controller_handler')
ch = logging.StreamHandler()
# create formatter and add it to the handlers.
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to loggers.
flags_logger.addHandler(ch)
frame_logger.addHandler(ch)
face_processor_logger.addHandler(ch)
back_ground_remover_logger.addHandler(ch)
detector_logger.addHandler(ch)
extractor_logger.addHandler(ch)
controller_logger.addHandler(ch)


class FlagsHandler:
    """Simple class for setting flags.  """

    def __init__(self):
        self.logger = logging.getLogger('flags_handler')
        self._key_board_input = None
        self.lifted = False
        self.takeoff_requested = False
        self.landing_requested = False
        self.quit_flag = False
        self.background_capture_required = True
        self.in_home_center = False
        self.is_bg_captured = False
        self.calibrated = False
        self.hand_control = False
        self.make_threshold_thinner = False
        self.make_threshold_thicker = False

    @property
    def keyboard_input(self):
        return self._key_board_input

    @keyboard_input.setter
    def keyboard_input(self, input_from_key_board):
        """State machine.  """
        if input_from_key_board == 27 and self.lifted is False:
            # press ESC to exit
            self.logger.info('!!!quiting!!!')
            self.quit_flag = True

        elif input_from_key_board == 27:
            self.logger.info('!!!cant quit without landing!!!')

        elif input_from_key_board == ord('b'):
            # press 'b' to capture the background
            self.calibrated = False
            self.background_capture_required = True
            self.is_bg_captured = True
            self.logger.info('!!!Background Captured!!!')

        elif input_from_key_board == ord('t') and self.calibrated is True:
            """Take off"""
            self.logger.info('!!!Take of!!!')
            self.lifted = True
            self.takeoff_requested = True

        elif input_from_key_board == ord('l'):
            """Land"""
            self.lifted = False
            self.landing_requested = True
            self.logger.info('!!!Landing!!!')

        elif input_from_key_board == ord('c'):
            """Control"""
            if self.hand_control is True:
                self.hand_control = False
                self.logger.info("control switched to keyboard")
            elif self.lifted is True:
                self.logger.info("control switched to detected hand")
                self.hand_control = True
            else:
                self.logger.info(
                    "Drone not in the air, can't change control to hand")

        elif input_from_key_board == ord('z'):
            """ calibrating Threshold from keyboard """
            self.make_threshold_thinner = True
            self.logger.info("made threshold thinner")

        elif input_from_key_board == ord('x'):
            """ calibrating Threshold from keyboard """
            self.logger.info("made threshold thicker")
            self.make_threshold_thicker = True


class FrameHandler:
    """FrameHandler handel input frame from controller,

    and perform some preprocessing.
    """
    _input_frame = ...  # type: np.ndarray

    def __init__(self):
        """Init preprocessing params.  """
        self.logger = logging.getLogger('frame_handler')
        self.logger.setLevel(logging.INFO)
        self._cap_region_x_begin = 0.6
        self._cap_region_y_end = 0.6
        self._input_frame = None

    @property
    def input_frame(self):
        # Returns the input frame, with drawn ROI on it.
        return self._input_frame

    @input_frame.setter
    def input_frame(self, input_frame_from_camera):
        """Setter with preprocessing.  """

        try:
            # make sure input is np.ndarray
            assert type(input_frame_from_camera).__module__ == np.__name__
        except AssertionError as error:
            self.logger.exception(error)
            return

        self._input_frame = cv2.bilateralFilter(
            input_frame_from_camera, 5, 50, 100)  # smoothing filter
        self._input_frame = cv2.flip(input_frame_from_camera, 1)
        self._draw_roi()

    def _draw_roi(self):
        """Function for drawing the ROI on input frame.  """
        cv2.rectangle(self._input_frame, (int(self._cap_region_x_begin * self._input_frame.shape[1]) - 20, 0),
                      (self._input_frame.shape[1], int(
                          self._cap_region_y_end * self._input_frame.shape[0]) + 20),
                      (255, 0, 0), 2)


class FaceProcessor:
    """FaceProcessor detect & cover faces in preprocessed input_frame.  """
    _preprocessed_input_frame = ...  # type: np.ndarray

    def __init__(self):
        self.logger = logging.getLogger('face_processor_handler')
        self.logger.setLevel(logging.INFO)
        self._face_detector = cv2.CascadeClassifier(
            utils.get_full_path('hallopy/config/haarcascade_frontalface_default.xml'))
        self._face_padding_x = 20
        self._face_padding_y = 60
        self._preprocessed_input_frame = None

    @property
    def face_covered_frame(self):
        """Return a face covered frame.  """
        return self._preprocessed_input_frame

    @face_covered_frame.setter
    def face_covered_frame(self, input_frame_with_faces):
        """Function to draw black recs over detected faces.

        This function remove eny 'noise' and help detector detecting palm.
        :param input_frame_with_faces (np.ndarray): a frame with faces, that needed to be covered.
        """

        try:
            # make sure input is np.ndarray
            assert type(input_frame_with_faces).__module__ == np.__name__
        except AssertionError as error:
            self.logger.exception(error)
            return

        # Preparation
        self._preprocessed_input_frame = input_frame_with_faces.copy()
        gray = cv2.cvtColor(self._preprocessed_input_frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(gray, 1.3, 5)

        # Black rectangle over faces to remove skin noises.
        for (x, y, w, h) in faces:
            self._preprocessed_input_frame[y - self._face_padding_y:y + h + self._face_padding_y,
                                           x - self._face_padding_x:x + w + self._face_padding_x, :] = 0


class BackGroundRemover:
    """BackGroundRemover removes background from inputted

     (preprocessed and face covered) frame.
     """
    _input_frame_with_hand = ...  # type: np.ndarray

    def __init__(self, flags_handler):
        self.logger = logging.getLogger('back_ground_remover_handler')
        self._cap_region_x_begin = 0.6
        self._cap_region_y_end = 0.6
        self._bg_Sub_Threshold = 50
        self._learning_Rate = 0
        self._bg_model = None
        self._input_frame_with_hand = None
        self.flag_handler = flags_handler

    @property
    def detected_frame(self):
        """Getter for getting the interest frame, with background removed.  """
        return self._input_frame_with_hand

    @detected_frame.setter
    def detected_frame(self, preprocessed_faced_covered_input_frame):
        """Function for removing background from input frame. """
        if self.flag_handler.background_capture_required is True:
            self._bg_model = cv2.createBackgroundSubtractorMOG2(
                0, self._bg_Sub_Threshold)
            self.flag_handler.background_capture_required = False
        if self._bg_model is not None:
            fgmask = self._bg_model.apply(
                preprocessed_faced_covered_input_frame, learningRate=self._learning_Rate)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            res = cv2.bitwise_and(preprocessed_faced_covered_input_frame, preprocessed_faced_covered_input_frame,
                                  mask=fgmask)
            self._input_frame_with_hand = res[
                0:int(
                    self._cap_region_y_end * preprocessed_faced_covered_input_frame.shape[0]),
                int(self._cap_region_x_begin * preprocessed_faced_covered_input_frame.shape[
                    1]):
                preprocessed_faced_covered_input_frame.shape[
                    1]]  # clip the ROI


class Detector:
    """Detector class detect hands contour and center of frame.

    Initiated object will receive a preprocessed frame, with detected & covered faces.
    """
    _input_frame_with_background_removed = ...  # type: np.ndarray

    def __init__(self, flags_handler):
        self.logger = logging.getLogger('detector_handler')
        self.flags_handler = flags_handler
        self._threshold = 50
        self._blur_Value = 41
        self.horiz_axe_offset = 60

        self._input_frame_with_background_removed = None
        self._detected_out_put = None
        self.raw_input_frame = None

        # max_area_contour: the contour of the detected hand.
        self.max_area_contour = None
        # Detected_out_put_center: the center point of the ROI
        self.detected_out_put_center = (0, 0)

    @property
    def input_frame_for_feature_extraction(self):
        return self._detected_out_put

    @input_frame_for_feature_extraction.setter
    def input_frame_for_feature_extraction(self, input_frame_with_background_removed):
        """Function for finding hand contour. """
        # Preparation
        # Update threshold
        self.raw_input_frame = input_frame_with_background_removed
        if self.flags_handler.make_threshold_thinner is True and self._threshold >= 0:
            self.flags_handler.make_threshold_thinner = False
            self._threshold = self._threshold - 1
        elif self.flags_handler.make_threshold_thicker is True and self._threshold <= 100:
            self.flags_handler.make_threshold_thicker = False
            self._threshold = self._threshold + 1

        temp_detected_gray = cv2.cvtColor(
            input_frame_with_background_removed, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(
            temp_detected_gray, (self._blur_Value, self._blur_Value), 0)
        thresh = cv2.threshold(blur, self._threshold,
                               255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Get the contours.
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            # Find the biggest area.
            self.max_area_contour = max(contours, key=cv2.contourArea)
            if self.max_area_contour is None:
                self.max_area_contour = [[(0, 0)]]
            self.detected_out_put_center = self._draw_axes(
                input_frame_with_background_removed)
        except (AttributeError, ValueError) as error:
            self.logger.debug(
                "something went wrong when Detector object received input_frame!: {}".format(error))

    def _draw_axes(self, detected):
        """Function for drawing axes on detected_out_put.

        :return detected_out_put_center (point): the center coord' of detected_out_put.
        """

        # Preparation
        temp_output = detected.copy()
        # np.array are opposite than cv2 row/cols indexing.
        detected_out_put_center = (
            int(temp_output.shape[1] / 2), int(temp_output.shape[0] / 2) + self.horiz_axe_offset)
        horiz_axe_start = (
            0, int(temp_output.shape[0] / 2) + self.horiz_axe_offset)
        horiz_axe_end = (
            temp_output.shape[1], int(temp_output.shape[0] / 2) + self.horiz_axe_offset)

        vertic_y_start = (int(temp_output.shape[1] / 2), 0)
        vertic_y_end = (int(temp_output.shape[1] / 2), temp_output.shape[0])

        # draw movement axes.
        cv2.line(temp_output, horiz_axe_start,
                 horiz_axe_end, (0, 0, 255), thickness=3)
        cv2.line(temp_output, vertic_y_start,
                 vertic_y_end, (0, 0, 255), thickness=3)

        self._draw_contours(temp_output)
        self._detected_out_put = temp_output
        return detected_out_put_center

    def _draw_contours(self, input_frame_with_background_removed):
        """Function for drawing contours of detected hand.

        contour color will accordingly to flags.hand_control flag.
        """
        hand_color = None
        if self.flags_handler.hand_control is True:
            hand_color = (0, 255, 0)
        else:
            hand_color = (0, 0, 255)
        assert hand_color is not None, self.logger.error(
            "No flags_handler.hand_control initiated")
        cv2.drawContours(input_frame_with_background_removed, [
                         self.max_area_contour], 0, hand_color, thickness=2)


class Extractor:
    """Extractor receives detected object,

    saves its 'center_of_frame' and 'max_contour'.
    and perform the following calculations:
    1. calculate palm center of mass --> palms center coordination.
    2. calculate distance between palms_center to frame_center.
    3. find contour extreme points coordination.
    4. calculate palms rotation.
    5. calculate top_ext_contour-palm_center max distance.
    """
    detector = ...  # type: Detector

    def __init__(self, flags_handler):
        self.logger = logging.getLogger('extractor_handler')
        self.flags_handler = flags_handler

        # detector hold: palms contour, frame_center, frame with drawn axes.
        self.detector = None
        # tracker tracks extractor palm point after calibration, using optical_flow
        self.tracker = None

        self._detected_hand = None
        self.calib_radius = 10

        self.calibration_time = 2
        self.time_captured = None

        self.palm_angle_in_degrees = 0
        self.palm_center_point = (0, 0)
        self.max_distance_from_ext_top_point_to_palm_center = 0

        self.forward_backward_movement_delta = 30
        self.zero_point = (0, 0)
        self.forward_point = (0, 0)
        self.backward_point = (0, 0)

        self.ext_left = (0, 0)
        self.ext_right = (0, 0)
        self.ext_top = (0, 0)
        self.ext_bot = (0, 0)

    @property
    def extract(self):
        return self._detected_hand

    @extract.setter
    def extract(self, detector):
        assert isinstance(detector, Detector), self.logger.error(
            "input is not Detector object!")
        self.detector = detector
        self._detected_hand = detector._detected_out_put
        # Calculate palm center of mass --> palms center coordination.
        self.palm_center_point = self._hand_center_of_mass(
            detector.max_area_contour)

        if self.flags_handler.calibrated is False:
            self.logger.info("calibrating...")
            # Determine the most extreme points along the contour.
            if detector.max_area_contour is not None:
                c = detector.max_area_contour
                self.ext_left = tuple(c[c[:, :, 0].argmin()][0])
                self.ext_right = tuple(c[c[:, :, 0].argmax()][0])
                self.ext_top = tuple(c[c[:, :, 1].argmin()][0])
                self.ext_bot = tuple(c[c[:, :, 1].argmax()][0])

                # Get max distance.
                if self.ext_top[1] == 0:
                    self.max_distance_from_ext_top_point_to_palm_center = 0
                else:
                    temp_distance = self.palm_center_point[1] - self.ext_top[1]
                    if temp_distance > self.max_distance_from_ext_top_point_to_palm_center:
                        self.max_distance_from_ext_top_point_to_palm_center = temp_distance

            if self.tracker is not None:
                self.tracker = None

        elif self.flags_handler.calibrated is True:
            self.logger.info("calibrated!")

            if self.tracker is None:
                # Initiate tracker.
                points_to_track = [self.ext_top,
                                   self.palm_center_point]  # [self.ext_left, self.ext_right, self.ext_bot, self.ext_top, self.palm_center_point]
                self.tracker = Tracker(
                    self.flags_handler, points_to_track, self.detector.raw_input_frame)

            else:
                # Use tracker to track.
                points_to_track = self.tracker.points_to_track
                self.tracker.track(
                    points_to_track, self.detector.raw_input_frame)
                points_to_draw = self.tracker.points_to_track
                try:
                    # Get only the contours middle-finger coordination.
                    self.ext_top = tuple(
                        points_to_draw[points_to_draw[:, :, 1].argmin()][0])
                except ValueError:
                    self.logger.debug("points_to_draw is empty")
        # Calculate palms angle.
        self._calculate_palm_angle()
        # Calculate distance between palms_center to frame_center.
        self._calculate_palm_distance_from_center()

    def get_drawn_extreme_contour_points(self):
        """Draw extreme contours points on a copy

        draw the outline of the object, then draw each of the
        extreme points, where the left-most is red, right-most
        is green, top-most is blue, and bottom-most is teal

        :returns image: image with drawn extreme contours point.
        """
        if self._detected_hand is not None:
            image = self._detected_hand.copy()

            if self.flags_handler.calibrated is True:
                cv2.circle(image, self.detector.detected_out_put_center,
                           self.calib_radius, (0, 255, 0), thickness=2)
            elif self.flags_handler.in_home_center is True:
                cv2.circle(image, self.detector.detected_out_put_center,
                           self.calib_radius, (0, 255, 0), thickness=-1)
            else:
                cv2.circle(image, self.detector.detected_out_put_center,
                           self.calib_radius, (0, 0, 255), thickness=2)

            self._draw_forward_backward_line(image)
            self._draw_palm_rotation(image)

            cv2.circle(image, (int(self.ext_top[0]), int(
                self.ext_top[1])), 8, (255, 0, 0), -1)
            cv2.circle(image, self.palm_center_point,
                       8, (255, 255, 255), thickness=-1)

            return image

    def _draw_forward_backward_line(self, image):
        """Draw forward/backward line.  """
        temp_delta = int(
            self.max_distance_from_ext_top_point_to_palm_center - self.max_distance_from_ext_top_point_to_palm_center / 5)
        self.zero_point = (
            self.ext_top[0], self.palm_center_point[1] - temp_delta)
        self.forward_backward_movement_delta = int(
            self.max_distance_from_ext_top_point_to_palm_center / 3)
        self.forward_point = (
            self.zero_point[0], self.zero_point[1] + self.forward_backward_movement_delta)
        self.backward_point = (
            self.zero_point[0], self.zero_point[1] - self.forward_backward_movement_delta)
        cv2.line(image, (int(self.forward_point[0]),int(self.forward_point[1])),
                 (int(self.zero_point[0]),int(self.zero_point[1])), (0, 255, 0), thickness=5)
        cv2.line(image, (int(self.zero_point[0]),int(self.zero_point[1])), (int(self.backward_point[0]),int(self.backward_point[1])),
                 (0, 0, 255), thickness=5)

    def _draw_palm_rotation(self, image):
        """To draw the ellipse, we need to pass several arguments.

        One argument is the center location (x,y).
        Next argument is axes lengths (major axis length, minor axis length).
        angle is the angle of rotation of ellipse in anti-clockwise direction.
        startAngle and endAngle denotes the starting and ending of ellipse arc measured in clockwise direction from major axis.
        i.e. giving values 0 and 360 gives the full ellipse. For more details, check the documentation of cv2.ellipse().
        """
        center_location = self.palm_center_point
        axis_length = int(self.max_distance_from_ext_top_point_to_palm_center)
        starting_angle = 270
        end_angle = 270 + (90 - self.palm_angle_in_degrees)
        cv2.ellipse(image, center_location, (axis_length, axis_length),
                    0, starting_angle, end_angle, (255, 0, 255), 3)

    def _hand_center_of_mass(self, hand_contour):
        """Find contours center of mass.  """
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        return cX, cY

    def _calculate_palm_angle(self):
        """Function for calculating palms angle.  """

        angelPointHelper = [self.palm_center_point[0] + 10,
                            self.palm_center_point[
                                1]]  # helper to calculate angle between middle finger and center of palm

        try:
            angle = self.simple_angle_calculator(
                self.ext_top, angelPointHelper, self.palm_center_point)
            self.palm_angle_in_degrees = np.rad2deg(angle)
        except ZeroDivisionError:
            pass

    def simple_angle_calculator(self, start, end, far):
        """Simple angle calculator.

        :returns angle: the angle in radians.
        """

        a = np.math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.math.acos((b ** 2 + c ** 2 - a ** 2) /
                             (2 * b * c))  # cosine theorem
        return angle

    def _calculate_palm_distance_from_center(self):
        """Simple radius calculator.  """
        frameCenter = self.detector.detected_out_put_center
        cX, cY = self.palm_center_point

        radius = np.math.sqrt((cX - frameCenter[0]) ** 2 + (
            cY - frameCenter[1]) ** 2)

        if radius <= self.calib_radius:
            # Palm is centered with self._detected_frame.
            if self.flags_handler.in_home_center is False:
                # First time entering into calib_circle, start timer.
                self.time_captured = time.time()
                self.flags_handler.in_home_center = True

            elif time.time() >= self.time_captured + self.calibration_time:
                # If inside calib_circle more than self.calibration_time, then set calibrated to True.
                self.flags_handler.calibrated = True
        else:
            self.flags_handler.in_home_center = False


class Tracker:
    """Tracker receives Extractor object, and track extracted points.  """

    def __init__(self, flags_handler, points_to_track, input_image):
        self.logger = logging.getLogger('tracker_handler')
        self.flags_handler = flags_handler
        self.points_to_track = points_to_track

        self._input_image = input_image
        self._old_gray = None
        self._p0 = None

        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.track(self.points_to_track, self._input_image)

    def track(self, points_to_track, input_image):
        if self._old_gray is None:
            self._old_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        points_reshaped = [list(elem) for elem in points_to_track]
        self.logger.debug("points_to_track: {}".format(points_reshaped))
        self._p0 = np.array(
            points_reshaped, dtype=np.float32).reshape(-1, 1, 2)

        # Capture current frame.
        frame_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        self._calculate_optical_flow(self._old_gray, frame_gray, self._p0)

        # Update tracking points.
        self.points_to_track = self._p0

    def _calculate_optical_flow(self, old_gray, frame_gray, p0):
        """This function tracks the edge of the Middle finger.

       points for tracking:
            expected_ext_left
            expected_ext_right
            expected_ext_top
            expected_ext_bot
            palm_center_point

        :param old_gray: old frame, gray scale
        :param frame_gray: current frame
        :return: p0- updated tracking point,

        """
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **self.lk_params)
        if p1 is None:
            good_new = p0[st == 1]
        else:
            good_new = p1[st == 1]

        # Now update the previous frame and previous points.
        self._old_gray = frame_gray.copy()
        self._p0 = good_new.reshape(-1, 1, 2)


class Controller(Icontroller):
    """Controller class holds all elements relevant for hand features extracting.

    :param icontroller.Icontroller: implemented interface
    """

    def __init__(self, drone=None):
        """Init a controller object.  """
        self.logger = logging.getLogger('controller_handler')
        self.move_up = 0
        self.move_left = 0
        self.move_right = 0
        self.move_down = 0
        self.move_forward = 0
        self.move_backward = 0
        self.rotate_left = 0
        self.rotate_right = 0

        # Initiate inner objects.
        self.flags_handler = FlagsHandler()
        self.frame_handler = FrameHandler()
        self.face_processor = FaceProcessor()
        self.back_ground_remover = BackGroundRemover(self.flags_handler)
        self.detector = Detector(self.flags_handler)
        self.extractor = Extractor(self.flags_handler)

        # Get initiated drone object
        self.drone = drone

    def start(self):
        """Function for starting image pipe processing.  """
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Controller')

        # cv2.namedWindow('Drone video stream')
        # Init video stream buffer.
        # container = av.open(self.drone.get_video_stream())
        # skip first 300 frames
        frame_skip = 300

        while self.flags_handler.quit_flag is False:
            # image = None
            # for frame in container.decode(video=0):
            #     if 0 < frame_skip:
            #         frame_skip = frame_skip - 1
            #         continue
            #     start_time = time.time()
            #     image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
            #
            #     frame_skip = int((time.time() - start_time) / frame.time_base)

            ret, frame = camera.read()
            # Controller processing pipe:
            # 1. Draw ROI on frame.
            self.frame_handler.input_frame = frame
            # 2. Cover faces, to remove detection noises.
            self.face_processor.face_covered_frame = self.frame_handler.input_frame
            # 3. Remove background from a covered-faces-frame.
            self.back_ground_remover.detected_frame = self.face_processor.face_covered_frame
            # 4. Detect a hand.
            self.detector.input_frame_for_feature_extraction = self.back_ground_remover.detected_frame
            # 5. Extract features, and track detected hand
            self.extractor.extract = self.detector

            inner_image = self.extractor.get_drawn_extreme_contour_points()
            if inner_image is not None:
                # Draw detected hand on outer image.
                outer_image = self.frame_handler.input_frame
                outer_image[0: inner_image.shape[0],
                            outer_image.shape[1] - inner_image.shape[1]: outer_image.shape[1]] = inner_image
                cv2.imshow('Controller', outer_image)
                self.get_drone_commands()

            self.flags_handler.keyboard_input = cv2.waitKey(1)

        if self.drone is not None:
            self.drone.quit()
        camera.release()
        cv2.destroyWindow('Controller')

    def get_up_param(self):
        """Return up parameter (int between 0..100). """
        temp_move_up = self.detector.detected_out_put_center[1] - \
            self.extractor.palm_center_point[1]
        self.move_up = temp_move_up
        if self.move_up <= 0:
            return 0
        return self.move_up if self.move_up <= 100 else 100

    def get_down_param(self):
        """Return down parameter (int between 0..100). """
        temp_move_down = self.extractor.palm_center_point[1] - \
            self.detector.detected_out_put_center[1]
        self.move_down = temp_move_down
        if self.move_down < 0:
            return 0
        return self.move_down if self.move_down <= 100 else 100

    def get_left_param(self):
        """Return left parameter (int between 0..100). """
        temp_move_left = self.detector.detected_out_put_center[0] - \
            self.extractor.palm_center_point[0]
        self.move_left = temp_move_left
        if self.move_left < 0:
            return 0
        return self.move_left if self.move_left <= 100 else 100

    def get_right_param(self):
        """Return right parameter (int between 0..100). """
        temp_move_right = self.extractor.palm_center_point[0] - \
            self.detector.detected_out_put_center[0]
        self.move_right = temp_move_right
        if self.move_right < 0:
            return 0
        return self.move_right if self.move_right <= 100 else 100

    def get_rotate_left_param(self):
        """Return rotate left parameter (int between 0..100). """
        temp_rotate_left = self.extractor.palm_angle_in_degrees - 90
        self.rotate_left = temp_rotate_left
        if self.rotate_left < 0:
            return 0
        return self.rotate_left if self.rotate_left <= 100 else 100

    def get_rotate_right_param(self):
        """Return rotate right parameter (int between 0..100). """
        temp_rotate_right = 90 - self.extractor.palm_angle_in_degrees
        self.rotate_right = temp_rotate_right
        if self.rotate_right < 0:
            return 0
        return self.rotate_right if self.rotate_right <= 100 else 100

    def get_forward_param(self):
        """Return move forward parameter (int between 0..100). """
        temp_forward_param = self.extractor.ext_top[1] - \
            self.extractor.zero_point[1]
        self.move_forward = temp_forward_param
        if self.move_forward < 0:
            return 0
        return self.move_forward if self.move_forward <= 100 else 100

    def get_backward_param(self):
        """Return move backward parameter (int between 0..100). """
        temp_backward_param = self.extractor.zero_point[1] - \
            self.extractor.ext_top[1]
        self.move_backward = temp_backward_param
        if self.move_backward < 0:
            return 0
        return self.move_backward if self.move_backward <= 100 else 100

    def get_drone_commands(self):
        try:
            # Send commands to drone
            if self.flags_handler.hand_control is False:
                # Make drone hover.
                self.drone.left(0)
                self.drone.right(0)
                self.drone.up(0)
                self.drone.down(0)
                self.drone.counter_clockwise(0)
                self.drone.clockwise(0)
                self.drone.forward(0)
                self.drone.backward(0)
            elif self.flags_handler.hand_control is True:
                # Send drone commands.
                if self.flags_handler.in_home_center is False:
                    # Send right_X and right_Y movements only when out of safety circle.
                    left = self.get_left_param()
                    if left != 0:
                        self.drone.left(left)
                    right = self.get_right_param()
                    if right != 0:
                        self.drone.right(right)
                    up = self.get_up_param()
                    if up != 0:
                        self.drone.up(up)
                    down = self.get_down_param()
                    if down != 0:
                        self.drone.down(down)

                counter_clockwise = self.get_rotate_left_param()
                if counter_clockwise != 0:
                    self.drone.counter_clockwise(counter_clockwise)
                clockwise = self.get_rotate_right_param()
                if clockwise != 0:
                    self.drone.clockwise(clockwise)

                forward = self.get_forward_param()
                if forward != 0:
                    self.drone.forward(forward)
                backward = self.get_backward_param()
                if backward != 0:
                    self.drone.backward(backward)

            if self.flags_handler.takeoff_requested is True:
                # Takeoff requested.
                self.drone.takeoff()
                time.sleep(3)
                self.flags_handler.takeoff_requested = False
            elif self.flags_handler.landing_requested is True:
                # Landing requested.
                self.drone.land()
                time.sleep(3)
                self.flags_handler.landing_requested = False
        except TypeError as error:
            self.logger.error(error)


if __name__ == '__main__':
    test = Controller()
    print(test.get_up_param())
