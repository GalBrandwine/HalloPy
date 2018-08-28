"""Detector class.

This module contains Detector class and all its relevant functionality,

"""
import cv2
import numpy as np
from HalloPy.util import files
import logging

# create logger
from hallopy.detector_data_class import DetectorDataClass

detector_logger = logging.getLogger('detector')
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to logger
detector_logger.addHandler(ch)


class Detector:
    """Detector class.  """

    def __init__(self):
        """Init inner algorithm params.  """
        self.logger = logging.getLogger('detector')
        self.data = DetectorDataClass()

    def set_frame(self, input_frame):
        """Function for getting frame from user.  """
        self.data.input_frame = cv2.bilateralFilter(input_frame, 5, 50, 100)  # smoothing filter
        self.data.input_frame = cv2.flip(input_frame, 1)
        self.data.out_put_frame = self.data.input_frame.copy()
        self.draw_ROI(self.data.out_put_frame)

    def draw_ROI(self, out_put_frame):
        """Function for drawing the ROI on input frame"""
        cv2.rectangle(out_put_frame, (int(self.cap_region_x_begin * out_put_frame.shape[1]) - 20, 0),
                      (out_put_frame.shape[1], int(self.cap_region_y_end * out_put_frame.shape[0]) + 20),
                      (255, 0, 0), 2)

        self.cover_faces(self.out_put_frame)

    def cover_faces(self, out_put_frame):
        """Function to draw black recs over detected faces.

        This function remove eny 'noise' and help detector detecting palm.
        """

        # Preparation
        self.detected = out_put_frame.copy()
        self.gray = cv2.cvtColor(self.detected, cv2.COLOR_BGR2GRAY)

        if self.face_detector is None:
            # Load the face detector cascade.
            self.face_detector = cv2.CascadeClassifier(
                files.get_full_path('hallopy/config/haarcascade_frontalface_default.xml'))
        self.faces = self.face_detector.detectMultiScale(self.gray, 1.3, 5)

        # Black rectangle over faces to remove skin noises.
        for (x, y, w, h) in self.faces:
            self.detected[y - self.face_padding_y:y + h + self.face_padding_y,
            x - self.face_padding_x:x + w + self.face_padding_x, :] = 0

        self.remove_back_ground(self.detected)

    def remove_back_ground(self, detected):
        """Function to remove back-ground from detected.

        Removing background help's find hand.
        """

        # todo: bg_model bgModel initation to controller's input_from_keyboard_thread.
        if self.bg_model is None:
            self.bg_model = cv2.createBackgroundSubtractorMOG2(0, self.bg_Sub_Threshold)

        fgmask = self.bg_model.apply(detected, learningRate=self.learning_Rate)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(detected, detected, mask=fgmask)
        self.detected = res[0:int(self.cap_region_y_end * self.detected.shape[0]),
                        int(self.cap_region_x_begin * self.detected.shape[1]):self.detected.shape[1]]  # clip the ROI

        self.find_largest_contour(self.detected)

    def find_largest_contour(self, detected):
        """Function for finding largest contour in contours.

        todo: remove 'draw it on self.detected' to controller's input_from_keyboard_thread:
        # cv2.drawContours(detected, self.max_area_contour, -1, (0, 255, 0), 3)
        """

        # Preparation
        self.detected_gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(self.detected_gray, (self.blur_Value, self.blur_Value), 0)
        _, thresh = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY)

        # Get the contours.
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the biggest area.
        self.max_area_contour = max(contours, key=cv2.contourArea)
        try:
            self.detected_out_put_center = self.draw_axes(self.detected)
        except AttributeError:
            self.logger.error("self.detected not initiated!")

    def draw_axes(self, detected):
        """Function for drawing axes on detected_out_put.

        Return detected_out_put_center (point): the center coord' of detected_out_put.
        """

        # Preparation
        self.detected_out_put = detected.copy()

        # np.array are opposite than cv2 row/cols indexing.
        detected_out_put_center = (
            int(self.detected_out_put.shape[1] / 2), int(self.detected_out_put.shape[0] / 2) + self.horiz_axe_offset)
        horiz_axe_start = (0, int(self.detected_out_put.shape[0] / 2) + self.horiz_axe_offset)
        horiz_axe_end = (
            self.detected_out_put.shape[1], int(self.detected_out_put.shape[0] / 2) + self.horiz_axe_offset)

        vertic_y_start = (int(self.detected_out_put.shape[1] / 2), 0)
        vertic_y_end = (int(self.detected_out_put.shape[1] / 2), self.detected_out_put.shape[0])

        # draw movement axe X
        cv2.line(self.detected_out_put, horiz_axe_start, horiz_axe_end
                 , (0, 0, 255), thickness=3)
        # draw movement axe Y
        cv2.line(self.detected_out_put, vertic_y_start, vertic_y_end
                 , (0, 0, 255), thickness=3)
        return detected_out_put_center

    """ At this point (in top-down data flow), 'self' has: 
        1. input_frame: a untouched inserted frame.
        2. detected: an image extracted from 'input_frame's ROI, and it's background subtrackted.
        3. max_area_contour: array of points, our palm contour.
        4. detected_out_put_center, which will help extractor know distance from center of ROI"""

    # todo: create a function thatembed detected_out_put_frame in out_put_frame.
