"""Detector class.

This module contains Detector class and all its relevant functionality,

"""
import cv2
import numpy as np
from HalloPy.util import files


class Detector:
    """Detector class.  """

    def __init__(self):
        """Init inner algorithm params.  """
        self.cap_region_x_begin = 0.6
        self.cap_region_y_end = 0.6

        self.threshold = 50
        self.blur_Value = 41
        self.bg_Sub_Threshold = 50
        self.learning_Rate = 0
        self.bg_model = None

        self.face_padding_x = 20
        self.face_padding_y = 60

        self.input_frame = None
        self.out_put_frame = None
        self.detected = None
        self.detected_gray = None
        self.gray = None
        self.face_detector = None
        self.faces = None

        self.max_area_contour = None

    def set_frame(self, input_frame):
        """Function for getting frame from user.  """
        self.input_frame = cv2.bilateralFilter(input_frame, 5, 50, 100)  # smoothing filter
        self.input_frame = cv2.flip(input_frame, 1)
        self.out_put_frame = self.input_frame.copy()
        self.draw_ROI(self.out_put_frame)

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

        # Remove back-ground
        self.remove_back_ground(self.detected)

    def remove_back_ground(self, detected):
        """Function to remove back-ground from detected.

        Removing background help's find hand.
        """

        # todo: bg_model bgModel initation to controller key_board_input thread.
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

        And draw it on self.detected.
        """

        # Preparation
        self.detected_gray = cv2.cvtColor(detected, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(self.detected_gray, (self.blur_Value, self.blur_Value), 0)
        _, thresh = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY)

        # Get the contours.
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the biggest area.
        self.max_area_contour = max(contours, key=cv2.contourArea)

        # todo: draw contour in controller, based on keyboard input.
        # cv2.drawContours(detected, self.max_area_contour, -1, (0, 255, 0), 3)

    """ At this point, 'self' has: 
        1. input_frame: a untouched inputed frame.
        2. detected: an image extracted from 'input_frame's ROI, and it's background subtrackted.
        3. max_area_contour: array of points, our palm contour"""

        # Copy img, before drawing on it, so OpticalFlow won't be affected.
        # extractedMovement = img.copy()
        # frameCenter = drawMovementsAxes(img)

        # if length > 0:
        #     for i in range(length):  # find the biggest contour (according to area)
        #         temp = contours[i]
        #         area = cv2.contourArea(temp)
        #         if area > maxArea + 30:
        #             maxArea = area
        #             ci = i
        #     try:
        #         res = contours[ci]
        #     except Exception as ex:  # sometimes ci is out-of range
        #         # print(ex)
        #         pass
