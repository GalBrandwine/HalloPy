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
        self.blurValue = 41
        self.bgSubThreshold = 50
        self.learningRate = 0
        self.bgModel = None

        self.face_padding_x = 20
        self.face_padding_y = 60

        self.input_frame = None
        self.out_put_frame = None
        self.detected = None
        self.gray = None
        self.face_detector = None
        self.faces = None

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

        fgmask = self.bgModel.apply(detected, learningRate=self.learningRate)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(detected, detected, mask=fgmask)
        self.detected = res[0:int(self.cap_region_y_end * self.detected.shape[0]),
                        int(self.cap_region_x_begin * self.detected.shape[1]):self.detected.shape[1]]  # clip the ROI

        # todo: call find_largest_contour


    def detect(self, input_frame):
        pass
        # raise NotImplementedError   # equal to '#todo:'

        # remove noise
        # todo: self.hide_faces(self.detected), make hide_faces function

    # @abc.abstractmethod(classmethod)
    # def _drawMovementsAxes(self, inputIFrame):
    #     """Private method, Draws movements Axes on frame.
    #
    #     Args:
    #         inputframe (openCV object): recieved frame from camera.
    #
    #     """
    #     pass
    #
    # @abc.abstractmethod(classmethod)
    # def _removeBG(self, frame):
    #     """Private method, Removes back ground.
    #
    #     Returns:
    #         The frame, with subtracted background.
    #
    #     """
    #     pass
    #
    # @abc.abstractmethod(classmethod)
    # def _simpleAngleCalculator(self, startPoint, endPoint, farPoint):
    #     """Private method, calculate angle of 3 given points.
    #
    #     Args:
    #         startPoint (Tuple):
    #         endPoint(Tuple):
    #         farPoint(Tuples):
    #
    #     Returns:
    #         angle (float): calculated angle in degrees.
    #
    #     """
    #     pass
