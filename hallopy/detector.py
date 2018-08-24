"""Detector class.

This module contains Detector class and all its relevant functionality,

"""
import cv2


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
        self.detected = None
        self.gray = None
        self.face_detector = None
        self.faces = None

    def set_frame(self, input_frame):
        """Function for getting frame from user.  """

        self.input_frame = cv2.bilateralFilter(input_frame, 5, 50, 100)  # smoothing filter
        self.input_frame = cv2.flip(input_frame, 1)
        self.draw_ROI(self.input_frame)
        self.cover_faces(self.input_frame)
        self.detect(self.input_frame)


    def draw_ROI(self, input_frame):
        """Function for drawing the ROI on input frame"""

        cv2.rectangle(input_frame, (int(self.cap_region_x_begin * input_frame.shape[1]) - 20, 0),
                      (input_frame.shape[1], int(self.cap_region_y_end * input_frame.shape[0]) + 20),
                      (255, 0, 0), 2)

    def cover_faces(self, input_frame):
        """Function to draw black recs over detected faces.

        This function remove eny 'noise' and help detector detecting palm.
        """

        # Preparation
        self.detected = input_frame
        self.gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

        if self.face_detector is None:
            # Load the face detector cascade.
            self.face_detector = cv2.CascadeClassifier(
                '/home/gal/PycharmProjects/droneTest1/HalloPy/hallopy/config/haarcascade_frontalface_default.xml')
        self.faces = self.face_detector.detectMultiScale(self.gray, 1.3, 5)

        # Black rectangle over faces to remove skin noises.
        for (x, y, w, h) in self.faces:
            self.detected[y - self.face_padding_y:y + h + self.face_padding_y,
            x - self.face_padding_x:x + w + self.face_padding_x, :] = 0

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
