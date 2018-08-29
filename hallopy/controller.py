"""Multi class incapsulation implementation.  """
import cv2
import logging
from HalloPy.hallopy.icontroller import Icontroller
from HalloPy.hallopy.detector import Detector

# Create loggers.
frame_logger = logging.getLogger('frame_handler')
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to loggers.
frame_logger.addHandler(ch)


class FrameHandler:
    """FrameHandler handel input frame from controller,

    and perform some preprocessing.
    """

    def __init__(self):
        """Init preprocessing params.  """
        self.logger = logging.getLogger('frame_handler')
        self.__cap_region_x_begin = 0.6
        self.__cap_region_y_end = 0.6
        self._input_frame = None

    @property
    def input_frame(self):
        return self._input_frame

    @input_frame.setter
    def input_frame(self, input_frame_from_camera):
        """Setter with preprocessing.  """
        try:
            self._input_frame = cv2.bilateralFilter(input_frame_from_camera, 5, 50, 100)  # smoothing filter
            self._input_frame = cv2.flip(input_frame_from_camera, 1)
            self._draw_roi()
        except cv2.error:
            self.logger.error("Input frame is None")

    def _draw_roi(self):
        """Function for drawing the ROI on input frame"""

        cv2.rectangle(self._input_frame, (int(self.__cap_region_x_begin * self._input_frame.shape[1]) - 20, 0),
                      (self._input_frame.shape[1], int(self.__cap_region_y_end * self._input_frame.shape[0]) + 20),
                      (255, 0, 0), 2)


class Controller(Icontroller):
    """Controller class holds a detector and a extractor.

    :param icontroller.Icontroller: implemented interface
    """

    def __init__(self):
        """Init a controller object.  """
        self.move_up = 0
        self.move_left = 0
        self.move_right = 0
        self.move_down = 0
        self.move_forward = 0
        self.move_backward = 0
        self.rotate_left = 0
        self.rotate_right = 0

        # Initiate inner objects
        self.frame_handler = FrameHandler()

    def get_up_param(self):
        """Return up parameter (int between 0..100). """
        if self.move_up <= 0:
            return 0
        return self.move_up if self.move_up <= 100 else 100

    def get_down_param(self):
        """Return down parameter (int between 0..100). """
        if self.move_down < 0:
            return 0
        return self.move_down if self.move_down <= 100 else 100

    def get_left_param(self):
        """Return left parameter (int between 0..100). """
        if self.move_left < 0:
            return 0
        return self.move_left if self.move_left <= 100 else 100

    def get_right_param(self):
        """Return right parameter (int between 0..100). """
        if self.move_right < 0:
            return 0
        return self.move_right if self.move_right <= 100 else 100

    def get_rotate_left_param(self):
        """Return rotate left parameter (int between 0..100). """
        if self.rotate_left < 0:
            return 0
        return self.rotate_left if self.rotate_left <= 100 else 100

    def get_rotate_right_param(self):
        """Return rotate right parameter (int between 0..100). """
        if self.rotate_right < 0:
            return 0
        return self.rotate_right if self.rotate_right <= 100 else 100

    def get_forward_param(self):
        """Return move forward parameter (int between 0..100). """
        if self.move_forward < 0:
            return 0
        return self.move_forward if self.move_forward <= 100 else 100

    def get_backward_param(self):
        """Return move backward parameter (int between 0..100). """
        if self.move_backward < 0:
            return 0
        return self.move_backward if self.move_backward <= 100 else 100


if __name__ == '__main__':
    test = Controller()
    print(test.get_up_param())
