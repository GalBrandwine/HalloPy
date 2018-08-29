"""Extractor class.

This module contains Extractor class and all its relevant functionality,

"""
import cv2
import numpy as np
from attr import dataclass

from HalloPy.util import files
import logging

# create logger
detector_logger = logging.getLogger('extractor')
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to logger
detector_logger.addHandler(ch)


@dataclass
class Extractor:
    """Detector class.  """

    def __init__(self):
        """Init inner algorithm params.  """
        self.logger = logging.getLogger('extractor')
        self.cap_region_x_begin = 0.6
        self.cap_region_y_end = 0.6

        self.threshold = 50
        self.blur_Value = 41
        self.bg_Sub_Threshold = 50
        self.learning_Rate = 0
        self.bg_model = None

        self.input_frame = None
        self.out_put_frame = None
        self.detected = None
        self.detected_gray = None
        self.detected_out_put = None
        self.detected_out_put_center = None
        self.horiz_axe_offset = 60

        self.face_padding_x = 20
        self.face_padding_y = 60
        self.gray = None
        self.face_detector = None
        self.faces = None

        self.max_area_contour = None
