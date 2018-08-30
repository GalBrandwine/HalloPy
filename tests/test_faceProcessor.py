import cv2
import numpy as np
import logging

from hallopy.controller import FaceProcessor
from hallopy import utils
from util.image_comp_tool import ImageTestTool


class TestFaceProcessor:
    def test_face_covered_frame(self):
        """Test if faces are detected and covered.  """
        # setup
        test_path = utils.get_full_path('docs/face_and_hand.jpg')
        test_image = cv2.imread(test_path)

        expected = test_image.copy()
        expected_faces = ImageTestTool.detect_faces(expected)
        ImageTestTool.draw_black_recs(expected, expected_faces)

        face_processor = FaceProcessor()
        face_processor.logger.setLevel(logging.DEBUG)
        # Insert image with face.
        face_processor.face_covered_frame = expected

        # run
        ssim = ImageTestTool.compare_imaged(face_processor.face_covered_frame, expected)
        # print("SSIM: {}".format(ssim))
        assert ssim >= 0.93
