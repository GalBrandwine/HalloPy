import logging
import cv2
import numpy as np
from hallopy.controller import FrameHandler
from hallopy import utils
from util.image_comp_tool import ImageTestTool


class TestFrameHandler:
    """TestFrameHandler tests FrameHandler functionality.  """

    def test_input_frame(self):
        """Test if input frame preprocessed correctly.  """

        # setup
        test_path = utils.get_full_path('docs/material_for_testing/face_and_hand.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from web-cam, a flip is needed,
        # inside frame_handler, a frame is supposed to be received from web-cam, hence it is flipped after receiving it.
        test_image = cv2.flip(test_image, 1)  # type: np.ndarray

        expected = test_image.copy()
        expected = cv2.bilateralFilter(expected, 5, 50, 100)  # smoothing filter
        expected = cv2.flip(expected, 1)

        frame_handler = FrameHandler()
        frame_handler.logger.setLevel(logging.DEBUG)

        # run
        # range [-1, 1] with a value of one being a “perfect match”.
        frame_handler.input_frame = test_image
        ssim = ImageTestTool.compare_imaged(frame_handler.input_frame, expected)
        # print("SSIM: {}".format(ssim))
        assert ssim >= 0.95
