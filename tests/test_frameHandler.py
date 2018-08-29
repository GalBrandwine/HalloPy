import cv2
import numpy as np
from hallopy.controller import FrameHandler
from HalloPy.util import files
from util.image_comp_tool import ImageTestTool


class TestFrameHandler:
    """TestFrameHandler tests FrameHandler functionality.  """

    def test_input_frame(self):
        """Test if input frame preprocessed correctly.  """

        # setup
        test_path = files.get_full_path('docs/face_and_hand.jpg')
        test_image = cv2.imread(test_path)
        # Because image loaded from local, and not received from web-cam, a flip is needed.
        test_image = cv2.flip(test_image, 1)  # type: np.ndarray

        expected = test_image.copy()
        expected = cv2.bilateralFilter(expected, 5, 50, 100)  # smoothing filter
        expected = cv2.flip(expected, 1)
        frame_handler = FrameHandler()

        # run
        # range [-1, 1] with a value of one being a “perfect match”.
        frame_handler.input_frame = test_image
        ssim = ImageTestTool.compare_imaged(frame_handler.input_frame, expected)
        # print("SSIM: {}".format(ssim))
        assert ssim >= 0.95
