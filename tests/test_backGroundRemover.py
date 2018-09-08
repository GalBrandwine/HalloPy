

import cv2

from hallopy import utils
from hallopy.controller import BackGroundRemover, FlagsHandler
from util.image_comp_tool import ImageTestTool


class TestBackGroundRemover:
    """TestBackGroundRemover tests BackgroundRemover functionality.  """

    def test_detected_frame(self):
        """Test if input frames background is being removed correctly.  """
        # setup
        expected_path = utils.get_full_path('docs/back_ground_removed_frame.jpg')
        expected = cv2.imread(expected_path)
        test_path = utils.get_full_path('docs/face_and_hand_0.avi')
        cap = cv2.VideoCapture(test_path)
        flags_handler = FlagsHandler()
        back_ground_remover = BackGroundRemover(flags_handler)
        ret = True

        # run
        while ret is True:
            ret, frame = cap.read()
            if ret is True:
                back_ground_remover.detected_frame = frame

        # write_path = utils.get_full_path('docs')
        # cv2.imwrite(write_path+'/back_ground_removed_frame.jpg',back_ground_remover.detected_frame)
        ssim = ImageTestTool.compare_imaged(back_ground_remover.detected_frame, expected)
        # print("SSIM: {}".format(ssim))
        assert ssim >= 0.95

        # teardown
        cap.release()
        cv2.destroyAllWindows()

    def test_back_ground_reset(self):
        """Test if background model is being reset correctly.

        resetting background model is via keyboard input,
        in Controller's flags_handler.
        """
        # setup
        # Input from camera.
        cv2.namedWindow('test')
        cap = cv2.VideoCapture(0)
        flags_handler = FlagsHandler()
        back_ground_remover = BackGroundRemover(flags_handler)

        # run
        while flags_handler.quit_flag is False:
            """
            Inside loop, remove back ground from frame using back_ground_remover,
            here we are testing for background model resetting.
            the reset flag is changed within Controller's flags_handler.
            
            Pressing 'b': will rest background.
            Pressing esc: break loop.
            """
            ret, frame = cap.read()
            if ret is True:
                back_ground_remover.detected_frame = frame
                if back_ground_remover.detected_frame is not None:
                    cv2.imshow('test', back_ground_remover.detected_frame)
                flags_handler.keyboard_input = cv2.waitKey(1)

        # teardown
        cap.release()
        cv2.destroyAllWindows()
