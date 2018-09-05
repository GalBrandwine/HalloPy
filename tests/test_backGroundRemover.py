import threading

import cv2

from hallopy import utils
from hallopy.controller import Controller, BackGroundRemover
from util.image_comp_tool import ImageTestTool


class TestBackGroundRemover:
    def test_detected_frame(self):
        """Test if input frames background is being removed correctly.  """
        # setup
        expected_path = utils.get_full_path('docs/back_ground_removed_frame.jpg')
        expected = cv2.imread(expected_path)
        test_path = utils.get_full_path('docs/face_and_hand_0.avi')
        cap = cv2.VideoCapture(test_path)
        back_ground_remover = BackGroundRemover()
        ret = True

        # run
        while (ret):
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
