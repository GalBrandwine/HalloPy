from unittest import TestCase
from hallopy.detector import Detector
from HalloPy.util.image_copm_tool import Image_Test_Tool


class TestDetector(TestCase):
    def test_set_frame(self):
        """Test if set_frame perform prepossessing correctly.  """

        # setup
        import cv2

        test_image = cv2.imread('/home/gal/PycharmProjects/droneTest1/HalloPy/docs/testing_img.jpg')
        detector_expected = Detector()
        expected = test_image.copy()

        expected = cv2.bilateralFilter(expected, 5, 50, 100)  # smoothing filter
        expected = cv2.flip(expected, 1)
        detector_expected.draw_ROI(expected)

        detector = Detector()
        detector.set_frame(test_image)

        # run
        # range [-1, 1] with a value of one being a “perfect match”.
        ssim = Image_Test_Tool.compare_imaged(detector.input_frame, expected)
        assert ssim >= 0.9

    def test_draw_ROI(self):
        self.fail()
