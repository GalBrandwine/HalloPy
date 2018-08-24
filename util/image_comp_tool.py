# import the necessary packages
from skimage.measure import compare_ssim
import imutils
import cv2


class ImageTestTool:
    """This class contain tools that helps test functionality"""

    @staticmethod
    def compare_imaged(img1, img2):
        """This function compare 2 images.

        Return SSIM:    Represents the structural similarity index between the two input images.
                        This value can fall into the range [-1, 1] with a value of one being a “perfect match”.
        """

        # load the two input images
        imageA = img1
        imageB = img2

        # convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        # diff = (diff * 255).astype("uint8")
        return score
        # print("SSIM: {}".format(score))
        #
        # # threshold the difference image, followed by finding contours to
        # # obtain the regions of the two input images that differ
        # thresh = cv2.threshold(diff, 0, 255,
        #                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        #                         cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        #
        # # loop over the contours
        # for c in cnts:
        #     # compute the bounding box of the contour and then draw the
        #     # bounding box on both input images to represent where the two
        #     # images differ
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #
        # # show the output images
        # cv2.imshow("Original", imageA)
        # cv2.imshow("Modified", imageB)
        # cv2.imshow("Diff", diff)
        # cv2.imshow("Thresh", thresh)
        # cv2.waitKey(0)
