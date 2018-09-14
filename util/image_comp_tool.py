# import the necessary packages
from skimage.measure import compare_ssim
import numpy as np
import cv2
from hallopy import utils


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

    @staticmethod
    def detect_faces(img):
        """Function for detecting faces.

        :returns faces: array with detected faces coordination's.
        """

        face_detector = cv2.CascadeClassifier(utils.get_full_path('hallopy/config/haarcascade_frontalface_default.xml'))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return face_detector.detectMultiScale(gray, 1.3, 5)

    @staticmethod
    def draw_black_recs(img, obj_coord):
        # Black rectangle over faces to remove skin noises.
        for (x, y, w, h) in obj_coord:
            img[y:y + h, x:x + w, :] = 0

    @staticmethod
    def clip_roi(img, roi):
        clipped = img[0:int(roi['cap_region_y_end'] * img.shape[0]),
                  int(roi['cap_region_x_begin'] * img.shape[1]):img.shape[1]]  # clip the ROI
        return clipped

    @staticmethod
    def get_max_area_contour(input_image):
        # Get the contours.
        expected_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(expected_gray, (41, 41), 0)
        thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the biggest area
        try:
            if len(contours) > 0:
                max_area_contour = max(contours, key=cv2.contourArea)
                return max_area_contour
        except ValueError as error:
            print(error)

    @staticmethod
    def get_contour_area(contour):
        return cv2.contourArea(contour)

    @staticmethod
    def get_center_of_mass(contour):
        """Find contours center of mass.  """
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        return cX, cY

    @staticmethod
    def get_middle_finger_edge_coord(contour):
        """Function for calculating middle finger edge coordination.
        :type contour: collection.iter
        """

        temp_y = 1000
        for point in contour:  # find highest point in contour, and track that point
            if point[0][1] < temp_y:
                temp_y = point[0][1]

        return point[0][0], point[0][1]

    @staticmethod
    def get_contour_extreme_points(contour):
        c = contour
        try:
            # determine the most extreme points along the contour
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
        except TypeError as error:
            extLeft = 0, 0
            extRight = 0, 0
            extTop = 0, 0
            extBot = 0, 0

        return extLeft, extRight, extTop, extBot

    @staticmethod
    def draw_contours(image, contours):
        cv2.drawContours(image, [contours], -1, (0, 255, 255), 2)

    @staticmethod
    def draw_tracking_points(image, points):

        # draw the outline of the object, then draw each of the
        # extreme points, where the left-most is red, right-most
        # is green, top-most is blue, and bottom-most is teal
        # determine the most extreme points along the contour
        c = points.reshape(-1, 1, 2)
        if points.size > 0:
            # only ext_contour points have been given.
            # ext_left = tuple(c[c[:, :, 0].argmin()][0])
            # ext_right = tuple(c[c[:, :, 0].argmax()][0])
            ext_top = tuple(c[c[:, :, 1].argmin()][0])
            ext_bot = tuple(c[c[:, :, 1].argmax()][0])
            # palm_center = points[4]
            # cv2.circle(image, ext_left, 8, (0, 0, 255), -1)
            # cv2.putText(image,'ext_left',ext_left, cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 255))

            # cv2.circle(image, ext_right, 8, (0, 255, 0), -1)
            # cv2.putText(image,'ext_right',ext_right, cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 0))

            cv2.circle(image, ext_top, 8, (255, 0, 0), -1)
            cv2.putText(image, 'ext_top', ext_top, cv2.FONT_HERSHEY_COMPLEX, .5, (255, 0, 0))

            cv2.circle(image, ext_bot, 8, (255, 255, 0), -1)
            cv2.putText(image, 'ext_bot', ext_bot, cv2.FONT_HERSHEY_COMPLEX, .5, (255, 255, 0))
            # cv2.circle(image, palm_center, 8, (255, 255, 255), thickness=-1)
