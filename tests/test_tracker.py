import cv2
import numpy as np
from hallopy import utils
from hallopy.controller import FlagsHandler, Tracker
from util.image_comp_tool import ImageTestTool


class TestTracker:
    """Unittests for a Tracker object.  """

    def test_track(self):
        """Test if tracker object tracks correctly after given set of points to track, and a frame."""

        # setup
        flags_handler = FlagsHandler()
        tracker = None

        bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
        cap = cv2.VideoCapture(0)
        while flags_handler.quit_flag is False:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Remove background from input frame.
            fgmask = bg_model.apply(frame, learningRate=0)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            res = cv2.bitwise_and(frame, frame, mask=fgmask)
            # Clip frames ROI.
            back_ground_removed_clipped = ImageTestTool.clip_roi(res,
                                                                 {'cap_region_x_begin': 0.6, 'cap_region_y_end': 0.6})

            if flags_handler.background_capture_required is True:
                bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
                flags_handler.background_capture_required = False

            max_area_contour = ImageTestTool.get_max_area_contour(back_ground_removed_clipped)

            extLeft, extRight, extTop, extBot = ImageTestTool.get_contour_extreme_points(max_area_contour)
            palm_center = ImageTestTool.get_center_of_mass(max_area_contour)

            """Stopped in test_tracker.
            still figuring out how to work with optical flow.
            It seems im not updating the tracked point.
            I can see that in debug.
            """
            # points = {"ext_left": extLeft, "ext_right": extRight, "ext_top": extTop, "ext_bot": extBot,
            #               "palm_canter_point": bpalm_center}

            if tracker is None:
                points = np.array([extLeft, extRight, extTop, extBot, palm_center])
                # print("tracker is None: {}".format(points))
            else:
                points = max_area_contour
                tracker.track(points, back_ground_removed_clipped)
                points = tracker.points_to_track
                # print(points)

            ImageTestTool.draw_tracking_points(back_ground_removed_clipped, points)
            cv2.imshow('test_track', back_ground_removed_clipped)
            keyboard_input = cv2.waitKey(1)
            flags_handler.keyboard_input = keyboard_input
            # run
            if flags_handler.background_capture_required is True:
                tracker = None
            if keyboard_input == ord('t'):
                tracker = Tracker(flags_handler, points, back_ground_removed_clipped)
