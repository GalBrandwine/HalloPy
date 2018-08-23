import time
import cv2


class Config(object):
    """Config class holds all Hallo configurations.  """

    def __init__(self, halloTitle='Hallo - a hand controlled Tello', debug=None, drone=None):
        """__init__ method
        Args:
            :arg halloTitle (string): name for the cv2.imshow window.
            :arg debug (bool): if True, run script without a drone object
        Attributes:
            halloTitle (str): App window title.
            cap_region_x_begin (float): ROI start point/total width.
            cap_region_y_end (float): ROI start point/total width.
            threshold (int): The BINARY threshold.
            blurValue (int): GaussianBlur parameter.
            bgSubThreshold (int): Bg param (default = 50).
            learningRate (int): LearningRate param (default = 0).
            bgModel (object): Trained model received from cv2.createBackgroundSubtractorKNN().

            # blackBox param to cover discovered faces:
                face_padding_x (int)
                face_padding_y (int)

            timeout (time.time()): A countDown timer before raising 'calibrated' flag.
            calibRadius (int):  ROI center-circle radius.
            tolerance (int): hand tolerance flexability, before sending commands to drone.
            palmCenterMiddleFingerMaxDistance (int): Maximum distance between palm center and middle finger edge.
            isBgCaptured (bool): whether the background captured.
            triggerSwitch (bool): if true, keyboard simulator works.
            calibrated (bool): If True, app is calibrated on detected palm.
            inHomeCenter (bool): If true, detected palm is in center of ROI.
            old_frame_captured (bool): A flag for OpticalFlow algorithm.
            handControl (bool): Flag for switching drone's control between detected-palm and keyboard.
            lifted (bool): Drone state flag.
            hover (bool): Drone state flag.
            drone (TelloPy_object): A drone object (assuming its connected to Tello_drone)
            debug (bool): Debug flag.
            lk_params (dict): Parameters for lucas kanade optical flow.

        """

        self.halloTitle = halloTitle
        self.cap_region_x_begin = 0.6
        self.cap_region_y_end = 0.6

        self.threshold = 50
        self.blurValue = 41
        self.bgSubThreshold = 50
        self.learningRate = 0
        self.bgModel = None

        self.face_padding_x = 20
        self.face_padding_y = 60

        self.timeout = time.time() + 5  # 5 seconds from now
        self.calibRadius = 15
        self.tolerance = 10
        self.palmCenterMiddleFingerMaxDistance = 0

        self.isBgCaptured = 0

        self.triggerSwitch = False
        self.calibrated = False
        self.inHomeCenter = False
        self.old_frame_captured = False
        self.handControl = False
        self.lifted = False
        self.hover = False
        self.drone = drone
        self.debug = debug
        self.desiredPoint = None  # todo add comment
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
