# USAGE
# python Hallo.py --cascade haarcascade_frontalface_default.xml ( make sure both filesd in same folder) --debug debug
#
# Press b - to detect hand
# Press t - to take off
# Toggle c - to change drone control keyboard\detected_hand
# Press l - to land
# Press esc - to exit (need to land first)

import argparse
import threading
from builtins import print
import tellopy
import cv2
import numpy as np
import copy
import math
import time

# Environment:
# OS    : Ubuntu 16.04
# python: 3.5.2
# opencv: 3.3.0


# construct the argument parse and parse the arguments
from temp._internal import config_old

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-d", "--debug", required=False, default='normal',
                help="enter: debug, for running as a simulation, without a Tello drone connected")
args = vars(ap.parse_args())

# load the face detector cascade
detector = cv2.CascadeClassifier(args["cascade"])


def keyboardInputThread(mainCamera, conf):
    while mainCamera.isOpened:
        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27 and conf.lifted is False:  # press ESC to exit
            print('!!!quiting!!!')
            if conf.drone is not None:
                conf.drone.quit()
            break
        elif k == 27:
            print('!!!cant quit without landing!!!')
        elif k == ord('b'):  # press 'b' to capture the background
            conf.bgModel = cv2.createBackgroundSubtractorMOG2(0, conf.bgSubThreshold)
            conf.isBgCaptured = 1
            print('!!!Background Captured!!!')
        elif k == ord('t') and conf.calibrated is True:
            """Take off"""
            print('!!!Take of!!!')
            if conf.drone is not None and conf.lifted is not True:
                print('Wait 5 seconds')
                conf.drone.takeoff()
                time.sleep(5)
            conf.lifted = True
        elif k == ord('l'):
            """Land"""
            conf.old_frame_captured = False
            conf.lifted = False
            print('!!!Landing!!!')
            if conf.drone is not None:
                print('Wait 5 seconds')
                conf.drone.land()
                time.sleep(5)
        elif k == ord('c'):
            """Control"""
            if conf.handControl is True:
                conf.handControl = False
                conf.old_frame_captured = False
                print("control switched to keyboard")
            elif conf.lifted is True:
                print("control switched to detected hand")
                conf.handControl = True
        elif k == ord('z'):
            """ calibrating Threshold from keyboard """
            tempThreshold = cv2.getTrackbarPos('trh1', 'trackbar') - 1
            if tempThreshold >= 0:
                cv2.setTrackbarPos('trh1', 'trackbar', tempThreshold)
        elif k == ord('x'):
            """ calibrating Threshold from keyboard """
            tempThreshold = cv2.getTrackbarPos('trh1', 'trackbar') + 1
            if tempThreshold <= 100:
                cv2.setTrackbarPos('trh1', 'trackbar', tempThreshold)

def printThreshold(thr):
    print("! Changed threshold to " + str(thr))


def removeBG(frame, conf):
    fgmask = conf.bgModel.apply(frame, learningRate=conf.learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def simpleAngleCalculator(start, end, far):
    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
    return angle


def handCenterOfMass(HandContour):
    """Findes largest contour center of mass"""
    M = cv2.moments(HandContour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    return cX, cY


def square_distance(x, y):
    return sum([(xi - yi) ** 2 for xi, yi in zip(x[0], y[0])])


def calculateOpticalFlow(old_gray, frame_gray, p0, conf):
    """
        This function tracks the edge of the Middle finger
        :param old_gray: old frame, gray scale
        :param frame_gray: current frame
        :param p0: previous point for tracking
        :return: p0 - updated tracking point,
    """
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **conf.lk_params)
    if p1 is None:
        conf.old_frame_captured = False
        good_new = p0[st == 1]
    else:
        good_new = p1[st == 1]

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    return p0, old_gray


def drawMovementsAxes(inputImage):
    outPutframeCenter = (int(inputImage.shape[0] / 2) - 20, int(inputImage.shape[1] / 2) + 60)
    axe_X = [(0, int(inputImage.shape[1] / 2) + 60),
             (inputImage.shape[0], int(inputImage.shape[1] / 2) + 60)]
    axe_Y = [(int(inputImage.shape[0] / 2) - 20, 0),
             (int(inputImage.shape[0] / 2) - 20, inputImage.shape[1] + 40)]
    # draw movement axe X
    cv2.line(inputImage, axe_X[0], axe_X[1]
             , (0, 0, 255), thickness=3)
    # draw movement axe Y
    cv2.line(inputImage, axe_Y[0], axe_Y[1]
             , (0, 0, 255), thickness=3)
    return outPutframeCenter


def exctractDroneCommands(palmCenter, middleFingerEdge, frameCenter, img,
                          conf):  # todo: break function to smaller functions
    """Extract drone movement commands"""
    move_left = 0
    move_right = 0
    move_up = 0
    move_down = 0
    move_forward = 0
    move_backward = 0
    rotate_left = 0
    rotate_right = 0

    move_left = frameCenter[0] - palmCenter[0]
    if move_left < 0:
        move_left = 0

    move_right = palmCenter[0] - frameCenter[0]
    if move_right < 0:
        move_right = 0

    move_down = palmCenter[1] - frameCenter[1]
    if move_down < 0:
        move_down = 0
    else:
        move_down = int(move_down ** 1.3)

    move_up = frameCenter[1] - palmCenter[1]
    if move_up < 0:
        move_up = 0
    else:
        move_up = int(move_up ** 1.3)

    angelPointHelper = [palmCenter[0] - 10,
                        palmCenter[1]]  # helper to calculate angle between middleFingre and center of palm
    angle = simpleAngleCalculator(middleFingerEdge[0][0], angelPointHelper, palmCenter)
    angleIdDegrees = np.rad2deg(angle)
    if angleIdDegrees <= 90:
        rotate_left = int((90 - angleIdDegrees) ** 1.3)
        rotate_right = 0
    else:
        rotate_right = int((angleIdDegrees - 90) ** 1.3)
        rotate_left = 0

    palmCenterMiddleFingerMaxDistance = conf.palmCenterMiddleFingerMaxDistance
    distance = palmCenter[1] - middleFingerEdge[0][0][1]
    if distance > palmCenterMiddleFingerMaxDistance:
        conf.palmCenterMiddleFingerMaxDistance = palmCenterMiddleFingerMaxDistance = distance

    normalizedDistence = distance - palmCenterMiddleFingerMaxDistance / 3.0
    if normalizedDistence < palmCenterMiddleFingerMaxDistance / 2.0:
        move_forward = (int(palmCenterMiddleFingerMaxDistance / 2.0 - normalizedDistence) ** 1.3)
        move_backward = 0
    else:
        move_backward = (int(normalizedDistence - palmCenterMiddleFingerMaxDistance / 2.0) ** 1.8)
        move_forward = 0

    '''Debug'''
    # draw on img the movement boundaries
    cv2.line(img, palmCenter, (middleFingerEdge[0][0][0], int(
        palmCenter[1] - palmCenterMiddleFingerMaxDistance + palmCenterMiddleFingerMaxDistance / 4.5)),
             (0, 255, 0), thickness=3)

    cv2.line(img, (middleFingerEdge[0][0][0], int(
        palmCenter[1] - palmCenterMiddleFingerMaxDistance + palmCenterMiddleFingerMaxDistance / 4.5)), (
                 middleFingerEdge[0][0][0],
                 int(palmCenter[1] - palmCenterMiddleFingerMaxDistance)), (0, 0, 255), thickness=3)

    return move_left, move_right, move_up, move_down, move_forward, move_backward, rotate_left, rotate_right


"""Tello data handler """


def handler(event, sender, data, **args):
    droneHndler = sender
    # if event is droneHndler.EVENT_FLIGHT_DATA:
    #     print(data)


"""Tello connection initiation"""


def init():
    drone = tellopy.Tello()

    try:
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)

        drone.connect()
        drone.wait_for_connection(60.0)

    except Exception as ex:
        print(ex)
        drone.quit()
        return None
    return drone


def main(conf):
    """ Camera preparations
    :param drone: Tello drone, after connection established
    """

    face_padding_y = conf.face_padding_y
    face_padding_x = conf.face_padding_x
    cap_region_x_begin = conf.cap_region_x_begin
    cap_region_y_end = conf.cap_region_y_end
    blurValue = conf.blurValue

    camera = cv2.VideoCapture(0)
    print("camera brightness: {}".format(camera.get(cv2.CAP_PROP_BRIGHTNESS)))
    camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)

    """ThreshHolder adjuster tracker"""
    cv2.namedWindow('trackbar')
    cv2.createTrackbar('trh1', 'trackbar', conf.threshold, 100, printThreshold)
    cv2.namedWindow(conf.halloTitle)

    keyBoardThread = threading.Thread(name='keyBoardThread', target=keyboardInputThread, args=(camera, conf))
    keyBoardThread.start()

    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally

        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        #  Black rectangle over faces to remove skin noises

        for (x, y, w, h) in faces:
            img[y - face_padding_y:y + h + face_padding_y, x - face_padding_x:x + w + face_padding_x, :] = 0

        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]) - 20, 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0]) + 20), (255, 0, 0), 2)
        cv2.imshow(conf.halloTitle, frame)

        #  Main operation
        if conf.isBgCaptured == 1:  # this part wont run until background captured
            img = removeBG(frame, conf)
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

            # convert the image into binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

            # get the contours
            thresh1 = copy.deepcopy(thresh)
            _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1

            """ Copy img, before drawing on it, so OpticalFlow won't be affected """
            extractedMovement = img.copy()
            frameCenter = drawMovementsAxes(img)

            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea + 30:
                        maxArea = area
                        ci = i
                try:
                    res = contours[ci]
                except Exception as ex:  # sometimes ci is out-of range
                    # print(ex)
                    pass

                (cX, cY) = handCenterOfMass(res)  # palm center of mass

                if conf.handControl is False:  # draw palm in red
                    palmColor = (0, 0, 255)
                else:
                    palmColor = (0, 255, 0)  # draw palm in green
                # draw the contour and center of the shape on the image
                cv2.circle(img, (cX, cY), 10, (255, 255, 255), -1)
                cv2.drawContours(img, [res], 0, palmColor, 2)

                # Implementing OpticalFlow only on one point: the highest (== smallest Y value) point of the contour,
                # which is corresponding to the middle finger
                if math.sqrt((cX - frameCenter[0]) ** 2 + (
                        cY - frameCenter[1]) ** 2) <= conf.calibRadius:
                    conf.inHomeCenter = True
                else:
                    conf.inHomeCenter = False

                if conf.calibrated or conf.inHomeCenter and time.time() > conf.timeout:
                    '''
                    If true:
                        We are calibrated or we have just finished calibrating
                        and we can start with movement extraction
                    '''
                    conf.calibrated = True

                    if conf.old_frame_captured is False:
                        conf.old_frame_captured = True
                        # Take first frame and find corners in it
                        old_frame = extractedMovement
                        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                        temp_y = old_frame.shape[1]
                        for point in res:  # find highest point in contour, and track that point
                            if point[0][1] < temp_y:
                                temp_y = point[0][1]
                                desiredPoint = point
                        p0 = np.array(desiredPoint, dtype=np.float32).reshape(-1, 1, 2)

                    # capture current frame
                    frame_gray = cv2.cvtColor(extractedMovement, cv2.COLOR_BGR2GRAY)
                    p0, old_gray = calculateOpticalFlow(old_gray, frame_gray, p0, conf)

                    desiredPoint = p0
                    if len(desiredPoint) > 0:
                        """
                            After finding 'desiredPoint'.
                            We now have:
                            * center of palm = (cX,cY)
                            * edge of middle finger = desiredPoint
                            * center of img = frameCenter
        
                            So we now can:
                            * track the edge of the middle finger, and extract palm angle - translate to drone rotation commands,
        
                            * palm leaning (towards/backwards), translate to drone forward/backward command
        
                            * translate distance(palm_center_of_mass, center_of_img) to left/right/move_up/move_down drone commands
                        """
                        move_left, move_right, move_up, move_down, move_forward, move_backward, rotate_left, rotate_right = exctractDroneCommands(
                            (cX, cY), desiredPoint, frameCenter, img, conf)

                        '''Debug'''

                        print(
                            "move_left: {}, move_right: {}, move_up: {}, move_down: {}, move_forward: {}, move_backward: {}, rotate_left: {}, rotate_right: {}".format(
                                move_left, move_right, move_up, move_down, move_forward, move_backward,
                                rotate_left,
                                rotate_right))

                        if conf.drone is not None:  # drone is None in debug
                            if conf.handControl is True and conf.lifted is True:
                                """Drone is in the air, and control directed to palm, only when center of palm is out of Frame center"""
                                conf.hover = False
                                # print(
                                #     "move_left: {}, move_right: {}, move_up: {}, move_down: {}, move_forward: {}, move_backward: {}, rotate_left: {}, rotate_right: {}".format(
                                #         move_left, move_right, move_up, move_down, move_forward, move_backward,
                                #         rotate_left,
                                #         rotate_right))
                                try:
                                    if conf.inHomeCenter is False:  # out of Home center
                                        if move_left > conf.tollerance:
                                            conf.drone.left(move_left)
                                        if move_right > conf.tollerance:
                                            conf.drone.right(move_right)
                                        if move_up > conf.tollerance:
                                            conf.drone.up(move_up)
                                        if move_down > conf.tollerance:
                                            conf.drone.down(move_down)
                                        pass
                                    elif conf.inHomeCenter is True:  # in Home center, don't move side ways nor up & down
                                        conf.drone.left(0)
                                        conf.drone.right(0)
                                        conf.drone.up(0)
                                        conf.drone.down(0)

                                    if rotate_left > conf.tollerance:
                                        if rotate_left > 100:
                                            conf.drone.counter_clockwise(100)
                                        else:
                                            conf.drone.counter_clockwise(rotate_left)
                                    if rotate_right > conf.tollerance:
                                        if rotate_right > 100:
                                            conf.drone.clockwise(100)
                                        else:
                                            conf.drone.clockwise(rotate_right)
                                    if move_forward > conf.tollerance:
                                        if move_forward > 100:
                                            conf.drone.forward(100)
                                        else:
                                            conf.drone.forward(move_forward)
                                    if move_backward > conf.tollerance:
                                        if move_backward > 100:
                                            conf.drone.backward(100)
                                        else:
                                            conf.drone.backward(move_backward)
                                except Exception as ex:
                                    print(ex)
                                    pass

                                # print("sending drone movement commands")
                            elif conf.hover is False:
                                """Control directed to keyboard or drone is not in the air"""
                                try:
                                    conf.hover = True
                                    conf.drone.left(0)
                                    conf.drone.right(0)
                                    conf.drone.up(0)
                                    conf.drone.down(0)
                                    conf.drone.counter_clockwise(0)
                                    conf.drone.clockwise(0)
                                    conf.drone.forward(0)
                                    conf.drone.backward(0)
                                except Exception as ex:
                                    print(ex)
                                    pass

                        # draw edge of middle finger
                        cv2.circle(img, (desiredPoint[0][0][0], desiredPoint[0][0][1]), 10, (0, 255, 255), -1)

                    # draw green filled circle
                    cv2.circle(img, frameCenter, conf.calibRadius, (0, 255, 0), -1)

            elif time.time() > conf.timeout:
                """
                    reset timer 
                    reset calibration flag
                    reset opticalFlow previous frame captured flag
                """
                conf.timeout = time.time() + 5
                conf.calibrated = False
                conf.old_frame_captured = False

                # TODO: reset all drone movements, because hand is not calibrated
                # reset palm movement globals
                conf.palmCenterMiddleFingerMaxDistance = 0
                print("calibration reseted!")
            else:
                # TODO: reset all drone movements, because hand is not calibrated
                conf.old_frame_captured = False

            cv2.circle(img, frameCenter, conf.calibRadius, (0, 0, 255), thickness=3)

            frame[0:int(cap_region_y_end * frame.shape[0]),
            int(cap_region_x_begin * frame.shape[1]):frame.shape[1], :] = img
            cv2.imshow(conf.halloTitle, frame)
        if keyBoardThread.is_alive() == False:
            break
            # operations using detected fingers, maybe good for later
            # if triggerSwitch is True:
            #     if isFinishCal is True and cnt <= 2:
            #         print(cnt)
            #         # app('System Events').keystroke(' ')  # simulate pressing blank space

        # # Keyboard OP
        # k = cv2.waitKey(10)
        # if k == 27 and lifted is False:  # press ESC to exit
        #     print('!!!quiting!!!')
        #     if drone is not None:
        #         drone.quit()
        #     break
        # elif k == 27:
        #     print('!!!cant quit without landing!!!')
        # elif k == ord('b'):  # press 'b' to capture the background
        #     bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        #     isBgCaptured = 1
        #     print('!!!Background Captured!!!')
        # # elif k == ord('r'):  # press 'r' to reset the background
        # #     bgModel = None
        # #     triggerSwitch = False
        # #     isBgCaptured = 0
        # #     print('!!!Reset BackGround!!!')
        # elif k == ord('t') and calibrated is True:
        #     """Take off"""
        #     print('!!!Take of!!!')
        #     if drone is not None and lifted is not True:
        #         print('Wait 5 seconds')
        #         drone.takeoff()
        #         time.sleep(5)
        #     lifted = True
        # elif k == ord('l'):
        #     """Land"""
        #     old_frame_captured = False
        #     lifted = False
        #     print('!!!Landing!!!')
        #     if drone is not None:
        #         print('Wait 5 seconds')
        #         drone.land()
        #         time.sleep(5)
        # elif k == ord('c'):
        #     """Control"""
        #     if handControl is True:
        #         handControl = False
        #         old_frame_captured = False
        #         print("control switched to keyboard")
        #     elif lifted is True:
        #         print("control switched to detected hand")
        #         handControl = True
        # elif k == ord('z'):
        #     """ calibrating Threshold from keyboard """
        #     tempThreshold = cv2.getTrackbarPos('trh1', 'trackbar') - 1
        #     if tempThreshold >= 0:
        #         cv2.setTrackbarPos('trh1', 'trackbar', tempThreshold)
        # elif k == ord('x'):
        #     """ calibrating Threshold from keyboard """
        #     tempThreshold = cv2.getTrackbarPos('trh1', 'trackbar') + 1
        #     if tempThreshold <= 100:
        #         cv2.setTrackbarPos('trh1', 'trackbar', tempThreshold)

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
    # kill drone threads
    if conf.drone is not None:
        conf.drone.quit()


# todo: make a class of constant variables that is being loaded from a config file and then passed to main and to keyBoardThread
if __name__ == '__main__':
    configHandler = config_old.Config()
    if args["debug"] == "debug":
        main(configHandler)
    else:
        configHandler.drone = init()
        if configHandler.drone is not None:
            main(configHandler)
