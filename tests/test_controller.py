import cv2

from hallopy.controller import Controller
import tellopy
"""Mockito usage:

when(os.path).exists('/foo').thenReturn(True)

# or:
import requests  # the famous library
# you actually want to return a Response-like obj, we'll fake it
response = mock({'status_code': 200, 'text': 'Ok'})
when(requests).get(...).thenReturn(response)

# use it
requests.get('http://google.com/')

# clean up
unstub()

Another example:

# setup
response = mock({
    'status_code': 200,
    'text': 'Ok'
}, spec=requests.Response)
when(requests).get('https://example.com/api').thenReturn(response)

# run
assert get_text('https://example.com/api') == 'Ok'

# done!
"""


class TestController:

    @staticmethod
    def handler(event, sender, data, **args):
        """Drone events handler, for testing.  """
        drone_handler = sender
        if event is drone_handler.EVENT_FLIGHT_DATA:
            print(data)

    @staticmethod
    def init_drone():
        """Drone initiation function for testing.  """
        drone = tellopy.Tello()

        try:
            drone.subscribe(drone.EVENT_FLIGHT_DATA, TestController.handler)
            drone.connect()
            drone.wait_for_connection(60.0)

        except Exception as ex:
            print(ex)
            drone.quit()
            return None
        return drone

    def test_controller_initiation(self):
        """Test if controller params initiated with 0. """

        # setup
        controller = Controller()

        # run
        assert controller.get_up_param() == 0
        assert controller.get_down_param() == 0
        assert controller.get_left_param() == 0
        assert controller.get_right_param() == 0
        assert controller.get_forward_param() == 0
        assert controller.get_backward_param() == 0
        assert controller.get_rotate_left_param() == 0
        assert controller.get_rotate_right_param() == 0

    def test_start(self):
        """Test if final image created correctly.

        :except final_image: a image with ROI drawn on it and the detected hand.
        """

        # setup
        controller = Controller()

        # run
        controller.start()

    def test_move_left(self):
        """Test if drone moves left properly.

        this test test if controller.move_left is corresponding to detected hand movements.
        """

        # setup
        controller = Controller()
        controller.logger.setLevel('DEBUG')

        # run
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Controller')
        while controller.flags_handler.quit_flag is False:
            ret, frame = camera.read()

            # A good practice is to mock all this processing in pipe.
            # Controller processing pipe:
            # 1. Draw ROI on frame.
            controller.frame_handler.input_frame = frame
            # 2. Cover faces, to remove detection noises.
            controller.face_processor.face_covered_frame = controller.frame_handler.input_frame
            # 3. Remove background from a covered-faces-frame.
            controller.back_ground_remover.detected_frame = controller.face_processor.face_covered_frame
            # 4. Detect a hand.
            controller.detector.input_frame_for_feature_extraction = controller.back_ground_remover.detected_frame
            # 5. Extract features, and track detected hand
            controller.extractor.extract = controller.detector

            inner_image = controller.extractor.get_drawn_extreme_contour_points()
            if inner_image is not None:
                # Draw detected hand on outer image.
                outer_image = controller.frame_handler.input_frame
                outer_image[0: inner_image.shape[0],
                outer_image.shape[1] - inner_image.shape[1]: outer_image.shape[1]] = inner_image
                cv2.imshow('Controller', outer_image)

                # For testing.
                move_left = controller.get_left_param()
                controller.logger.debug("move_left: {}".format(move_left))
                if move_left < 0:
                    assert False
                    break
                elif move_left > 100:
                    assert False
                    break

            controller.flags_handler.keyboard_input = cv2.waitKey(1)

        camera.release()
        cv2.destroyWindow('Controller')

    def test_move_right(self):
        """Test if drone moves right properly.

        this test test if controller.move_right is corresponding to detected hand movements.
        """

        # setup
        controller = Controller()
        controller.logger.setLevel('DEBUG')

        # run
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Controller')
        while controller.flags_handler.quit_flag is False:
            ret, frame = camera.read()

            # A good practice is to mock all this processing in pipe.
            # Controller processing pipe:
            # 1. Draw ROI on frame.
            controller.frame_handler.input_frame = frame
            # 2. Cover faces, to remove detection noises.
            controller.face_processor.face_covered_frame = controller.frame_handler.input_frame
            # 3. Remove background from a covered-faces-frame.
            controller.back_ground_remover.detected_frame = controller.face_processor.face_covered_frame
            # 4. Detect a hand.
            controller.detector.input_frame_for_feature_extraction = controller.back_ground_remover.detected_frame
            # 5. Extract features, and track detected hand
            controller.extractor.extract = controller.detector

            inner_image = controller.extractor.get_drawn_extreme_contour_points()
            if inner_image is not None:
                # Draw detected hand on outer image.
                outer_image = controller.frame_handler.input_frame
                outer_image[0: inner_image.shape[0],
                outer_image.shape[1] - inner_image.shape[1]: outer_image.shape[1]] = inner_image
                cv2.imshow('Controller', outer_image)

                # For testing.
                move_right = controller.get_right_param()
                controller.logger.debug("move_right: {}".format(move_right))
                if move_right < 0:
                    assert False
                    break
                elif move_right > 100:
                    assert False
                    break

            controller.flags_handler.keyboard_input = cv2.waitKey(1)

        camera.release()
        cv2.destroyWindow('Controller')

    def test_move_up(self):
        """Test if drone moves up properly.

        this test test if controller.move_up is corresponding to detected hand movements.
        """

        # setup
        controller = Controller()
        controller.logger.setLevel('DEBUG')

        # run
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Controller')
        while controller.flags_handler.quit_flag is False:
            ret, frame = camera.read()

            # A good practice is to mock all this processing in pipe.
            # Controller processing pipe:
            # 1. Draw ROI on frame.
            controller.frame_handler.input_frame = frame
            # 2. Cover faces, to remove detection noises.
            controller.face_processor.face_covered_frame = controller.frame_handler.input_frame
            # 3. Remove background from a covered-faces-frame.
            controller.back_ground_remover.detected_frame = controller.face_processor.face_covered_frame
            # 4. Detect a hand.
            controller.detector.input_frame_for_feature_extraction = controller.back_ground_remover.detected_frame
            # 5. Extract features, and track detected hand
            controller.extractor.extract = controller.detector

            inner_image = controller.extractor.get_drawn_extreme_contour_points()
            if inner_image is not None:
                # Draw detected hand on outer image.
                outer_image = controller.frame_handler.input_frame
                outer_image[0: inner_image.shape[0],
                outer_image.shape[1] - inner_image.shape[1]: outer_image.shape[1]] = inner_image
                cv2.imshow('Controller', outer_image)

                # For testing.
                move_up = controller.get_up_param()
                controller.logger.debug("move_up: {}".format(move_up))
                if move_up < 0:
                    assert False
                    break
                elif move_up > 100:
                    assert False
                    break

            controller.flags_handler.keyboard_input = cv2.waitKey(1)

        camera.release()
        cv2.destroyWindow('Controller')

    def test_move_down(self):
        """Test if drone moves down properly.

        this test test if controller.move_down is corresponding to detected hand movements.
        """

        # setup
        controller = Controller()
        controller.logger.setLevel('DEBUG')

        # run
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Controller')
        while controller.flags_handler.quit_flag is False:
            ret, frame = camera.read()

            # A good practice is to mock all this processing in pipe.
            # Controller processing pipe:
            # 1. Draw ROI on frame.
            controller.frame_handler.input_frame = frame
            # 2. Cover faces, to remove detection noises.
            controller.face_processor.face_covered_frame = controller.frame_handler.input_frame
            # 3. Remove background from a covered-faces-frame.
            controller.back_ground_remover.detected_frame = controller.face_processor.face_covered_frame
            # 4. Detect a hand.
            controller.detector.input_frame_for_feature_extraction = controller.back_ground_remover.detected_frame
            # 5. Extract features, and track detected hand
            controller.extractor.extract = controller.detector

            inner_image = controller.extractor.get_drawn_extreme_contour_points()
            if inner_image is not None:
                # Draw detected hand on outer image.
                outer_image = controller.frame_handler.input_frame
                outer_image[0: inner_image.shape[0],
                outer_image.shape[1] - inner_image.shape[1]: outer_image.shape[1]] = inner_image
                cv2.imshow('Controller', outer_image)

                # For testing.
                move_down = controller.get_down_param()
                controller.logger.debug("move_down: {}".format(move_down))
                if move_down < 0:
                    assert False
                    break
                elif move_down > 100:
                    assert False
                    break

            controller.flags_handler.keyboard_input = cv2.waitKey(1)

        camera.release()
        cv2.destroyWindow('Controller')

    def test_rotate_left(self):
        """Test if drone rotates left properly.

        this test test if controller.rotate_left is corresponding to detected hand movements.
        """

        # setup
        controller = Controller()
        controller.logger.setLevel('DEBUG')

        # run
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Controller')
        while controller.flags_handler.quit_flag is False:
            ret, frame = camera.read()

            # A good practice is to mock all this processing in pipe.
            # Controller processing pipe:
            # 1. Draw ROI on frame.
            controller.frame_handler.input_frame = frame
            # 2. Cover faces, to remove detection noises.
            controller.face_processor.face_covered_frame = controller.frame_handler.input_frame
            # 3. Remove background from a covered-faces-frame.
            controller.back_ground_remover.detected_frame = controller.face_processor.face_covered_frame
            # 4. Detect a hand.
            controller.detector.input_frame_for_feature_extraction = controller.back_ground_remover.detected_frame
            # 5. Extract features, and track detected hand
            controller.extractor.extract = controller.detector

            inner_image = controller.extractor.get_drawn_extreme_contour_points()
            if inner_image is not None:
                # Draw detected hand on outer image.
                outer_image = controller.frame_handler.input_frame
                outer_image[0: inner_image.shape[0],
                outer_image.shape[1] - inner_image.shape[1]: outer_image.shape[1]] = inner_image
                cv2.imshow('Controller', outer_image)

                # Testing.
                rotate_left = controller.get_rotate_left_param()
                controller.logger.debug("rotate_left: {}".format(rotate_left))
                if rotate_left < 0:
                    assert False
                    break
                elif rotate_left > 100:
                    assert False
                    break

            controller.flags_handler.keyboard_input = cv2.waitKey(1)

        camera.release()
        cv2.destroyWindow('Controller')

    def test_rotate_right(self):
        """Test if drone rotates right properly.

        this test test if controller.rotate_right is corresponding to detected hand movements.
        """

        # setup
        controller = Controller()
        controller.logger.setLevel('DEBUG')

        # run
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Controller')
        while controller.flags_handler.quit_flag is False:
            ret, frame = camera.read()

            # A good practice is to mock all this processing in pipe.
            # Controller processing pipe:
            # 1. Draw ROI on frame.
            controller.frame_handler.input_frame = frame
            # 2. Cover faces, to remove detection noises.
            controller.face_processor.face_covered_frame = controller.frame_handler.input_frame
            # 3. Remove background from a covered-faces-frame.
            controller.back_ground_remover.detected_frame = controller.face_processor.face_covered_frame
            # 4. Detect a hand.
            controller.detector.input_frame_for_feature_extraction = controller.back_ground_remover.detected_frame
            # 5. Extract features, and track detected hand
            controller.extractor.extract = controller.detector

            inner_image = controller.extractor.get_drawn_extreme_contour_points()
            if inner_image is not None:
                # Draw detected hand on outer image.
                outer_image = controller.frame_handler.input_frame
                outer_image[0: inner_image.shape[0],
                outer_image.shape[1] - inner_image.shape[1]: outer_image.shape[1]] = inner_image
                cv2.imshow('Controller', outer_image)

                # Testing.
                rotate_right = controller.get_rotate_right_param()
                controller.logger.debug("rotate_right: {}".format(rotate_right))
                if rotate_right < 0:
                    assert False
                    break
                elif rotate_right > 100:
                    assert False
                    break

            controller.flags_handler.keyboard_input = cv2.waitKey(1)

        camera.release()
        cv2.destroyWindow('Controller')

    def test_move_forward(self):
        """Test if drone moves forward properly.

        this test test if controller.move_forward is corresponding to detected hand movements.
        """

        # setup
        controller = Controller()
        controller.logger.setLevel('DEBUG')

        # run
        camera = cv2.VideoCapture(0)
        cv2.namedWindow('Controller')
        while controller.flags_handler.quit_flag is False:
            ret, frame = camera.read()

            # A good practice is to mock all this processing in pipe.
            # Controller processing pipe:
            # 1. Draw ROI on frame.
            controller.frame_handler.input_frame = frame
            # 2. Cover faces, to remove detection noises.
            controller.face_processor.face_covered_frame = controller.frame_handler.input_frame
            # 3. Remove background from a covered-faces-frame.
            controller.back_ground_remover.detected_frame = controller.face_processor.face_covered_frame
            # 4. Detect a hand.
            controller.detector.input_frame_for_feature_extraction = controller.back_ground_remover.detected_frame
            # 5. Extract features, and track detected hand
            controller.extractor.extract = controller.detector

            inner_image = controller.extractor.get_drawn_extreme_contour_points()
            if inner_image is not None:
                # Draw detected hand on outer image.
                outer_image = controller.frame_handler.input_frame
                outer_image[0: inner_image.shape[0],
                outer_image.shape[1] - inner_image.shape[1]: outer_image.shape[1]] = inner_image
                cv2.imshow('Controller', outer_image)

                # Testing.
                move_forward = controller.get_forward_param()
                controller.logger.debug("move_forward: {}".format(move_forward))
                if move_forward < 0:
                    assert False
                    break
                elif move_forward > 100:
                    assert False
                    break

            controller.flags_handler.keyboard_input = cv2.waitKey(1)

        camera.release()
        cv2.destroyWindow('Controller')
