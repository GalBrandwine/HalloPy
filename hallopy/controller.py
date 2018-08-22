from HalloPy.hallopy import extractor


class Controller(extractor.Extractor):
    """Controller class holds a detector and a extractor.

    :param extractor.Extractor: implemented interface
    """

    def __init__(self):
        """Init a controller object.  """
        self.move_up = 0
        # todo: init self with param:    move_left = 0
        # todo: init self with param:    move_right = 0
        # todo: init self with param:    move_down = 0
        # todo: init self with param:    move_forward = 0
        # todo: init self with param:    move_backward = 0
        # todo: init self with param:    rotate_left = 0
        # todo: init self with param:    rotate_right = 0

    # todo: make a unittest for checking inheritence correctness
    def get_up_param(self):
        """Return up parameter (int between 0..100). """
        if self.move_up <= 0:
            return 0
        return self.move_up if self.move_up <= 100 else 100

    # todo: make a unittest for checking inheritence correctness
    def get_down_param(self):
        """Return down parameter (int between 0..100). """
        if self.x < 0:
            return 0
        return self.x if self.x <= 100 else 100

    # todo: make a unittest for checking inheritence correctness
    def get_left_param(self, x):
        """Return left parameter (int between 0..100). """
        if self.x < 0:
            return 0
        return self.x if self.x <= 100 else 100

    # todo: make a unittest for checking inheritence correctness
    def get_right_param(self, x):
        """Return right parameter (int between 0..100). """
        if x < 0:
            return 0
        return x if x <= 100 else 100

    # todo: make a unittest for checking inheritence correctness
    def get_rotate_left_param(self, x):
        """Return rotate left parameter (int between 0..100). """
        if x < 0:
            return 0
        return x if x <= 100 else 100

    # todo: make a unittest for checking inheritence correctness
    def get_rotate_right_param(self, x):
        """Return rotate right parameter (int between 0..100). """
        if x < 0:
            return 0
        return x if x <= 100 else 100

    # todo: make a unittest for checking inheritence correctness
    def get_forward_param(self, x):
        """Return move forward parameter (int between 0..100). """
        if x < 0:
            return 0
        return x if x <= 100 else 100

    # todo: make a unittest for checking inheritence correctness
    def get_backward_param(self, x):
        """Return move backward parameter (int between 0..100). """
        if x < 0:
            return 0
        return x if x <= 100 else 100


if __name__ == '__main__':
    test = Controller()
    print(test.get_up_param())
