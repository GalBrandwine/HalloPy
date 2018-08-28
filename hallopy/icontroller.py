class Icontroller(object):
    """Extractor interface.

    This module contain the methods a Controller need.

    """

    def get_up_param(self):
        """Return up parameter (int between 0..100). """
        raise NotImplementedError

    def get_down_param(self):
        """Return down parameter (int between 0..100). """
        raise NotImplementedError

    def get_left_param(self):
        """Return left parameter (int between 0..100). """
        raise NotImplementedError

    def get_right_param(self):
        """Return right parameter (int between 0..100). """
        raise NotImplementedError

    def get_rotate_left_param(self):
        """Return rotate left parameter (int between 0..100). """
        raise NotImplementedError

    def get_rotate_right_param(self):
        """Return rotate right parameter (int between 0..100). """
        raise NotImplementedError

    def get_forward_param(self):
        """Return move forward parameter (int between 0..100). """
        raise NotImplementedError

    def get_backward_param(self):
        """Return move backward parameter (int between 0..100). """
        raise NotImplementedError
