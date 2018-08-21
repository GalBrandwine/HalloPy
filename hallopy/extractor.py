class Extractor(object):
    """Extractor interface.

    This module contain the methods a Controller need.

    """

    def get_up_param(self, x):
        """Return up parameter (int between 0..100)."""
        if x < 0:
            return 0
        return x if x <= 100 else 100

    def get_down_param(self, x):
        """Return down parameter (int between 0..100)."""
        if x < 0:
            return 0
        return x if x <= 100 else 100

    def get_left_param(self, x):
        """Return left parameter (int between 0..100)."""
        if x < 0:
            return 0
        return x if x <= 100 else 100

    def get_right_param(self, x):
        """Return right parameter (int between 0..100)."""
        if x < 0:
            return 0
        return x if x <= 100 else 100

    def get_rotate_left_param(self, x):
        """Return rotate left parameter (int between 0..100)."""
        if x < 0:
            return 0
        return x if x <= 100 else 100

    def get_rotate_right_param(self, x):
        """Return rotate right parameter (int between 0..100)."""
        if x < 0:
            return 0
        return x if x <= 100 else 100

    def get_forward_param(self, x):
        """Return move forward parameter (int between 0..100)."""
        if x < 0:
            return 0
        return x if x <= 100 else 100

    def get_backward_param(self, x):
        """Return move backward parameter (int between 0..100)."""
        if x < 0:
            return 0
        return x if x <= 100 else 100
