from unittest import TestCase
from hallopy.controller import Controller

"""Mockito usage.

from mockito import when, mock, unstub

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
"""
class TestController(TestCase):
    def test_get_up_param(self):
        """Test if up param above 0. """

        temp_controller = Controller()

        self.fail()

    def test_get_down_param(self):
        self.fail()

    def test_get_left_param(self):
        self.fail()

    def test_get_right_param(self):
        self.fail()

    def test_get_rotate_left_param(self):
        self.fail()

    def test_get_rotate_right_param(self):
        self.fail()

    def test_get_forward_param(self):
        self.fail()

    def test_get_backward_param(self):
        self.fail()
