from unittest import TestCase
from mockito import when, mock, unstub
from hallopy.controller import Controller

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
class TestController(TestCase):
    def test_get_up_param(self):
        """Test if up param up can be negative. """
        # setup
        controller = Controller()
        when(controller).get_up_param().thenReturn(-1)
        # todo: I didnt understand mockito well enoufgh, tommorow ill try again
        # run
        assert controller.get_up_param() == -1

        # clean up
        unstub()

    def test_get_down_param(self):
        """Test if down param initiate with 0 value. """
        # setup
        controller = Controller()
        when(controller).get_down_param().thenReturn(0)

        # run
        assert controller.get_down_param() == 0

        # clean up
        unstub()

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
