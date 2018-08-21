"""Detector Interface.

This module contains Interface for detectors,
Interface were built using abc class

"""
import abc


class Detector(abc.ABC):
    """abc.ABC is Abstract Base Class.  """

    @abc.abstractmethod(classmethod)
    def _drawMovementsAxes(self, inputIFrame):
        """Private method, Draws movements Axes on frame.

        Args:
            inputframe (openCV object): recieved frame from camera.

        """
        pass

    @abc.abstractmethod(classmethod)
    def _removeBG(self, frame):
        """Private method, Removes back ground.

        Returns:
            The frame, with subtracted background.

        """
        pass

    @abc.abstractmethod(classmethod)
    def _simpleAngleCalculator(self, startPoint, endPoint, farPoint):
        """Private method, calculate angle of 3 given points.

        Args:
            startPoint (Tuple):
            endPoint(Tuple):
            farPoint(Tuples):

        Returns:
            angle (float): calculated angle in degrees.

        """
        pass
