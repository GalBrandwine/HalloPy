"""hallopy script is an example how to use the Hand_recognition_controller.  """

import os
import sys

cwd = os.getcwd()
sys.path.insert(0, cwd)
sys.path.insert(0,cwd+"/thirdparty/TelloPy")
import tellopy
from hallopy.controller import Controller


def handler(event, sender, data, **args):
    """Drone events handler, for testing.  """
    drone_handler = sender
    if event is drone_handler.EVENT_FLIGHT_DATA:
        print(data)


def init_drone():
    """Drone initiation function for testing.  """
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


def main():
    drone = init_drone()
    controller = Controller(drone)
    controller.start()


if __name__ == '__main__':
    main()
