# Hallo - a hand controlled Tello
DJI Tello controller using hand gestures python app


Environment:
* ubuntu 16.4
* python 3.5.2

Dependency libraries:
* OpenCV 3.3.2
* tellopy repository - https://github.com/hanyazou/TelloPy.git
* numpy 1.14.5

# Usage:

1. make sure you have all dependency libraries.
  for great openCV installation tutorial refer to:
  https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/

2.1 turn on Tello drone and connect to it's wifi

2.2 there's no setup.py yet, so in order to run this project, open the project in an IDE and run: hallo.py

3
  Application usage:
   
    press 'b' - to detect palm ( depend on the environment lights, a Thresh Hold slider tuning may be needed)
      center the detected palm at the center of the detection_frame FOR 5 seconds 
      pressing b again will reset calibration.
    
  After center-circle become GREEN (meaning we are now calibrated):
  
    press 't' - to take off ( a 3 second hold-up, until drone is in the air)
      after landing - if program is calibrated, press t again to take-off
      
    press 'c' - to toggle drone's control between key-board and detected-palm
      toggling back to keyboard will force drone to hover in the same place
      
    press 'l' - to land ( at any time)

    press 'x' - to adjust background' threshold - for thicker palm recognition
    press 'z' - to adjust background' threshold - for thinner palm recognition
  
    press 'esc' to exit program ( only after drone landed)
    

# Tanks to:
* OpenCV - for the greatest computer Vision library in this world ( and others)

* tellopy repo - for making a super friendly drone api

* Adrian and his crew at - https://www.pyimagesearch.com/ for the best How-to's tutorials
  and email support.
  
* Izane for his FingerDetection - https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python
