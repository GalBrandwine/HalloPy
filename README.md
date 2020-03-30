# Hallo - a hand controlled Tello
DJI Tello controller using hand gestures python app


## Docker installation in 3 steps:
Tested on ubuntu 19.

### step 1
```shell script
git clone github/GalBrandwine/hallopy.git
cd HalloPy
```
### step 2
```shell script
docker build --network=host  --tag hallopy:1.3 .
```
(make sure you have docker [installed](https://docs.docker.com/get-started/).)
### step 3
(Make sure you're connected to the TELLO wifi)
```shell script
xhost +  && docker run --rm -it --net=host --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --privileged -v /dev/video0:/dev/video0 --name hallopy hallopy:1.3 
```

For flags explanation: [running docker with gui](https://marcosnietoblog.wordpress.com/2017/04/30/docker-image-with-opencv-with-x11-forwarding-for-gui/)   

## None docker users:
Environment:
* ubuntu 19.10
* python 3.7

Dependency libraries installation:
* pip install av==6.1.2
* pip install opencv-python
* pip install tellopy

### Run
From directory `/HalloPy/`:
```shell script
python ./hallopy/hallo.py
```
## Controller Usage:

1. Make sure you have all dependency libraries.
  for great openCV installation tutorial refer to:
  https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/
2. Turn on Tello drone and connect to it's wifi
    1. there's no setup.py yet, so in order to run this project, open the project in an IDE and run: hallo.py
3. Application usage:
   ```
    press 'b' - to detect palm ( depend on the environment lights, a Thresh Hold slider tuning may be needed)
    center the detected palm at the center of the detection_frame FOR 5 seconds 
    pressing b again will reset calibration.
   ```
   
   After center-circle become GREEN (meaning we are now calibrated):
    ```
    press 't' - to take off ( a 3 second hold-up, until drone is in the air)
      after landing - if program is calibrated, press t again to take-off
      
    press 'c' - to toggle drone's control between key-board and detected-palm
      toggling back to keyboard will force drone to hover in the same place
      
    press 'l' - to land ( at any time)

    press 'x' - to adjust background' threshold - for thicker palm recognition
    press 'z' - to adjust background' threshold - for thinner palm recognition
  
    press 'esc' to exit program ( only after drone landed)
    ```
4. Video explaining hands movements for controlling the drone can be found [here](https://youtu.be/NSwKCzxFBv4)


# Tanks to:
* OpenCV - for the greatest computer Vision library in this world ( and others)

* tellopy repo - for making a super friendly drone api

* Adrian and his crew at - https://www.pyimagesearch.com/ for the best How-to's tutorials
  and email support.
  
* Izane for his FingerDetection - https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python
