xhost +
sudo docker run --rm -it --net=host  \
   --privileged \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v /dev/video0:/dev/video0 \
   -v /dev/video1:/dev/video1 \
   -n hallopy \
   hallopy:1.3
