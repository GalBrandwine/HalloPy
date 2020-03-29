xhost +
sudo docker run --rm -ti --net=host --ipc=host \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -v /dev/video0:/dev/video0 \
   hallopy:1.0

#docker run --network=host --privileged --ipc=host -v /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY:$DISPLAY hallopy:1.0
