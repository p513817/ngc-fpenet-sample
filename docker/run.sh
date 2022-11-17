#!/bin/bash


xhost + > /dev/null 2>&1
docker run \
-it --rm \
-v $(pwd):/workspace \
-v /tmp/.x11-unix:/tmp/.x11-unix:rw -e DISPLAY=unix${DISPLAY} \
-v /etc/localtime:/etc/localtime:ro \
--net=host --ipc=host \
--privileged -v /dev:/dev \
fpenet-sample
