#!/bin/sh

# 1. argument: path to a video directory on the host.
# 2. argument: filename of the video to be processed as found in the path specified by the first argument.
#
# Returns: the output video placed in the video directory specified above.

CONTAINER="trtpose-activity-demo:1"

docker run --runtime nvidia -v $1:/videos $CONTAINER /bin/bash -c "cd IW276WS20-P12/src && python3 demo.py --video_path /videos/ --video_filename $2"

echo "Resulting video can be found in $1"
