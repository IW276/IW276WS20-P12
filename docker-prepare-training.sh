#!/bin/sh

# 1. argument: path to a video directory on the host where the sample videos will be downloaded to.
# 2. argument: path to the training-data directory where the sample data will be placed in.
#
# Returns: the sample data.

CONTAINER="trtpose-activity-demo:1"

docker run --runtime nvidia -v $1:/videos -v $2:/training_data_dir $CONTAINER /bin/bash -c "cd IW276WS20-P12/src && python3 train_1_download_videos.py  /videos/ && python3 train_2_generate_annotations.py /videos/ /IW276WS20-P12/datasets/MPII_youtube_offline.csv --training_data_dir /training_data_dir --drop_dirty_pose"

echo "Resulting training data can be found in $2"
