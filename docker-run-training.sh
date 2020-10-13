#!/bin/sh

# 1. argument: path to the training-data directory where the sample data can be found.
# 2. argument: path to the model directory the resulting pre-trained model .pth will be placed in.
#
# Returns: the pre-trained model.

CONTAINER="trtpose-activity-demo:1"

docker run --runtime nvidia  -v $1:/training_data_dir -v $2:/model_dir $CONTAINER /bin/bash -c "cd IW276WS20-P12/src && python3 train_3_train_model.py --train_dir /training_data_dir/ --model_dir /model_dir/"

echo "Resulting training data can be found in $2"
