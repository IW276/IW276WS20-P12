# 2D Pose Based Action Recognition

This Project aims at realizing real time 2D Pose Based Action Recognition for the following five human actions:
* running
* walking
* walking the dog
* jogging
* bicycling

<p align="center">
  <img src="/walking-the-dog.png" />
</p>

 We wanted to achieve this by completing the following tasks:
- [x] using docker for building an automated training pipeline based on the previous work done by [Group 1](https://github.com/IW276/IW276SS20-P1) and the open-source [MPII](http://human-pose.mpi-inf.mpg.de/) dataset.
- [x] using docker for building an automated real time activity recognition video stream pipeline.
- [x] using [trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) for extracting the skeletons.
- [ ] using [ddnet](https://github.com/fandulu/DD-Net) for recognizing the activities.

> This work was done by ZhengChen Guan, Karsten Rudolf and Tobias Heilig during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

## Table of Contents

* [Prerequisites](#prerequisites)
* [Running](#running)
* [Acknowledgments](#acknowledgments)

## Requirements
* Python 3.6 (or above)
* OpenCV
* TensorFlow
* Scipy
* Docker
* Jetson Nano

## Running ...

The demo aka video stream pipeline as well as the training pipeline have both been containerized with docker to provide a clean runtime environment.

**Run <a href="/docker-build.sh">docker-build.sh</a> to build the docker image.**  

### ... the demo

**Run <a href="/docker-run.sh">docker-run.sh</a> to execute the video stream pipeline.**  

_First argument_ - path to a video directory on the host.  
_Second argument_ - filename of the video to be processed as found in the path specified by the first argument.  
_Returns_ - the output video placed in the video directory specified above.  

Example
```
./docker-run.sh /path/to/videos video.mp4
```

### ... the training

We did split the training execution pipeline into two separate steps: prepare the training data and execute the training. This was done to
keep it open for the user to actually start the training on his own machine or Google Collab for example (where no docker runs are supported -
the training-data has to be uploaded to Google Drive in that case and train_3_train_model.py has to be executed directly there).


**1. Run <a href="/docker-prepare-training.sh">docker-prepare-training.sh</a> to prepare the training data.**  

_First argument_ - path to a video directory on the host where the sample videos will be downloaded to.  
_Second argument_ - path to the training-data directory where the sample data will be placed in.  
_Returns_ - the sample data.  

Example
```
./docker-prepare-training.sh /path/to/videos /path/to/training-data
```

**2. Run <a href="/docker-run-training.sh">docker-run-training.sh</a> to start the training.**  

_First argument_ - path to the training-data directory where the sample data can be found. 
_Second argument_ - path to the model directory the resulting pre-trained model .pth will be placed in.  
_Returns_ - the pre-trained model.  

Example
```
./docker-run-training.sh /path/to/training-data /path/to/model
```

## TODO's

Unfortunately, there is a yet unresolved bug when starting the training. Therefore, we were not able to obtain a pre-trained model and actually use ddnet for activity recognition. In the current state of the project the recognized activities drawn onto the frames come from the pre-processed training data samples. To make the project work lively as intended the following problems have to be solved first:
- [ ] Fix segmentation fault when starting the training to obtain a model. See the internal Wiki, contact `mickael.cormier AT iosb.fraunhofer.de` for more information.
- [ ] Execute the model in the demo to recognize the activities and draw the actual results onto the frames.

## Acknowledgments

This repo is based on
  - [IWI276/IW276SS20P1](https://github.com/IW276/IW276SS20-P1)
  - [TRT_Pose](https://github.com/NVIDIA-AI-IOT/trt_pose)
  - [DD-Net](https://github.com/fandulu/DD-Net)

Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
