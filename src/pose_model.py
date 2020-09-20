"""
This module contains a wrapper for the trt_pose model to estimate human poses.
"""

from os import path
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
import cv2
import torchvision.transforms as transforms
import PIL.Image
import numpy as np
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

DATASETS_DIR = '../datasets/'
MODELS_DIR = '../pretrained-models/'

DATASET_POSE = 'human_pose.json'
MODEL_RESNET18 = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
MODEL_RESNET18_OPTIMIZED = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

WIDTH = 224
HEIGHT = 224

class PoseModel():
    """
    Class for estimating poses with trt_pose.
    """
    def __init__(self):
        # load json containing human pose tasks
        with open(DATASETS_DIR + DATASET_POSE, 'r') as human_pose_file:
            human_pose = json.load(human_pose_file)
        # set topology
        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        # load model
        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
        # load model weights
        model.load_state_dict(torch.load(MODELS_DIR + MODEL_RESNET18))
        # optimize the model
        data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
        if not path.exists(MODELS_DIR + MODEL_RESNET18_OPTIMIZED):
            model.load_state_dict(torch.load(MODELS_DIR + MODEL_RESNET18))
            self.model_trt = torch2trt.torch2trt(\
                model, [data], fp16_mode=True, max_workspace_size=1<<25)
            torch.save(self.model_trt.state_dict(), MODELS_DIR + MODEL_RESNET18_OPTIMIZED)
        self.model_trt = torch2trt.TRTModule()
        self.model_trt.load_state_dict(torch.load(MODELS_DIR + MODEL_RESNET18_OPTIMIZED))
        # setup
        self.parse_objects = ParseObjects(topology)
        self.draw_objects = DrawObjects(topology)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')

    @classmethod
    def get_keypoints(cls, image, counts, objects, peaks):
        """
        peaks: 1x18x100x2
        """
        # COCO order
        coco_indices = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        # List of all {pose, score}s within each frame
        frame_dets = []
        height, width = image.shape[:2]
        count = int(counts[0])
        for i in range(count):
            obj = objects[0][i]
            pose = np.empty((0, 2), float)
            for j in range(obj.shape[0]):
                k = int(obj[j])
                if k >= 0:
                    peak = peaks[0][j][k]
                    k_x = float(peak[1]) * width
                    k_y = float(peak[0]) * height
                    pose = np.vstack([pose, [k_x, k_y]])
                else:
                    pose = np.vstack([pose, [0, 0]])
            pose = pose[coco_indices]
            for j in range(pose.shape[0]):
                coords = tuple(np.round(pose[j]).astype(int))
                cv2.circle(image, coords, 3, (0, 255, 0), 2)
            det = {'pose': pose, 'score': 1.0}
            frame_dets.append(det)
        return frame_dets

    def preprocess(self, image):
        """
        Preprocesses an image before handing it over to be
        processed by the NN.
        """
        self.device = torch.device('cuda')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def estimate_pose(self, frame):
        """
        Passes an image through the NN to estimate the poses on it.
        Returns the frame resized and its keypoints.
        """
        image = cv2.resize(frame, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        data = self.preprocess(image)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)
        counts = counts.detach().cpu().numpy()
        objects = objects.detach().cpu().numpy()
        peaks = peaks.detach().cpu().numpy()
        height, width = image.shape[:2]
        image_resized = cv2.resize(image, (int(width) * 5, int(height) * 5))
        keypoints = PoseModel.get_keypoints(image_resized, counts, objects, peaks)
        return image_resized, keypoints
        
