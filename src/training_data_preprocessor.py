import json
import csv
import time
import argparse
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
import cv2
import torchvision.transforms as transforms
import PIL.Image
import numpy as np

from os import path
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

DATASETS_DIR = '../datasets/'
MODELS_DIR = '../pretrained-models/'

DATASET_POSE = 'human_pose.json'
MODEL_RESNET18 = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
MODEL_RESNET18_OPTIMIZED = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

WIDTH = 224
HEIGHT = 224

parser = argparse.ArgumentParser(description='Video clips processor for generating training data.')
parser.add_argument('--video_dir', type=str)
parser.add_argument('--csv_path', type=str)
args = parser.parse_args()

# load json containing human pose tasks
with open(DATASETS_DIR + DATASET_POSE, 'r') as f:
    human_pose = json.load(f)

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
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), MODELS_DIR + MODEL_RESNET18_OPTIMIZED)

model_trt = torch2trt.TRTModule()
model_trt.load_state_dict(torch.load(MODELS_DIR + MODEL_RESNET18_OPTIMIZED))

# parse objects from the NN
parse_objects = ParseObjects(topology)
# draw parsed objects onto IMG
draw_objects = DrawObjects(topology)

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

# transform IMG to tensor for NN
def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

# get keypoints from image
def get_keypoints(image, counts, objects, peaks):
    """
    peaks: 1x18x100x2
    """
    COCO_inds = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]  # COCO order
    frame_dets = []  # List of all {pose, score}s within each frame
    height, width = image.shape[:2]
    count = int(counts[0])
    for i in range(count):
        obj = objects[0][i]
        C = obj.shape[0]  # 18
        pose = np.empty((0, 2), float)
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                peak = peaks[0][j][k]
                x = float(peak[1]) * width
                y = float(peak[0]) * height
                pose = np.vstack([pose, [x, y]])
            else:
                pose = np.vstack([pose, [0, 0]])
        pose = pose[COCO_inds]  # Reorder to COCO format
        # print(pose)
        for j in range(pose.shape[0]):
            coords = tuple(np.round(pose[j]).astype(int))
            color = (0, 255, 0)
            cv2.circle(image, coords, 3, color, 2)
            # cv2.putText(image, str(j), coords, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        score = 1.0
        det = {'pose': pose, 'score': score}
        frame_dets.append(det)
    return frame_dets

# execute the NN
def executeNN(frame):
    image = cv2.resize(frame, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)
    counts = counts.detach().cpu().numpy()
    objects = objects.detach().cpu().numpy()
    peaks = peaks.detach().cpu().numpy()
    height, width = image.shape[:2]
    image_resized = cv2.resize(image, (int(width) * 5, int(height) * 5))
    keypoints = get_keypoints(image_resized, counts, objects, peaks)
    return image_resized, keypoints

def video_capture_init(file_path):
    return cv2.VideoCapture(file_path)

def video_capture_destroy():
    cv2.destroyAllWindows()
    video_capture.release()

# iterate frames of a video
def iter_frames(video_capture):
    id = 0
    while (video_capture.isOpened()):
        ret, frame = video_capture.read()
        if not ret:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        id =+ 1
        yield frame, 

# parse keypoints for each frame of a video clip into a json
def parse_videoclip(video_clip):
    '''
    Generates training data for a video clip.
    IN:
        videoclip as csv data row
    OUT:
        a json file in COCO format containing the keypoints
        for each frame of the given video clip as well as the
        represented action category.
    '''
    json = {
        "images": [],
        "annotations": []
    }
    csv_header = {'Activity': 0, 'Category': 1, 'StartOfFrame': 2, 'Directory': 3}
    category = video_clip[csv_header['Category']]
    activity = video_clip[csv_header['Activity']]
    directory = video_clip[csv_header['Directory']]
    frame_start = video_clip[csv_header['StartOfFrame']]
    frame_end = frame_start + 3
    file_path = path.join(args.video_dir, category, activity, directory, "clip_{0}_{1}.mp4".format(frame_start, frame_end))
    print(file_path)
    video_capture = video_capture_init(file_path)
    for frame, frame_id in iter_frames(video_capture):
        image, keypoints = executeNN(frame)
        json["images"].append({
            "id": frame_id,
            "width": image.width,
            "height": image.height
        })
        json["annotations"].append({
            "id": frame_id,
            "image_id": frame_id,
            "category_id": activity,
            "keypoints": keypoints
        })
    video_capture_destroy()
    return json

if __name__ == '__main__':
    """
    Generates training data for all video clips.
    """

    # open csv containing video clip locations and category information
    with open(args.csv_path) as csv_file:
        csv_data = csv.reader(csv_file, delimiter=';')

        for clip in csv_data:
            json = parse_videoclip(clip)
            print(json)
