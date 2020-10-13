import json
import time
import argparse
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
import cv2
import torchvision.transforms as transforms
import PIL.Image

from utils.frame_iterator import iter_frames
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

CLASS_NAMES = {
        1: "walking, general",
        2: "walking the dog",
        3: "running",
        4: "jogging",
        5: "bicycling, general"
}

parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--video_path', type=str)
parser.add_argument('--video_filename', type=str)
parser.add_argument('--video_json', type=str)
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

# execute the NN
def execute(image, frame, t, annotation, frame_info):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)

    # TODO draw activity detection onto the frame with DDNet actually
    #
    # training data is available. At the time of writing unfortunately
    # there is a problem with ddnet which could not be resolved in the
    # scope of this labour. Therefore, we show at least a preview of the
    # expected output based on our training data.

    for kp in annotation["keypoints"]:
        pose = [y for y in kp["pose"] if int(y[0]) != 0 and int(y[1]) !=0]
        min_y = int(min(pose, key = lambda t: t[1])[1]/5/HEIGHT*frame_info["height"]) - 10 
        max_y = int(max(pose, key = lambda t: t[1])[1]/5/HEIGHT*frame_info["height"]) + 10
        min_x = int(min(pose, key = lambda t: t[0])[0]/5/WIDTH*frame_info["width"])
        max_x = int(max(pose, key = lambda t: t[0])[0]/5/WIDTH*frame_info["width"])
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        cv2.putText(frame, "%s" % (CLASS_NAMES[annotation["category_id"]]), (min_x, min_y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    fps = 1.0 / (time.time() - t)
    draw_objects(frame, counts, objects, peaks)
    return fps

def video_capture_init():
    video_capture = cv2.VideoCapture(args.video_path + args.video_filename)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter(\
            args.video_path + 'output.mp4',\
            fourcc,\
            video_capture.get(cv2.CAP_PROP_FPS),\
            (int(video_capture.get(3)),int(video_capture.get(4))))
    return video_capture, out_video

def video_capture_destroy():
    cv2.destroyAllWindows()
    out_video.release()
    video_capture.release()



if __name__ == '__main__':
    '''
    For each frame of the given video source this will
    - Preprocess the image
    - Execute the neural network
    - Parse the objects from the neural network output
    - Draw the objects onto the frame
    and write the frames back into a new video.
    '''

    print("Initializing video capture") 

    video_capture, out_video = video_capture_init()

    print("Processing frames ...")
    print("Hint: This may take a while depending on the size of the video.") 

    with open(path.join(DATASETS_DIR, "training-data", args.video_json)) as json_file:
        json_data = json.loads(json_file.read())
        annotations = json_data["annotations"]
        images_info = json_data["images"]

    for frame_id, frame in iter_frames(video_capture):
        image = cv2.resize(frame, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        image_info = next(i for i in images_info if i["id"] == frame_id)
        annotation = next(annotation for annotation in annotations if annotation["image_id"] == frame_id)
        fps = execute(image, frame, time.time(), annotation, image_info)
        cv2.putText(frame, "FPS: %f" % (fps), (20, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        out_video.write(frame)

    video_capture_destroy()

    print("Done") 
