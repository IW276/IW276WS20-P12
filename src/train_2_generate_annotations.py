"""
This module generates annotations in COCO format as json for each
video clip supplied via the csv file obtained by the video downloader.
"""

import os
import json
import csv
import argparse
import cv2

from utils.frame_iterator import iter_frames
from utils.pose_model import PoseModel

DATASETS_DIR = '../datasets/'
TRAINING_DATA_DIR = os.path.join(DATASETS_DIR, "training-data")

MAX_ZERO_POSES = 6

model = PoseModel()

clip_annotations_template = {
        "categories": [
            {
                "supercategory": "walking",
                "id": 1,
                "name": "walking, general"
            },
            {
                "supercategory": "walking",
                "id": 2,
                "name": "walking the dog"
            },
            {
                "supercategory": "running",
                "id": 3,
                "name": "running"
            },
            {
                "supercategory": "running",
                "id": 4,
                "name": "jogging"
            },
            {
                "supercategory": "bicycling",
                "id": 5,
                "name": "bicycling, general"
            },
        ],
        "images": [],
        "annotations": []
    }

category_ids = { c['name']:c['id'] for c in clip_annotations_template["categories"] }

def generate_annotations(video_clip, training_data_dir):
    '''
    Generates ddnet training data for the given video clip
    by returning a json in COCO format containing annotations
    for each frame.
    '''
    video_clip_annotations = dict(clip_annotations_template)
    video_clip_annotations["annotations"] = []
    video_clip_annotations["images"] = []
    csv_header = {'Activity': 0, 'Category': 1, 'StartOfFrame': 2, 'VideoHash': 3}
    category = video_clip[csv_header['Category']]
    activity = video_clip[csv_header['Activity']]
    video_hash = video_clip[csv_header['VideoHash']]
    frame_start = video_clip[csv_header['StartOfFrame']]
    frame_end = int(frame_start) + 3
    clip_dir = os.path.join(args.video_dir, category, activity, video_hash)
    clip_name = "clip_{0}_{1}.mp4".format(frame_start, frame_end)
    clip_path = os.path.join(clip_dir, clip_name)
    print("processing clip {}".format(clip_path))
    video_capture = cv2.VideoCapture(clip_path)
    annotation_id = 0
    for frame_id, frame in iter_frames(video_capture):
        image, keypoints = model.estimate_pose(frame)
        if args.drop_dirty_pose:
            i = 0
            n = len(keypoints)
            while i < n:
                if keypoints[i]["pose"].count([0,0]) > MAX_ZERO_POSES or keypoints[i]["pose"].count([0.0, 0.0]) > MAX_ZERO_POSES:
                    keypoints.pop(i)
                    n -= 1
                else:
                    i += 1
        if not keypoints:
            continue
        if args.skip_dirty and len(keypoints) > 1:
            return
        video_clip_annotations["images"].append({
            "id": frame_id,
            "width": frame.shape[1],
            "height": frame.shape[0]
        })
        video_clip_annotations["annotations"].append({
            "image_id": frame_id,
            "category_id": category_ids[activity],
            "keypoints": keypoints
        })
    annotations_path = os.path.join(training_data_dir, "{}_clip_{}_{}.json".format(video_hash, frame_start, frame_end))
    if len(video_clip_annotations["annotations"]) < 1:
        return
    with open(annotations_path, 'w') as video_clip_json:
        json.dump(video_clip_annotations, video_clip_json, sort_keys=True, indent=2)
    cv2.destroyAllWindows()
    video_capture.release()

DESCRIPTION = """
Processes each video clip and generates a json
in COCO format containing annotations for training.
"""
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('video_dir', type=str)
parser.add_argument('csv_path', type=str)
parser.add_argument('--skip_dirty', help='skip video clips with more than a single person in it.', action='store_true')
parser.add_argument('--drop_dirty_pose', help='drop poses with multipe zero Values in poses', action='store_true')
parser.add_argument('--training_data_dir', type=str, default=TRAINING_DATA_DIR)
args = parser.parse_args()

if __name__ == '__main__':
    """
    {}
    """.format(DESCRIPTION)
    # create sub folder in datasets
    os.makedirs(args.training_data_dir, exist_ok=True)
    # open csv containing video clip locations and category information
    with open(args.csv_path) as csv_file:
        video_clip_data = csv.reader(csv_file, delimiter=';')
        # generate annotations for each video clip
        for clip in video_clip_data:
            generate_annotations(clip, args.training_data_dir)
