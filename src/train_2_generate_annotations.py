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
                "name": "jogging"
            },
            {
                "supercategory": "running",
                "id": 4,
                "name": "running"
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

def generate_annotations(video_clip):
    '''
    Generates ddnet training data for the given video clip
    by returning a json in COCO format containing annotations
    for each frame.
    '''
    video_clip_annotations = clip_annotations_template
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
        video_clip_annotations["images"].append({
            "id": frame_id,
            "width": image.shape[0],
            "height": image.shape[1]
        })
        for k in keypoints:
            if args.skip_dirty and annotation_id == frame_id:
                return
            annotation_id += 1
            video_clip_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": frame_id,
                "category_id": activity,
                "keypoints": k
            })
    annotations_path = os.path.join(TRAINING_DATA_DIR, "{}_clip_{}_{}.json".format(video_hash, frame_start, frame_end))
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
args = parser.parse_args()

if __name__ == '__main__':
    """
    {}
    """.format(DESCRIPTION)
    # create sub folder in datasets
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    # open csv containing video clip locations and category information
    with open(args.csv_path) as csv_file:
        video_clip_data = csv.reader(csv_file, delimiter=';')
        # generate annotations for each video clip
        for clip in video_clip_data:
            generate_annotations(clip)
