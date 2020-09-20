"""
This module generates annotations in COCO format as json for each
video clip supplied via the csv file obtained by the video downloader.
"""

import os
import json
import csv
import argparse
import cv2

from frame_iterator import iter_frames
from pose_model import PoseModel

model = PoseModel()

def generate_annotations(video_clip):
    '''
    Generates ddnet training data for the given video clip
    by returning a json in COCO format containing annotations
    for each frame.
    '''
    video_clip_annotations = {
        "images": [],
        "annotations": []
    }
    csv_header = {'Activity': 0, 'Category': 1, 'StartOfFrame': 2, 'Directory': 3}
    category = video_clip[csv_header['Category']]
    activity = video_clip[csv_header['Activity']]
    directory = video_clip[csv_header['Directory']]
    frame_start = video_clip[csv_header['StartOfFrame']]
    frame_end = int(frame_start) + 3
    clip_dir = os.path.join(args.video_dir, category, activity, directory)
    clip_name = "clip_{0}_{1}.mp4".format(frame_start, frame_end)
    clip_path = os.path.join(clip_dir, clip_name)
    print("processing clip {}".format(clip_path))
    video_capture = cv2.VideoCapture(clip_path)
    for frame_id, frame in iter_frames(video_capture):
        image, keypoints = model.estimate_pose(frame)
        if len(keypoints) == 0:
            continue
        video_clip_annotations["images"].append({
            "id": frame_id,
            "width": image.shape[0],
            "height": image.shape[1]
        })
        video_clip_annotations["annotations"].append({
            "id": frame_id,
            "image_id": frame_id,
            "category_id": activity,
            "keypoints": {
                "pose" : keypoints[0]["pose"].tolist(),
                "score" : keypoints[0]["score"]
            }
        })
    annotations_path = os.path.join(clip_dir, clip_name + ".json")
    with open(annotations_path, 'w') as video_clip_json:
        json.dump(video_clip_annotations, video_clip_json, sort_keys=True, indent=2)
    cv2.destroyAllWindows()
    video_capture.release()

DESCRIPTION = """
Processes each video clip and generates a json
in COCO format containing annotations for training.
"""
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('--video_dir', type=str)
parser.add_argument('--csv_path', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    """
    {}
    """.format(DESCRIPTION)

    # open csv containing video clip locations and category information
    with open(args.csv_path) as csv_file:
        video_clip_data = csv.reader(csv_file, delimiter=';')

        for clip in video_clip_data:
            generate_annotations(clip)
