"""
This module downloads the training videos from YouTube
and cuts them into smaller clips to be processed for
training by the NN.
"""

import csv
import os
import re
import argparse
from tqdm import tqdm
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

DATASETS_DIR = '../../datasets/'

CSV_HEADER = {'Activity': 0, 'Category': 1, 'StartOfFrame': 2, 'Directory': 3}

def importcsv(path):
    """
    Import csv converted from mat_converter.
    """
    importedcsv = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            importedcsv.append(row)
    return importedcsv

def download(csv_data):
    """
    Download training videos from YouTube.
    """
    downloaded_files = []
    count_failed = 0
    failed_videos = []
    for entry in tqdm(csv_data):
        category = entry[CSV_HEADER['Category']]
        activity = entry[CSV_HEADER['Activity']]
        directory = entry[CSV_HEADER['Directory']]
        print(entry)
        if "https://www.youtube.com/watch?v=" + category in failed_videos:
            count_failed += 1
            continue
        path = os.path.join(args.video_dir, category, activity, directory)
        if os.path.exists(os.path.join(args.video_dir, path)):
            continue
        os.makedirs(os.path.join(args.video_dir, path), exist_ok=True)
        try:
            YouTube("https://www.youtube.com/watch?v=" + directory).streams.first().download(
                os.path.join(args.video_dir, path))
        except:
            if not "https://www.youtube.com/watch?v=" + directory in failed_videos:
                failed_videos.append("https://www.youtube.com/watch?v=" + directory)
                count_failed += 1
        downloaded_files.append(directory)
    print("Failed to download {0} videos.".format(count_failed))
    with open("failed_videos.txt", "w") as f:
        for entry in failed_videos:
            f.writelines(entry + '\n')
    return [x for x in csv_data \
        if "https://www.youtube.com/watch?v={0}".format(x[CSV_HEADER['Directory']]) \
        not in failed_videos]

def save(csv_list):
    """
    Save csv.
    """
    with open(os.path.join(DATASETS_DIR, 'MPII_youtube_offline.csv'), 'w', newline='') as file:
        file_writer = csv.writer(file, delimiter=';')
        for entry in csv_list:
            file_writer.writerow(entry)

def lookup_file(orig, path):
    """
    Lookup file.
    """
    for entry in orig:
        base = os.path.dirname(entry)
        if base == path:
            return entry
    print("could not find correct path")
    print("path was {0}".format(path))
    return 0

def cut_videos(csv_data):
    """
    Extract small video clips from videos for training.
    """
    orig_videos = []
    for entry in tqdm(csv_data):
        category = entry[CSV_HEADER['Category']]
        activity = entry[CSV_HEADER['Activity']]
        directory = entry[CSV_HEADER['Directory']]
        path = os.path.join(args.video_dir, category, activity, directory)
        try:
            for file in os.listdir(os.path.join(args.video_dir, path)):
                if not re.search("clip", file):
                    if not os.path.join(args.video_dir, path, file) in orig_videos:
                        orig_videos.append(os.path.join(args.video_dir, path, file))
        except:
            print("Could not find file at {0}".format(path))
    print('Start cutting videos to be of length {0} seconds'.format(args.clips_len))
    for entry in tqdm(csv_data):
        category = entry[CSV_HEADER['Category']]
        activity = entry[CSV_HEADER['Activity']]
        directory = entry[CSV_HEADER['Directory']]
        frame_start = entry[CSV_HEADER['StartOfFrame']]
        path = os.path.join(args.video_dir, category, activity, directory)
        original = lookup_file(orig_videos, os.path.join(args.video_dir, path))
        if original == 0:
            print("Could not find file at {0}".format(path))
        start_time = int(frame_start)
        end_time = int(frame_start) + args.clips_len
        clipname = "clip_{0}_{1}.mp4".format(start_time, end_time)
        destination = os.path.join(os.path.dirname(original), clipname)
        try:
            ffmpeg_extract_subclip(original, start_time, end_time, targetname=destination)
        except Exception as e:
            print(e)
            print("Something went wrong with {0}".format(original))

DESCRIPTION="""
Download training videos from Youtube and extract clips.
"""
parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument("video_dir", type=str)
parser.add_argument("--clips_len", type=int, default=3)
parser.add_argument("--retry_count", type=int, default=1)
args = parser.parse_args()

if __name__ == "__main__":
    """
    {}
    """.format(DESCRIPTION)
    # import csv
    csv_data = importcsv(os.path.join(DATASETS_DIR, "MPII_youtube.csv"))
    # Sometimes Youtube downloads don't work out very well
    # if connections is terminated or something like that.
    # Use args.retry_count to repeat the whole thing
    if args.retry_count != 0:
        original_csv = csv_data
        for i in range(args.retry_count):
            csv_data = download(original_csv)
    else:
        csv_data = download(csv_data)
    # Save the csv containing the video clips annotations
    save(csv_data)
    # extract videoclips
    cut_videos(csv_data)
