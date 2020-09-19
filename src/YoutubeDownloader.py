import csv
import os
import re
from pytube import YouTube
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def importcsv(path):
    importedcsv = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            importedcsv.append(row)
    return importedcsv


def download(imp_csv):
    dict = {'Activity': 0, 'Category': 1, 'frame_sec': 2, 'id': 3}
    downloaded_files = []
    count_failed = 0
    failed_videos = []
    for entry in tqdm(imp_csv):
        if "https://www.youtube.com/watch?v=" + entry[dict['id']] in failed_videos:
            count_failed += 1
            continue
        path = 'videos\{0}\{1}\{2}'.format(entry[dict['Category']], entry[dict['Activity']], entry[dict['id']])
        if os.path.exists(os.path.join(os.path.dirname(__file__), path)):
            continue
        # if entry[dict['id']] in downloaded_files and os.path.exists(os.path.join(os.path.dirname(__file__), path)):
        #     continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            YouTube("https://www.youtube.com/watch?v=" + entry[dict['id']]).streams.first().download(
                os.path.join(os.path.dirname(__file__), path))
        except:
            if not "https://www.youtube.com/watch?v=" + entry[dict['id']] in failed_videos:
                failed_videos.append("https://www.youtube.com/watch?v=" + entry[dict['id']])
                count_failed += 1
        downloaded_files.append(entry[dict['id']])
    print("Failed to download {0} videos.".format(count_failed))
    with open("failed_videos.txt", "w") as f:
        for entry in failed_videos:
            f.writelines(entry + '\n')

    return [x for x in imp_csv if not "https://www.youtube.com/watch?v={0}".format(x[dict['id']]) in failed_videos]


def save(csv_list):
    with open('MPII_youtube_offline.csv', 'w', newline='') as file:
        file_writer = csv.writer(file, delimiter=';')
        for entry in csv_list:
            file_writer.writerow(entry)


def lookup_file(orig, path):
    for entry in orig:
        base = os.path.dirname(entry)
        if base == path:
            return entry
    print("could not find correct path")
    print("path was {0}".format(path))
    return 0


def cut_videos(imp_csv, sec):
    # Create list of all original videos:
    orig_videos = []
    dict = {'Activity': 0, 'Category': 1, 'frame_sec': 2, 'id': 3}
    for entry in tqdm(imp_csv):
        path = 'videos\{0}\{1}\{2}'.format(entry[dict['Category']], entry[dict['Activity']], entry[dict['id']])
        count = 0
        try:
            for file in os.listdir(os.path.abspath(path)):
                x = re.search("clip", file)
                if not x:
                    if not os.path.join(os.path.abspath(path), file) in orig_videos:
                        orig_videos.append(os.path.join(os.path.abspath(path), file))
        except:
            print("Could not find file at {0}".format(path))

    # start cutting
    print('Start cutting videos to be of length {0} seconds'.format(sec))
    for entry in tqdm(imp_csv):
        path = 'videos\{0}\{1}\{2}'.format(entry[dict['Category']], entry[dict['Activity']], entry[dict['id']])
        original = lookup_file(orig_videos, os.path.abspath(path))

        if original == 0:
            print("Could not find file at {0}".format(path))

        start_time = int(entry[dict['frame_sec']])
        end_time = int(entry[dict['frame_sec']]) + sec
        clipname = "clip_{0}_{1}.mp4".format(start_time, end_time)
        destination = os.path.join(os.path.dirname(original), clipname)

        try:
            ffmpeg_extract_subclip(original, start_time, end_time, targetname=destination)
        except Exception as e:
            print(e)
            print("Something went wrong with {0}".format(original))


def main():
    # Sometimes Youtube downloads don't work out very well if connections is terminated or something like that.
    # Use repeat_yt_download to repeat the whole thing
    repeat_yt_download = 1
    print('Importing csv.')
    imp_csv = importcsv("MPII_youtube.csv")
    print('Downloading Youtube videos.')
    if repeat_yt_download != 0:
        original_csv = imp_csv
        for i in range(repeat_yt_download):
            imp_csv = download(original_csv)
    else:
        imp_csv = download(imp_csv)

    # Save the new correct csv without the offline yt links
    save(imp_csv)
    # cut videos, second parameter is the length of the videos.
    cut_videos(imp_csv, 3)


if __name__ == "__main__":
    main()
