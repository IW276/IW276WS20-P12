"""
This module contains an iterator for enumerating the
frames of a given video capture.
"""

import cv2

def iter_frames(video_capture):
    """
    Iterates the frames of a given video capture.
    """
    frame_id = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        frame_id += 1
        yield frame_id, frame
