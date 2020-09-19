import cv2
import numpy as np
import os
import json
from torch.utils.data import Dataset

COCO_INDICES = [18, 19, 16, 17, 32, 33, 24, 25, 34, 35, 12, 13, 4, 5, 14, 15, 28, 29,
                30, 31, 20, 21, 8, 9, 10, 11, 0, 1, 26, 27, 6, 7, 22, 23, 2, 3]  # COCO sequence
ACTION_NAMES = ["Other", "Punch", "Kick", "Wave", "Check watch", "Cross arms", "Film", "Get up", "Hand shake", "Hit",
           "Hug", "Pick up", "Point", "Push", "Scratch head", "Sit down", "Throw over head", "Turn around", "Walk"]

class DDNetJsonDataset(Dataset):
    def __init__(self, directory, n_input_frames, pose_json):
        self.pose_json = pose_json
        self._dir = directory
        self.n_input_frames = n_input_frames
        self._skeletons = list()  # List of n_input_frames x P x 2
        self._filenames = list()
        # Load json
        self.prepare_dataset()
        self._jcd, self._pose = DDNetJsonDataset.data_generator(self._skeletons)
        # Data dimensions
        self.n_joints, self.d_joints = self._pose.shape[2:]

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, idx):
        motions = self.pose_motion(self._pose[idx, ...])
        return (self._filenames[idx], self._jcd[idx, ...], *motions)

    @staticmethod
    def data_generator(skeletons):
        from scipy.spatial.distance import cdist

        def get_cg(p):
            def norm_scale(x):
                return (x - np.mean(x)) / np.mean(x)

            m = []
            frame_l, joint_n, _ = p.shape
            iu = np.triu_indices(joint_n, 1, joint_n)
            for f in range(frame_l):
                d_m = cdist(p[f], p[f], 'euclidean')
                d_m = d_m[iu]
                m.append(d_m)
            m = np.stack(m)
            m = norm_scale(m)
            return m

        x_0 = []
        x_1 = []
        for skeleton in skeletons:
            m = get_cg(skeleton)

            x_0.append(m)
            x_1.append(skeleton)

        x_0 = np.stack(x_0)
        x_1 = np.stack(x_1)
        return x_0, x_1

    @staticmethod
    def poses_diff(poses):  # poses: frame_l x joint_n x joint_d
        l, n = poses.shape[:2]
        x = poses[0, ...] - poses[:, ...]
        frames = cv2.split(x)
        frames = [cv2.resize(frame, (n, l), interpolation=cv2.INTER_NEAREST) for frame in frames]
        x = np.stack(frames, 2)
        return x

    @staticmethod
    def pose_motion(poses):
        frame_l = poses.shape[0]
        diff_slow = DDNetJsonDataset.poses_diff(poses)
        diff_slow = diff_slow.reshape((frame_l, -1))
        fast = poses[::2, ...]
        diff_fast = DDNetJsonDataset.poses_diff(fast)
        diff_fast = diff_fast.reshape((frame_l // 2, -1))
        return diff_slow, diff_fast

    @staticmethod
    def get_poses(pose_json):
        poses = list()
        filenames = list()
        data = json.load(pose_json)
        all_annotations = [x['annotations'] for x in data]
        all_filenames = [x['filename'] for x in data]
        for annotation, filename in zip(all_annotations, all_filenames):
                scores = [x['score'] for x in annotation]
                if scores:
                        filenames.append(filename)
                        idx = scores.index(max(scores))  # Find max score in the image
                        pose = list(annotation[idx].values())
                        pose = [pose[i] for i in COCO_INDICES]
                        pose = np.array(pose).astype(np.float)
                        pose = pose.reshape(-1, 2)
                        pose = np.insert(pose, 8, (pose[8, :] + pose[11, :]) / 2, axis=0)  # Add middle hip
                        pose = pose[0:15, :]
                        poses.append(pose)
        return poses, filenames

    @staticmethod
    def get_bboxes(poses):
        bbs = list()
        for keypoints in poses:
            keypoints = np.copy(keypoints)
            keypoints[keypoints == 0.0] = np.nan
            x_min, y_min = np.nanmin(keypoints, axis=0)
            x_max, y_max = np.nanmax(keypoints, axis=0)
            width, height = x_max - x_min, y_max - y_min
            # Expand 30% in all sides
            center_x, center_y = x_min + width / 2, y_min + height / 2
            width *= 1.3
            height *= 1.3
            x_min, y_min = round(center_x - width / 2), round(center_y - height / 2)
            x_max, y_max = round(center_x + width / 2), round(center_y + height / 2)
            width, height = x_max - x_min, y_max - y_min
            bbs.append((x_min, y_min, width, height))
        return bbs

    def prepare_dataset(self):
        poses, filenames = DDNetJsonDataset.get_poses(self.pose_json)
        # Create accessible dataset
        k = self.n_input_frames
        for i in range(len(filenames) - k + 1):
                skeletons = list()
                self._filenames.append(list())
                max_height = 0
                for j in range(k):
                        skeletons.append(poses[i + j])
                        self._filenames[i].append(filenames[i + j])
                        bb = DDNetJsonDataset.get_bboxes((poses[i + j], ))
                        height = bb[0][3]
                        if height > max_height:
                                max_height = height
        # Normalize to the center of the body (hip) & largest height in sequence
        skeletons = [x - x[8, :] for x in skeletons]
        skeletons = [x / max_height for x in skeletons]
        self._skeletons.append(np.stack(skeletons))