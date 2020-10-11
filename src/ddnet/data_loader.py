import csv
import json
import os
import re

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


DATASETS_DIR = '../../datasets/'
TRAINING_DATA_DIR = os.path.join(DATASETS_DIR, "training-data")


RGB_MEAN = np.array([0.485, 0.456, 0.406])  # ImageNet RGB
RGB_STDDEV = np.array([0.229, 0.224, 0.225])  # ImageNet RGB
PAF_MEAN = np.array([0.5])
PAF_STDDEV = np.array([0.5])

random_index = np.random.randint


class DDNetDataset(Dataset):

    def __init__(self, training_data_dir, phase, n_input_frames=6):
        """
        Initializer
        :param csv_file: csv file containing image list
        """
        self._training_data_dir = training_data_dir
        self._phase = phase
        self.n_input_frames = n_input_frames
        self._skeletons = list()  # List of n_input_frames x P x 2
        self._labels = list()
        self._indices = range(18)  # BODY_25 w/0 face & feet
        # Load csv
        self.prepare_dataset()
        self._jcd, self._pose, self._y = DDNetDataset.data_generator(self._skeletons, self._labels)
        # Data dimensions
        self.n_joints, self.d_joints = self._pose.shape[2:]

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        """
        Return image tensor (n_input_frames x 5 PAFs) x H x W & label
        """
        motions = self.pose_motion(self._pose[idx, ...])
        return (self._jcd[idx, ...], *motions, self._y[idx, ...])

    @staticmethod
    def data_generator(skeletons, labels):
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
        y = []
        for skeleton, label in zip(skeletons, labels):
            m = get_cg(skeleton)

            x_0.append(m)
            x_1.append(skeleton)
            y.append(label)

        x_0 = np.stack(x_0)
        x_1 = np.stack(x_1)
        y = np.stack(y)
        return x_0, x_1, y

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
        diff_slow = DDNetDataset.poses_diff(poses)
        diff_slow = diff_slow.reshape((frame_l, -1))
        fast = poses[::2, ...]
        diff_fast = DDNetDataset.poses_diff(fast)
        diff_fast = diff_fast.reshape((frame_l // 2, -1))
        return diff_slow, diff_fast

    @staticmethod
    def get_poses(annotations):
        poses = list()
        for pose in annotations["keypoints"]:
            for p in pose["pose"]:
                if p[0] > 0.0 and p[1] > 0.0:
                    kp = np.asarray(pose["pose"])#.reshape(-1, 3)[:, 0:2]  # P x 2
                    poses.append(kp)
                    break
        return poses

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
        data = dict()
        for train_file in os.listdir(self._training_data_dir):
            with open(os.path.join(self._training_data_dir, train_file)) as train_json:
                print(train_file)
                train_data = json.loads(train_json.read())
                activity = train_data["annotations"][0]["category_id"]

                sequence = sorted([(i, j) for i in train_data["images"] for j in train_data["annotations"] if j["image_id"] == i["id"]], key=lambda k: k[0]["id"])
                k = self.n_input_frames
                if len(sequence) > k:
                    for i in range(len(sequence) - k + 1):
                        sub_sequence = sequence[i:i + k]
                        skeletons = list()
                        max_height = 0
                        for img_fn, json_fn in sub_sequence:
                            poses = DDNetDataset.get_poses(json_fn)
                            bbs = DDNetDataset.get_bboxes(poses)
                            if bbs:
                                if len(bbs) == 1:
                                    skeletons.append(poses[0])
                                    height = bbs[0][3]
                                else:  # Remove duplicate bounding boxes (skeletons)
                                    # Check which cropped image size == bounding box size
                                    width = img_fn["width"]
                                    height = img_fn["height"]
                                    onehot = [x[2:] == [width, height] for x in bbs]
                                    if sum(onehot) == 1:
                                        idx = onehot.index(True)
                                        skeletons.append(poses[idx])
                                if height > max_height:
                                    max_height = height
                        if len(skeletons) == self.n_input_frames:  # Add to dataset if complete
                            # Normalize to the center of the body (hip) & largest height in sequence
                            skeletons = [x - x[8, :] for x in skeletons]
                            skeletons = [x / max_height for x in skeletons]
                            self._labels.append(activity)
                            self._skeletons.append(np.stack(skeletons)[:, self._indices, :])



def ddnet_data_main():
    phase = 'test'
    dataset = DDNetDataset(TRAINING_DATA_DIR, phase)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(data_loader):
        [print(x.shape) for x in sample_batched]
        break


if __name__ == '__main__':
    ddnet_data_main()
