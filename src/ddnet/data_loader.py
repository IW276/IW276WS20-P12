import csv
import json
import os
import re
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DATASETS_DIR = '../datasets/'
TRAINING_DATA_DIR = os.path.join(DATASETS_DIR, "training-data")

RGB_MEAN = np.array([0.485, 0.456, 0.406])  # ImageNet RGB
RGB_STDDEV = np.array([0.229, 0.224, 0.225])  # ImageNet RGB
PAF_MEAN = np.array([0.5])
PAF_STDDEV = np.array([0.5])

random_index = np.random.randint

class DDNetDataset(Dataset):
    def __init__(self, annotations_file, phase, n_input_frames=6):
        self._phase = phase
        self.n_input_frames = n_input_frames
        self._skeletons = list()  # List of n_input_frames x P x 2
        self._labels = list()
        self._indices = range(15)  # BODY_25 w/0 face & feet
        # parse training data
        self.prepare_dataset(annotations_json)
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
    def get_bbox(pose):
        pose = np.copy(pose)
        pose[pose == 0.0] = np.nan
        x_min, y_min = np.nanmin(pose, axis=0)
        x_max, y_max = np.nanmax(pose, axis=0)
        width, height = x_max - x_min, y_max - y_min
        center_x, center_y = x_min + width / 2, y_min + height / 2
        width *= 1.3
        height *= 1.3
        x_min, y_min = round(center_x - width / 2), round(center_y - height / 2)
        x_max, y_max = round(center_x + width / 2), round(center_y + height / 2)
        width, height = x_max - x_min, y_max - y_min
        return (x_min, y_min, width, height)

    def prepare_dataset(self, train_data):
        skeletons = []
        max_height = 0
        images = train_data['images']
        annotations = train_data['annotations']
        annotated_frames = [(i, a) for i in images for a in annotations if a['image_id'] == i['id']]
        for (image, annotation) in annotated_frames:
            if len(skeletons) > self.n_input_frames:
                # Normalize to the center of the body (hip) & largest height in sequence
                skeletons = [x - x[8, :] for x in skeletons]
                skeletons = [x / max_height for x in skeletons]
                self._labels.append(annotation['category_id'])
                self._skeletons.append(np.stack(skeletons)[:, self._indices, :])
                return
            pose = np.asarray(annotation['keypoints']['pose']).reshape(-1, 3)[:, 0:2]  # P x 2
            bbox = DDNetDataset.get_bbox(pose)
            if bbox:
                skeletons.append(pose)
                height = bbox[3]
                if height > max_height:
                    max_height = height

if __name__ == '__main__':
    PHASE = 'train'
    for train_file in os.listdir(TRAINING_DATA_DIR):
        with open(train_file) as train_json:
            train_data = json.loads(train_json.read())
            dataset = DDNetDataset(train_data, PHASE)
            data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
            for i_batch, sample_batched in enumerate(data_loader):
                [print(x.shape) for x in sample_batched]
                break