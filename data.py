import os
import glob
import random
import math
import logging

import numpy as np
import scipy.io as sio
from PIL import Image
import imageio
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision.transforms import functional
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from torch.utils.serialization import load_lua
from torchvision.transforms.functional import to_tensor

from utils.dataset import DuplicateBatchSampler
from utils.regime import Regime
from img_utils import crop
from preprocess import get_transform

__DATASETS_DEFAULT_PATH = '/home/ANT.AMAZON.COM/nadavb/datasets'


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
    train = (split == 'train')
    root = os.path.join(datasets_path, name)
    if name == 'cmu_segm':
        root = os.path.join(root.rsplit('/', 1)[0], 'SURREAL/data/cmu', )
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        root = os.path.join(root, 'run0')
        return Cmu(root=root,
                   train=train,
                   transform=transform,
                   target_transform=target_transform)


joints_idx_tmp = [8, 5, 2, 3, 6, 9, 1, 7, 13, 16, 21, 19, 17, 18, 20, 22]
JOINTS_IDX = [i - 1 for i in joints_idx_tmp]

smpl_2_segm = [0, 2, 12, 9, 2, 13, 10, 2, 14, 11, 2, 14, 11, 2, 2, 2, 1, 6, 3, 7, 4, 8, 5, 8, 5]


class BadImageError(Exception):
    pass


class Cmu(data.Dataset):
    scale = 0.25
    rotate = 30

    def __init__(self, root, train, transform, target_transform):
        self.img_paths = glob.glob('{}/**/*.mp4'.format(root), recursive=True)[1:]
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        mean_rgb = load_lua('meanstd/meanRgb.t7')
        self.mean = mean_rgb['mean']
        self.std = mean_rgb['std']
        self.out_size = [64, 64]
        self.in_size = [3, 256, 256]

    def __getitem__(self, seq_index):
        try:
            seq_path = self.img_paths[seq_index]
            seq_length = self.get_duration(seq_path)
            if self.train:
                frame_idx = random.randint(0, seq_length - 1)
            else:
                frame_idx = seq_length // 2

            img = self.load_rgb(seq_path, frame_idx)
            segm = self.get_segm(seq_path, frame_idx)

            im_size = img.shape

            joints2d = self.get_joints_2d(seq_path, frame_idx)

            scale = self.get_scale(joints2d)
            center = self.get_center(joints2d)
            if center[0] < 1 or center[1] < 1 or center[0] > im_size[1] or center[1] > im_size[0]:
                raise BadImageError('Human out of image {} center: {}, {}'
                                    .format(seq_path, center[0], center[1]))

            rot = 0
            if self.train:
                rnd = lambda x: max(-2 * x, min(2 * x, torch.randn(1) * x))
                scale = scale * (2 ** rnd(Cmu.scale))
                rot = rnd(Cmu.rotate)
                if torch.bernoulli(torch.tensor(0.4)):
                    rot = 0

            img = crop(img, center.float(), scale, rot, self.in_size[1], Image.BILINEAR)
            img = functional.to_tensor(img).float()
            img = self.color_augmentation(img)
            img = functional.normalize(img, mean=self.mean, std=self.std)

            segm = crop(segm, center.float(), scale, rot, self.out_size[0], Image.NEAREST)
            label = torch.from_numpy(segm).long()

            return img, label

        except (BadImageError, TypeError) as e:
            logging.debug(e)

    @staticmethod
    def load_rgb(path, t):
        return imageio.get_reader(path, 'ffmpeg').get_data(t)

    @staticmethod
    def get_duration(path):
        path = path.replace('.mp4', '_info')
        info = sio.loadmat(path)
        return len(info['zrot'])

    @staticmethod
    def get_segm(path, t):
        path = path.replace('.mp4', '_segm.mat')
        seg_dict = sio.loadmat(path)
        try:
            segm = seg_dict['segm_{}'.format(t)]
        except KeyError:
            raise BadImageError('Segm not loaded {}'.format(path))
        if len(segm.nonzero()) == 0:
            raise BadImageError('no segmentation available')
        # merge body parts
        segm2 = Cmu.change_segm_idx(segm, smpl_2_segm)
        return segm2

    @staticmethod
    def change_segm_idx(segm, s):
        out = np.zeros_like(segm)
        for i in range(len(s)):
            out[segm == i] = s[i]
        return out

    @staticmethod
    def get_tight_box(label):
        # Tightest bound box covering the joint positions
        min, _ = label.min(0)
        max, _ = label.max(0)
        x_min, y_min = min[0], min[1]
        x_max, y_max = max[0], max[1]

        human_width = x_max - x_min + 1
        human_height = y_max - y_min + 1

        # Slightly larger are to cover the head/heat of the human
        x_min = x_min - 0.25 * human_width
        y_min = y_min - 0.35 * human_height
        x_max = x_max + 0.25 * human_width
        y_max = y_max + 0.25 * human_height

        human_width = x_max - x_min + 1
        human_height = y_max - y_min + 1

        return x_min, y_min, human_width, human_height

    @staticmethod
    def get_scale(label):
        x_min, y_min, human_width, human_height = Cmu.get_tight_box(label)
        return max(float(human_height) / 240, float(human_width) / 240)

    @staticmethod
    def get_center(label):
        x_min, y_min, human_width, human_height = Cmu.get_tight_box(label)
        center_x = x_min + human_width / 2
        center_y = y_min + human_height / 2
        return torch.tensor((center_x, center_y))

    @staticmethod
    def get_joints_2d(path, t):
        path = path.replace('.mp4', '_info')
        info = sio.loadmat(path)
        joints2d = info['joints2D'][:, :, t][:, JOINTS_IDX]
        joints2d = torch.from_numpy(joints2d).t().long()
        joints2d.add_(1)  # it was 0-based

        if len(joints2d.nonzero()) == 0:
            raise BadImageError('all elements are zero')

        return joints2d

    @staticmethod
    def color_augmentation(img):
        for c in range(3):
            img[:, :, c].mul_(torch.empty_like(img[:, :, c]).uniform_(0.6, 1.4)) \
                .clamp_(0, 1)
        return img

    def __len__(self):
        return len(self.img_paths)


def ignore_exceptions_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


_DATA_ARGS = {'name', 'split', 'transform',
              'target_transform', 'download', 'datasets_path'}
_DATALOADER_ARGS = {'batch_size', 'shuffle', 'sampler', 'batch_sampler',
                    'num_workers', 'collate_fn', 'pin_memory', 'drop_last',
                    'timeout', 'worker_init_fn'}
_TRANSFORM_ARGS = {'transform_name', 'input_size',
                   'scale_size', 'normalize', 'augment', 'num_crops'}
_OTHER_ARGS = {'distributed', 'duplicates'}


class DataRegime(object):
    def __init__(self, regime, defaults={}):
        self.regime = Regime(regime, defaults)
        self.epoch = 0
        self.steps = None
        self.get_loader(True)

    def get_setting(self):
        setting = self.regime.setting
        loader_setting = {k: v for k,
                                   v in setting.items() if k in _DATALOADER_ARGS}
        data_setting = {k: v for k, v in setting.items() if k in _DATA_ARGS}
        transform_setting = {
            k: v for k, v in setting.items() if k in _TRANSFORM_ARGS}
        other_setting = {k: v for k, v in setting.items() if k in _OTHER_ARGS}
        transform_setting.setdefault('transform_name', data_setting['name'])
        return {'data': data_setting, 'loader': loader_setting,
                'transform': transform_setting, 'other': other_setting}

    def get_loader(self, force_update=False):
        if force_update or self.regime.update(self.epoch, self.steps):
            setting = self.get_setting()
            self._transform = get_transform(**setting['transform'])
            setting['data'].setdefault('transform', self._transform)
            self._data = get_dataset(**setting['data'])
            if setting['other'].get('distributed', False):
                setting['loader']['sampler'] = DistributedSampler(self._data)
                setting['loader']['shuffle'] = None
                # pin-memory currently broken for distributed
                setting['loader']['pin_memory'] = False
            if setting['other'].get('duplicates', 0) > 1:
                setting['loader']['shuffle'] = None
                sampler = setting['loader'].get(
                    'sampler', RandomSampler(self._data))
                setting['loader']['sampler'] = DuplicateBatchSampler(sampler, setting['loader']['batch_size'],
                                                                     duplicates=setting['other']['duplicates'],
                                                                     drop_last=setting['loader'].get('drop_last',
                                                                                                     False))

            self._sampler = setting['loader'].get('sampler', None)
            self._loader = torch.utils.data.DataLoader(
                self._data, **setting['loader'])
            if setting['other'].get('duplicates', 0) > 1:
                self._loader.batch_sampler = self._sampler
        return self._loader

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self._sampler is not None and hasattr(self._sampler, 'set_epoch'):
            self._sampler.set_epoch(epoch)

    def __len__(self):
        return len(self._data)
