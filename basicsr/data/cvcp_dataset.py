import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CVCPDataset(data.Dataset):
    """CVCP dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_CVCP_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number
    Examples:
    100002264_01 32
    100002264_02 32 
    ...

    Key examples: "LD_100002264_01_32F_QP37.yuv/00000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            compresseded_type (str): Compressed by LD or RA
            qp_value (str): QP value for compression. 22,27,32,37 
            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(CVCPDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        assert opt['num_frame'] % 2 == 1, (f'num_frame should be odd number, but got {opt["num_frame"]}')
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2
        self.qp_value=opt['qp_value']
        self.compresseded_type=opt['compresseded_type']
        self.gt_keys = []
        self.lq_keys = []
        qp_value=self.qp_value
        compresseded_type=self.compresseded_type
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num= line.replace('\n','').split(' ')
                self.gt_keys.extend([f'{folder}/{i:05d}' for i in range(int(frame_num))]) # key example: 100002264_01/00000
                self.lq_keys.extend([f'{compresseded_type}/{qp_value}/{compresseded_type}_{folder}_{frame_num}F_{qp_value}.yuv/{i:05d}' for i in range(int(frame_num))]) # key example: LD/QP37/LD_100002264_01_32F_QP37.yuv/00000

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        interval_str = ','.join(str(x) for x in opt['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        gt_key = self.gt_keys[index]
        lq_key = self.lq_keys[index]
        clip_name, frame_name = gt_key.split('/')  # key example: 100002264_01/00000
        type_name, qp_name , lq_clip_name, _ = lq_key.split('/')
        center_frame_idx = int(frame_name)

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        # each clip has 32 frames starting from 0 to 31
        while (start_frame_idx < 0) or (end_frame_idx > 31):
            center_frame_idx = random.randint(0, 31)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = center_frame_idx + self.num_half_frames * interval
        frame_name = f'{center_frame_idx:05d}'
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        assert len(neighbor_list) == self.num_frame, (f'Wrong length of neighbor list: {len(neighbor_list)}')

        # get the GT frame (as the center frame)
        if self.is_lmdb:
            img_gt_path = f'{clip_name}/{frame_name}'
        else:
            img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{lq_clip_name}/{neighbor:05d}'
            else:
                img_lq_path = self.lq_root / type_name / qp_name / lq_clip_name / f'{neighbor:05d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gt, 'key': gt_key}

    def __len__(self):
        return len(self.gt_keys)

