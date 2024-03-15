from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch
from basicsr.data.data_util import paths_from_lmdb
from basicsr.data.transforms import augment, single_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.bsrgan_light import degradation_bsrgan_variant
from functools import partial
from basicsr.utils.matlab_functions import imresize

@DATASET_REGISTRY.register()
class DegradationEndocderDataset(data.Dataset):
    """Read only gt images in the train phase
    Read GT. Crop two degraded LR part. Merge them into one batch. 

    Read GT.

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(DegradationEndocderDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))
        self.deg_fn = partial(degradation_bsrgan_variant, sf=opt['scale'])

    def __getitem__(self, index):
        scale = self.opt["scale"]
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # flip, rotation
            img_gt = augment(img_gt, self.opt['use_flip'], self.opt['use_rot'])
            # random crop
            img_gt = single_random_crop(img_gt, gt_size, scale, gt_path)

        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            
            
        #print(img_gt)
        #input()
        img_lq = self.deg_fn(img_gt)
        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] !=  'train':
            img_gt = img_gt[0:img_gt.shape[0] * scale, 0:img_gt.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lq = imresize(img_gt,1/scale)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)
            
        return {'gt': img_gt, 'gt_path': gt_path, 'lq': img_lq}


    def __len__(self):
        return len(self.paths)
