# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import os
import os.path as osp
import random
import parse
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

extention = './data/'


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, show_extra_info=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.show_extra_info = show_extra_info

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.flow_list[index] is not None:
            if self.sparse:
                flow, valid = frame_utils.read_flow_sparse(self.flow_list[index])
            else:
                flow = frame_utils.read_gen(self.flow_list[index])
        else:
            flow, valid = None, None

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        if flow is not None:
            flow = np.array(flow).astype(np.float32)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        if flow is not None:
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid).float()
        elif flow is not None:
            valid = ((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float()

        if self.show_extra_info:
            return img1, img2, flow, valid, self.extra_info[index]
        else:
            return img1, img2, flow, valid

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='sintel', dstype='clean', show_extra_info=False):
        root = extention + root
        super(MpiSintel, self).__init__(aug_params, show_extra_info=show_extra_info)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [[image_list[i], image_list[i+1]]]
                self.extra_info += [(scene, i)]  # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='fc/data'):
        root = extention + root
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2*i], images[2*i+1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='fth', dstype='frames_cleanpass'):
        root = extention + root
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i+1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i+1], images[i]]]
                            self.flow_list += [flows[i+1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='kitti15/dataset'):
        root = extention + root
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class KITTI_split(FlowDataset):
    def __init__(self, aug_params=None, split='kitti_train', root='kitti_split/'):
        root = extention + root
        super(KITTI_split, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='HD1k'):
        root = extention + root
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i+1]]]

            seq_ix += 1


class Middlebury(FlowDataset):
    def __init__(self, aug_params=None, root='middlebury/training', full=False, show_extra_info=False):
        super(Middlebury, self).__init__(aug_params, show_extra_info=show_extra_info, sparse=True)

        root = osp.join(extention, root)
        img_root = osp.join(root, 'img')
        flow_root = osp.join(root, 'flow')

        flows = []
        imgs = []
        info = []

        for scene in sorted(os.listdir(img_root)):
            for img1 in sorted(os.listdir(osp.join(img_root, scene))):
                num = parse.parse("frame{:02d}.png", img1)[0]

                img2 = f"frame{num + 1:02d}.png"
                flow = f"flow{num:02d}.flo"

                img1 = osp.join(img_root, scene, img1)
                img2 = osp.join(img_root, scene, img2)
                flow = osp.join(flow_root, scene, flow)

                if not osp.exists(img2):
                    continue

                if not osp.exists(flow):
                    flow = None

                if not full and flow is None:
                    continue

                imgs += [(img1, img2)]
                flows += [flow]
                info += [(scene, f"frame{num:02d}")]

        self.image_list = imgs
        self.flow_list = flows
        self.extra_info = info


class Viper(FlowDataset):
    def __init__(self, aug_params=None, root='viper', split='train', show_extra_info=False):
        super(Viper, self).__init__(aug_params, show_extra_info=show_extra_info, sparse=True)

        root = osp.join(extention, root, split)
        img_root = osp.join(root, 'img')
        flow_root = osp.join(root, 'flow')

        if split == 'train':            # ignore large outliers
            self.max_flow = 2048

        flows = []
        imgs = []
        info = []

        for group in sorted(os.listdir(img_root)):
            for img1 in sorted(os.listdir(osp.join(img_root, group))):
                num = parse.parse(group + "_{:05d}.jpg", img1)[0]

                img1 = osp.join(img_root, group, img1)
                img2 = osp.join(img_root, group, f"{group}_{num + 1:05d}.jpg")
                flow = osp.join(flow_root, group, f"{group}_{num:05d}.npz")

                if not osp.exists(img2):
                    continue

                if osp.getsize(img1) == 0 or osp.getsize(img2) == 0:
                    continue

                if split == 'test':
                    flow = None
                else:
                    if not osp.exists(flow) or osp.getsize(flow) == 0:
                        continue

                imgs += [(img1, img2)]
                flows += [flow]
                info += [(group, num)]

        self.image_list = imgs
        self.flow_list = flows
        self.extra_info = info
