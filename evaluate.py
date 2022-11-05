import sys
sys.path.append('core')

import argparse
import logging
import numpy as np
import os
import os.path as osp
import torch
from tqdm import tqdm

import datasets
from config.config_loader import cpy_eval_args_to_config
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
from raft import RAFT


@torch.no_grad()
def create_kitti_submission(model, iters=[4, 8, 24], output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)
    coarsest_scale = 16

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in tqdm(range(len(test_dataset))):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti', coarsest_scale=coarsest_scale)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def create_sintel_submission(model, iters=[10, 15, 20], warm_start=False, output_path='sintel_submission', split="test"):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    coarsest_scale = 16

    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split=split, aug_params=None, dstype=dstype, show_extra_info=True)

        flow_prev, sequence_prev = None, None
        for test_id in tqdm(range(len(test_dataset))):
            if split == "test":
                image1, image2, (sequence, frame) = test_dataset[test_id]
            elif split == "training":
                image1, image2, _, _, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            _, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                init_flow_for_next_low, _ = model(image1, image2, iters=iters, flow_init=None, test_mode=True)
                flow_prev = forward_interpolate(init_flow_for_next_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame_%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_middlebury_submission(model, iters=[10, 15, 20], warm=False, output_path='middlebury_submission'):
    """Create submission for the Middlebury benchmark"""

    model.eval()
    coarsest_scale = 16

    val_dataset = datasets.Middlebury(full=True, show_extra_info=True, root='middlebury/test')

    prev_flow = None
    prev_scene = '<invalid>'

    for data_id in tqdm(range(len(val_dataset))):
        image1, image2, _, _, (scene, sample_id) = val_dataset[data_id]

        if scene != prev_scene:
            prev_flow = None

        prev_scene = scene

        # only submit SCENE/frame10.flo
        if sample_id != 'frame10' and (not warm or sample_id != 'frame09'):
            continue

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
        image1, image2 = padder.pad(image1, image2)

        if sample_id == 'frame10':
            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=prev_flow, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        if warm and sample_id == 'frame09':
            prev_flow_low, prev_flow_pr = model(image1, image2, iters=iters, flow_init=None, test_mode=True)
            prev_flow = forward_interpolate(prev_flow_low[0])[None].cuda()

        # only submit SCENE/frame10.flo
        if sample_id == 'frame10':
            path_out = osp.join(output_path, scene, f'{sample_id}.flo')

            os.makedirs(osp.dirname(path_out), exist_ok=True)
            frame_utils.writeFlow(path_out, flow)


@torch.no_grad()
def create_viper_submission(model, iters=[10, 15, 20], output_path='viper_submission'):
    """ Peform validation using the (official) Viper validation split"""

    model.eval()
    coarsest_scale = 16

    val_dataset = datasets.Viper(split="test", show_extra_info=True)

    for data_id in tqdm(range(len(val_dataset))):
        image1, image2, _, _, (scene, sample_id) = val_dataset[data_id]
        _, h, w = image1.shape

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=None, test_mode=True)
        flow = padder.unpad(flow_pr[0])

        flow = flow.permute(1, 2, 0).cpu().numpy()

        # save
        path_out = osp.join(output_path, f'{scene}_{sample_id:05d}.flo')

        os.makedirs(osp.dirname(path_out), exist_ok=True)
        frame_utils.writeFlow(path_out, flow)


@torch.no_grad()
def validate_sintel(model, warm=True, iters=[10, 15, 20]):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    coarsest_scale = 16

    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, show_extra_info=True)
        epe_list = []
        efel_list = []

        flow_prev, sequence_prev = None, None

        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _, (sequence, frame) = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            if warm:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            mag = torch.sum(flow_gt**2, dim=0).sqrt()
            epe = epe.view(-1)
            mag = mag.view(-1)
            efel = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
            efel_list.append(efel.cpu().numpy())

            sequence_prev = sequence

        efel_list = np.concatenate(efel_list)
        FL = 100 * np.mean(efel_list)

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation WARM: %s (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, FL_error: %f" % (str(warm), dstype, epe, px1, px3, px5, FL))
        if warm:
            results['warm' + dstype] = np.mean(epe_list)
            results['warm' + dstype + 'FL_error'] = FL
        else:
            results[dstype] = np.mean(epe_list)
            results[dstype + 'FL_error'] = FL

    return results


@torch.no_grad()
def validate_kitti(model, iters=[4, 8, 24]):
    """ Peform validation using the KITTI-2015 (train) split """
    logger = logging.getLogger("eval.kitti")
    model.eval()
    val_dataset = datasets.KITTI(split='training')
    out_list, epe_list = [], []
    coarsest_scale = 16

    for data_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt = val_dataset[data_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape, mode='kitti', coarsest_scale=coarsest_scale)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    logger.warning("Kitti iters:%d validation:%f, %f" % (sum(iters), epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_middlebury(model, iters=[10, 15, 20], warm=True):
    """ Peform validation using the (full) Middlebury dataset"""
    logger = logging.getLogger("eval.middlebury")

    model.eval()

    val_dataset = datasets.Middlebury(full=warm,  show_extra_info=True)

    epe_list = []
    coarsest_scale = 16

    prev_flow = None
    prev_scene = '<invalid>'

    for data_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt, (scene, sample_id) = val_dataset[data_id]

        if flow_gt is None and not warm:
            continue

        if scene != prev_scene:
            prev_flow = None

        prev_scene = scene

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
        image1, image2 = padder.pad(image1, image2)

        if flow_gt is not None:
            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=prev_flow, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

        if warm:
            prev_flow_low, prev_flow_pr = model(image1, image2, iters=iters, flow_init=None, test_mode=True)
            prev_flow = forward_interpolate(prev_flow_low[0])[None].cuda()

        if flow_gt is None:
            continue

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()

        epe = epe.view(-1)
        val = valid_gt.view(-1) >= 0.5

        epe_list.append(epe[val].mean().item())

    epe = np.mean(epe_list)

    tag = 'warm' if warm else 'cold'
    logger.warning(f"Middlebury ({tag}) iters: {sum(iters)} epe: {epe}")
    return {f"middlebury-{tag}-epe": epe}


@torch.no_grad()
def validate_viper(model, iters=[10, 15, 20], wauc_bins=100):
    """ Peform validation using the (official) Viper validation split"""
    logger = logging.getLogger("eval.viper")

    model.eval()

    val_dataset = datasets.Viper(split="val", show_extra_info=True)

    out_list, epe_list, wauc_list = [], [], []
    coarsest_scale = 16

    for data_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt, (scene, sample_id) = val_dataset[data_id]
        _, h, w = flow_gt.shape

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        flow_gt = flow_gt.cuda()

        padder = InputPadder(image1.shape, coarsest_scale=coarsest_scale)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=None, test_mode=True)
        flow = padder.unpad(flow_pr[0])

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt >= 0.5
        val_e = val & ((flow_gt[0].abs() < 1000) & (flow_gt[1].abs() < 1000)).cpu()
        val_e = val_e.reshape(-1)
        val = val.reshape(-1)

        # weighted area under curve
        wauc = 0.0
        w_total = 0.0
        for i in range(1, wauc_bins + 1):
            w = 1.0 - (i - 1.0) / wauc_bins
            d = 5 * (i / wauc_bins)

            wauc += w * (epe[val] <= d).float().mean()
            w_total += w

        wauc = (100.0 / w_total) * wauc

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val_e].mean().cpu().item())
        out_list.append(out[val_e].cpu().numpy())
        wauc_list.append(wauc.cpu().item())

    wauc_list = np.array(wauc_list)
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    wauc = np.mean(wauc_list)
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    logger.warning(f"Viper: iters: {sum(iters)}, wauc: {wauc} ({wauc_bins} bins), epe: {epe}, f1: {f1}")
    return {"viper-epe": epe, "viper-f1": f1, f"viper-wauc-{wauc_bins}": wauc}


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1", "True")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--warm', action='store_true', help="use warm start", default=True)
    parser.add_argument('--iters', type=int, nargs='+', default=[10, 15, 20])
    parser.add_argument('--lookup_pyramid_levels', type=int, default=2)
    parser.add_argument('--lookup_radius', default=4)
    parser.add_argument('--mixed_precision', help='use mixed precision', type=str2bool, default=True)
    parser.add_argument('--cuda_corr', help="use cuda kernel for on-demand cost computation", action='store_true', default=False)

    args = parser.parse_args()
    config = cpy_eval_args_to_config(args)
    print(config)
    model = torch.nn.DataParallel(RAFT(config))
    model.load_state_dict(torch.load(config["model"]))

    model.cuda()
    model.eval()
    print("**********MIXED:", config['mixed_precision'])
    print('WARM: ', config["warm"])

    with torch.no_grad():

        if config["dataset"] == 'sintel_test':
            print(osp.join(os.path.dirname(config["model"]), f'sintel_test_iters{sum(config["iters"])}_mixed{config["mixed_precision"]}_warm{config["warm"]}'))
            out_path = osp.join(os.path.dirname(config["model"]), f'sintel_test_iters{sum(config["iters"])}_mixed{config["mixed_precision"]}_warm{config["warm"]}')
            create_sintel_submission(model, iters=config["iters"], warm_start=config['warm'], output_path=out_path)

        elif config["dataset"] == 'sintel':
            validate_sintel(model.module, iters=config["iters"], warm=config['warm'])

        elif config["dataset"] == "kitti_test":
            print(osp.join(os.path.dirname(config["model"]), f'kitti_test_iters{sum(config["iters"])}_mixed{config["mixed_precision"]}'))
            out_path = osp.join(os.path.dirname(config["model"]), f'kitti_test_iters{sum(config["iters"])}_mixed{config["mixed_precision"]}')
            create_kitti_submission(model, output_path=osp.join(out_path, 'flow'), iters=config["iters"])

        elif config["dataset"] == 'kitti':
            validate_kitti(model.module, iters=config["iters"])

        elif config["dataset"] == 'middlebury_test':
            print(osp.join(os.path.dirname(config["model"]), f'middlebury_test_iters{sum(config["iters"])}_mixed{config["mixed_precision"]}_warm{config["warm"]}'))
            out_path = osp.join(os.path.dirname(config["model"]), f'middlebury_test_iters{sum(config["iters"])}_mixed{config["mixed_precision"]}_warm{config["warm"]}')
            create_middlebury_submission(model, iters=config["iters"], warm=config['warm'], output_path=out_path)

        elif config["dataset"] == 'middlebury':
            validate_middlebury(model.module, iters=config["iters"])

        elif config["dataset"] == 'viper_test':
            print(osp.join(os.path.dirname(config["model"]), f'viper_test_iters{sum(config["iters"])}_mixed{config["mixed_precision"]}'))
            out_path = osp.join(os.path.dirname(config["model"]), f'viper_test_iters{sum(config["iters"])}_mixed{config["mixed_precision"]}')
            create_viper_submission(model, iters=config["iters"], output_path=out_path)

        elif config["dataset"] == 'viper':
            validate_viper(model.module, iters=config["iters"])
