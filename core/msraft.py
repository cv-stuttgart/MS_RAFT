import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from update import BasicUpdateBlock
from extractor import BasicEncoder, Basic_Context_Encoder
from corr import AlternateCorrBlock, CorrBlock
from utils.utils import coords_grid, upflow2

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


def downflow(flow, mode='bilinear', factor=0.125):
    old_size = (flow.shape[2], flow.shape[3])
    new_size = (int(factor * flow.shape[2]), int(factor * flow.shape[3]))
    u_scale = new_size[1]/old_size[1]
    v_scale = new_size[0]/old_size[0]
    resized_flow = F.interpolate(flow, size=new_size, mode=mode, align_corners=True)  # b 2 h w
    resized_flow_split = torch.split(resized_flow, 1,  dim=1)
    rescaled_flow = torch.cat([u_scale*resized_flow_split[0], v_scale*resized_flow_split[1]], dim=1)

    return rescaled_flow


class MS_RAFT(nn.Module):
    def __init__(self, args):
        super(MS_RAFT, self).__init__()
        self.args = args

        self.correlation_depth = 2*81

        self.hidden_dim = 128
        self.context_dim = 128

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn="group")
        self.cnet = Basic_Context_Encoder(output_dim=256, norm_fn="group")
        self.update_block = BasicUpdateBlock(self.args, self.correlation_depth, hidden_dim=128, scale=2)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow16(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//16, W//16).to(img.device)
        coords1 = coords_grid(N, H//16, W//16).to(img.device)

        return coords0, coords1

    def get_grid(self, img, scale):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//scale, W//scale).to(img.device)
        return coords0

    def upsample_flow(self, flow, mask, scale=8):
        """ Upsample flow field [H/scale, W/scale, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale*H, scale*W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        with autocast(enabled=self.args["mixed_precision"]):
            # run the feature network
            fnet_pyramid = self.fnet([image1, image2])
            # run the context network
            cnet_pyramid = self.cnet(image1)

        coords0, coords1 = self.initialize_flow16(image1)
        flow_predictions = []
        if flow_init is not None:
            coords1 = coords1 + flow_init

        assert len(fnet_pyramid) == len(cnet_pyramid), 'fnet and cnet pyramid should have the same length.'
        assert len(fnet_pyramid) == len(iters), 'pyramid levels and the length of GRU iteration lists should be the same.'

        for index, (fmap1, fmap2) in enumerate(fnet_pyramid):
            fmap1 = fmap1.float()
            fmap2 = fmap2.float()

            if self.args['cuda_corr']:
                corr_fn = AlternateCorrBlock(fmap1, fmap2)
            else:
                corr_fn = CorrBlock(fmap1, fmap2)

            net, inp = torch.split(cnet_pyramid[index], [128, 128], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

            for itr in range(iters[index]):

                coords1 = coords1.detach()
                if index >= 1 and itr == 0:
                    coords1 = self.upsample_flow(coords1, up_mask, scale=2)
                    coords0 = self.get_grid(image1, scale=16/(2**index))

                corr = corr_fn(coords1)
                flow = coords1 - coords0
                with autocast(enabled=self.args["mixed_precision"]):
                    net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

                # F(t+1) = F(t) + \Delta(t)
                coords1 = coords1 + delta_flow
                # upsample predictions
                flow_up = self.upsample_flow(coords1 - coords0, up_mask, scale=2)
                for i in range(len(fnet_pyramid) - index):
                    flow_up = upflow2(flow_up)

                flow_predictions.append(flow_up)

        if test_mode:

            flow_low = downflow(flow_up, factor=0.0625)
            return flow_low, flow_up

        return flow_predictions
