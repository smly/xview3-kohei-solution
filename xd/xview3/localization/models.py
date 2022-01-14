import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from xd.xview3.localization.models_hrnet import get_seg_model


class BinarizedF(Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, threshold)
        a = torch.ones_like(input).cuda()
        b = torch.zeros_like(input).cuda()
        output = torch.where(input >= threshold, a, b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # print('grad_output',grad_output)
        input, threshold = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = 0.2 * grad_output
        if ctx.needs_input_grad[1]:
            grad_weight = -grad_output
        return grad_input, grad_weight


class compressedSigmoid(nn.Module):
    def __init__(self, para=2.0, bias=0.2):
        super(compressedSigmoid, self).__init__()

        self.para = para
        self.bias = bias

    def forward(self, x):
        output = 1.0 / (self.para + torch.exp(-x)) + self.bias
        return output


class BinarizedModule(nn.Module):
    def __init__(self, input_channels=720):
        super(BinarizedModule, self).__init__()

        self.Threshold_Module = nn.Sequential(
            nn.Conv2d(
                input_channels, 256, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PReLU(),
            # nn.AvgPool2d(15, stride=1, padding=7),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            # nn.AvgPool2d(15, stride=1, padding=7),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.AvgPool2d(15, stride=1, padding=7),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(15, stride=1, padding=7),
        )

        self.sig = compressedSigmoid()
        self.weight = nn.Parameter(torch.Tensor(1).fill_(0.5), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(1).fill_(0), requires_grad=True)

    def forward(self, feature, pred_map):
        p = F.interpolate(pred_map.detach(), scale_factor=0.125)
        f = F.interpolate(feature.detach(), scale_factor=0.5)
        f = f * p
        threshold = self.Threshold_Module(f)
        threshold = self.sig(threshold * 10.0)  # fixed factor
        threshold = F.interpolate(threshold, scale_factor=8)
        binar_map = BinarizedF.apply(pred_map, threshold)
        return threshold, binar_map


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Conv2d(1, 1, 3, stride=1, padding=1)

    def forward(self, input):
        output = self.cnn(input)
        return output


class CrowdLocator(nn.Module):
    def __init__(self, net_name, gpu_id, binar_input_channels=720):
        super(CrowdLocator, self).__init__()

        if net_name == "hrnet":
            self.extractor = get_seg_model(net_name)
        else:
            assert False

        self.binar = BinarizedModule(input_channels=binar_input_channels)

        if len(gpu_id) > 1:
            self.extractor = torch.nn.DataParallel(self.extractor).cuda()
            self.binar = torch.nn.DataParallel(self.binar).cuda()
        else:
            self.extractor = self.extractor.cuda()
            self.binar = self.binar.cuda()

    @property
    def loss(self):
        return self.head_map_loss, self.binar_map_loss

    def forward(self, img, mask_gt, mode="train"):
        # print(size_map_gt.max())
        feature, pre_map = self.extractor(img)
        threshold_matrix, binar_map = self.binar(feature, pre_map)

        if mode == "train":
            assert pre_map.size(2) == mask_gt.size(2)
            self.binar_map_loss = (torch.abs(binar_map - mask_gt)).mean()
            self.head_map_loss = F.mse_loss(pre_map, mask_gt)

        return threshold_matrix, pre_map, binar_map


class CrowdLocatorV2(CrowdLocator):
    def __init__(self, backbone="hrnet", gpu_id="0,1", binar_input_channels=720):
        super(CrowdLocatorV2, self).__init__(backbone, gpu_id, binar_input_channels)
