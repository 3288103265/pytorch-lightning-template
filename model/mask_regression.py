import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SyncBatchNorm as SynchronizedBatchNorm2d


def masks_to_layout(boxes, masks, H, W=None):
    """
    Inputs:
        - boxes: Tensor of shape (b, num_o, 4) giving bounding boxes in the format
            [x0, y0, x1, y1] in the [0, 1] coordinate space
        - masks: Tensor of shape (b, num_o, M, M) giving binary masks for each object
        - H, W: Size of the output image.
    Returns:
        - out: Tensor of shape (N, num_o, H, W)
    """
    b, num_o, _ = boxes.size()
    M = masks.size(2)
    assert masks.size() == (b, num_o, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes.view(b*num_o, -1), H,
                          W).float().cuda(device=masks.device)

    img_in = masks.float().view(b*num_o, 1, M, M)
    sampled = F.grid_sample(img_in, grid, mode='bilinear')

    return sampled.view(b, num_o, H, W)


def _boxes_to_grid(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output
    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    ww, hh = boxes[:, 2], boxes[:, 3]

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid


class MaskRegressNet(nn.Module):
    def __init__(self, obj_feat=128, mask_size=16, map_size=64):
        super(MaskRegressNet, self).__init__()
        self.mask_size = mask_size
        self.map_size = map_size

        self.fc = nn.utils.spectral_norm(nn.Linear(obj_feat, 128 * 4 * 4))
        conv1 = list()
        conv1.append(nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)))
        conv1.append(SynchronizedBatchNorm2d(128))
        conv1.append(nn.ReLU())
        self.conv1 = nn.Sequential(*conv1)

        conv2 = list()
        conv2.append(nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)))
        conv2.append(SynchronizedBatchNorm2d(128))
        conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*conv2)

        conv3 = list()
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)))
        conv3.append(SynchronizedBatchNorm2d(128))
        conv3.append(nn.ReLU())
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(128, 1, 1, 1)))
        conv3.append(nn.Sigmoid())
        self.conv3 = nn.Sequential(*conv3)

    def forward(self, obj_feat, bbox):
        """
        :param obj_feat: (b*num_o, feat_dim)
        :param bbox: (b, num_o, 4)
        :return: bbmap: (b, num_o, map_size, map_size)
        """
        b, num_o, _ = bbox.size()
        obj_feat = obj_feat.view(b * num_o, -1)
        x = self.fc(obj_feat)
        x = self.conv1(x.view(b * num_o, 128, 4, 4))
        x = F.interpolate(x, size=8, mode='bilinear')
        x = self.conv2(x)
        x = F.interpolate(x, size=16, mode='bilinear')
        x = self.conv3(x)
        x = x.view(b, num_o, 16, 16)

        bbmap = masks_to_layout(bbox, x, self.map_size).view(
            b, num_o, self.map_size, self.map_size)
        return bbmap


class MaskRegressNetv2(nn.Module):
    def __init__(self, obj_feat=128, mask_size=16, map_size=64):
        super(MaskRegressNetv2, self).__init__()
        self.mask_size = mask_size
        self.map_size = map_size

        self.fc = nn.utils.spectral_norm(nn.Linear(obj_feat, 256 * 4 * 4))
        conv1 = list()
        conv1.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv1.append(nn.InstanceNorm2d(256))
        conv1.append(nn.ReLU())
        self.conv1 = nn.Sequential(*conv1)

        conv2 = list()
        conv2.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv2.append(nn.InstanceNorm2d(256))
        conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*conv2)

        conv3 = list()
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(256, 256, 3, 1, 1)))
        conv3.append(nn.InstanceNorm2d(256))
        conv3.append(nn.ReLU())
        conv3.append(nn.utils.spectral_norm(nn.Conv2d(256, 1, 1, 1)))
        conv3.append(nn.Sigmoid())
        self.conv3 = nn.Sequential(*conv3)

    def forward(self, obj_feat, bbox):
        """
        :param obj_feat: (b*num_o, feat_dim)
        :param bbox: (b, num_o, 4)
        :return: bbmap: (b, num_o, map_size, map_size)
        """
        b, num_o, _ = bbox.size()
        obj_feat = obj_feat.view(b * num_o, -1)
        x = self.fc(obj_feat)
        x = self.conv1(x.view(b * num_o, 256, 4, 4))
        x = F.interpolate(x, size=8, mode='bilinear')
        x = self.conv2(x)
        x = F.interpolate(x, size=16, mode='bilinear')
        x = self.conv3(x)
        x = x.view(b, num_o, 16, 16)

        bbmap = masks_to_layout(bbox, x, self.map_size).view(
            b, num_o, self.map_size, self.map_size)
        return bbmap
