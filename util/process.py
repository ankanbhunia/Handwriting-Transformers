import argparse
from collections import namedtuple
import numpy as np
import torch
import cv2,os
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.cluster import DBSCAN

"""
taken from https://github.com/githubharald/WordDetectorNN
Download the models from https://www.dropbox.com/s/mqhco2q67ovpfjq/model.zip?dl=1 and pass the path to word_segment(.) as argument.
"""

from typing import Type, Any, Callable, Union, List, Optional

import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        out1 = self.relu(x)
        x = self.maxpool(out1)

        out2 = self.layer1(x)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        return out5, out4, out3, out2, out1

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def compute_iou(ra, rb):
    """intersection over union of two axis aligned rectangles ra and rb"""
    if ra.xmax < rb.xmin or rb.xmax < ra.xmin or ra.ymax < rb.ymin or rb.ymax < ra.ymin:
        return 0

    l = max(ra.xmin, rb.xmin)
    r = min(ra.xmax, rb.xmax)
    t = max(ra.ymin, rb.ymin)
    b = min(ra.ymax, rb.ymax)

    intersection = (r - l) * (b - t)
    union = ra.area() + rb.area() - intersection

    iou = intersection / union
    return iou

def compute_dist_mat(aabbs):
    """Jaccard distance matrix of all pairs of aabbs"""
    num_aabbs = len(aabbs)

    dists = np.zeros((num_aabbs, num_aabbs))
    for i in range(num_aabbs):
        for j in range(num_aabbs):
            if j > i:
                break

            dists[i, j] = dists[j, i] = 1 - compute_iou(aabbs[i], aabbs[j])

    return dists


def cluster_aabbs(aabbs):
    """cluster aabbs using DBSCAN and the Jaccard distance between bounding boxes"""
    if len(aabbs) < 2:
        return aabbs

    dists = compute_dist_mat(aabbs)
    clustering = DBSCAN(eps=0.7, min_samples=3, metric='precomputed').fit(dists)

    clusters = defaultdict(list)
    for i, c in enumerate(clustering.labels_):
        if c == -1:
            continue
        clusters[c].append(aabbs[i])

    res_aabbs = []
    for curr_cluster in clusters.values():
        xmin = np.median([aabb.xmin for aabb in curr_cluster])
        xmax = np.median([aabb.xmax for aabb in curr_cluster])
        ymin = np.median([aabb.ymin for aabb in curr_cluster])
        ymax = np.median([aabb.ymax for aabb in curr_cluster])
        res_aabbs.append(AABB(xmin, xmax, ymin, ymax))

    return res_aabbs


class AABB:
    """axis aligned bounding box"""

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def scale(self, fx, fy):
        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = fx * new.xmin
        new.xmax = fx * new.xmax
        new.ymin = fy * new.ymin
        new.ymax = fy * new.ymax
        return new

    def scale_around_center(self, fx, fy):
        cx = (self.xmin + self.xmax) / 2
        cy = (self.ymin + self.ymax) / 2

        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = cx - fx * (cx - self.xmin)
        new.xmax = cx + fx * (self.xmax - cx)
        new.ymin = cy - fy * (cy - self.ymin)
        new.ymax = cy + fy * (self.ymax - cy)
        return new

    def translate(self, tx, ty):
        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = new.xmin + tx
        new.xmax = new.xmax + tx
        new.ymin = new.ymin + ty
        new.ymax = new.ymax + ty
        return new

    def as_type(self, t):
        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = t(new.xmin)
        new.xmax = t(new.xmax)
        new.ymin = t(new.ymin)
        new.ymax = t(new.ymax)
        return new

    def enlarge_to_int_grid(self):
        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = np.floor(new.xmin)
        new.xmax = np.ceil(new.xmax)
        new.ymin = np.floor(new.ymin)
        new.ymax = np.ceil(new.ymax)
        return new

    def clip(self, clip_aabb):
        new = AABB(self.xmin, self.xmax, self.ymin, self.ymax)
        new.xmin = min(max(new.xmin, clip_aabb.xmin), clip_aabb.xmax)
        new.xmax = max(min(new.xmax, clip_aabb.xmax), clip_aabb.xmin)
        new.ymin = min(max(new.ymin, clip_aabb.ymin), clip_aabb.ymax)
        new.ymax = max(min(new.ymax, clip_aabb.ymax), clip_aabb.ymin)
        return new

    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def __str__(self):
        return f'AABB(xmin={self.xmin},xmax={self.xmax},ymin={self.ymin},ymax={self.ymax})'

    def __repr__(self):
        return str(self)

class MapOrdering:
    """order of the maps encoding the aabbs around the words"""
    SEG_WORD = 0
    SEG_SURROUNDING = 1
    SEG_BACKGROUND = 2
    GEO_TOP = 3
    GEO_BOTTOM = 4
    GEO_LEFT = 5
    GEO_RIGHT = 6
    NUM_MAPS = 7


def encode(shape, gt, f=1.0):
    gt_map = np.zeros((MapOrdering.NUM_MAPS,) + shape)
    for aabb in gt:
        aabb = aabb.scale(f, f)

        # segmentation map
        aabb_clip = AABB(0, shape[0] - 1, 0, shape[1] - 1)

        aabb_word = aabb.scale_around_center(0.5, 0.5).as_type(int).clip(aabb_clip)
        aabb_sur = aabb.as_type(int).clip(aabb_clip)
        gt_map[MapOrdering.SEG_SURROUNDING, aabb_sur.ymin:aabb_sur.ymax + 1, aabb_sur.xmin:aabb_sur.xmax + 1] = 1
        gt_map[MapOrdering.SEG_SURROUNDING, aabb_word.ymin:aabb_word.ymax + 1, aabb_word.xmin:aabb_word.xmax + 1] = 0
        gt_map[MapOrdering.SEG_WORD, aabb_word.ymin:aabb_word.ymax + 1, aabb_word.xmin:aabb_word.xmax + 1] = 1

        # geometry map TODO vectorize
        for x in range(aabb_word.xmin, aabb_word.xmax + 1):
            for y in range(aabb_word.ymin, aabb_word.ymax + 1):
                gt_map[MapOrdering.GEO_TOP, y, x] = y - aabb.ymin
                gt_map[MapOrdering.GEO_BOTTOM, y, x] = aabb.ymax - y
                gt_map[MapOrdering.GEO_LEFT, y, x] = x - aabb.xmin
                gt_map[MapOrdering.GEO_RIGHT, y, x] = aabb.xmax - x

    gt_map[MapOrdering.SEG_BACKGROUND] = np.clip(1 - gt_map[MapOrdering.SEG_WORD] - gt_map[MapOrdering.SEG_SURROUNDING],
                                                 0, 1)

    return gt_map


def subsample(idx, max_num):
    """restrict fg indices to a maximum number"""
    f = len(idx[0]) / max_num
    if f > 1:
        a = np.asarray([idx[0][int(j * f)] for j in range(max_num)], np.int64)
        b = np.asarray([idx[1][int(j * f)] for j in range(max_num)], np.int64)
        idx = (a, b)
    return idx


def fg_by_threshold(thres, max_num=None):
    """all pixels above threshold are fg pixels, optionally limited to a maximum number"""

    def func(seg_map):
        idx = np.where(seg_map > thres)
        if max_num is not None:
            idx = subsample(idx, max_num)
        return idx

    return func


def fg_by_cc(thres, max_num):
    """take a maximum number of pixels per connected component, but at least 3 (->DBSCAN minPts)"""

    def func(seg_map):
        seg_mask = (seg_map > thres).astype(np.uint8)
        num_labels, label_img = cv2.connectedComponents(seg_mask, connectivity=4)
        max_num_per_cc = max(max_num // (num_labels + 1), 3)  # at least 3 because of DBSCAN clustering

        all_idx = [np.empty(0, np.int64), np.empty(0, np.int64)]
        for curr_label in range(1, num_labels):
            curr_idx = np.where(label_img == curr_label)
            curr_idx = subsample(curr_idx, max_num_per_cc)
            all_idx[0] = np.append(all_idx[0], curr_idx[0])
            all_idx[1] = np.append(all_idx[1], curr_idx[1])
        return tuple(all_idx)

    return func


def decode(pred_map, comp_fg=fg_by_threshold(0.5), f=1):
    idx = comp_fg(pred_map[MapOrdering.SEG_WORD])
    pred_map_masked = pred_map[..., idx[0], idx[1]]
    aabbs = []
    for yc, xc, pred in zip(idx[0], idx[1], pred_map_masked.T):
        t = pred[MapOrdering.GEO_TOP]
        b = pred[MapOrdering.GEO_BOTTOM]
        l = pred[MapOrdering.GEO_LEFT]
        r = pred[MapOrdering.GEO_RIGHT]
        aabb = AABB(xc - l, xc + r, yc - t, yc + b)
        aabbs.append(aabb.scale(f, f))
    return aabbs


def main():
    import matplotlib.pyplot as plt
    aabbs_in = [AABB(10, 30, 30, 60)]
    encoded = encode((50, 50), aabbs_in, f=0.5)
    aabbs_out = decode(encoded, f=2)
    print(aabbs_out[0])
    plt.subplot(151)
    plt.imshow(encoded[MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND + 1].transpose(1, 2, 0))

    plt.subplot(152)
    plt.imshow(encoded[MapOrdering.GEO_TOP])
    plt.subplot(153)
    plt.imshow(encoded[MapOrdering.GEO_BOTTOM])
    plt.subplot(154)
    plt.imshow(encoded[MapOrdering.GEO_LEFT])
    plt.subplot(155)
    plt.imshow(encoded[MapOrdering.GEO_RIGHT])

    plt.show()


def compute_scale_down(input_size, output_size):
    """compute scale down factor of neural network, given input and output size"""
    return output_size[0] / input_size[0]


def prob_true(p):
    """return True with probability p"""
    return np.random.random() < p


class UpscaleAndConcatLayer(torch.nn.Module):
    """
    take small map with cx channels
    upscale to size of large map (s*s)
    concat large map with cy channels and upscaled small map
    apply conv and output map with cz channels
    """

    def __init__(self, cx, cy, cz):
        super(UpscaleAndConcatLayer, self).__init__()
        self.conv = torch.nn.Conv2d(cx + cy, cz, 3, padding=1)

    def forward(self, x, y, s):
        x = F.interpolate(x, s)
        z = torch.cat((x, y), 1)
        z = F.relu(self.conv(z))
        return z


class WordDetectorNet(torch.nn.Module):
    # fixed sizes for training
    input_size = (448, 448)
    output_size = (224, 224)
    scale_down = compute_scale_down(input_size, output_size)

    def __init__(self):
        super(WordDetectorNet, self).__init__()

        self.backbone = resnet18()

        self.up1 = UpscaleAndConcatLayer(512, 256, 256)  # input//16
        self.up2 = UpscaleAndConcatLayer(256, 128, 128)  # input//8
        self.up3 = UpscaleAndConcatLayer(128, 64, 64)  # input//4
        self.up4 = UpscaleAndConcatLayer(64, 64, 32)  # input//2

        self.conv1 = torch.nn.Conv2d(32, MapOrdering.NUM_MAPS, 3, 1, padding=1)

    @staticmethod
    def scale_shape(s, f):
        assert s[0] % f == 0 and s[1] % f == 0
        return s[0] // f, s[1] // f

    def output_activation(self, x, apply_softmax):
        if apply_softmax:
            seg = torch.softmax(x[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND + 1], dim=1)
        else:
            seg = x[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND + 1]
        geo = torch.sigmoid(x[:, MapOrdering.GEO_TOP:]) * self.input_size[0]
        y = torch.cat([seg, geo], dim=1)
        return y

    def forward(self, x, apply_softmax=False):
        # x: BxCxHxW
        # eval backbone with 448px: bb1: 224px, bb2: 112px, bb3: 56px, bb4: 28px, bb5: 14px
        s = x.shape[2:]
        bb5, bb4, bb3, bb2, bb1 = self.backbone(x)

        x = self.up1(bb5, bb4, self.scale_shape(s, 16))
        x = self.up2(x, bb3, self.scale_shape(s, 8))
        x = self.up3(x, bb2, self.scale_shape(s, 4))
        x = self.up4(x, bb1, self.scale_shape(s, 2))
        x = self.conv1(x)

        return self.output_activation(x, apply_softmax)


def ceil32(val):
    if val % 32 == 0:
        return val
    val = (val // 32 + 1) * 32
    return val

def word_segment(path, output_folder, model_path):
        
    os.makedirs(output_folder, exist_ok = True)

    max_side_len = 5000
    thres = 0.5
    max_aabbs = 1000

    orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    net = WordDetectorNet()
    net.load_state_dict(torch.load(model_path, map_location='cuda'))
    net.eval()
    net.cuda()

    f = min(max_side_len / orig.shape[0], max_side_len / orig.shape[1])
    if f < 1:
        orig = cv2.resize(orig, dsize=None, fx=f, fy=f)
    img = np.ones((ceil32(orig.shape[0]), ceil32(orig.shape[1])), np.uint8) * 255
    img[:orig.shape[0], :orig.shape[1]] = orig

    img = (img / 255 - 0.5).astype(np.float32)
    imgs = img[None, None, ...]
    imgs = torch.from_numpy(imgs).cuda()
    with torch.no_grad():
        y = net(imgs, apply_softmax=True)
        y_np = y.to('cpu').numpy()
    scale_up = 1 / compute_scale_down(WordDetectorNet.input_size, WordDetectorNet.output_size)

    img_np = imgs[0, 0].to('cpu').numpy()
    pred_map = y_np[0]

    aabbs = decode(pred_map, comp_fg=fg_by_cc(thres, max_aabbs), f=scale_up)
    h, w = img_np.shape
    aabbs = [aabb.clip(AABB(0, w - 1, 0, h - 1)) for aabb in aabbs]  # bounding box must be inside img
    clustered_aabbs = cluster_aabbs(aabbs)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    for idx,bb in enumerate(clustered_aabbs):
        bb1 = bb
        im_i = (img_np[int(bb1.ymin):int(bb1.ymax),int(bb1.xmin):int(bb1.xmax)]+0.5)*255
        cv2.imwrite(f'{output_folder}/im_{idx}.png',im_i)
