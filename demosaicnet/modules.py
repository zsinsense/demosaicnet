"""Models for [Gharbi2016] Deep Joint demosaicking and denoising."""
import os
from collections import OrderedDict
from pkg_resources import resource_filename

import numpy as np
import torch as th
import torch.nn as nn


__all__ = ["BayerDemosaick", "XTransDemosaick"]


_BAYER_WEIGHTS = resource_filename(__name__, 'data/bayer.pth')
_XTRANS_WEIGHTS = resource_filename(__name__, 'data/xtrans.pth')


class BayerDemosaick(nn.Module):
  """Released version of the network, best quality.

  This model differs from the published description. It has a mask/filter split
  towards the end of the processing. Masks and filters are multiplied with each
  other. This is not key to performance and can be ignored when training new
  models from scratch.
  """
  def __init__(self, depth=15, width=64, pretrained=True, pad=False, min_noise=0, max_noise=0, pattern="GRBG"):
    super(BayerDemosaick, self).__init__()

    self.depth = depth
    self.width = width

    self.add_noise = (min_noise > 0 or max_noise > 0) and (min_noise < max_noise)
    self.min_noise = min_noise
    self.max_noise = max_noise
    self.pattern = pattern

    if pad:
      pad = 1
    else:
      pad = 0

    self.downsample = nn.Conv2d(3, 4, 2, stride=2)
    layers = OrderedDict([
        # ("pack_mosaic", nn.Conv2d(3, 4, 2, stride=2)),  # Downsample 2x2 to re-establish translation invariance
      ])
    for i in range(depth):
      n_out = width
      n_in = width
      if i == 0:
        n_in = 5 if self.add_noise else 4
      if i == depth-1:
        n_out = 2*width
      layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3, padding=pad)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)
    self.residual_predictor = nn.Conv2d(width, 12, 1)
    self.upsampler = nn.ConvTranspose2d(12, 3, 2, stride=2, groups=3)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(6, width, 3, padding=pad)),
      ("post_relu", nn.ReLU(inplace=True)),
      ("output", nn.Conv2d(width, 3, 1)),
      ]))

    # Load weights
    if pretrained:
      assert depth == 15, "pretrained bayer model has depth=15."
      assert width == 64, "pretrained bayer model has width=64."
      state_dict = th.load(_BAYER_WEIGHTS)
      self.load_state_dict(state_dict)

  def forward(self, mosaic):
    """Demosaicks a Bayer image.

    Args:
      mosaic (th.Tensor):  input Bayer mosaic

    Returns:
      th.Tensor: the demosaicked image
    """
    if self.add_noise:
      mosaic_ = self._addnoise(mosaic)
    else:
      mosaic_ = self.downsample(mosaic)

    # 1/4 resolution features
    features = self.main_processor(mosaic_)
    filters, masks = features[:, 0:self.width], features[:, self.width:2*self.width]
    filtered = filters * masks
    residual = self.residual_predictor(filtered)

    # Match mosaic and residual
    upsampled = self.upsampler(residual)
    cropped = _crop_like(mosaic, upsampled)

    packed = th.cat([cropped, upsampled], 1)  # skip connection
    output = self.fullres_processor(packed)
    return output

  def _upsample(self, x):
    sz = x.shape
    output = th.zeros((sz[0], 3, sz[2]*2, sz[3]*2), dtype= x.dtype)
    for c in range(3):
      output[:, c, ::2, ::2] = x[:, 4*c, :, :]
      output[:, c, ::2, 1::2] = x[:, 4*c+1, :, :]
      output[:, c, 1::2, ::2] = x[:, 4*c+2, :, :]
      output[:, c, 1::2, 1::2] = x[:, 4*c+3, :, :]

    return output

  # def _unpack(self, mosaic):
  #   sz = mosaic.shape
  #   output = th.zeros((sz[0], 3, sz[2]*2, sz[3]*2))
  #   for c in range(4):
  #     output[:, "RGB".index(self.pattern[c]), c//2::2, c%2::2] = mosaic[:, c, :, :]
  #   return output

  def _addnoise(self, bayer3):
      sz = bayer3.shape
      noise_levels = np.random.rand(sz[0])
      noise_levels *= self.max_noise - self.min_noise
      noise_levels += self.min_noise
      noise = th.randn(sz)
      for i in range(sz[0]):
          # noise[i] = th.randn(sz[1], sz[2], sz[3]) * noise_levels[i]
          noise[i] *= noise_levels[i]
      mask = th.zeros_like(bayer3, dtype=bool)
      for c in range(4):
        mask[..., "RGB".index(self.pattern[c]), c//2::2, c%2::2] = True
      bayer3[mask] += noise[mask]
      mosaic_downsample = self.downsample(bayer3)
      sz = mosaic_downsample.shape
      sigma_layer = th.empty(sz[0], 1, sz[2], sz[3])
      for n in range(sz[0]):
        sigma_layer[n] = noise_levels[n]
      mosaic_pack = th.cat((mosaic_downsample, sigma_layer), 1)
      return mosaic_pack


class XTransDemosaick(nn.Module):
  """Released version of the network.

  There is no downsampling here.

  """
  def __init__(self, depth=11, width=64, pretrained=True, pad=False):
    super(XTransDemosaick, self).__init__()

    self.depth = depth
    self.width = width

    if pad:
      pad = 1
    else:
      pad = 0

    layers = OrderedDict([])
    for i in range(depth):
      n_in = width
      n_out = width
      if i == 0:
        n_in = 3
      layers["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3, padding=pad)
      layers["relu{}".format(i+1)] = nn.ReLU(inplace=True)

    self.main_processor = nn.Sequential(layers)

    self.fullres_processor = nn.Sequential(OrderedDict([
      ("post_conv", nn.Conv2d(3+width, width, 3, padding=pad)),
      ("post_relu", nn.ReLU(inplace=True)),
      ("output", nn.Conv2d(width, 3, 1)),
      ]))

    # Load weights
    if pretrained:
      assert depth == 11, "pretrained xtrans model has depth=11."
      assert width == 64, "pretrained xtrans model has width=64."
      state_dict = th.load(_XTRANS_WEIGHTS)
      self.load_state_dict(state_dict)


  def forward(self, mosaic):
    """Demosaicks an XTrans image.

    Args:
      mosaic (th.Tensor):  input XTrans mosaic

    Returns:
      th.Tensor: the demosaicked image
    """

    features = self.main_processor(mosaic)
    cropped = _crop_like(mosaic, features)  # Match mosaic and residual
    packed = th.cat([cropped, features], 1)  # skip connection
    output = self.fullres_processor(packed)
    return output


def _crop_like(src, tgt):
    """Crop a source image to match the spatial dimensions of a target.

    Args:
        src (th.Tensor or np.ndarray): image to be cropped
        tgt (th.Tensor or np.ndarray): reference image
    """
    src_sz = np.array(src.shape)
    tgt_sz = np.array(tgt.shape)

    # Assumes the spatial dimensions are the last two
    crop = (src_sz[-2:]-tgt_sz[-2:])
    crop_t = crop[0] // 2
    crop_b = crop[0] - crop_t
    crop_l = crop[1] // 2
    crop_r = crop[1] - crop_l
    crop //= 2
    if (np.array([crop_t, crop_b, crop_r, crop_l])> 0).any():
        return src[..., crop_t:src_sz[-2]-crop_b, crop_l:src_sz[-1]-crop_r]
    else:
        return src

