"""Dataset loader for demosaicnet."""
import json
import os
import platform
import subprocess
import shutil
import hashlib


import numpy as np
from imageio import imread
from torch.utils.data import Dataset as TorchDataset
import wget

import ttools

from .mosaic import bayer, xtrans

__all__ = ["BAYER_MODE", "XTRANS_MODE", "Dataset",
           "TRAIN_SUBSET", "VAL_SUBSET", "TEST_SUBSET"]


LOG = ttools.get_logger(__name__)
ttools.set_logger(True)

BAYER_MODE = "bayer"
"""Applies a Bayer mosaic pattern."""

XTRANS_MODE = "xtrans"
"""Applies an X-Trans mosaic pattern."""

TRAIN_SUBSET = "train"
"""Loads the 'train' subset of the data."""

VAL_SUBSET = "val"
"""Loads the 'val' subset of the data."""

TEST_SUBSET = "test"
"""Loads the 'test' subset of the data."""


class Dataset(TorchDataset):
    """Dataset of challenging image patches for demosaicking.

    Args:
        download(bool): if True, automatically download the dataset.
        mode(:class:`BAYER_MODE` or :class:`XTRANS_MODE`): mosaic pattern to apply to the data.
        subset(:class:`TRAIN_SUBET`, :class:`VAL_SUBSET` or :class:`TEST_SUBSET`): subset of the data to load.
    """

    def __init__(self, root, download=False,
                 mode=BAYER_MODE, subset="train", pattern="GRBG"):

        super(Dataset, self).__init__()

        # self.add_noise = (min_noise > 0 or max_noise > 0) and min_noise < max_noise
        # self.min_noise = min_noise
        # self.max_noise = max_noise

        self.pattern = pattern
        # self.cfa_array = []
        # for i in range(4):
        #     self.cfa_array.append("RGB".index(pattern[i]))

        self.root = os.path.abspath(root)

        if subset not in [TRAIN_SUBSET, VAL_SUBSET, TEST_SUBSET]:
            raise ValueError("Dataset subet should be '%s', '%s' or '%s', got"
                             " %s" % (TRAIN_SUBSET, TEST_SUBSET, VAL_SUBSET,
                                      subset))

        if mode not in [BAYER_MODE, XTRANS_MODE]:
            raise ValueError("Dataset mode should be '%s' or '%s', got"
                             " %s" % (BAYER_MODE, XTRANS_MODE, mode))
        self.mode = mode

        listfile = os.path.join(self.root, subset, "filelist.txt")
        LOG.debug("Reading image list from %s", listfile)

        if not os.path.exists(listfile):
            if download:
                _download(self.root)
            else:
                LOG.error("Filelist %s not found", listfile)
                raise ValueError("Filelist %s not found" % listfile)
        else:
            LOG.debug("No need no download the data, filelist exists.")

        self.files = []
        with open(listfile, "r") as fid:
            for fname in fid.readlines():
                self.files.append(os.path.join(self.root, subset, fname.strip()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Fetches a mosaic / demosaicked pair of images.

        Returns
            mosaic(np.array): with size [3, h, w] the mosaic data with separated color channels.
            img(np.array): with size [3, h, w] the groundtruth image.
        """
        fname = self.files[idx]
        img = np.array(imread(fname)).astype(np.float32) / (2**8-1)
        img = np.transpose(img, [2, 0, 1])

        img_ = img

        if self.mode == BAYER_MODE:
            mosaic = bayer(img_, pattern=self.pattern)
        else:
            mosaic = xtrans(img_)

        return mosaic, img

    # def _packBayerMosaic(self, bayer3):
    #     sz = bayer3.shape
    #     output = np.zeros((4, sz[1]//2, sz[2]//2), dtype=bayer3.dtype)
    #     output[0, :, :] = bayer3[self.cfa_array[0], ::2, ::2]
    #     output[1, :, :] = bayer3[self.cfa_array[1], ::2, 1::2]
    #     output[2, :, :] = bayer3[self.cfa_array[2], 1::2, ::2]
    #     output[3, :, :] = bayer3[self.cfa_array[3], 1::2, 1::2]
    #     return output


CHECKSUMS = json.load(open('demosaicnet/data/checksums'))


def _download(dst):
    dst = os.path.abspath(dst)
    files = CHECKSUMS.keys()
    fullzip = os.path.join(dst, "datasets.zip")
    joinedzip = os.path.join(dst, "joined.zip")

    URL_ROOT = "https://data.csail.mit.edu/graphics/demosaicnet"

    if not os.path.exists(joinedzip):
        LOG.info("Dowloading %d files to %s (This will take a while, and ~80GB)", len(
            files), dst)

        os.makedirs(dst, exist_ok=True)
        for f in files:
            fname = os.path.join(dst, f)
            url = os.path.join(URL_ROOT, f)

            do_download = True
            if os.path.exists(fname):
                checksum = md5sum(fname)
                if checksum == CHECKSUMS[f]:  # File is is and correct
                    LOG.info('%s already downloaded, with correct checksum', f)
                    do_download = False
                else:
                    LOG.warning('%s checksums do not match, got %s, should be %s',
                                f, checksum, CHECKSUMS[f])
                    try:
                        os.remove(fname)
                    except OSError as e:
                        LOG.error("Could not delete broken part %s: %s", f, e)
                        raise ValueError

            if do_download:
                LOG.info('Downloading %s', f)
                wget.download(url, fname)

            checksum = md5sum(fname)

            if checksum == CHECKSUMS[f]:
                LOG.info("%s MD5 correct", f)
            else:
                LOG.error('%s checksums do not match, got %s, should be %s. Downloading failed',
                          f, checksum, CHECKSUMS[f])

        LOG.info("Joining zip files")
        cmd = " ".join(["zip", "-FF", fullzip, "--out", joinedzip])
        subprocess.check_call(cmd, shell=True)

        # Cleanup the parts
        for f in files:
            fname = os.path.join(dst, f)
            try:
                os.remove(fname)
            except OSError as e:
                LOG.warning("Could not delete file %s", f)

    # Extract
    wd = os.path.abspath(os.curdir)
    os.chdir(dst)
    LOG.info("Extracting files from %s", joinedzip)
    cmd = " ".join(["unzip", joinedzip])
    subprocess.check_call(cmd, shell=True)

    try:
        os.remove(joinedzip)
    except OSError as e:
        LOG.warning("Could not delete file %s", f)

    LOG.info("Moving subfolders")
    for k in ["train", "test", "val"]:
        shutil.move(os.path.join(dst, "images", k), os.path.join(dst, k))
    images = os.path.join(dst, "images")
    LOG.info("removing '%s' folder", images)
    shutil.rmtree(images)


def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()
