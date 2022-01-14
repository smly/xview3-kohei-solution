from argparse import Namespace
from pathlib import Path
from dataclasses import dataclass
import tarfile
import math

import cv2
import numpy as np
import pandas as pd
import torch
import rasterio
from omegaconf import DictConfig
from rasterio.enums import Resampling
from torch.utils.data import Dataset

from xd.utils.configs import dynamic_load
from xd.utils.timer import timer


@dataclass
class XView3DataSource:
    # Given files by the competition host.
    trainval_input_dir: Path = Path("data/input/xview3/downloaded")
    train_csv: Path = Path("data/input/xview3/train.csv")
    validation_csv: Path = Path("data/input/xview3/validation.csv")


@dataclass
class XView3PreprocessedDatasetV2:
    prefix: str = "data/working/xview3/preprocess_vh_vv_bathymetry_v2"

    train_image_dir: Path = Path(
        f"{prefix}/train"
    )
    # no pseudo-labeling
    train_mask_dir: Path = Path(
        f"{prefix}/train_masks"
    )
    train_csv: Path = Path(
        f"{prefix}/train.csv"
    )
    validation_image_dir: Path = Path(
        f"{prefix}/validation"
    )
    validation_mask_dir: Path = Path(
        f"{prefix}/validation_masks"
    )
    validation_csv: Path = Path(
        f"{prefix}/validation.csv"
    )
    validation_kfold_csv: Path = Path(
        f"{prefix}/validation_kfold.csv"  # noqa
    )

    @classmethod
    def get_input_dir(cls, mode: str = "train") -> Path:
        # assert mode in ["train", "val", "public"]
        assert mode in ["train", "val"]
        input_dir = XView3DataSource().trainval_input_dir
        return input_dir

    @classmethod
    def get_image(cls, scene_id: str, mode: str = "train") -> np.ndarray:
        input_dir = cls.get_input_dir(mode=mode)
        return load_image_ppv2(input_dir, scene_id)


def _load_image_layers_v2(scene_id, base_dir="data/input/xview3/downloaded"):
    imgs = {}
    proc_imgs = {}

    channels = [
        "VH_dB",
        "VV_dB",
        "bathymetry",
        "owiMask",
    ]

    with tarfile.open(f"{base_dir}/{scene_id}.tar.gz", "r") as f:
        for fl in channels:
            print(f"Loading {scene_id}/{fl}...")
            with rasterio.open(f.extractfile(f"{scene_id}/{fl}.tif"), "r") as dataset:
                imgs[fl] = dataset.read(1)
                if imgs[fl].shape != imgs[channels[0]].shape:
                    imgs[fl] = dataset.read(
                        out_shape=imgs[channels[0]].shape,
                        resampling=Resampling.bilinear,
                    ).squeeze()
                assert imgs[fl].shape == imgs[channels[0]].shape

    # mask
    im_tmp = imgs["VH_dB"]
    im_mask = np.where(im_tmp == -(2 ** 15), 0, 1)
    proc_imgs["mask"] = im_mask

    # see mask
    im_tmp = imgs["owiMask"]
    im_tmp = np.where(im_mask > 0, im_tmp == 0, 0)
    proc_imgs["owiMask"] = (im_tmp * 255).astype(np.uint8)

    # VH
    im_tmp = imgs["VH_dB"]
    # min_val, max_val = np.percentile(im_tmp[im_mask > 0].ravel(), 0.5), np.percentile(
    #     im_tmp[im_mask > 0].ravel(), 99.5
    # )
    min_val, max_val = -36, -9

    im_tmp = np.where(im_mask > 0, im_tmp, min_val)
    im_tmp = np.clip(im_tmp, min_val, max_val)
    im_tmp = (im_tmp - min_val) / (max_val - min_val)
    im_tmp = (im_tmp * 255).astype(np.uint8)
    proc_imgs["VH_dB"] = im_tmp

    # VV
    im_tmp = imgs["VV_dB"]
    # min_val, max_val = np.percentile(im_tmp[im_mask > 0].ravel(), 0.5), np.percentile(
    #     im_tmp[im_mask > 0].ravel(), 99.5
    # )
    min_val, max_val = -34, 1.3

    im_tmp = np.where(im_mask > 0, im_tmp, min_val)
    im_tmp = np.clip(im_tmp, min_val, max_val)
    im_tmp = (im_tmp - min_val) / (max_val - min_val)
    im_tmp = (im_tmp * 255).astype(np.uint8)
    proc_imgs["VV_dB"] = im_tmp

    # bathymetry
    im_tmp = imgs["bathymetry"]
    min_bathymetry, max_bathymetry = -255, 255
    im_tmp = np.where(im_tmp < min_bathymetry, min_bathymetry, im_tmp)
    im_tmp = np.where(im_tmp > max_bathymetry, max_bathymetry, im_tmp)
    im_tmp = (im_tmp - min_bathymetry) / (max_bathymetry - min_bathymetry)
    im_tmp = (im_tmp * 255).astype(np.uint8)
    proc_imgs["bathymetry"] = im_tmp
    return proc_imgs


def pad(vh, rows, cols):
    r, c = vh.shape
    to_rows = math.ceil(r / rows) * rows
    to_cols = math.ceil(c / cols) * cols
    pad_rows = to_rows - r
    pad_cols = to_cols - c
    vh_pad = np.pad(
        vh, pad_width=((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0
    )
    return vh_pad, pad_rows, pad_cols


def load_image_ppv2(input_dir: Path, scene_id: str, crop_size: int = 800) -> np.ndarray:
    with timer(f"Loading scene image ({scene_id}) from archive file."):
        proc_imgs = _load_image_layers_v2(scene_id, base_dir=input_dir)

    with timer("Stack layers."):
        im = np.stack(
            [
                np.where(proc_imgs["mask"] > 0, proc_imgs["VV_dB"], 0),
                np.where(proc_imgs["mask"] > 0, proc_imgs["VH_dB"], 0),
                np.where(proc_imgs["mask"] > 0, proc_imgs["bathymetry"], 0),
            ],
            axis=2,
        )
        im_pad = np.stack(
            [
                pad(im[..., 0], crop_size, crop_size)[0],
                pad(im[..., 1], crop_size, crop_size)[0],
                pad(im[..., 2], crop_size, crop_size)[0],
            ],
            axis=2,
        )

    return im_pad


class PPV2VO(Dataset):
    def __init__(
        self, args: Namespace, conf: DictConfig, is_test: bool = False
    ):  # noqa
        self.args: Namespace = args
        self.fold: int = args.fold
        self.conf: DictConfig = conf
        self.is_test: bool = is_test

        self.datainfo: XView3PreprocessedDatasetV2 = XView3PreprocessedDatasetV2()  # noqa
        self.img_path, self.mask_path, self.gt_points = self.prepare_filelist(
            args.fold
        )  # noqa

        self.num_samples = len(self.img_path)

        self.train_transform, self.test_transform = dynamic_load(
            conf.train.augmentation.func
        )(**conf.train.augmentation.kwargs)

    def load_base_dataframe(self, fold_idx: int):
        df = pd.read_csv(self.datainfo.validation_kfold_csv)
        df = (
            df[df["fold_idx"] == fold_idx]
            if self.is_test
            else df[df["fold_idx"] != fold_idx]
        )
        return df

    def prepare_filelist(self, fold_idx: int):
        df = self.load_base_dataframe(fold_idx)

        # TODO: object が存在する画像のみ対象としている。
        # scene_id の全景（陸上）も対象として検証したい
        df_unique_chips = df.drop_duplicates(
            subset=["scene_id", "chip_yidx", "chip_xidx"]
        )

        img_path_list = [
            (
                self.datainfo.validation_image_dir
                / "{}_{}_{}.png".format(
                    r["scene_id"],
                    r["chip_yidx"],
                    r["chip_xidx"],
                )
            )
            for _, r in df_unique_chips.iterrows()
        ]
        mask_path_list = [
            (
                self.datainfo.validation_mask_dir
                / "{}_{}_{}.png".format(
                    r["scene_id"],
                    r["chip_yidx"],
                    r["chip_xidx"],
                )
            )
            for _, r in df_unique_chips.iterrows()
        ]
        point_gt_list = [
            list(
                zip(
                    df[
                        (df["scene_id"] == r["scene_id"])
                        & (df["chip_yidx"] == r["chip_yidx"])
                        & (df["chip_xidx"] == r["chip_xidx"])
                    ]["chip_ship_x"],
                    df[
                        (df["scene_id"] == r["scene_id"])
                        & (df["chip_yidx"] == r["chip_yidx"])
                        & (df["chip_xidx"] == r["chip_xidx"])
                    ]["chip_ship_y"],
                )
            )
            for _, r in df_unique_chips.iterrows()
        ]
        assert len(img_path_list) == len(mask_path_list)
        assert len(img_path_list) == len(point_gt_list)
        return img_path_list, mask_path_list, point_gt_list

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # read_image_and_gt
        img_path, mask_path = self.img_path[index], self.mask_path[index]
        assert img_path.exists()
        assert mask_path.exists()

        im = cv2.imread(str(img_path))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.is_test:
            aug = self.test_transform(image=im, mask=mask)
            im = aug["image"]
            mask = aug["mask"]
        else:
            aug = self.train_transform(image=im, mask=mask)
            im = aug["image"]
            mask = aug["mask"]

        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        target = torch.tensor(mask > 0).unsqueeze(0).float()

        return im, target, index
