from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from xd.utils.configs import dynamic_load


@dataclass
class XView3DataSource:
    train_csv: Path = Path("data/input/xview3/train.csv")
    train_shore_dir: Path = Path("data/input/xview3/train")
    validation_csv: Path = Path("data/input/xview3/validation.csv")
    validation_shore_dir: Path = Path("data/input/xview3/validation")


@dataclass
class XView3PreprocessedV2:
    train_image_dir: Path = Path(
        "data/working/xview3/preprocess_vh_vv_bathymetry_v2/train"
    )
    train_csv: Path = Path(
        "data/working/xview3/preprocess_vh_vv_bathymetry_v2/train.csv"
    )
    validation_image_dir: Path = Path(
        "data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation"
    )
    validation_mask_dir: Path = Path(
        "data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation_masks"
    )
    validation_csv: Path = Path(
        "data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation.csv"
    )
    validation_kfold_csv: Path = Path(
        "data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation_kfold.csv"  # noqa
    )


def get_dataloaders(args, conf):
    ds = XView3VesselLengthDataset(conf, fold=args.fold, is_test=False)
    dl = DataLoader(
        ds, num_workers=32, batch_size=64, drop_last=True, shuffle=True
    )  # noqa

    val_ds = XView3VesselLengthDataset(conf, fold=args.fold, is_test=True)
    val_dl = DataLoader(
        val_ds, num_workers=32, batch_size=64, drop_last=False, shuffle=False
    )

    dataloaders = {}
    dataloaders["train"] = dl
    dataloaders["val"] = val_dl

    return dataloaders


class XView3VesselLengthDataset(Dataset):
    def __init__(self, conf: DictConfig, fold: int = 0, is_test: bool = False):
        self.crop_size = conf.dataset.crop_size
        self.chip_image_size = conf.dataset.chip_image_size
        self.length_upper = conf.dataset.length_upper  # 95 percentile point
        self.length_lower = conf.dataset.length_lower  # 5 percentile point

        assert self.crop_size % 2 == 0
        self.kfold_csv = XView3PreprocessedV2().validation_kfold_csv
        self.train_csv = XView3DataSource().train_csv
        self.train_imdir = XView3PreprocessedV2().train_image_dir
        self.val_imdir = XView3PreprocessedV2().validation_image_dir
        self.is_test = is_test

        if is_test:
            self.df = self.load_valtst_scaled_vessel_length(fold=fold)
        else:
            self.df = self.load_train_val_scaled_vessel_length(fold=fold)

        self.train_transform, self.test_transform = dynamic_load(
            conf.train.augmentation.func
        )(**conf.train.augmentation.kwargs)

    def __len__(self):
        return len(self.df)

    def decode_vessel_length(self, vals: np.ndarray):
        return np.expm1(vals) + self.length_lower

    def load_valtst_scaled_vessel_length(self, fold: int = 0):
        dfv = pd.read_csv(self.kfold_csv)
        dfv = dfv[~dfv["vessel_length_m"].isna()][
            [
                "vessel_length_m",
                # "is_vessel",
                # "is_fishing",
                "scene_id",
                "detect_scene_row",
                "detect_scene_column",
                "fold_idx",
            ]
        ]
        dfv["vessel_length_log1p_encoded"] = np.log1p(
            np.minimum(
                np.maximum(dfv["vessel_length_m"], self.length_lower),
                self.length_upper,
            )  # noqa
            - self.length_lower
        )
        dfv = dfv[dfv["fold_idx"] == fold]
        return dfv

    def load_train_val_scaled_vessel_length(self, fold: int = 0):
        df = pd.read_csv(self.train_csv)
        df = df[~df["vessel_length_m"].isna()][
            [
                "vessel_length_m",
                # "is_vessel",
                # "is_fishing",
                "scene_id",
                "detect_scene_row",
                "detect_scene_column",
            ]
        ]
        dfv = pd.read_csv(self.kfold_csv)
        dfv = dfv[~dfv["vessel_length_m"].isna()][
            [
                "vessel_length_m",
                # "is_vessel",
                # "is_fishing",
                "scene_id",
                "detect_scene_row",
                "detect_scene_column",
                "fold_idx",
            ]
        ]
        dfv = dfv[dfv["fold_idx"] != fold]
        df = pd.concat([df, dfv])

        # Rescale
        df["vessel_length_log1p_encoded"] = np.log1p(
            np.minimum(
                np.maximum(df["vessel_length_m"], self.length_lower),
                self.length_upper,
            )  # noqa
            - self.length_lower
        )
        return df

    def __getitem__(self, index):
        r = self.df.iloc[index]

        filename = "{}_{}_{}.png".format(
            r["scene_id"],
            r["detect_scene_row"] // self.chip_image_size,
            r["detect_scene_column"] // self.chip_image_size,
        )
        yc = r["detect_scene_row"] % self.chip_image_size
        xc = r["detect_scene_column"] % self.chip_image_size

        if np.isnan(r["fold_idx"]):
            im_orig = cv2.imread(str(self.train_imdir / filename))
        else:
            im_orig = cv2.imread(str(self.val_imdir / filename))

        im_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        d = int(self.crop_size / 2)

        y0, y1, x0, x1 = yc - d, yc + d, xc - d, xc + d
        top, left, bottom, right = 0, 0, self.crop_size, self.crop_size
        if yc - d < 0:
            top = d - yc
            y0 = 0
        if xc - d < 0:
            left = d - xc
            x0 = 0
        if yc + d > self.chip_image_size:
            bottom = self.chip_image_size - d - yc
            y1 = self.chip_image_size
        if xc + d > self.chip_image_size:
            right = self.chip_image_size - d - xc
            x1 = self.chip_image_size

        im_crop[top:bottom, left:right] = im_orig[y0:y1, x0:x1]
        if self.is_test:
            im_crop = self.test_transform(image=im_crop)["image"]
        else:
            im_crop = self.train_transform(image=im_crop)["image"]

        im_crop = torch.from_numpy(im_crop.transpose((2, 0, 1))).float()

        return im_crop, torch.from_numpy(
            np.array([r["vessel_length_log1p_encoded"]])
        )  # noqa
