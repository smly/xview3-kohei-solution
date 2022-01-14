from argparse import Namespace
from pathlib import Path

from omegaconf import DictConfig
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from xd.utils.configs import load_config, dynamic_load


class PPV6(Dataset):
    def __init__(self,
                 conf: DictConfig,
                 fold: int = 0,
                 is_test: bool = False):
        self.is_test = is_test

        self.crop_size = conf.dataset.crop_size
        assert self.crop_size == 256

        self.img_dir = "data/working/xview3/images"
        self.src_fmt = "{img_dir}/ppv6/{mode:s}/{scene_id:s}.png"
        self.out_fmt = "{img_dir}/ppv6/crop_{mode:s}/{scene_id:s}/{i:06d}.png"

        self.df = self.preprocess_crop_images()

        # Augmentation
        self.train_transform, self.test_transform = dynamic_load(
            conf.train.augmentation.func
        )(**conf.train.augmentation.kwargs)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        r = self.df.iloc[index]
        fp = self.out_fmt.format(
            img_dir=self.img_dir,
            mode=r["dirname"],
            scene_id=r["scene_id"],
            i=index,
        )
        assert Path(fp).exists()
        im = cv2.imread(fp)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.is_test:
            im = self.test_transform(image=im)["image"]
        else:
            im = self.train_transform(image=im)["image"]

        im = torch.from_numpy(im.transpose((2, 0, 1))).float()

        return im, r["ship_class"]  # noqa

    def preprocess_crop_images(self, force_preproc=False):
        df = self.load_val() if self.is_test else self.load_trn()

        # Check first image
        check_fp = self.out_fmt.format(
            img_dir=self.img_dir,
            mode=df.iloc[0]["dirname"],
            scene_id=df.iloc[0]["scene_id"],
            i=0,
        )
        if not Path(check_fp).exists() or force_preproc:
            # Crop and save images
            print("Crop and save images...")
            scene_ids = list(sorted(df["scene_id"].unique()))
            for scene_id in scene_ids:
                df_scene = df[df["scene_id"] == scene_id]
                scene_dirname = df_scene.iloc[0]["dirname"]
                src_fp = self.src_fmt.format(
                    img_dir=self.img_dir,
                    mode=scene_dirname,
                    scene_id=scene_id,
                )
                assert Path(src_fp).exists()

                check_output_dir = self.out_fmt.format(
                    img_dir=self.img_dir,
                    mode=df_scene.iloc[0]["dirname"],
                    scene_id=df_scene.iloc[0]["scene_id"],
                    i=df_scene.index[0],
                )
                if Path(check_output_dir).parent.exists():
                    print("skip")
                    continue
                else:
                    print("run", check_output_dir)
                im_scene = cv2.imread(src_fp)
                for idx, r in df_scene.iterrows():
                    self.preproc_per_image(idx, r, im_scene)

        return df

    def preproc_per_image(self, idx, r, im_scene):
        yc = int(r["detect_scene_row"])
        xc = int(r["detect_scene_column"])

        im = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        # distance from center point
        d = int(self.crop_size / 2)

        # 全体マップからの位置
        y0, y1, x0, x1 = yc - d, yc + d, xc - d, xc + d

        # 切り抜き後の位置
        top, left, bottom, right = 0, 0, self.crop_size, self.crop_size

        # 全体マップからはみ出る場合
        if yc - d < 0:
            # 上にはみ出る場合
            top = d - yc
            y0 = 0
        if xc - d < 0:
            # 左にはみ出る場合
            left = d - xc
            x0 = 0
        if yc + d > im_scene.shape[0]:
            # 下にはみ出る場合
            bottom = im_scene.shape[0] - d - yc
            y1 = im_scene.shape[0]
        if xc + d > im_scene.shape[1]:
            right = im_scene.shape[1] - d - xc
            x1 = im_scene.shape[1]

        check_fp = self.out_fmt.format(
            img_dir=self.img_dir,
            mode=r["dirname"],
            scene_id=r["scene_id"],
            i=idx,
        )
        Path(check_fp).parent.mkdir(parents=True, exist_ok=True)
        im[top:bottom, left:right] = im_scene[y0:y1, x0:x1]
        cv2.imwrite(check_fp, im, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def load_val(self, fold: int = 0):
        kfold_csv = "data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation_kfold.csv"
        dfv = pd.read_csv(kfold_csv)
        dfv = dfv[~dfv["is_vessel"].isna()][
            [
                "is_vessel",
                "is_fishing",
                "scene_id",
                "detect_scene_row",
                "detect_scene_column",
                "fold_idx",
            ]
        ]
        dfv["dirname"] = "validation"
        dfv["is_fishing"].fillna(False, inplace=True)
        dfv = dfv[dfv["fold_idx"] == fold]
        dfv["ship_class"] = dfv["is_vessel"].astype(int) + dfv[
            "is_fishing"
        ].astype(int)
        return dfv.reset_index(drop=True)

    def load_trn(self, fold: int = 0):
        train_csv = Path("data/input/xview3/train.csv")
        kfold_csv = Path("data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation_kfold.csv")
        df = pd.read_csv(train_csv)
        df = df[~df["is_vessel"].isna()][
            [
                "is_vessel",
                "is_fishing",
                "scene_id",
                "detect_scene_row",
                "detect_scene_column",
            ]
        ]
        df["dirname"] = "train"
        df["is_fishing"].fillna(False, inplace=True)

        dfv = pd.read_csv(kfold_csv)
        dfv = dfv[~dfv["is_vessel"].isna()][
            [
                "is_vessel",
                "is_fishing",
                "scene_id",
                "detect_scene_row",
                "detect_scene_column",
                "fold_idx",
            ]
        ]
        dfv["dirname"] = "validation"
        dfv["is_fishing"].fillna(False, inplace=True)
        dfv = dfv[dfv["fold_idx"] != fold]
        df = pd.concat([df, dfv], sort=False)
        df["ship_class"] = df["is_vessel"].astype(int) + df["is_fishing"].astype(int)
        return df.reset_index(drop=True)
