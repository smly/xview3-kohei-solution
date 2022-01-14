import math
import tarfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import tqdm
from rasterio.enums import Resampling


@dataclass
class XView3DataSource:
    train_csv: Path = Path("data/input/xview3/train.csv")
    validation_csv: Path = Path("data/input/xview3/validation.csv")


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


def _pad(vh, rows, cols):
    r, c = vh.shape
    to_rows = math.ceil(r / rows) * rows
    to_cols = math.ceil(c / cols) * cols
    pad_rows = to_rows - r
    pad_cols = to_cols - c
    vh_pad = np.pad(
        vh, pad_width=(
            (0, pad_rows),
            (0, pad_cols)
        ), mode="constant", constant_values=0
    )
    return vh_pad, pad_rows, pad_cols


def load_image_v2(scene_id, base_dir="data/input/xview3/downloaded"):
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
            with rasterio.open(
                f.extractfile(f"{scene_id}/{fl}.tif"), "r"
            ) as dataset:
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
    min_val, max_val = -36, -9

    im_tmp = np.where(im_mask > 0, im_tmp, min_val)
    im_tmp = np.clip(im_tmp, min_val, max_val)
    im_tmp = (im_tmp - min_val) / (max_val - min_val)
    im_tmp = (im_tmp * 255).astype(np.uint8)
    proc_imgs["VH_dB"] = im_tmp

    # VV
    im_tmp = imgs["VV_dB"]
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


def _internal_preprocess_v2(orig_csv, out_csv, out_image_dir, crop_size=800):
    df = pd.read_csv(orig_csv)
    df_chip_list = []
    for scene_id in df["scene_id"].unique():
        df_chip = df[df["scene_id"] == scene_id].copy()
        df_chip["chip_yidx"] = df_chip["detect_scene_row"].apply(
            lambda x: x // crop_size
        )
        df_chip["chip_xidx"] = df_chip["detect_scene_column"].apply(
            lambda x: x // crop_size
        )
        df_chip["chip_ship_y"] = df_chip["detect_scene_row"].apply(
            lambda x: x % crop_size
        )
        df_chip["chip_ship_x"] = df_chip["detect_scene_column"].apply(
            lambda x: x % crop_size
        )
        df_chip_list.append(df_chip)

    print(df)
    assert False

    df = pd.concat(df_chip_list, sort=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    scene_ids = df["scene_id"].unique()
    for scene_id in tqdm.tqdm(scene_ids):
        proc_imgs = load_image_v2(scene_id)
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
                _pad(im[..., 0], crop_size, crop_size)[0],
                _pad(im[..., 1], crop_size, crop_size)[0],
                _pad(im[..., 2], crop_size, crop_size)[0],
            ],
            axis=2,
        )

        df_chip = df[df["scene_id"] == scene_id]

        detected_chip_locations = {}
        for _, r in df_chip.iterrows():
            detected_chip_locations[(r["chip_yidx"], r["chip_xidx"])] = 1

        for yidx in range(int(im_pad.shape[0] / crop_size)):
            for xidx in range(int(im_pad.shape[1] / crop_size)):
                water_pixel_count = (
                    proc_imgs["owiMask"][
                        yidx * crop_size: (yidx + 1) * crop_size,
                        xidx * crop_size: (xidx + 1) * crop_size,
                    ].ravel()
                    == 255
                ).sum()

                fn = out_image_dir / f"{scene_id}_{yidx}_{xidx}.png"
                Path(fn).parent.mkdir(parents=True, exist_ok=True)
                if (yidx, xidx) in detected_chip_locations:
                    im_crop = im_pad[
                        yidx * crop_size: (yidx + 1) * crop_size,
                        xidx * crop_size: (xidx + 1) * crop_size,
                    ]
                    cv2.imwrite(str(fn), im_crop,
                                [cv2.IMWRITE_PNG_COMPRESSION, 1])
                elif water_pixel_count > crop_size * crop_size * 0.5:
                    im_crop = im_pad[
                        yidx * crop_size: (yidx + 1) * crop_size,
                        xidx * crop_size: (xidx + 1) * crop_size,
                    ]
                    cv2.imwrite(str(fn), im_crop,
                                [cv2.IMWRITE_PNG_COMPRESSION, 1])


def preprocess_vh_vv_bathymetry_v2():
    data_source = XView3DataSource()
    preprocessed_info = XView3PreprocessedV2()

    _internal_preprocess_v2(
        data_source.validation_csv,
        preprocessed_info.validation_csv,
        preprocessed_info.validation_image_dir,
        crop_size=800,
    )

    _internal_preprocess_v2(
        data_source.train_csv,
        preprocessed_info.train_csv,
        preprocessed_info.train_image_dir,
        crop_size=800,
    )


def preproc_v2():
    preprocess_vh_vv_bathymetry_v2()


def preproc_v6():
    pass


def main():
    # Generate data/working/xview3/preprocess_vh_vv_bathymetry_v2/*
    preproc_v2()

    # Generate data/working/xview3/images/ppv6/*
    preproc_v6()


if __name__ == "__main__":
    main()
