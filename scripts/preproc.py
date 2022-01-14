import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from logging import getLogger

import cv2
import numpy as np
import pandas as pd
import rasterio
import tqdm
from rasterio.enums import Resampling

from xd.utils.timer import timer
from xd.utils.logger import set_logger


logger = getLogger("xd")


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
    if out_csv.exists():
        print(" => Skip: output csv is already exists.")
        return

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

    df = pd.concat(df_chip_list, sort=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    scene_ids = df["scene_id"].unique()
    for scene_id in tqdm.tqdm(scene_ids):
        if len(list(out_image_dir.glob(f"{scene_id}_*.png"))) > 0:
            # Skip
            continue

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


def euclidean_dist(test_matrix, train_matrix):
    num_test = test_matrix.shape[0]
    num_train = train_matrix.shape[0]
    dists = np.zeros((num_test, num_train))
    d1 = -2 * np.dot(test_matrix, train_matrix.T)
    d2 = np.sum(np.square(test_matrix), axis=1, keepdims=True)
    d3 = np.sum(np.square(train_matrix), axis=1)
    dists = np.sqrt(d1 + d2 + d3)
    return dists


def preprocess_validation_masks_v2():
    df_val = pd.read_csv(
        "data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation.csv"
    )
    image_dir = "data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation/"
    mask_dir = "data/working/xview3/preprocess_vh_vv_bathymetry_v2/validation_masks/"
    if Path(mask_dir).exists():
        print(" => Skip: output validation_masks/ is already exists.")
        return

    assert False
    Path(mask_dir).mkdir(parents=True, exist_ok=True)

    for image_path in list(sorted(Path(image_dir).glob("*.png"))):
        id_, yidx, xidx = image_path.stem.split("_")
        yidx, xidx = int(yidx), int(xidx)
        df_part = df_val[
            (df_val["scene_id"] == id_)
            & (df_val["chip_yidx"] == yidx)
            & (df_val["chip_xidx"] == xidx)
        ]

        if (Path(mask_dir) / Path(image_path).name).exists():
            continue

        im = cv2.imread(str(image_path))
        mask_map = np.zeros(im.shape, dtype=np.uint8)

        ImgInfo = {
            "img_id": id_,
            "human_num": len(df_part),
        }
        h, w = im.shape[:2]

        centroid_list = []
        wh_list = []
        for _, r in df_part.iterrows():
            yc, xc = r["chip_ship_y"], r["chip_ship_x"]
            y0, x0, y1, x1 = int(yc - 2), int(xc - 2), int(yc + 2), int(xc + 2)
            # 中心点
            centroid_list.append([xc, yc])
            # width and height (最低でも 3)
            wh_list.append([max((x1 - x0) / 2, 3), max((y1 - y0) / 2, 3)])

        centroids = np.array(centroid_list.copy(), dtype="int")
        wh = np.array(wh_list.copy(), dtype="int")
        # print(centroids, wh)

        # 幅、高さどちらも 25 以上の場合は 25 に固定する（最大サイズの固定）
        wh[wh > 25] = 25
        human_num = ImgInfo["human_num"]
        for point in centroids:
            point = point[None, :]

            # 中心からの距離
            dists = euclidean_dist(point, centroids)
            dists = dists.squeeze()
            ids = np.argsort(dists)

            for start, first in enumerate(ids, 0):
                if start > 0 and start < 5:
                    src_point = point.squeeze()
                    dst_point = centroids[first]

                    src_w, src_h = wh[ids[0]][0], wh[ids[0]][1]
                    dst_w, dst_h = wh[first][0], wh[first][1]

                    count = 0
                    if (src_w + dst_w) - np.abs(src_point[0] - dst_point[0]) > 0 and (
                        src_h + dst_h
                    ) - np.abs(src_point[1] - dst_point[1]) > 0:
                        w_reduce = (
                            (src_w + dst_w) - np.abs(src_point[0] - dst_point[0])
                        ) / 2
                        h_reduce = (
                            (src_h + dst_h) - np.abs(src_point[1] - dst_point[1])
                        ) / 2
                        threshold_w, threshold_h = max(
                            -int(max(src_w - w_reduce, dst_w - w_reduce) / 2.0), -60
                        ), max(-int(max(src_h - h_reduce, dst_h - h_reduce) / 2.0), -60)

                    else:
                        threshold_w, threshold_h = max(
                            -int(max(src_w, dst_w) / 2.0), -60
                        ), max(-int(max(src_h, dst_h) / 2.0), -60)
                    # threshold_w, threshold_h = -5, -5
                    while (src_w + dst_w) - np.abs(
                        src_point[0] - dst_point[0]
                    ) > threshold_w and (src_h + dst_h) - np.abs(
                        src_point[1] - dst_point[1]
                    ) > threshold_h:

                        if (dst_w * dst_h) > (src_w * src_h):
                            wh[first][0] = max(int(wh[first][0] * 0.9), 2)
                            wh[first][1] = max(int(wh[first][1] * 0.9), 2)
                            dst_w, dst_h = wh[first][0], wh[first][1]
                        else:
                            wh[ids[0]][0] = max(int(wh[ids[0]][0] * 0.9), 2)
                            wh[ids[0]][1] = max(int(wh[ids[0]][1] * 0.9), 2)
                            src_w, src_h = wh[ids[0]][0], wh[ids[0]][1]

                        if human_num > 3:
                            # print(human_num, centroids.shape, ids.shape, start)
                            dst_point_ = centroids[ids[start + 1]]
                            dst_w_, dst_h_ = (
                                wh[ids[start + 1]][0],
                                wh[ids[start + 1]][1],
                            )
                            if (dst_w_ * dst_h_) > (src_w * src_h) and (
                                dst_w_ * dst_h_
                            ) > (dst_w * dst_h):
                                if (src_w + dst_w_) - np.abs(
                                    src_point[0] - dst_point_[0]
                                ) > -3 and (src_h + dst_h_) - np.abs(
                                    src_point[1] - dst_point_[1]
                                ) > -3:
                                    wh[ids[start + 1]][0] = max(
                                        int(wh[ids[start + 1]][0] * 0.9), 2
                                    )
                                    wh[ids[start + 1]][1] = max(
                                        int(wh[ids[start + 1]][1] * 0.9), 2
                                    )

                        count += 1
                        if count > 40:
                            break

        for (center_w, center_h), (width, height) in zip(centroids, wh):
            assert width > 0 and height > 0

            if (0 < center_w < w) and (0 < center_h < h):
                h_start = center_h - height
                h_end = center_h + height

                w_start = center_w - width
                w_end = center_w + width
                #
                if h_start < 0:
                    h_start = 0

                if h_end > h:
                    h_end = h

                if w_start < 0:
                    w_start = 0

                if w_end > w:
                    w_end = w

                mask_map[h_start:h_end, w_start:w_end] = 1

        mask_map = mask_map * 255
        cv2.imwrite(
            str(Path(mask_dir) / Path(image_path).name),
            mask_map,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )


def preprocess_vh_vv_bathymetry_v2():
    data_source = XView3DataSource()
    preprocessed_info = XView3PreprocessedV2()

    # Cropped images
    print("# (1) Generating preprocess_vh_vv_bathymetry_v2/validation/*")
    print("# (2) Generating preprocess_vh_vv_bathymetry_v2/validation.csv")
    _internal_preprocess_v2(
        data_source.validation_csv,
        preprocessed_info.validation_csv,
        preprocessed_info.validation_image_dir,
        crop_size=800,
    )

    # Cropped images
    print("# (3) Generating preprocess_vh_vv_bathymetry_v2/train/*")
    print("# (4) Generating preprocess_vh_vv_bathymetry_v2/train.csv")
    _internal_preprocess_v2(
        data_source.train_csv,
        preprocessed_info.train_csv,
        preprocessed_info.train_image_dir,
        crop_size=800,
    )


def preproc_v2():
    # Preprocess images
    preprocess_vh_vv_bathymetry_v2()

    # Preprocess masks
    print("# (5) Generating preprocess_vh_vv_bathymetry_v2/validation_masks/*")
    preprocess_validation_masks_v2()


def validation_scene_ids():
    datainfo = XView3DataSource()
    df = pd.read_csv(datainfo.validation_csv)
    scene_ids = list(sorted(df["scene_id"].unique()))
    return scene_ids


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


def load_image_ppv6(input_dir: Path, scene_id: str, crop_size: int = 800) -> np.ndarray:
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
                _pad(im[..., 0], crop_size, crop_size)[0],
                _pad(im[..., 1], crop_size, crop_size)[0],
                _pad(im[..., 2], crop_size, crop_size)[0],
            ],
            axis=2,
        )

    return im_pad


def get_scale(shape, max_size):
    return max_size / max(shape[:2])


def processing_ppv6(scene_ids,
                    input_dir: Path,
                    setname: str):
    out_prefix = "data/working/xview3/images/ppv6"
    out_dir = Path(f"{out_prefix}/{setname}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_out_dir = Path(f"{out_prefix}/thumb_{setname}/")
    thumb_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Total num of scene_ids: {len(scene_ids)}")
    for scene_id in tqdm.tqdm(scene_ids, total=len(scene_ids)):
        with timer(f"Load scene_image: {scene_id}", logger=logger):
            im_pad = load_image_ppv6(input_dir, scene_id)
            im_pad = cv2.cvtColor(im_pad, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(out_dir / f"{scene_id}.png"),
                        im_pad,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
            scale = get_scale(im_pad.shape, 1080)
            h = int(im_pad.shape[0] * scale)
            w = int(im_pad.shape[1] * scale)
            im_pad = cv2.resize(im_pad, (w, h))
            cv2.imwrite(str(thumb_out_dir / f"{scene_id}.png"),
                        im_pad,
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])


def preproc_v6():
    datainfo = XView3DataSource()
    processing_ppv6(validation_scene_ids(),
                    datainfo.trainval_input_dir,
                    "validation")


def main():
    # Generate data/working/xview3/preprocess_vh_vv_bathymetry_v2/*
    preproc_v2()

    # Generate data/working/xview3/images/ppv6/*
    preproc_v6()


if __name__ == "__main__":
    set_logger(logger)
    main()
