import sys
import time
import math
from typing import List, Dict
from contextlib import contextmanager
from pathlib import Path
import warnings
import tarfile

import gshhg
import timm
import albumentations as albu
import cv2
import rasterio
from rasterio.enums import Resampling
import rasterio.warp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from pyproj import Transformer

from models_hrnet import get_seg_model


def get_boxinfo_from_binar_map(out_bin, min_area=3):
    binar_numpy = out_bin.squeeze().astype(np.uint8)
    assert binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binar_numpy, connectivity=4
    )

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = boxes[:, 4] >= min_area
    boxes = boxes[index]
    points = points[index]

    return {
        "num": len(points),
        "points": points.tolist(),
        "boxes": boxes.tolist(),
    }


@contextmanager
def timer(name, logger=None):
    t0 = time.time()

    if logger:
        logger.info(f"[{name}] start.")
    else:
        print(f"[{name}] start.")
    yield

    if logger:
        logger.info(f"[{name}] done in {time.time() - t0:.0f} s")
    else:
        print(f"[{name}] done in {time.time() - t0:.0f} s")


def _load_image_layers_v2(scene_id, base_dir="data/input/xview3/downloaded"):
    imgs = {}
    proc_imgs = {}

    channels = [
        "VH_dB",
        "VV_dB",
        "bathymetry",
        "owiMask",
    ]

    for fl in channels:
        tif_path = str(base_dir / scene_id / f"{fl}.tif")
        with rasterio.open(tif_path, "r") as dataset:
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


def load_image_ppv6(input_dir: Path, scene_id: str, crop_size: int = 800) -> np.ndarray:
    with timer(f"Loading scene image ({scene_id})..."):
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

    return im_pad, proc_imgs


def gen_crop_locations(im: np.ndarray) -> List[Dict[str, int]]:
    crop_size = 800
    h, w = im.shape[:2]

    rows = []
    for yidx in range(int(h / crop_size)):
        for xidx in range(int(w / crop_size)):
            rows.append(
                {
                    "crop_size": crop_size,
                    "yidx": yidx,
                    "xidx": xidx,
                    "y0": yidx * crop_size,
                    "x0": xidx * crop_size,
                }
            )

    return rows


class PPV2InferenceDataset(Dataset):
    def __init__(self, rows, im):
        self.rows = rows
        self.im = im
        self.test_transform = albu.Compose([
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        im = self.im[
            row["y0"] : row["y0"] + row["crop_size"],
            row["x0"] : row["x0"] + row["crop_size"],
        ]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        aug = self.test_transform(image=im)
        im = aug["image"]

        im = torch.from_numpy(im.transpose((2, 0, 1))).float()
        return im


def localize_singlemodel(scene_id, im, rows):
    weight_path = Path("v13ep59.pth")
    assert weight_path.exists()

    model = CrowdLocatorV2(backbone="hrnet")
    model.load_state_dict(torch.load(weight_path))

    # Single GPU
    model = model.to("cuda")
    model.eval()

    test_batch_size = 8
    ds = PPV2InferenceDataset(rows, im)
    infer_dl = DataLoader(
        ds,
        num_workers=4,
        batch_size=test_batch_size,
        drop_last=False,
        shuffle=False
    )

    locator_results = []
    for idx, X in enumerate(infer_dl):
        if idx % 20 == 0 and idx > 0:
            print("Processing... {} of {}".format(
                idx, len(infer_dl)))

        with torch.no_grad():
            X = X.to("cuda")
            batch_size = X.size(0)
            th_out, pre_map, _ = model(X, mask_gt=None, mode="val")

            for batch_idx in range(batch_size):
                im_thresh = th_out[batch_idx].squeeze()
                im_pred = pre_map[batch_idx].squeeze()
                out_bin = (
                    torch.where(
                        im_pred >= im_thresh,
                        torch.ones_like(im_pred) * 255,
                        torch.zeros_like(im_pred),
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                boxes = get_boxinfo_from_binar_map(out_bin)

                y0 = infer_dl.dataset.rows[idx * test_batch_size + batch_idx]["y0"]
                x0 = infer_dl.dataset.rows[idx * test_batch_size + batch_idx]["x0"]
                for pt in boxes["points"]:
                    center_x, center_y = pt
                    locator_results.append(
                        {
                            "scene_id": scene_id,
                            "detect_scene_column": x0 + center_x,  # not x, y
                            "detect_scene_row": y0 + center_y,  # not y, x
                        }
                    )

    df = pd.DataFrame(locator_results)
    return df


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


class CrowdLocator(nn.Module):
    def __init__(self, net_name, gpu_id, binar_input_channels=720):
        super(CrowdLocator, self).__init__()

        self.extractor = get_seg_model(net_name)
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


class PPV2ShipCropInferenceDataset(Dataset):
    def __init__(self, df, im):
        self.crop_size = 256
        self.df = df
        self.im = im
        self.test_transform = albu.Compose([
            albu.CenterCrop(128, 128, p=1.0),
            albu.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
        ])

    def __len__(self):
        return len(self.df)

    def decode_vessel_length(self, vals: np.ndarray, length_lower):
        return np.expm1(vals) + length_lower

    def __getitem__(self, index):
        r = self.df.iloc[index]
        yc = int(r["detect_scene_row"])
        xc = int(r["detect_scene_column"])

        im_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
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
        if yc + d > self.im.shape[0]:
            # 下にはみ出る場合
            bottom = self.im.shape[0] - d - yc
            y1 = self.im.shape[0]
        if xc + d > self.im.shape[1]:
            right = self.im.shape[1] - d - xc
            x1 = self.im.shape[1]

        im_crop[top:bottom, left:right] = self.im[y0:y1, x0:x1]
        im_crop = self.test_transform(image=im_crop)["image"]
        im_crop = torch.from_numpy(im_crop.transpose((2, 0, 1))).float()
        return im_crop


def _internal_classify_ensemble(df, im):
    sigmoid_outputs = {
        "class0": [],
        "class1": [],
        "class2": [],
    }
    for fold in range(10):
        path = Path(f"v77f{fold}ep9.pth")
        assert path.exists()

        model = timm.create_model(
            model_name="resnet50d",
            pretrained=False,
            num_classes=3,
        )
        model = model.to("cuda")
        model.load_state_dict(torch.load(path))
        model.eval()

        ds_clf = PPV2ShipCropInferenceDataset(df, im)
        infer_dl = DataLoader(
            ds_clf, num_workers=4, batch_size=8, drop_last=False, shuffle=False
        )

        y_pred_class0_confidence_ = []
        y_pred_class1_confidence_ = []
        y_pred_class2_confidence_ = []
        for idx, X in enumerate(infer_dl):
            softmax_func = torch.nn.Softmax(dim=1)
            with torch.no_grad():
                X = X.to("cuda")
                out = softmax_func(model(X))
                y_pred_class0_confidence_ += out[:, 0].cpu().numpy().ravel().tolist()
                y_pred_class1_confidence_ += out[:, 1].cpu().numpy().ravel().tolist()
                y_pred_class2_confidence_ += out[:, 2].cpu().numpy().ravel().tolist()
        sigmoid_outputs["class0"].append(y_pred_class0_confidence_)
        sigmoid_outputs["class1"].append(y_pred_class1_confidence_)
        sigmoid_outputs["class2"].append(y_pred_class2_confidence_)

    y_pred_class0_confidence_ = np.mean(sigmoid_outputs["class0"], axis=0)
    y_pred_class1_confidence_ = np.mean(sigmoid_outputs["class1"], axis=0)
    y_pred_class2_confidence_ = np.mean(sigmoid_outputs["class2"], axis=0)
    y_pred_ = np.array([
        y_pred_class0_confidence_,
        y_pred_class1_confidence_,
        y_pred_class2_confidence_,
    ]).argmax(axis=0)

    return {
        "predict": y_pred_,
        "class0_confidence": y_pred_class0_confidence_,
        "class1_confidence": y_pred_class1_confidence_,
        "class2_confidence": y_pred_class2_confidence_,
    }


def classify(df, scene_id, im_pad, rows):
    with timer("Run classifier..."):
        pred_dict = _internal_classify_ensemble(df, im_pad)
        # 0: non-vessel/non-fishing, 1: vessel/non-fishing, 2: fishing
        df["detection_class"] = np.array(pred_dict["predict"]).ravel()
        # TODO: tuning thresholdfor for is_vessel and is_fishing flags.
        df["detection_class0_confidence"] = np.array(pred_dict["class0_confidence"]).ravel()
        df["detection_class1_confidence"] = np.array(pred_dict["class1_confidence"]).ravel()
        df["detection_class2_confidence"] = np.array(pred_dict["class2_confidence"]).ravel()
        return df


def regression(df, scene_id, im_pad, rows):
    with timer("Run regressor..."):
        path = Path("v16ep19.pth")
        assert path.exists()

        model = timm.create_model(
            model_name="resnet50", pretrained=False, num_classes=1
        )
        model = model.to("cuda")
        model.load_state_dict(torch.load(path))
        model.eval()

        ds_reg = PPV2ShipCropInferenceDataset(df, im_pad)
        infer_dl = DataLoader(
            ds_reg,
            num_workers=4,
            batch_size=8,
            drop_last=False,
            shuffle=False
        )
        y_pred_ = []
        for idx, X in enumerate(infer_dl):
            with torch.no_grad():
                X = X.to("cuda")
                y_pred_ += model(X).cpu().numpy().ravel().tolist()

        length_lower = 15
        length_upper = 200

        y_pred = ds_reg.decode_vessel_length(np.array(y_pred_), length_lower)
        y_pred = np.clip(y_pred, length_lower, length_upper)
        df["vessel_length_m"] = y_pred.ravel()
        return df


def main():
    print("args:", sys.argv)

    scene_id: str = sys.argv[2]
    input_dir, output_csv = Path(sys.argv[1]), Path(sys.argv[3])
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    im_pad, proc_imgs = load_image_ppv6(input_dir, scene_id)
    # im_pad = cv2.cvtColor(im_pad, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f"/dev/shm/{scene_id}.png",
    #             im_pad,
    #             [cv2.IMWRITE_PNG_COMPRESSION, 1])
    # im_pad = cv2.imread(f"/dev/shm/{scene_id}.png")
    # im_pad = cv2.cvtColor(im_pad, cv2.COLOR_BGR2RGB)
    # print(im_pad.shape)

    rows = gen_crop_locations(im_pad)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Localize
        df = localize_singlemodel(scene_id, im_pad, rows)

    df = classify(df, scene_id, im_pad, rows)
    df = regression(df, scene_id, im_pad, rows)

    df["is_vessel"] = df["detection_class"].apply(
        lambda x: "true" if x > 0 else "false")
    df["is_fishing"] = df["detection_class"].apply(
        lambda x: "true" if x == 2 else "false")

    df = land_cover_mask(scene_id, df, input_dir, mask_distance_thresh=0)

    df[[
        "scene_id",
        "detect_scene_row",
        "detect_scene_column",
        "is_vessel",
        "is_fishing",
        "vessel_length_m",
    ]].to_csv(output_csv, index=False)


def land_cover_mask(scene_id, df, input_dir, mask_distance_thresh=0):
    shorelines = gshhg.GSHHG(
        "/home/xview3/GSHHS_shp", resolution="full")

    tile_path = input_dir / scene_id / "VV_dB.tif"
    with rasterio.open(tile_path, "r") as rasterdata:
        inv_transformer = Transformer.from_crs(
            rasterdata.crs, "EPSG:4326", always_xy=True)
        transformer = Transformer.from_crs(
            "EPSG:4326", rasterdata.crs, always_xy=True)

    x = df["detect_scene_row"].values
    y = df["detect_scene_column"].values
    lon, lat = inv_transformer.transform(*rasterdata.xy(x, y))

    df["mask_level"] = shorelines.mask(lon, lat)
    df["distance_to_nearest"] = shorelines.distance_to_nearest(lon, lat)

    df = df[~((df["mask_level"] != 0) & (df["distance_to_nearest"] > mask_distance_thresh))]
    return df


def land_cover_mask_tarfile(scene_id, df, input_dir, mask_distance_thresh=0):
    shorelines = gshhg.GSHHG(
        "/home/xview3/GSHHS_shp", resolution="full")

    tar_path = input_dir / f"{scene_id}.tar.gz"
    with tarfile.open(tar_path, "r") as f:
        with rasterio.open(f.extractfile(f"{scene_id}/VV_dB.tif"), "r") as rasterdata:
            inv_transformer = Transformer.from_crs(
                rasterdata.crs, "EPSG:4326", always_xy=True)
            transformer = Transformer.from_crs(
                "EPSG:4326", rasterdata.crs, always_xy=True)

    x = df["detect_scene_row"].values
    y = df["detect_scene_column"].values
    lon, lat = inv_transformer.transform(*rasterdata.xy(x, y))

    df["mask_level"] = shorelines.mask(lon, lat)
    df["distance_to_nearest"] = shorelines.distance_to_nearest(lon, lat)

    df = df[~((df["mask_level"] != 0) & (df["distance_to_nearest"] > mask_distance_thresh))]
    return df


if __name__ == "__main__":
    main()
