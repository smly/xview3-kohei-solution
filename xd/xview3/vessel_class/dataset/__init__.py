from argparse import Namespace
from pathlib import Path

from omegaconf import DictConfig
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from xd.utils.configs import load_config, dynamic_load

from xd.xview3.vessel_class.dataset.ppv6 import PPV6


def get_dataloaders(args):
    ds = get_dataset(args, is_test=False)
    dl = DataLoader(
        ds,
        num_workers=32,
        batch_size=64,
        drop_last=True,
        shuffle=True
    )  # noqa

    val_ds = get_dataset(args, is_test=True)
    val_dl = DataLoader(
        val_ds,
        num_workers=32,
        batch_size=64,
        drop_last=False,
        shuffle=False
    )

    dataloaders = {}
    dataloaders["train"] = dl
    dataloaders["val"] = val_dl

    return dataloaders


def get_dataset(args: Namespace, is_test=False) -> Dataset:
    conf = load_config(args.configs)
    dataset = dynamic_load(conf.dataset.fqdn)(
        conf,
        fold=args.fold,
        is_test=is_test,
    )
    return dataset


def get_scene_dataset(
    regressor_conf: DictConfig,
    location_dataframe: pd.DataFrame,
    input_array: np.ndarray,
) -> Dataset:
    klass = dynamic_load(regressor_conf.dataset.scene_fqdn)
    dataset = klass(
        regressor_conf,
        location_dataframe,
        input_array,
    )
    return dataset
