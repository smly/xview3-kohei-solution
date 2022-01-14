import argparse
import os
from logging import getLogger
from pathlib import Path

import cv2
import torch
import wandb
from omegaconf import OmegaConf
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from xd.xview3.localization.metrics import (
    AverageMeter,
    calculate_p_r_f,
    evaluate_metrics,
    get_boxinfo_from_binar_map,
)
from xd.utils.configs import dynamic_load, load_config
from xd.utils.logger import set_logger

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

logger = getLogger("xd")


def parse_args():
    parser = argparse.ArgumentParser(description="train_iim")
    parser.add_argument("-c", "--configs", type=str, required=True)
    parser.add_argument("-f", "--fold", type=int, default=0)
    return parser.parse_args()


def get_dataloaders(args, conf):
    trn_ds = dynamic_load(conf.dataset.fqdn)(args, conf, is_test=False)
    val_ds = dynamic_load(conf.dataset.fqdn)(args, conf, is_test=True)
    dataloaders = {}

    dataloaders["train"] = DataLoader(
        dataset=trn_ds,
        sampler=RandomSampler(trn_ds),
        batch_size=conf.train.batch_size,
        pin_memory=True,
        num_workers=conf.train.num_workers,
        drop_last=True,
    )
    dataloaders["val"] = DataLoader(
        dataset=val_ds,
        sampler=SequentialSampler(val_ds),
        batch_size=conf.test.batch_size,
        pin_memory=True,
        num_workers=conf.test.num_workers,
        drop_last=False,
    )
    return dataloaders


def train(epoch, model, dataloaders, optimizer, scheduler):
    model.train()
    trn_metrics = {
        "loss": AverageMeter(),
        "head_loss": AverageMeter(),
        "binar_loss": AverageMeter(),
        "trn_dice": AverageMeter(),
        "trn_jaccard": AverageMeter(),
    }
    trn_metrics_count = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
    for i, (X, y, idxs) in enumerate(dataloaders["train"]):
        X = X.to("cuda")
        y = y.to("cuda")

        optimizer.zero_grad()
        threshold_matrix, pre_map, _ = model(X, y)
        head_map_loss, binary_map_loss = model.loss
        all_loss = head_map_loss + binary_map_loss
        all_loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = X.size(0)

        with torch.no_grad():
            trn_metrics["trn_dice"].update(
                1.0 - DiceLoss(mode="binary", from_logits=False)(pre_map, y).item(), count=1
            )
            trn_metrics["trn_jaccard"].update(
                1.0 - JaccardLoss(mode="binary", from_logits=False)(pre_map, y).item(), count=1
            )

        trn_metrics["loss"].update(all_loss.item(), count=batch_size)
        trn_metrics["head_loss"].update(head_map_loss.item(), count=batch_size)  # noqa
        trn_metrics["binar_loss"].update(
            binary_map_loss.item(), count=batch_size
        )  # noqa

        for im_i, index in enumerate(idxs):
            im_thresh = threshold_matrix[im_i].squeeze()
            im_pred = pre_map[im_i].squeeze()
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

            gt_pred = y[im_i].squeeze()
            gt_out_bin = (
                torch.where(
                    gt_pred > 0,
                    torch.ones_like(gt_pred) * 255,
                    torch.zeros_like(gt_pred),
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            gt_boxes = get_boxinfo_from_binar_map(gt_out_bin)
            metrics = evaluate_metrics(boxes["points"], gt_boxes["points"])

            trn_metrics_count["tp"] += metrics["tp"]
            trn_metrics_count["fp"] += metrics["fp"]
            trn_metrics_count["fn"] += metrics["fn"]

        if i % 100 == 0 and i > 0:
            report_line = "Epoch {}, Iter {}/{}".format(
                epoch, i, len(dataloaders["train"])
            )
            for name in trn_metrics.keys():
                report_line += " {}: {:.6f}".format(name, trn_metrics[name].avg)  # noqa
            logger.info(report_line)

    prec, recall, fscore = calculate_p_r_f(
        trn_metrics_count["tp"],
        trn_metrics_count["fp"],
        trn_metrics_count["fn"],
    )
    wandb.log(
        {
            "epoch": epoch,
            "trn_loss": trn_metrics["loss"].avg,
            "trn_head_loss": trn_metrics["head_loss"].avg,
            "trn_binar_loss": trn_metrics["binar_loss"].avg,
            "trn_precision": prec,
            "trn_recall": recall,
            "trn_fscore": fscore,
            "trn_dice": trn_metrics["trn_dice"].avg,
            "trn_jaccard": trn_metrics["trn_jaccard"].avg,
            "trn_tp": trn_metrics_count["tp"],
            "trn_fp": trn_metrics_count["fp"],
            "trn_fn": trn_metrics_count["fn"],
        }
    )


def validate(epoch, model, dataloaders):
    model.eval()

    val_meters = {
        "val_dice": AverageMeter(),
        "val_jaccard": AverageMeter(),
    }
    val_metrics = {"tp": 0, "fp": 0, "fn": 0}
    for _, (X, y_orig, idxs) in enumerate(dataloaders["val"]):
        with torch.no_grad():
            X = X.to("cuda")
            y = y_orig.to("cuda")
            thresh, pred, _ = model(X, mask_gt=None, mode="val")

        with torch.no_grad():
            val_meters["val_dice"].update(
                1.0 - DiceLoss(mode="binary", from_logits=False)(pred, y).item(), count=1
            )
            val_meters["val_jaccard"].update(
                1.0 - JaccardLoss(mode="binary", from_logits=False)(pred, y).item(), count=1
            )

        for i, index in enumerate(idxs):
            im_thresh = thresh[i].squeeze()
            im_pred = pred[i].squeeze()
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

            gt_pred = y_orig[i].squeeze()
            gt_out_bin = (
                torch.where(
                    gt_pred > 0,
                    torch.ones_like(gt_pred) * 255,
                    torch.zeros_like(gt_pred),
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            gt_boxes = get_boxinfo_from_binar_map(gt_out_bin)
            metrics = evaluate_metrics(boxes["points"], gt_boxes["points"])
            for name in val_metrics.keys():
                val_metrics[name] += metrics[name]

    prec, recall, fscore = calculate_p_r_f(
        val_metrics["tp"],
        val_metrics["fp"],
        val_metrics["fn"],
    )

    wandb.log(
        {
            "epoch": epoch,
            "val_precision": prec,
            "val_recall": recall,
            "val_fscore": fscore,
            "val_dice": val_meters["val_dice"].avg,
            "val_jaccard": val_meters["val_jaccard"].avg,
            "val_tp": val_metrics["tp"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
        }
    )
    val_dice = val_meters["val_dice"].avg
    val_jaccard = val_meters["val_jaccard"].avg

    logger.info(
        f"Epoch {epoch}, "
        f"val_prec: {prec:.6f}, val_recall: {recall:.6f}, "
        f"val_fscore: {fscore:.6f}, "
        f"val_dice: {val_dice:.6f}, "
        f"val_jaccard: {val_jaccard:.6f}"
    )


def custom_optimizer(model, model_init_lr=1e-5):
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.extractor.parameters(),
                "lr": model_init_lr,
                "weight_decay": 1e-5,
            },
            {
                "params": model.binar.parameters(),
                "lr": model_init_lr * 0.1,
                "weight_decay": 1e-5,
            },
        ]
    )
    return optimizer


def main(args: argparse.Namespace):
    conf = load_config(args.configs)
    wandb.init(
        project="xview3",
        tags=[f"fold{args.fold}", "detector"],
        config=OmegaConf.to_object(conf),
    )  # noqa
    logger.info("Config file: " + args.configs + "\n" + OmegaConf.to_yaml(conf))  # noqa

    dataloaders = get_dataloaders(args, conf)

    model = dynamic_load(conf.train.model.fqdn)(**conf.train.model.kwargs)
    model = model.to("cuda")

    # Cluster model
    if conf.dataset.fqdn.startswith("xd.xview3.localization.dataset.PPV2VO_C"):
        assert Path(conf.train.model.pretrained).exists()
        model.load_state_dict(torch.load(conf.train.model.pretrained))

    if conf.train.optimizer.fqdn.startswith("torch.optim"):
        optimizer = dynamic_load(conf.train.optimizer.fqdn)(
            model.parameters(), **conf.train.optimizer.kwargs
        )
    else:
        optimizer = dynamic_load(conf.train.optimizer.fqdn)(
            model, **conf.train.optimizer.kwargs
        )

    scheduler_kwargs = conf.train.scheduler.kwargs
    if conf.train.scheduler.fqdn.endswith("CosineAnnealingLR"):
        scheduler_kwargs.update(
            {"T_max": len(dataloaders["train"]) * conf.train.total_epochs}
        )
    scheduler = dynamic_load(conf.train.scheduler.fqdn)(
        optimizer, **scheduler_kwargs
    )  # noqa

    for epoch in range(conf.train.total_epochs):
        train(epoch, model, dataloaders, optimizer, scheduler)
        validate(epoch, model, dataloaders)

        model_dir = (
            Path("data/working/xview3/models/") / conf.name / f"fold{args.fold}"
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            model.state_dict(),
            str(model_dir / f"ep{epoch}.pth"),
        )


if __name__ == "__main__":
    set_logger(logger)
    main(parse_args())
