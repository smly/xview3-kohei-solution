import argparse
import os
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
# import wandb
from omegaconf import OmegaConf
from xd.xview3.vessel_class.dataset import get_dataloaders
from xd.xview3.vessel_class.models import get_model
from xd.utils.meters import AverageMeter
from xd.utils.configs import dynamic_load, load_config
from xd.utils.logger import set_logger

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

logger = getLogger("xd")


def parse_args():
    parser = argparse.ArgumentParser(description="vessel class estimator")
    parser.add_argument("-c", "--configs", type=str, required=True)
    parser.add_argument("-f", "--fold", type=int, default=0)
    return parser.parse_args()


def main(args: argparse.Namespace):
    assert "vessel_class" in args.configs
    conf = load_config(args.configs)
    # wandb.init(
    #     project="xview3",
    #     tags=[f"fold{args.fold}", "train+val", "classify"],
    #     config=OmegaConf.to_object(conf),
    # )  # noqa
    dataloaders = get_dataloaders(args)

    model = get_model(conf)
    # model = nn.DataParallel(model)

    total_epochs = conf.train.total_epochs

    criterion = dynamic_load(conf.train.loss.fqdn)(**conf.train.loss.kwargs)
    optimizer = dynamic_load(conf.train.optimizer.fqdn)(
        model.parameters(), **conf.train.optimizer.kwargs
    )

    scheduler_kwargs = conf.train.scheduler.kwargs
    if conf.train.scheduler.fqdn.endswith("CosineAnnealingLR"):
        scheduler_kwargs.update(
            {"T_max": len(dataloaders["train"]) * conf.train.total_epochs}
        )
    scheduler = dynamic_load(conf.train.scheduler.fqdn)(
        optimizer, **scheduler_kwargs
    )  # noqa

    for epoch in range(total_epochs):
        model.train()
        trn_metrics = {
            "loss": AverageMeter(),
            "acc": AverageMeter(),
        }
        has_numerical = False
        for idx, batch_data in enumerate(dataloaders["train"]):
            if len(batch_data) == 2:
                X, y = batch_data
                X, y = X.to("cuda"), y.to("cuda").long()
            elif len(batch_data) == 3:
                has_numerical = True
                X, X_num, y = batch_data
                X, X_num, y = X.to("cuda"), X_num.to("cuda").float(), y.to("cuda").long()
            else:
                raise RuntimeError

            optimizer.zero_grad()

            if not has_numerical:
                out = model(X)
            else:
                out = model(X, X_num)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            trn_metrics["loss"].update(loss.item(), count=X.size(0))
            trn_metrics["acc"].update((out.argmax(dim=1) == y).float().mean())  # noqa

            if idx % 100 == 0 and idx > 0:
                max_iter = len(dataloaders["train"])
                loss_val = trn_metrics["loss"].avg
                acc_val = trn_metrics["acc"].avg
                logger.info(
                    f"Ep {epoch}, iter {idx}/{max_iter}, trn_loss {loss_val:.6f}"
                    f", trn_acc {acc_val:.6f}"
                )  # noqa

        model.eval()
        val_metrics = {
            "loss": AverageMeter(),
            "acc": AverageMeter(),
        }
        for idx, batch_data in enumerate(dataloaders["val"]):
            if len(batch_data) == 2:
                X, y = batch_data
                X, y = X.to("cuda"), y.to("cuda").long()
            elif len(batch_data) == 3:
                X, X_num, y = batch_data
                X, X_num, y = X.to("cuda"), X_num.to("cuda").float(), y.to("cuda").long()
            else:
                raise RuntimeError

            with torch.no_grad():
                if not has_numerical:
                    out = model(X)
                else:
                    out = model(X, X_num)

                loss = criterion(out, y)
                val_metrics["loss"].update(loss.item(), count=X.size(0))
                val_metrics["acc"].update(
                    (out.argmax(dim=1) == y).float().mean()
                )  # noqa

        logger.info(
            "val_loss: {:.6f}, val_acc: {:.6f}".format(
                val_metrics["loss"].avg, val_metrics["acc"].avg
            )
        )
        # wandb.log(
        print(
            {
                "epoch": epoch,
                "trn_loss": trn_metrics["loss"].avg,
                "trn_acc": trn_metrics["acc"].avg,
                "val_loss": val_metrics["loss"].avg,
                "val_acc": val_metrics["acc"].avg,
            }
        )  # noqa

        model_dir = Path("data/working/xview3/models/") / conf.name
        model_dir = model_dir / f"fold{args.fold}"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            model.state_dict(),
            str(model_dir / f"ep{epoch}.pth"),
        )


if __name__ == "__main__":
    set_logger(logger)
    main(parse_args())
