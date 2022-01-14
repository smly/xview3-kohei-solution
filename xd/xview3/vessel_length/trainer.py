import argparse
import os
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
# import wandb
from omegaconf import OmegaConf

from xd.xview3.vessel_length.dataset import get_dataloaders
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
    parser = argparse.ArgumentParser(description="vessel length estimator")
    parser.add_argument("-c", "--configs", type=str, required=True)
    parser.add_argument("-f", "--fold", type=int, default=0)
    return parser.parse_args()


def main(args: argparse.Namespace):
    conf = load_config(args.configs)
    # wandb.init(
    #     project="xview3",
    #     tags=[f"fold{args.fold}", "train+val", "regressor"],
    #     config=OmegaConf.to_object(conf),
    # )  # noqa
    dataloaders = get_dataloaders(args, conf)

    model = timm.create_model(
        conf.train.model.kwargs.model_name, pretrained=True, num_classes=1
    )  # noqa
    model = model.to("cuda")

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
        }
        for idx, (X, y) in enumerate(dataloaders["train"]):
            X, y = X.to("cuda"), y.to("cuda").float()
            optimizer.zero_grad()

            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            trn_metrics["loss"].update(loss.item(), count=X.size(0))

            if idx % 100 == 0 and idx > 0:
                max_iter = len(dataloaders["train"])
                loss_val = trn_metrics["loss"].avg
                logger.info(
                    f"Ep {epoch}, iter {idx}/{max_iter}, loss {loss_val:.6f}"
                )  # noqa

        y_pred_ = []

        model.eval()
        for idx, (X, y) in enumerate(dataloaders["val"]):
            X, y = X.to("cuda"), y.to("cuda").float()
            with torch.no_grad():
                out = model(X)
                out = out.cpu().numpy().ravel()
                y_pred_ += out.tolist()

        y_pred = np.array(y_pred_)
        y_pred = dataloaders["val"].dataset.decode_vessel_length(y_pred)
        y_gt = dataloaders["val"].dataset.df["vessel_length_m"].values
        y_pred = np.clip(
            y_pred,
            dataloaders["val"].dataset.length_lower,
            dataloaders["val"].dataset.length_upper,
        )
        pe_l = 1.0 - min(np.mean(np.abs(y_gt - y_pred) / y_gt), 1.0)
        logger.info(f"PE_L={pe_l:.6f}")
        # wandb.log(
        print(
            {"epoch": epoch, "trn_loss": trn_metrics["loss"].avg, "val_pe": pe_l}
        )  # noqa

        model_dir = Path("data/working/xview3/models/") / conf.name
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            model.state_dict(),
            str(model_dir / f"ep{epoch}.pth"),
        )


if __name__ == "__main__":
    set_logger(logger)
    main(parse_args())
