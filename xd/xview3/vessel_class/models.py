import numpy as np
import torch
import timm
from timm.models.layers import SelectAdaptivePool2d, Linear
import torch.nn as nn

from xd.utils.configs import dynamic_load


def get_model(conf):
    conf_model = conf.train.model
    model = dynamic_load(conf_model.fqdn)(**conf_model.kwargs)
    model = model.to("cuda")
    return model
