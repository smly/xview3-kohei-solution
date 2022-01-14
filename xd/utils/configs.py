import importlib
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(config: str) -> Union[DictConfig, ListConfig]:
    conf = OmegaConf.load(config)
    conf.name = Path(config).stem

    OmegaConf.set_readonly(conf, False)
    return conf


def dynamic_load(model_class_fqn):
    module_name = ".".join(model_class_fqn.split(".")[:-1])
    class_name = model_class_fqn.split(".")[-1]
    mod = importlib.import_module(module_name)
    class_obj = getattr(mod, class_name)
    return class_obj


def load_optimizer(parameters, config):
    class_obj = dynamic_load(config["optimizer"]["fqdn"])
    optimizer = class_obj(parameters, **config["optimizer"]["kwargs"])
    return optimizer


def load_scheduler(optimizer, config, auto_resume=False, last_iters=100):
    class_obj = dynamic_load(config["scheduler"]["fqdn"])
    kwargs = dict(config["scheduler"]["kwargs"])
    if auto_resume and last_iters > 0 and False:
        kwargs.update({"last_epoch": last_iters})
    scheduler = class_obj(optimizer, **kwargs)
    return scheduler
