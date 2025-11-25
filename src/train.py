import typing as tp
import logging
import traceback

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from model import BSModel
from data.augmentations import *
from pytorch_lightning.strategies.ddp import DDPStrategy

from data import collate_fn
from model import PLModel

log = logging.getLogger(__name__)


def initialize_loaders(cfg: DictConfig) -> tp.Tuple[DataLoader, DataLoader]:
    """
    Initializes train and validation dataloaders from configuration file.
    """
    train_dataset = instantiate(cfg.train_dataset)
    train_loader = DataLoader(train_dataset, **cfg.train_loader, collate_fn=collate_fn)
    val_dataset = instantiate(cfg.val_dataset)
    val_loader = DataLoader(val_dataset, **cfg.val_loader, collate_fn=collate_fn)

    return train_loader, val_loader


def initialize_featurizer(cfg: DictConfig) -> tp.Tuple[nn.Module, nn.Module]:
    """
    Initializes direct and inverse featurizers for audio.
    """
    featurizer = instantiate(
        cfg.featurizer.direct_transform,
    )
    inv_featurizer = instantiate(
        cfg.featurizer.inverse_transform,
    )
    multi_featurizer = []
    for window_size in cfg.featurizer.multi_win_length:
        n_fft = max(window_size, cfg.featurizer.multi_n_fft)
        multi_featurizer.append(
            instantiate(
                cfg.featurizer.multi_transform, n_fft=n_fft, win_length=window_size
            ).to("cuda")
        )

    return featurizer, inv_featurizer, torch.nn.ModuleList(multi_featurizer)


def initialize_augmentations(cfg: DictConfig) -> nn.Module:
    """
    Initializes augmentations.
    """
    augs = instantiate(cfg.augmentations)
    augs = nn.Sequential(*augs.values())

    return augs


def initialize_model(
    cfg: DictConfig,
) -> tp.Tuple[nn.Module, Optimizer, lr_scheduler._LRScheduler]:
    """
    Initializes model from configuration file.
    """
    # initialize model
    num_stems = len(cfg.train_dataset.target)
    if cfg.model_type == "bandsplitmodel":
        model = BSModel(**cfg.model, num_stems=num_stems, audio_cfg=cfg.featurizer)
    else:
        print(cfg.model_type)
        raise ValueError(
            f"{cfg.model_type} is not valid."
        )

    # initialize optimizer
    opt = instantiate(cfg.opt, params=model.parameters())

    # initialize scheduler
    if hasattr(cfg, "sch"):
        # other than LambdaLR
        sch = instantiate(cfg.sch, optimizer=opt)
    else:
        sch = None
    return model, opt, sch


def initialize_utils(cfg: DictConfig):
    # change model and logs saving directory to logging directory of hydra
    if HydraConfig.instance().cfg is not None:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        save_dir = hydra_cfg["runtime"]["output_dir"]
        cfg.logger.save_dir = save_dir + cfg.logger.save_dir
        if hasattr(cfg.callbacks, "model_ckpt"):
            cfg.callbacks.model_ckpt.dirpath = (
                save_dir + cfg.callbacks.model_ckpt.dirpath
            )
    # delete early stopping if there is no validation dataset
    if not hasattr(cfg, "val_dataset") and hasattr(cfg.callbacks, "early_stop"):
        del cfg.callbacks.early_stop
    # initialize logger and callbacks
    logger = instantiate(cfg.logger)
    callbacks = list(instantiate(cfg.callbacks).values())
    return logger, callbacks


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    pl.seed_everything(42, workers=True)

    torch.set_float32_matmul_precision("high")

    log.info(OmegaConf.to_yaml(cfg))

    # learning rate Scaling Rule
    # cfg.opt.lr *=  (
    #     cfg.train_loader.batch_size * cfg.trainer.accumulate_grad_batches / 128
    # )

    log.info("Initializing loaders, featurizers.")
    train_loader, val_loader = initialize_loaders(cfg)

    featurizer, inverse_featurizer, multi_featurizer = initialize_featurizer(cfg)
    augs = initialize_augmentations(cfg)

    log.info("Initializing model, optimizer, scheduler.")
    model, opt, sch = initialize_model(cfg)

    log.info("Initializing Lightning logger and callbacks.")
    logger, callbacks = initialize_utils(cfg)

    log.info("Initializing Lightning modules.")
    plmodel = PLModel(
        model,
        featurizer,
        inverse_featurizer,
        multi_featurizer,
        augs,
        opt,
        sch,
        cfg,
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    log.info("Starting training...")

                    
    trainer.fit(
        plmodel,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path,
    )


    log.info("Training finished!")


if __name__ == "__main__":
    my_app()
