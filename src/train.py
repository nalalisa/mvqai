from __future__ import annotations

import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf


def build_logger(cfg: DictConfig):
    if cfg.logger.backend == "wandb":
        return WandbLogger(
            project=cfg.logger.project,
            save_dir=cfg.logger.save_dir,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    return CSVLogger(save_dir=cfg.logger.save_dir, name="csv_logs")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.dataset)
    task = instantiate(cfg.task, _recursive_=False)
    logger = build_logger(cfg)
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:03d}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
