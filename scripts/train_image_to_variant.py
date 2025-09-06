import logging
import os
from datetime import datetime
import time
import sys
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from RP_2026.logger_utils import setup_logger_and_tensorboard, set_seed
from RP_2026.trainers.utils import cross_validate
from RP_2026.data.toy import make_toy_dataset
from pdb import set_trace as st  # noqa: F401

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("run_name", None)
    logger, writer = setup_logger_and_tensorboard(run_name=run_name, log_dir=log_dir, base_dir=base_dir, tensorboard=cfg.get("tensorboard", True))

    cmd = " ".join(sys.argv)
    logger.info(f"Command: {cmd}")
    
    logger.info("Config:\n%s", cfg)

    start_time = datetime.now()
    logger.info(f"Run started at {start_time}")

    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Using seed: {seed}")

    if "_target_" in cfg.dataset:
        train_dataset, val_dataset, test_dataset = hydra.utils.instantiate(cfg.dataset, seed=cfg.seed)
    # else:
    #     # Assume pre-split datasets exist on disk
    #     train_dataset = load_dataset(cfg.dataset.train_path)
    #     val_dataset = load_dataset(cfg.dataset.val_path)
    #     test_dataset = load_dataset(cfg.dataset.test_path)


    # --- Build model and trainer from Hydra configs ---
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, writer=writer)

    # --- Run training ---
    # trainer.fit(model, dataset)
    valid_modes = {"train", "test", "infer"}
    
    logger.info(f"Mode = {cfg.mode}")
    if cfg.mode == "train":
        if val_dataset is None:
            logger.warning("No validation dataset provided. Training without validation.")
        if not cfg.get("crossval", True) and val_dataset is not None:
            logger.warning("crossval is False. Will not perform cross-validation.")
            val_dataset = None
        trainer.fit(model, train_dataset, val_dataset)
    elif cfg.mode == "test":
        test_score = trainer.evaluate(model, test_dataset)
        logger.info(f"Test score: {test_score:.4f}")
    elif cfg.mode == "infer":
        preds = trainer.predict(model, test_dataset)
        logger.info(f"Inference done: {len(preds)} samples predicted")
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Valid options are: {valid_modes}")

    end_time = datetime.now()
    logger.info(f"Run finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total runtime: {end_time - start_time}")

    writer.close()

if __name__ == "__main__":
    main()
