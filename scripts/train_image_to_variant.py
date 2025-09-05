import logging
import os
from datetime import datetime
import time
import sys
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from RP_2026.logger_utils import setup_logger_and_tensorboard, set_seed
from pdb import set_trace as st  # noqa: F401

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("run_name", None)
    logger, writer = setup_logger_and_tensorboard(run_name=run_name, log_dir=log_dir, base_dir=base_dir)

    cmd = " ".join(sys.argv)
    logger.info(f"Command: {cmd}")
    
    logger.info("Config:\n%s", cfg)

    start_time = datetime.now()
    logger.info(f"Run started at {start_time}")

    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Using seed: {seed}")

    # --- Dummy dataset (replace later with real loader) ---
    X = torch.randn(1000, cfg.dataset.n_features)
    y = torch.randint(0, cfg.dataset.n_classes, (1000,))
    dataset = TensorDataset(X, y)

    # --- Build model and trainer from Hydra configs ---
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, writer=writer)

    # --- Run training ---
    trainer.fit(model, dataset)

    end_time = datetime.now()
    logger.info(f"Run finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total runtime: {end_time - start_time}")

    writer.close()

if __name__ == "__main__":
    main()
