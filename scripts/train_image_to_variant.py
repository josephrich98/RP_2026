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
from pdb import set_trace as st  # noqa: F401

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

def setup_logger_and_tensorboard(run_name = None):
    start_time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    if run_name is None or run_name == "":
        run_name = f"run_{start_time_string}"

    log_file_path = os.path.join(log_dir, f"{run_name}.log")

    if os.path.exists(log_file_path):
        raise FileExistsError(f"Log file {log_file_path} already exists. Please choose a different run name.")
    print(f"Logging to {log_file_path}")

    tensorboard_dir = os.path.join(base_dir, "runs", run_name)
    if os.path.exists(tensorboard_dir):
        print(f"Warning: TensorBoard log directory {tensorboard_dir} already exists and may be overwritten.")
    writer = SummaryWriter(tensorboard_dir)
    os.makedirs(tensorboard_dir, exist_ok=True)
    print(f"TensorBoard logs to {tensorboard_dir} - visualize with `tensorboard --logdir {tensorboard_dir}`")

    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel("INFO")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, writer


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("run_name", None)
    logger, writer = setup_logger_and_tensorboard(run_name)

    cmd = " ".join(sys.argv)
    logger.info(f"Command: {cmd}")
    
    logger.info("Config:\n%s", cfg)

    start_time = datetime.now()
    logger.info(f"Run started at {start_time}")

    # --- Dummy dataset (replace with your real loader later) ---
    X = torch.randn(1000, cfg.dataset.n_features)
    y = torch.randint(0, cfg.dataset.n_classes, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=cfg.trainer.batch_size, shuffle=True)

    # --- Build model from Hydra config ---
    model = hydra.utils.instantiate(cfg.model)
    if model == "pupl":
        from pupl import PairedUnpairedTrainer  # local import to only if needed

         # --- Build trainer from Hydra config ---
        trainer = PairedUnpairedTrainer(
            epochs=cfg.trainer.epochs,
            lr=cfg.trainer.lr,
            batch_size=cfg.trainer.batch_size,
            logger=logger,
            writer=writer,
        )

        # --- Run training ---
        trainer.fit(model, loader)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.trainer.lr)

        # --- Training loop ---
        for epoch in range(cfg.trainer.epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            writer.add_scalar("Loss/train", total_loss, epoch+1)
            logger.info(f"Epoch {epoch+1}/{cfg.trainer.epochs}, Loss: {total_loss:.4f}")
    
    end_time = datetime.now()
    logger.info(f"Run finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total runtime: {end_time - start_time}")


if __name__ == "__main__":
    main()
