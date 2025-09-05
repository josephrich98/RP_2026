import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging


class SupervisedTrainer:
    """
    Generic supervised trainer for feedforward models
    (works with Logistic Regression, MLP, or any nn.Module
    returning class logits).
    """

    def __init__(self, epochs: int, lr: float, batch_size: int,
                 logger: logging.Logger = None,
                 writer: SummaryWriter = None):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        self.writer = writer

    def fit(self, model: nn.Module, dataset):
        """
        Train a model on the given dataset.

        Args:
            model (nn.Module): PyTorch model (LogisticRegression, MLP, etc.)
            dataset (torch.utils.data.Dataset): Dataset object
        """
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Logging
            if self.writer:
                self.writer.add_scalar("Loss/train", total_loss, epoch)
            self.logger.info(f"Epoch {epoch}/{self.epochs}, Loss: {total_loss:.4f}")
