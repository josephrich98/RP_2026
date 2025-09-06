import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from sklearn.model_selection import KFold


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
        self.last_val_score = None

    def fit(self, model: nn.Module, train_dataset, val_dataset=None):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            total_loss = 0
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Log training loss
            if self.writer:
                self.writer.add_scalar("Loss/train", total_loss, epoch)
            self.logger.info(f"Epoch {epoch}/{self.epochs}, Train Loss: {total_loss:.4f}")

            # --- Validation phase ---
            if val_dataset is not None:
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
                model.eval()
                correct, total, val_loss = 0, 0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        preds = model(xb)
                        loss = criterion(preds, yb)
                        val_loss += loss.item()
                        correct += (preds.argmax(dim=1) == yb).sum().item()
                        total += yb.size(0)
                val_acc = correct / total
                self.last_val_score = val_acc

                if self.writer:
                    self.writer.add_scalar("Loss/val", val_loss, epoch)
                    self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.logger.info(f"Epoch {epoch}/{self.epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    @torch.no_grad()
    def evaluate(self, model, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                preds = model(xb)
                predicted = preds.argmax(dim=1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        accuracy = correct / total
        return accuracy
    
    @torch.no_grad()
    def predict(self, model, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size)
        model.eval()
        all_preds = []
        for xb, _ in loader:
            preds = model(xb)
            all_preds.append(preds.argmax(dim=1))
        return torch.cat(all_preds)
