import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from sklearn.model_selection import KFold

def cross_validate(trainer, model_class, dataset, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        model = model_class(
            in_features=dataset.tensors[0].shape[1],
            out_features=len(torch.unique(dataset.tensors[1]))
        )

        trainer.fit(model, train_subset, val_subset)  # extend trainer to accept val data
        fold_results.append(trainer.last_val_score)

    return sum(fold_results) / len(fold_results)