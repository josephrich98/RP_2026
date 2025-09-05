import torch.nn as nn

class LogisticRegression(nn.Module):
    """
    Simple logistic regression model using a single linear layer.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
