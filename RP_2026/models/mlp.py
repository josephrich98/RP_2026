import torch.nn as nn

class MLP(nn.Module):
    """
    A simple multi-layer perceptron with one hidden layer.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.net(x)
