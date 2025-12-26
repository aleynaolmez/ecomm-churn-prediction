import torch
import torch.nn as nn

class ChurnMLP(nn.Module):
    """Simple MLP for tabular churn prediction."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, 1)]  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # (batch,)
