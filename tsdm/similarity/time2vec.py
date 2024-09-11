import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(Time2Vec, self).__init__()
        self.output_size = output_size
        self.linear = nn.Linear(input_size, 1, bias=False)
        self.sinusoid = nn.Linear(input_size, output_size - 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear transformation
        lin = self.linear(x)
        # Apply the sinusoidal transformation
        sin = torch.sin(self.sinusoid(x))
        return torch.cat([lin, sin], dim=-1)