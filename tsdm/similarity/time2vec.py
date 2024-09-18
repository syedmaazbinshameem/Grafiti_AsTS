import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    def __init__(self, seq_len, d):
        super(Time2Vec, self).__init__()
        self.seq_len = seq_len
        self.d = d
        
        self.time_linear = nn.Linear(1, d)
        self.time_periodic = nn.Parameter(torch.randn(1, d))

    def forward(self, time_steps):
        time_steps = time_steps.unsqueeze(-1)  #BxTx1
        
        linear_out = self.time_linear(time_steps)
        periodic_out = torch.sin(time_steps * self.time_periodic)
        
        return linear_out + periodic_out
