import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionReduce(nn.Module):
    def __init__(self, dim, mid_dim, glimpses):
        super(AttentionReduce, self).__init__()
        self.mlp = nn.Sequential(
            *[
                nn.Linear(dim, mid_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mid_dim, glimpses),
            ]
        )
        self.glimpses = glimpses

    def forward(self, x, y):
        x_reduced = self.mlp(x)
        x_reduced = F.softmax(x_reduced, dim=1)

        att_list = []
        for i in range(self.glimpses):
            att_list.append(torch.sum(x_reduced[:, :, i : i + 1] * y, dim=1))
        x_atted = torch.cat(att_list, dim=1)

        return x_atted


class PJF(nn.Module):
    def __init__(self, input_dim=768, num_heads=16, output_dim=4096, dropout=0.4):
        super(PJF, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            input_dim=input_dim, num_heads=num_heads, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
        )
        self.attention_2 = nn.MultiheadAttention(
            input_dim=input_dim, num_heads=num_heads, dropout=dropout
        )
        self.atte_reduce = AttentionReduce(input_dim, input_dim // 2, 1)

        self.ffn_2 = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
        )

        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, output_dim),
        )

    def forward(self, x):
        y, _ = self.attention(x, x, x)
        x = self.layer_norm(x + y)
        x = self.ffn(x)
        y, _ = self.attention_2(x, x, x)
        x = self.atte_reduce(y, x)
        x = self.layer_norm(x + self.dropout(self.ffn_2(x)))
        x = self.fc(x)
        return x


class PJFTrainModel(nn.Module):
    def __init__(self, input_dim=768, num_heads=16, output_dim=4096, dropout=0.4):
        super(PJFTrainModel, self).__init__()
        self.model_1 = PJF(input_dim, num_heads, output_dim, dropout)
        self.model_2 = PJF(input_dim, num_heads, output_dim, dropout)

    def forward(self, x, y):
        x = self.model_1(x)
        y = self.model_2(y)

        return (x - y).square()
