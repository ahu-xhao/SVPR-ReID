# coding=utf-8
'''
@Time     : 2025/04/08 11:30:55
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MLPExpert(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4.):
        super().__init__()
        self.mid_dim = int(input_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.mid_dim),
            QuickGELU(),
            nn.Linear(self.mid_dim, input_dim),
            QuickGELU(),
            # nn.BatchNorm1d(input_dim),
            nn.LayerNorm(input_dim, eps=1e-6)  # Use LayerNorm instead of BatchNorm
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class MOE_simple(nn.Module):
    def __init__(self, input_dim, num_experts, topk: int | float = 0.5):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([MLPExpert(input_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.topk = int(self.num_experts * topk) if isinstance(topk, float) else topk

    def forward(self, x):
        B, C = x.shape
        route_scores = self.gate(x)                  # [B, num_experts]
        route_scores = route_scores.softmax(dim=-1)  # [B, num_experts]
        topk_scores, topk_indices = torch.topk(route_scores, self.topk, dim=-1)  # [B, top_k]

        expert_outputs = [expert(x) for expert in self.experts]  # list of [B, C]
        expert_outputs = torch.stack(expert_outputs, dim=1)      # [B, num_experts, C]

        topk_expert_outputs = expert_outputs.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, C)
        )

        topk_weights = topk_scores.unsqueeze(-1)  # [B, top_k, 1]
        weighted_output = (topk_expert_outputs * topk_weights).sum(dim=1)

        return weighted_output
