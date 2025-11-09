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
from .attention import CrossAttentionBlock, SCAttentionBlock


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class simpleNet(nn.Module):
    def __init__(self, input_dim):
        super(simpleNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            QuickGELU(),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Expert(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4.):
        super(Expert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * mlp_ratio),
            QuickGELU(),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class ExpertHead(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(ExpertHead, self).__init__()
        self.expertHead = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])

    def forward(self, x_chunk, gate_head):
        expert_outputs = [expert(x_chunk[i]) for i, expert in enumerate(self.expertHead)]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_outputs = expert_outputs * gate_head.squeeze(1).unsqueeze(2)
        return expert_outputs


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, y):
        _, N_x, C = x.shape
        B, N, C = y.shape
        # x = self.linear_re(x)
        q = self.q_(x).reshape(B, N_x, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        gates = attn.softmax(dim=-1)
        return gates

    def forward_(self, x):
        x = self.direct_gate(x)
        return x.unsqueeze(1)


class GatingNetwork_CA(nn.Module):
    def __init__(self, input_dim, head):
        super().__init__()
        self.gate = CrossAttention(input_dim, head)

    def forward(self, x, y):
        x_i = torch.chunk(x, x.shape[1], dim=1)
        gates = torch.stack([self.gate(x_i[i], y) for i in range(x.shape[1])], dim=1)
        return gates


class GatingNetwork_linear(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.gate = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        gates = self.gate(x)
        return gates


class MoE_head(nn.Module):
    def __init__(self, input_dim, num_experts, head=8):
        super().__init__()
        self.head_dim = input_dim // head
        self.head = head
        self.experts = nn.ModuleList([ExpertHead(self.head_dim, num_experts) for _ in range(head)])
        self.gating_network = GatingNetwork_CA(input_dim, head)

        self.linear_re = nn.Sequential(nn.Linear(4 * input_dim, input_dim), QuickGELU(), nn.BatchNorm1d(input_dim))

    def forward(self, x, q=None):
        emd_dim = self.head_dim * self.head
        xi_list = torch.chunk(x, emd_dim, dim=1)
        xi_chunks = [torch.chunk(img, self.head, dim=-1) for img in xi_list]
        head_input = [[img_chunk[i] for img_chunk in xi_chunks] for i in range(self.head)]

        query = torch.cat(xi_list, dim=-1).squeeze(1)
        query = self.linear_re(query).unsqueeze(1)
        key = torch.cat(xi_list, dim=1)
        gate_heads = self.gating_network(query, key)

        expert_outputs = [expert(head_input[i], gate_heads[:, i]) for i, expert in enumerate(self.experts)]
        outputs = torch.cat(expert_outputs, dim=-1).flatten(start_dim=1, end_dim=-1)

        return outputs


class AttentionExpert(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()

        # self.expert = CrossAttentionBlock(
        #     dim=dim,
        #     num_heads=num_heads,
        #     qkv_bias=False,
        #     qk_scale=scale,
        # )

        self.expert = SCAttentionBlock(
            dim=dim,
            num_heads=num_heads,
        )

        # self.expert = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     QuickGELU(),
        #     nn.BatchNorm1d(dim),
        # )

    def forward(self, q, x):
        x = self.expert(q, x, x)
        return x


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


class Router(nn.Module):
    def __init__(self, dim, num_heads=12, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.router = CrossAttention(dim, num_heads=num_heads)

    def forward(self, x, y):
        self.router(x, y)
        return x


class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, expert_type='linear'):
        super().__init__()
        if expert_type == 'linear':
            self.experts = nn.ModuleList([MLPExpert(input_dim, num_heads=8) for _ in range(num_experts)])
        else:
            self.experts = nn.ModuleList([AttentionExpert(input_dim, num_heads=8) for _ in range(num_experts)])
        self.gating_network = Router(input_dim, num_heads=8)

    def forward_attnExpert(self, x, q, top_k=1.):
        B, N, C = x.shape
        expert_outputs = torch.cat([expert(q, x[:, i:i + 1, :])
                                   for i, expert in enumerate(self.experts)], dim=1)  # [B, num_experts, C]
        gate_scores = self.gating_network(x, q)  # [B, num_experts, C]
        route_scores = gate_scores.mean(dim=-1)  # [B, num_experts]

        top_k = int(N * 0.5)
        _, topk_indices = torch.topk(route_scores, top_k, dim=-1)  # [B, top_k]

        topk_expert_outputs = expert_outputs.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(2)))  # [B, top_k, C]
        topk_route_scores = gate_scores.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, gate_scores.size(2)))  # [B, top_k, C]

        # weighted_expert_output = topk_expert_outputs * topk_route_scores  # [B, top_k, C] 加权输出
        weighted_expert_output = torch.einsum('bkh,bkc->bkc', topk_route_scores, topk_expert_outputs)  # [B, top_k, C] 加权输出

        outputs = weighted_expert_output.mean(dim=1, keepdim=True)  # [B, C] 可以取平均或者其他方式合并
        return outputs

    def forward_mlpExpert(self, x, q=None, top_k=1.):
        B, N, C = x.shape
        expert_outputs = torch.cat([expert(q, x[:, i:i + 1, :])
                                   for i, expert in enumerate(self.experts)], dim=1)  # [B, num_experts, C]
        gate_scores = self.gating_network(x, q)  # [B, num_experts, C]
        route_scores = gate_scores.mean(dim=-1)  # [B, num_experts]

        top_k = int(N * 0.5)
        _, topk_indices = torch.topk(route_scores, top_k, dim=-1)  # [B, top_k]

        topk_expert_outputs = expert_outputs.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(2)))  # [B, top_k, C]
        topk_route_scores = gate_scores.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, gate_scores.size(2)))  # [B, top_k, C]

        # weighted_expert_output = topk_expert_outputs * topk_route_scores  # [B, top_k, C] 加权输出
        weighted_expert_output = torch.einsum('bkh,bkc->bkc', topk_route_scores, topk_expert_outputs)  # [B, top_k, C] 加权输出

        outputs = weighted_expert_output.mean(dim=1, keepdim=True)  # [B, C] 可以取平均或者其他方式合并
        return outputs


class MOE_simple(nn.Module):
    def __init__(self, input_dim, num_experts, topk: int | float = 0.5):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([MLPExpert(input_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.topk = int(self.num_experts * topk) if isinstance(topk, float) else topk

    def forward_error(self, x):
        B, C = x.shape
        # 计算门控网络的输出，得到每个专家的权重
        route_scores = self.gate(x)  # (B, num_experts)
        route_scores = route_scores.softmax(dim=-1)  # 归一化权重

        # top_k = int(self.num_experts * 0.5)
        _, topk_indices = torch.topk(route_scores, self.topk, dim=-1)  # [B, top_k]
        # 获取每个专家的输出
        expert_outputs = [expert(x) for expert in self.experts]  # 每个专家的输出 (B, input_dim)
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (B, input_dim, num_experts)
        topk_expert_outputs = expert_outputs.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(2)))  # [B, top_k, C]
        topk_route_scores = route_scores.gather(1, topk_indices)  # [B, top_k, C]

        # weighted_expert_output = topk_expert_outputs * topk_route_scores  # [B, top_k, C] 加权输出
        weighted_expert_output = torch.einsum('bkh,bkc->bkc', topk_route_scores, topk_expert_outputs)  # [B, top_k, C] 加权输出

        return weighted_expert_output

    def forward(self, x):
        B, C = x.shape
        # 计算路由分数
        route_scores = self.gate(x)                  # [B, num_experts]
        route_scores = route_scores.softmax(dim=-1)  # [B, num_experts]

        # Top-k 路由索引
        topk_scores, topk_indices = torch.topk(route_scores, self.topk, dim=-1)  # [B, top_k]

        # 所有专家前向（共用同一个输入）
        expert_outputs = [expert(x) for expert in self.experts]  # list of [B, C]
        expert_outputs = torch.stack(expert_outputs, dim=1)      # [B, num_experts, C]

        # gather top-k 输出：[B, top_k, C]
        topk_expert_outputs = expert_outputs.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, C)
        )

        # top-k scores reshape 为 [B, top_k, 1] 用于加权
        topk_weights = topk_scores.unsqueeze(-1)  # [B, top_k, 1]

        # 加权求和：最终输出 [B, C]
        weighted_output = (topk_expert_outputs * topk_weights).sum(dim=1)

        return weighted_output

