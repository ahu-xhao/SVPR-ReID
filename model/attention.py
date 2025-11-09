# coding=utf-8
'''
@Time     : 2024/09/22 13:17:51
@Author   : XHao
@Email    : 2510383889@qq.com
'''
# here put the import lib

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, output_dim=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()

        hidden_dim = hidden_dim or in_dim
        output_dim = output_dim or in_dim
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        self.norm_layer = norm_layer(in_dim)
        self.layers = nn.ModuleList()
        self.activation = act_layer()
        self.drop = nn.Dropout(drop)

        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(in_dim, hidden_dim[0]))

        # 隐藏层之间
        for i in range(1, len(hidden_dim)):
            self.layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))

        # 最后一个隐藏层到输出层
        self.layers.append(nn.Linear(hidden_dim[-1], output_dim))

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x, norm=False):
        if norm:
            x = self.norm_layer(x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.drop(x)
        x = self.layers[-1](x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # mask = torch.ones((B, 1, N, N), dtype=torch.bool).cuda()
        # attn = attn.masked_fill(~mask.bool(), torch.tensor(-1e3, dtype=torch.float16))  # mask
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v)
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False
        )  # [B, num_heads, N_q, head_dim]
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaskAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.ones((B, 1, N, N), dtype=torch.bool).cuda()
        attn = attn.masked_fill(~mask.bool(), torch.tensor(-1e3, dtype=torch.float16))  # mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.tem = TokenEnhancement_CNN(monument=0.1)

    def forward(self, x, use_attention=True, use_mlp=True):
        if use_attention:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        # else:
        #     x = x + self.drop_path(self.tem(self.norm1(x)))
        if use_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim,  num_heads=8, downsample_rate: int = 1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.normy = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.hidden_dim = dim // downsample_rate
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, self.hidden_dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, self.hidden_dim, bias=qkv_bias)
        self.v_ = nn.Linear(dim, self.hidden_dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(self.hidden_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v=None,return_attn=False):
        q_dim = q.dim()
        k_dim = k.dim()

        if q_dim == 2:q = q.unsqueeze(1)
        if k_dim == 2:k = k.unsqueeze(1)
        if v is None: v = k
        if v.dim() == 2: v = v.unsqueeze(1)

        B_q, N_q, _ = q.shape
        B_k, N_k, _ = k.shape
        C = self.hidden_dim

        q = self.q_(q).reshape(B_q, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(k).reshape(B_k, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_(v).reshape(B_k, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v)
        # ⚡️ scaled_dot_product_attention automatically scales & applies softmax/dropout
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False
        )  # [B, num_heads, N_q, head_dim]
        
        x = x.transpose(1, 2).reshape(B_q, N_q, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, k, v=None, use_mlp=True,return_attn=False):
        if return_attn:
            q, attn = self.attn(self.norm1(q), k, v if v is not None else k, return_attn=True)
        else:
            q = self.attn(self.norm1(q), k, v if v is not None else k, return_attn=False)
        q = q + self.drop_path(q)
        if use_mlp:
            q = q + self.drop_path(self.mlp(self.norm2(q)))
        if return_attn:
            return q, attn
        else:
            return q


class ViewAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.CAB = CrossAttentionBlock(dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                                       attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

    def forward(self, x, y, use_mlp=True):
        assert x.dim() == 3 and y.dim() == 3, 'Input tensor dim must be 3. but got x: {}, y: {}'.format(x.shape, y.shape)
        x_cls = self.CAB(x[:, :1, :], y[:, 1:, :],y[:, 1:, :], use_mlp)
        y_cls = self.CAB(y[:, :1, :], x[:, 1:, :],x[:, 1:, :], use_mlp)
        x = torch.cat([x_cls, x[:, 1:, :]], dim=-2)
        y = torch.cat([y_cls, y[:, 1:, :]], dim=-2)
        return x, y


class SCAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=mlp_hidden_dim)
        self.norm3 = norm_layer(dim)

    def forward_self_cross(self, q, k, v=None):
        _, N_q, _ = q.shape
        # self attention
        q_self = torch.cat((q, k), dim=1)
        self_out = self.attn(q_self)
        queries = self_out[:, :N_q, :]
        queries = q + queries
        # cross attention
        k_cross = self_out[:, N_q:, :]
        q_cross = self.norm1(queries)
        cross_out = self.cross_attn(q_cross, k_cross)
        queries = queries + cross_out

        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        return queries

    def forward_cross_self(self, q, k, v):
        attn_out = self.cross_attn(q, k)
        queries = q + attn_out
        queries = self.norm1(queries)
        queries = queries + self.attn(queries)
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        return queries
    
    def forward_self(self, q, k=None, v=None):
        queries = self.norm1(q)
        queries = queries + self.attn(queries)
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        return queries
    
    def forward_cross(self, q, k, v=None):
        attn_out = self.cross_attn(q, k)
        queries = q + attn_out
        queries = self.norm1(queries)
        
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        
        return queries
    
    def forward(self, q, k, v):
        return self.forward_cross_self(q, k, v)


class PromptRecapBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 activation=nn.ReLU,
                 mlp_ratio=4.0,
                 method='attn'):
        super().__init__()
        self.cross_attn = CrossAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn = Attention(embedding_dim, num_heads)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = MLP(in_dim=embedding_dim, hidden_dim=mlp_hidden_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.method = method

        self.attn2 = Attention(embedding_dim, num_heads)

    def forward(self, queries, keys):
        if self.method == 'cat':
            queries = torch.cat((queries, keys), dim=1)
            attn_out = self.attn2(queries)
            queries = queries + attn_out
            queries = self.norm1(queries)
        elif self.method == 'add':
            queries = queries + keys
            queries = self.norm1(queries)
        elif self.method == 'attn':
            attn_out = self.cross_attn(q=queries, k=keys, v=keys)
            queries = queries + attn_out
            queries = self.norm1(queries)
        else:
            attn_out = self.cross_attn(q=queries, k=keys, v=keys)
            queries = queries + attn_out
            queries = self.norm1(queries)

        attn_out = self.attn(queries)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        if self.method == 'cat':
            k_len = keys.shape[1]
            # print(queries[:-k_len].shape)
            return queries[:, :-k_len]
        return queries


class PromptBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int,         # 输入channel
                 num_heads: int,             # attention的head数
                 mlp_ratio: float = 4.0,        # MLP中间channel
                 activation=nn.ReLU,      # 激活层
                 attention_downsample_rate: int = 2,         # 下采样
                 skip_first_layer_pe: bool = False,):
        super().__init__()
        self.cross_attn = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        # self.cross_attn = CrossAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.self_attn = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        # self.self_attn = CrossAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = MLP(in_dim=embedding_dim, hidden_dim=mlp_hidden_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,         # 输入channel
        num_heads: int,             # attention的head数
        mlp_ratio: float = 4.0,        # MLP中间channel
        activation=nn.ReLU,      # 激活层
        attention_downsample_rate: int = 2,         # 下采样
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        # self.self_attn = CrossAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        # self.cross_attn_token_to_image = CrossAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.mlp = MLP(in_dim=embedding_dim, hidden_dim=mlp_hidden_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        # self.cross_attn_image_to_token = CrossAttention(embedding_dim, num_heads)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):

        # queries：标记点编码相关(原始标记点编码经过一系列特征提取)
        # keys：原始图像编码相关(原始图像编码经过一系列特征提取)
        # query_pe：原始标记点编码
        # key_pe：原始图像位置编码
        # 第一轮本身queries==query_pe没比较再"残差"
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        # 层数
        depth: int,
        # 输入channel
        embedding_dim: int,
        # attention的head数
        num_heads: int,
        # MLP内部channel
        mlp_ratio: float,
        activation=nn.ReLU,
        attention_downsample_rate: int = 2,
        use_to_way=True,
        out_method=None,
    ) -> None:
        super().__init__()
        self.depth = depth      # 层数
        self.embedding_dim = embedding_dim          # 输入channel
        self.num_heads = num_heads                  # attention的head数
        self.mlp_ratio = mlp_ratio                      # MLP内部隐藏channel
        self.layers = nn.ModuleList()
        for i in range(depth):
            if use_to_way:
                self.layers.append(
                    TwoWayAttentionBlock(
                        embedding_dim=embedding_dim,    # 输入channel
                        num_heads=num_heads,            # attention的head数
                        mlp_ratio=mlp_ratio,                # MLP中间channel
                        activation=activation,          # 激活层
                        attention_downsample_rate=attention_downsample_rate,      # 下采样
                        skip_first_layer_pe=(i == 0),
                    )
                )
            else:
                self.layers.append(
                    PromptBlock(
                        embedding_dim=embedding_dim,    # 输入channel
                        num_heads=num_heads,            # attention的head数
                        mlp_ratio=mlp_ratio,                # MLP中间channel
                        activation=activation,          # 激活层
                        attention_downsample_rate=attention_downsample_rate,      # 下采样
                        skip_first_layer_pe=(i == 0),
                    )
                )
        self.out_method = out_method
        if out_method is None:
            self.final_attn_token_to_image = SCAttentionBlock(embedding_dim, num_heads, mlp_ratio=self.mlp_ratio)
        elif out_method == 'Atten':
            self.final_attn_token_to_image = CrossAttention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.norm_final_attn = nn.LayerNorm(embedding_dim)
        self.out_token = nn.Parameter(torch.zeros(1, self.embedding_dim))
        trunc_normal_(self.out_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        image_embedding,
        image_pe,
        point_embedding,
        modality_flag
    ):
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, l, c = image_embedding.shape
        # 图像编码(image_encoder的输出)
        # BxHWxC=>B,N,C
        # image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        # 图像位置编码
        # BxHWxC=>B,N,C
        # image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # 标记点编码
        # B,N,C
        queries = point_embedding
        queries_len = queries.shape[1]
        keys = image_embedding
        # -----TwoWayAttention-----

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        # -----TwoWayAttention-----

        q = queries + point_embedding
        # out_token = self.out_token.unsqueeze(0).expand(q.size(0), -1, -1)
        # q = torch.cat((out_token, q), dim=1)
        k = keys + image_pe
        if self.out_method == 'no_out':
            return queries, keys
        # -----Attention-----
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        # -----Attention-----
        queries = queries + attn_out
        # queries = attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class AttentionScoreMask(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        # self.linear_re = nn.Sequential(nn.Linear(dim, dim), QuickGELU(), nn.BatchNorm1d(dim))
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, q, k, top_k=1., method='mean'):
        B, Nq, C = q.shape
        B, Nk, C = k.shape
        query = self.q_(q).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = self.k_(k).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)   # (B, num_heads, Nx, Ny)
        # attn = attn[:, :, :1, :]  # (B, num_heads, 1, Ny)
        mask, topk_indices = self.gather_multihead_attn(attn, topk=top_k, method=method)  # mask：(B, Nx, Ny)
        mask = mask[:, 0, :]  # (B, Ny)
        return mask, topk_indices

    def gather_multihead_attn(self, attn, topk=1., method='mean'):
        B, H, Nq, Nkv = attn.shape
        mask = None
        if method == 'mean':
            scroes = torch.mean(attn, dim=1)  # (B, Nx, Ny)
            _, topk_indices = torch.topk(scroes, int(Nkv * topk), dim=2)
            selected_tokens_mask = torch.zeros((B, Nq, Nkv), dtype=torch.bool).cuda()
            selected_tokens_mask.scatter_(2, topk_indices, 1)
            mask = selected_tokens_mask
            return mask, topk_indices

        elif method == 'union':
            # attn = attn[:, :, 0, :]  # Nq==1
            for i in range(H):
                _, topk_indices = torch.topk(attn[:, i, :, :], int(Nkv * topk), dim=2)
                topk_indices = torch.sort(topk_indices, dim=2).values
                selected_tokens_mask = torch.zeros((B, Nq, Nkv), dtype=torch.bool).cuda()
                selected_tokens_mask.scatter_(2, topk_indices, 1)
                if i == 0:
                    mask = selected_tokens_mask
                else:
                    mask = mask | selected_tokens_mask
            return mask, None


if __name__ == '__main__':
    
    CA = CrossAttentionBlock(768)
    q = torch.randn(128, 1, 768)
    kv = torch.randn(128, 128, 768)
    outputs = CA(q, kv)
    print(q == outputs)
    print(kv == outputs)
