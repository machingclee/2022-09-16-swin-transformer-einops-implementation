import numbers
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from typing import Tuple


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim):
        super(PatchMerging, self).__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.resolution
        x = rearrange(x, "b (h w) c -> b h w c", h=h, w=w)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = rearrange(x, "b h w c2 -> b (h w) c2")
        x = self.norm(x)
        x = self.reduction(x)

        return x


class MLP(nn.Sequential):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.):
        super(MLP, self).__init__(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )


def windows_partition(x, window_size):
    x = rearrange(
        x,
        "b (win_size1 h2) (win_size2 w2) c -> (b h2 w2) (win_size1 win_size2) c",
        win_size1=window_size,
        win_size2=window_size
    )
    return x


def windows_reverse(windows, window_size, h, w):
    h2 = h // window_size
    w2 = w // window_size
    # logically we should reverse to the shape "b (win_size1 h2) (win_size2 w2) c",
    # but technically we are going to add the result with skip connection,
    # therefore we reshape directly to "b (win_size1 h2 win_size2 w2) c"
    x = rearrange(
        windows,
        "(b h2 w2) (win_size1 win_size2) c -> b (win_size1 h2 win_size2 w2) c",
        win_size1=window_size,
        win_size2=window_size,
        h2=h2,
        w2=w2
    )
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(-1)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [b, num_img_tokens, embed_dim]
        x = self.qkv(x)
        qkv = rearrange(x, "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = torch.einsum("bhqd, bhkd -> bhqk", q, k)  # attn = Q * K^T
        attn = self.softmax(attn)
        out = torch.einsum("bhai, bhid -> bhad", attn, v)  # attn * V
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size):
        super(SwinBlock, self).__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        # x: [b, n, d]
        H, W = self.resolution
        h = x
        x = self.attn_norm(x)
        x = rearrange(x, "b (h w) c -> b h w c", h=H, w=W)
        x_windows = windows_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        attn_windows = windows_reverse(attn_windows, window_size=self.window_size, h=H, w=W)
        x = h + attn_windows

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x


if __name__ == "__main__":
    t = torch.randn([4, 3, 224, 224])
    patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    swin_block = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7)
    patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)

    out = patch_embedding(t)  # result: [4, 56*56, 96]
    out = swin_block(out)     # result: [4, 56*56, 96]
    out = patch_merging(out)  # result: [4, 784, 192]
    print(out.shape)
