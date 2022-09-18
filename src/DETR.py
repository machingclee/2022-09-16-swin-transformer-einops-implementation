from telnetlib import DET
from turtle import pos
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from typing import Tuple, Optional


class MLP(nn.Sequential):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.):
        super(MLP, self).__init__(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 #  window_size
                 ):
        super(Attention, self).__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(-1)

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, query, key, value):
        # x: [b, num_img_tokens, embed_dim]
        # mask: [n, ws*ws, ws*ws]
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        attn = torch.einsum("bhqd, bhkd -> bhqk", q, k)  # attn = Q * K^T
        attn = self.softmax(attn)
        attn = attn * self.scale

        out = torch.einsum("bhai, bhid -> bhad", attn, v)  # attn * V
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, mlp_ratio=4.0):
        super(EncoderLayer, self).__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x, pos=None):
        h = x
        x = self.attn_norm(x)
        q = x + pos if pos is not None else x
        k = x + pos if pos is not None else x
        x = self.attn(q, k, x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, mlp_ratio=4.0):
        super(DecoderLayer, self).__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.enc_dec_attn_norm = nn.LayerNorm(embed_dim)
        self.enc_dec_attn = Attention(embed_dim, num_heads)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x, enc_out, pos=None, query_pos=None):
        h = x
        x = self.attn_norm(x)
        q = x + query_pos if pos is not None else x
        k = x + query_pos if pos is not None else x
        x = self.attn(q, k, x)
        x = x + h

        h = x
        x = self.enc_dec_attn_norm(x)
        q = x + query_pos if pos is not None else x
        k = enc_out + pos if pos is not None else x
        v = enc_out
        x = self.attn(q, k, v)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h

        return x


class PositionEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.row_embed = nn.Parameter(torch.rand(50, embed_dim))
        self.col_embed = nn.Parameter(torch.rand(50, embed_dim))

    def forward(self, x):
        # (b, features, h, w)
        h, w = x.shape[-2:]

        pos = torch.cat([
            self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1),
            self.row_embed[:h].unsqueeze(1).repeat(1, w, 1)
        ], dim=-1)

        pos = pos.permute([2, 0, 1])
        pos = pos.unsqueeze(0)
        pos = pos.expand([x.shape[0]] + list(pos.shape[1::]))
        return pos


class Transformer(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, num_encoders=2, num_decoders=2):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.encoders = nn.ModuleList([EncoderLayer(embed_dim, num_heads) for _ in range(num_encoders)])
        self.decoders = nn.ModuleList([DecoderLayer(embed_dim, num_heads) for _ in range(num_decoders)])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, query_embed, pos_embed):
        B, _, _, _ = x.shape
        x = rearrange(x, "b c h w -> (h w) b c")
        pos_embed = rearrange(pos_embed, "b d h w -> (h w) b d")
        query_embed = repeat(query_embed.unsqueeze(1), "nq () d -> nq b d", b=B)  # nq = num_queries

        target = torch.zeros_like(query_embed)

        encoder_out = x

        for encoder in self.encoders:
            encoder_out = encoder(encoder_out, pos_embed)
        encoder_out = self.encoder_norm(encoder_out)

        decoder_out = target
        for decoder in self.decoders:
            decoder_out = decoder(target, encoder_out, pos_embed, query_embed)
        decoder_out = self.decoder_norm(decoder_out)

        encoder_out = rearrange(encoder_out, "(h w) b d -> b d h w")
        decoder_out = rearrange(decoder_out.unsequeeze(0), "() nq b d -> () b nq d")

        return encoder_out, decoder_out


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.num_channels = 512
        self.resnet_model = models.resnet18()
        self.net = nn.Sequential(
            self.resnet_model.conv1,
            self.resnet_model.bn1,
            self.resnet_model.relu,
            self.resnet_model.maxpool,
            self.resnet_model.layer1,
            self.resnet_model.layer2,
            self.resnet_model.layer3,
            self.resnet_model.layer4
        )

    def forward(self, x):
        return self.net(x)


class BboxEmbed(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(BboxEmbed, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


class DETR(nn.Module):
    def __init__(self,
                 backbone: Backbone,
                 pos_embed: PositionEmbedding,
                 transformer: Transformer,
                 num_classes,
                 num_queries
                 ):
        super(DETR, self).__init__()
        self.num_quries = num_queries
        self.transformer = transformer
        embed_dim = transformer.embed_dim

        self.class_embed = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed = BboxEmbed(embed_dim, embed_dim, 4)
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, embed_dim, kernel_size=1)
        self.backbone = backbone
        self.pos_embed = pos_embed

    def forward(self, x):
        feat = self.backbone(x)
        pos_embed = self.pos_embed(feat)

        feat = self.input_proj(feat)
        _, decoder_out = self.transformer(feat, self.query_embed.weight, pos_embed)

        out_classes = self.class_embed(decoder_out)
        out_coords = self.bbox_embed(decoder_out)

        return out_classes, out_coords


if __name__ == "__main__":
    # # x = torch.randn(1, 3, 224, 224)
    # # feature_extractor = Resnet18FeatureExtractor()
    # # out = feature_extractor(x)
    # # print(out.shape)  # ([1, 256, 14, 14])

    backbone = Backbone()
    transformer = Transformer()
    pos_embed = PositionEmbedding(transformer.embed_dim // 2)
    detr = DETR(backbone, pos_embed, transformer, 10, 100)
    t = torch.randn(3, 3, 224, 224)
    out = detr(t)
    print(out.shape)
