"""
Implementations from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_flash_attn_vit.py
"""

from collections import namedtuple
from packaging import version

import numbers

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange

from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
import math

from ksmm_py.layer.get_linear_layer import LinearBsl

from typing import Tuple

# constants

Config = namedtuple(
    "FlashAttentionConfig",
    ["enable_flash", "enable_math", "enable_mem_efficient"],
)


# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    # patches.shape = (bs, h, w, dim)
    # return pe.shape = (h * w, dim)
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    assert (
        dim % 4
    ) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_2d_bsl(patchs, temperature=10000, dtype=torch.float32):
    # patches.shape = (dim, h, w, bs)
    dim, h, w, _, device, dtype = *patchs.shape, patchs.device, patchs.dtype

    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    assert (
        dim % 4
    ) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)  # (h * w, dim)

    pe = pe.t().unsqueeze(-1)

    return pe.type(dtype)


# main class


class Attend(nn.Module):
    def __init__(self, sdpa_version="flash"):
        super().__init__()
        self.sdpa_version = sdpa_version
        use_flash = sdpa_version == "flash"
        assert not (
            use_flash
            and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(
            torch.device("cuda")
        )

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # flash attention - https://arxiv.org/abs/2205.14135

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v)

        return out

    def forward(self, q, k, v):
        n, device, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5

        if self.sdpa_version == "flash":
            return self.flash_attn(q, k, v)

        elif self.sdpa_version == "equivalent":
            return scaled_dot_product_attention(q, k, v)

        elif self.sdpa_version == "default":
            # similarity

            sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

            # attention

            attn = sim.softmax(dim=-1)

            # aggregate values

            out = einsum("b h i j, b h j d -> b h i d", attn, v)

            return out


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    bias=False,
) -> torch.Tensor:
    """input of the form (bs, heads, seq, dim_head)"""
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    if bias is True:
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
    else:
        attn_bias = None
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        if attn_bias is not None:
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            if attn_bias is not None:
                attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if attn_bias is not None:
        attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def scaled_dot_product_attention_bsl(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    bias=False,
):
    """
    in scaled_dot_product_attention the query, key, value have shape (bs, heads, seq, dim_head)
    do the same but assuming that the input is of shape (seq, dim_heads, heads, bs). Is it the right shape to consider?
    """
    L, S = query.size(0), key.size(0)
    scale_factor = 1 / math.sqrt(query.size(1)) if scale is None else scale
    if bias is True:
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
    else:
        attn_bias = None

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        if attn_bias is not None:
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            if attn_bias is not None:
                attn_bias += attn_mask

    attn_weight = (
        torch.einsum("ijhb,kjhb->ikhb", query, key) * scale_factor
    )  # (seq, seq, heads, bs)
    if attn_bias is not None:
        attn_weight += attn_bias.view(*attn_bias.shape, 1, 1)
    attn_weight = torch.softmax(attn_weight, dim=1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return torch.einsum(
        "ikhb,kjhb->ijhb", attn_weight, value
    )  # (seq, dim_heads, heads, bs)


# classes


class CustomLayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # # bias and weight has the shape of normalized_shape except for the last dimension set to 1
        # shape_bias_weight = list(normalized_shape)
        # shape_bias_weight[-1] = 1
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        d = len(self.normalized_shape)
        assert tuple(input.shape[-d:]) == self.normalized_shape
        mean = input.mean(
            dim=tuple([-i for i in range(1, d + 1)]), keepdim=True
        )
        var = input.var(
            dim=tuple([-i for i in range(1, d + 1)]),
            correction=0,
            keepdim=True,
        )
        out = (input - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            out = out * self.weight + self.bias  # broadcast
        return out

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class CustomLayerNormBsl(nn.Module):
    """like nn.LayerNorm but it assumes the forward method is encoded from scratch assuming that the inputs has the batch size as the last dimension"""

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # # bias and weight has the shape of normalized_shape except for the last dimension set to 1
        # shape_bias_weight = list(normalized_shape)
        # shape_bias_weight[-1] = 1
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        d = len(self.normalized_shape)
        d_c = len(input.shape) - d
        assert tuple(input.shape[:d]) == self.normalized_shape
        # average over the first dimensions except the last one:
        # mean = input.mean(dim=tuple(range(len(input.shape) - 1)), keepdim=True)
        # var = input.var(dim=tuple(range(len(input.shape) - 1)), keepdim=True)
        # normalized the first D dimension where D is the length of normalized_shape
        mean = input.mean(
            dim=tuple(range(len(self.normalized_shape))), keepdim=True
        )
        var = input.var(
            dim=tuple(range(len(self.normalized_shape))),
            unbiased=False,
            keepdim=True,
        )
        # the next thing would be batch norm, not layer norm
        # mean = input.mean(dim=-1, keepdim=True)
        # std = input.std(dim=-1, keepdim=True)
        out = (input - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(
                *self.weight.shape, *tuple([1 for _ in range(d_c)])
            ) + self.bias.view(
                *self.bias.shape, *tuple([1 for _ in range(d_c)])
            )
        return out

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )

class FeedForwardBsl(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
    ):
        super().__init__()
        self.net = nn.Sequential(
                CustomLayerNormBsl(dim),
                LinearBsl(dim, hidden_dim),
                nn.GELU(),
                LinearBsl(hidden_dim, dim),
            )

    def forward(self, x):
        assert len(x.shape) == 3
        dim, seq, bs = x.shape
        x = x.view(dim, -1)
        return self.net(x).view(dim, seq, bs)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
    ):
        super().__init__()
        self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        sdpa_version="flash",
        split_qkv=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        assert sdpa_version in ["flash", "equivalent", "default"]

        self.attend = Attend(sdpa_version=sdpa_version)

        self.split_qkv = split_qkv

        if self.split_qkv:
            self.to_q = nn.Linear(
                dim, inner_dim, bias=False
            )
            self.to_k = nn.Linear(
                dim, inner_dim, bias=False,
            )
            self.to_v = nn.Linear(
                dim, inner_dim, bias=False,
            )
        else:
            self.to_qkv = nn.Linear(
                dim, inner_dim * 3, bias=False,
            )

        self.to_out = nn.Linear(
            inner_dim, dim, bias=False,
        )

    def forward(self, x):
        x = self.norm(x)

        if self.split_qkv:
            qkv = [self.to_q(x), self.to_k(x), self.to_v(x)]
        else:
            qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        out = self.attend(q, k, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class AttentionBsl(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        split_qkv=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = CustomLayerNormBsl(dim)

        self.split_qkv = split_qkv

        if self.split_qkv:
            self.to_q = LinearBsl(
                dim, inner_dim, bias=False,
            )
            self.to_k = LinearBsl(
                dim, inner_dim, bias=False,
            )
            self.to_v = LinearBsl(
                dim, inner_dim, bias=False,
            )
        else:
            self.to_qkv = LinearBsl(
                dim, inner_dim * 3, bias=False,
            )
        self.to_out = LinearBsl(
            inner_dim, dim, bias=False,
        )

    def forward(self, x):
        # x.shape = (d, n, bs)
        x = self.norm(x)

        if self.split_qkv:
            qkv = [self.to_q(x), self.to_k(x), self.to_v(x)]
        else:
            qkv = self.to_qkv(x).chunk(3, dim=0)

        q, k, v = map(
            lambda t: rearrange(t, "(h d) n b -> n d h b", h=self.heads), qkv
        )

        out = scaled_dot_product_attention_bsl(q, k, v)  # (n, d, h, bs)

        out = rearrange(out, "n d h b -> (h d) n b")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        sdpa_version,
        split_qkv=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            sdpa_version=sdpa_version,
                            split_qkv=split_qkv,
                        ),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerBsl(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, split_qkv=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        AttentionBsl(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            split_qkv=split_qkv,
                        ),
                        FeedForwardBsl(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class FlashAttentionSimpleViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        sdpa_version="flash",  # "flash", "equivalent" or "default"
        split_qkv=False,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert sdpa_version in ["flash", "equivalent", "default"]

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b h w (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            sdpa_version=sdpa_version,
            split_qkv=split_qkv,
        )

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, "b ... d -> b (...) d") + pe

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)


class SimpleViTBsl(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        split_qkv=False,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        # assume that image input is of shape (height, width, c, b) ?
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "(h p1) (w p2) c b -> (p1 p2 c) h w b",
                p1=patch_height,
                p2=patch_width,
            ),
            CustomLayerNormBsl(patch_dim),
            LinearBsl(patch_dim, dim),
            CustomLayerNormBsl(dim),
        )

        self.transformer = TransformerBsl(
            dim, depth, heads, dim_head, mlp_dim, split_qkv=split_qkv
        )

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            CustomLayerNormBsl(dim), LinearBsl(dim, num_classes)
        )

    def forward(self, img):
        # img of shape (height, width, c, b)
        h, w, *_, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)  # (dim, h, w, b)
        pe = posemb_sincos_2d_bsl(x)  # check this
        x = rearrange(x, "d ... b -> d (...) b") + pe

        x = self.transformer(x)
        x = x.mean(dim=1)  # after mean: x of shape (dim, b)

        x = self.to_latent(x)
        return self.linear_head(x)


def simple_vit_s16_in1k(sdpa_version, split_qkv=False, bs_last=False):
    assert sdpa_version in ["flash", "equivalent", "default"]

    dim = 384
    heads = 6

    if bs_last:
        assert sdpa_version == "equivalent"
        return SimpleViTBsl(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=dim,
            depth=12,
            heads=heads,
            mlp_dim=4 * dim,
            dim_head=dim // heads,
            split_qkv=split_qkv,
        )
    else:
        return FlashAttentionSimpleViT(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=dim,
            depth=12,
            heads=heads,
            mlp_dim=4 * dim,
            dim_head=dim // heads,
            sdpa_version=sdpa_version,
            split_qkv=split_qkv,
        )


def simple_vit_b16_in1k(sdpa_version, split_qkv=False, bs_last=False):
    assert sdpa_version in ["flash", "equivalent", "default"]

    dim = 768
    heads = 12
    if bs_last:
        assert sdpa_version == "equivalent"
        return SimpleViTBsl(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=dim,
            depth=12,
            heads=heads,
            mlp_dim=4 * dim,
            dim_head=dim // heads,
            split_qkv=split_qkv,
        )
    else:
        return FlashAttentionSimpleViT(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=dim,
            depth=12,
            heads=heads,
            mlp_dim=4 * dim,
            dim_head=dim // heads,
            sdpa_version=sdpa_version,
            split_qkv=split_qkv,
        )


def simple_vit_l16_in1k(sdpa_version, split_qkv=False, bs_last=False):
    assert sdpa_version in ["flash", "equivalent", "default"]

    dim = 1024
    heads = 16

    if bs_last:
        assert sdpa_version == "equivalent"
        return SimpleViTBsl(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=dim,
            depth=24,
            heads=heads,
            mlp_dim=4 * dim,
            dim_head=dim // heads,
            split_qkv=split_qkv,
        )
    else:
        return FlashAttentionSimpleViT(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=dim,
            depth=24,
            heads=heads,
            mlp_dim=4 * dim,
            dim_head=dim // heads,
            sdpa_version=sdpa_version,
            split_qkv=split_qkv,
        )


def simple_vit_h14_in1k(sdpa_version, split_qkv=False, bs_last=False):
    assert sdpa_version in ["flash", "equivalent", "default"]

    dim = 1280
    heads = 16

    if bs_last:
        assert sdpa_version == "equivalent"
        return SimpleViTBsl(
            image_size=224,
            patch_size=14,
            num_classes=1000,
            dim=dim,
            depth=32,
            heads=heads,
            mlp_dim=4 * dim,
            dim_head=dim // heads,
            split_qkv=split_qkv,
        )
    else:
        return FlashAttentionSimpleViT(
            image_size=224,
            patch_size=14,
            num_classes=1000,
            dim=dim,
            depth=32,
            heads=heads,
            mlp_dim=4 * dim,
            dim_head=dim // heads,
            sdpa_version=sdpa_version,
            split_qkv=split_qkv,
        )