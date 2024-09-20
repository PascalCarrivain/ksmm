import torch

from ksmm_py.benchmark.utils import get_in_size_out_size


from ksmm_py.layer.get_linear_layer import (
    get_linear_layer,
    LinearBsl,
)

from ksmm_py.model.vit import (
    CustomLayerNorm,
    CustomLayerNormBsl,
)

import ksmm_py.model.vit as vit


def kslinear_replacement(
    linear_layer,
    algo,
    patterns,
    bias,
    dtype: torch.dtype = torch.float16,
    bs_last=False,
    device: str = "cpu",
):
    """
    Replaces a Linear layer (either the PyTorch default torch.nn.Linear in batch-size-first, or vit.LinearBsl,
    an equivalent layer in batch-size-last) by a Kronecker-sparse Linear (KSLinear) layer.
    It checks that the input and output sizes of the linear layer are compatible with the patterns.
    """
    if bs_last:
        assert isinstance(linear_layer, vit.LinearBsl)
    else:
        assert isinstance(linear_layer, torch.nn.Linear)
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    # Check that the patterns are compatible with the input and output sizes of the linear layer
    assert in_features, out_features == get_in_size_out_size(patterns)
    return get_linear_layer(
        patterns=patterns,
        algo=algo,
        dtype=dtype,
        bs_last=bs_last,
        bias=bias,
        device=device,
    )


def kslinear_surgery_ffn(
    ffn,
    algo,
    patterns_down,
    patterns_up,
    bs_last=False,
):
    """
    Replaces the first and second linear layers (up and down projections respectively)
    of a FeedForward module in a Vision Transformer (ViT) model by a Kronecker-sparse Linear (KSLinear) layer.

    Args:
        ffn: vit.FeeadForwardBsl or vit.FeedForward
            Instance of vit.
        algo: str
            Name of the algorithm.
        patterns_down: list
            List of (a, b, c, d) patterns for a Kronecker-sparse Linear (KSLinear) layer used to represent
            the down projection matrix (which has input size = 4 * output size).
        patterns_up: list
            List of (a, b, c, d) patterns for a Kronecker-sparse Linear (KSLinear) layer used to represent
            the up projection matrix (which has output size = 4 * input size).
        bs_last: bool, optional
           Batch-size first (default) or batch-size last.
    """
    if bs_last:
        assert isinstance(ffn, vit.FeedForwardBsl)
    else:
        assert isinstance(ffn, vit.FeedForward)

    if patterns_up is not None:
        ffn.net[1] = kslinear_replacement(
            ffn.net[1],
            algo=algo,
            patterns=patterns_up,
            bias=True,
            dtype=ffn.net[1].weight.dtype,
            bs_last=bs_last,
            device=ffn.net[1].weight.device,
        )
    if patterns_down is not None:
        ffn.net[3] = kslinear_replacement(
            ffn.net[3],
            algo=algo,
            patterns=patterns_down,
            bias=True,
            dtype=ffn.net[3].weight.dtype,
            bs_last=bs_last,
            device=ffn.net[3].weight.device,
        )


class FeedForwardResidual(torch.nn.Module):
    """
    Wraps a FeedForward module in a Vision Transformer (ViT) model with a residual connection.
    """

    def __init__(self, feedforward):
        super(FeedForwardResidual, self).__init__()
        assert isinstance(feedforward, vit.FeedForward) or isinstance(
            feedforward, vit.FeedForwardBsl
        )
        self.feedforward = feedforward

    def forward(self, x):
        x = x + self.feedforward(x)
        return x


def get_ffn(
    dim,
    mlp_dim,
    residual,
    algo,
    patterns_down,
    patterns_up,
    dtype: torch.dtype = torch.float16,
    bs_last=False,
    device: str = "cpu",
):
    """
    Returns a FeedForward module in a Vision Transformer (ViT) model whose up and down
    projection matrices have been replaced by Kronecker-sparse Linear (KSLinear) layers.
    """
    if bs_last:
        model = vit.FeedForwardBsl(dim, mlp_dim).to(device).to(dtype)
    else:
        model = vit.FeedForward(dim, mlp_dim).to(device).to(dtype)
    kslinear_surgery_ffn(
        model, algo, patterns_down, patterns_up, bs_last
    )
    if residual:
        return FeedForwardResidual(model)
    return model


class AttentionResidual(torch.nn.Module):
    """
    Wraps an Attention module in a Vision Transformer (ViT) model with a residual connection.
    """

    def __init__(self, attention):
        super(AttentionResidual, self).__init__()
        assert isinstance(attention, vit.Attention) or isinstance(
            attention, vit.AttentionBsl
        )
        self.attention = attention

    def forward(self, x):
        x = x + self.attention(x)
        return x


def kslinear_surgery_attention(
    attention,
    algo,
    patterns_attention,
    bs_last=False,
):
    """
    Replaces the up and down projection matrices of an Attention module in a
    Vision Transformer (ViT) model by Kronecker-sparse Linear (KSLinear) layers.

    Args:
        patterns_attention: list
            List of (a, b, c, d) patterns for a Kronecker-sparse Linear (KSLinear) layer used to represent
            the linear projections in an attention module (Query, Key, Value).
    """
    if patterns_attention is None:
        return
    if bs_last:
        assert isinstance(attention, vit.AttentionBsl)
    else:
        assert isinstance(attention, vit.Attention) or isinstance(
            attention, AttentionWithOnlyLinear
        )
    assert attention.split_qkv  # cannot replace if not split

    attention.to_q = kslinear_replacement(
        attention.to_q,
        algo=algo,
        patterns=patterns_attention,
        bias=False,
        dtype=attention.to_q.weight.dtype,
        bs_last=bs_last,
        device=attention.to_q.weight.device,
    )
    attention.to_k = kslinear_replacement(
        attention.to_k,
        algo=algo,
        patterns=patterns_attention,
        bias=False,
        dtype=attention.to_k.weight.dtype,
        bs_last=bs_last,
        device=attention.to_k.weight.device,
    )
    attention.to_v = kslinear_replacement(
        attention.to_v,
        algo=algo,
        patterns=patterns_attention,
        bias=False,
        dtype=attention.to_v.weight.dtype,
        bs_last=bs_last,
        device=attention.to_v.weight.device,
    )


def get_attention(
    dim,
    heads,
    residual,
    algo,
    patterns_attention,
    dtype: torch.dtype = torch.float16,
    bs_last=False,
    device: str = "cpu",
    sdpa_version="equivalent",
    split_qkv=False,
):
    """
    Returns an Attention module in a Vision Transformer (ViT) model whose
    linear projection matrices have been replaced by Kronecker-sparse Linear (KSLinear) layers.
    """
    assert dim % heads == 0
    if bs_last:
        assert sdpa_version == "equivalent"
        model = vit.AttentionBsl(
            dim, heads, dim_head=dim // heads, split_qkv=split_qkv,
        ).to(device).to(dtype)
    else:
        model = vit.Attention(
            dim,
            heads,
            dim_head=dim // heads,
            sdpa_version=sdpa_version,
            split_qkv=split_qkv,
        ).to(device).to(dtype)

    kslinear_surgery_attention(model, algo, patterns_attention, bs_last)
    if residual:
        return AttentionResidual(model)
    return model


class Block(torch.nn.Module):
    """
    Wraps a Block in a Vision Transformer (ViT) model with a residual connection.
    """

    def __init__(self, attention_residual, feedforward_residual):
        super(Block, self).__init__()
        assert isinstance(attention_residual, AttentionResidual)
        assert isinstance(feedforward_residual, FeedForwardResidual)
        self.attention_residual = attention_residual
        self.feedforward_residual = feedforward_residual

    def forward(self, x):
        x = self.attention_residual(x)
        x = self.feedforward_residual(x)
        return x


def get_block(
    dim,
    heads,
    mlp_dim,
    algo,
    patterns_attention,
    patterns_down,
    patterns_up,
    dtype: torch.dtype = torch.float16,
    bs_last=False,
    device: str = "cpu",
    sdpa_version="equivalent",
    split_qkv=False,
):
    """
    Returns a Block in a Vision Transformer (ViT) model whose linear projections of
    attention and feedforward modules have been replaced by Kronecker-sparse Linear (KSLinear) layers.
    """
    residual = True
    ffn = get_ffn(
        dim,
        mlp_dim,
        residual,
        algo,
        patterns_down,
        patterns_up,
        dtype=dtype,
        bs_last=bs_last,
        device=device,
    )
    attention = get_attention(
        dim,
        heads,
        residual,
        algo,
        patterns_attention,
        dtype=dtype,
        bs_last=bs_last,
        device=device,
        sdpa_version=sdpa_version,
        split_qkv=split_qkv,
    )
    return Block(attention, ffn)


def kslinear_surgery_vit(
    model,
    algo,
    patterns_attention,
    patterns_down,
    patterns_up,
    bs_last=False,
    surgery_part="all",
):
    """
    Replaces the linear projection matrices of the attention and feedforward
    modules in a Vision Transformer (ViT) model by Kronecker-sparse Linear (KSLinear) layers.
    """
    if bs_last == "first":
        assert isinstance(model, vit.SimpleViTBsl)
    else:
        assert isinstance(model, vit.FlashAttentionSimpleViT)
    for attn, fnn in model.transformer.layers:
        if surgery_part == "all":
            kslinear_surgery_attention(
                attn,
                algo,
                patterns_attention,
                bs_last=bs_last,
            )
            kslinear_surgery_ffn(
                fnn, algo, patterns_down, patterns_up, bs_last,
            )
        elif surgery_part == "ffn_only":
            kslinear_surgery_ffn(
                fnn, algo, patterns_down, patterns_up, bs_last,
            )
        else:
            raise NotImplementedError


def get_vit(
    arch,
    algo,
    patterns_attention,
    patterns_down,
    patterns_up,
    dtype: torch.dtype = torch.float16,
    bs_last=False,
    device: str = "cpu",
    surgery_part="all",
    sdpa_version="equivalent",
    split_qkv=False,
):
    """
    Returns a Vision Transformer (ViT) model whose linear projection
    matrices have been replaced by Kronecker-sparse Linear (KSLinear) layers.
    """
    assert sdpa_version in ["flash", "equivalent", "default"]
    if arch == "simple_vit_s16_in1k":
        model = vit.simple_vit_s16_in1k(sdpa_version, split_qkv, bs_last)
    elif arch == "simple_vit_b16_in1k":
        model = vit.simple_vit_b16_in1k(sdpa_version, split_qkv, bs_last)
    elif arch == "simple_vit_l16_in1k":
        model = vit.simple_vit_l16_in1k(sdpa_version, split_qkv, bs_last)
    elif arch == "simple_vit_h14_in1k":
        model = vit.simple_vit_h14_in1k(sdpa_version, split_qkv, bs_last)
    else:
        raise NotImplementedError

    model = model.to(device).to(dtype)

    if surgery_part is not None:
        kslinear_surgery_vit(
            model,
            algo,
            patterns_attention,
            patterns_down,
            patterns_up,
            bs_last,
            surgery_part,
            split_qkv=split_qkv,
        )
    return model


def get_only_linear_in_ffn(
    dim,
    mlp_dim,
    algo,
    patterns_down,
    patterns_up,
    dtype: torch.dtype = torch.float16,
    bs_last=False,
    device: str = "cpu",
):
    """
    Returns two Kronecker-sparse Linear (KSLinear) layers that replace the up and down projection matrices of a FeedForward module in a Vision Transformer (ViT) model.
    Used to benchmark only the time of the linear layers in the FeedForward module.
    """
    bias = False
    assert dim, mlp_dim == get_in_size_out_size(patterns_down)
    assert mlp_dim, dim == get_in_size_out_size(patterns_up)
    return torch.nn.Sequential(
        get_linear_layer(
            patterns=patterns_up,
            algo=algo,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
            bias=bias,
        ),
        get_linear_layer(
            patterns=patterns_down,
            algo=algo,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
            bias=bias,
        ),
    )


class AttentionWithOnlyLinear(torch.nn.Module):
    """
    Attention module with only the linear projection matrices (Query, Key, Value) and no softmax or other operations.
    Used to benchmark only the time of the linear layers in the Attention module.
    """

    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.split_qkv = True
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False, dtype=dtype, device=device)
        self.to_q = torch.nn.Linear(dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(dim, inner_dim, bias=False)

    def forward(self, x):
        qkv = [self.to_q(x), self.to_k(x), self.to_v(x)]
        v = qkv[-1]
        return v


def get_only_linear_in_attention(
    dim,
    heads,
    algo,
    patterns_attention,
    dtype: torch.dtype = torch.float16,
    bs_last=False,
    device: str = "cpu",
):
    """
    Returns an Attention module in a Vision Transformer (ViT) model with only the linear projection matrices (Query, Key, Value) and no softmax or other operations. Used to benchmark only the time of the linear layers in the Attention module.
    """
    assert not bs_last
    model = AttentionWithOnlyLinear(dim, heads).to(device).to(dtype)
    kslinear_surgery_attention(
        model, algo, patterns_attention, dtype, bs_last, device
    )
    return model


def get_only_linear_in_block(
    dim,
    heads,
    mlp_dim,
    algo,
    patterns_attention,
    patterns_down,
    patterns_up,
    dtype: torch.dtype = torch.float16,
    bs_last=False,
    device: str = "cpu",
):
    """
    Returns a Block in a Vision Transformer (ViT) model with only the linear projection matrices of the attention and feedforward modules and no softmax or other operations. Used to benchmark only the time of the linear layers in the attention and feedforward modules.
    """
    linear_in_ffn = get_only_linear_in_ffn(
        dim, mlp_dim, algo, patterns_down, patterns_up, dtype, bs_last, device
    )
    linear_in_attention = get_only_linear_in_attention(
        dim, heads, algo, patterns_attention, dtype, bs_last, device
    )
    return torch.nn.Sequential(linear_in_attention, linear_in_ffn)


def get_only_linear_in_vit(
    depth,
    dim,
    heads,
    mlp_dim,
    algo,
    patterns_attention,
    patterns_down,
    patterns_up,
    dtype: torch.dtype = torch.float16,
    bs_last=False,
    device: str = "cpu",
):
    """
    Returns a Vision Transformer (ViT) model with only the linear projection matrices of the attention and feedforward modules and no softmax or other operations. Used to benchmark only the time of the linear layers in the attention and feedforward modules
    """
    assert not bs_last
    blocks = [
        get_only_linear_in_block(
            dim,
            heads,
            mlp_dim,
            algo,
            patterns_attention,
            patterns_down,
            patterns_up,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
        )
        for _ in range(depth)
    ]
    return torch.nn.Sequential(*blocks)


def get_vit_config(arch):
    if arch == "simple_vit_s16_in1k":
        return {
            "image_size": 224,
            "patch_size": 16,
            "num_classes": 1000,
            "dim": 384,
            "depth": 12,
            "heads": 6,
            "mlp_dim": 1536,
        }
    if arch == "simple_vit_b16_in1k":
        return {
            "image_size": 224,
            "patch_size": 16,
            "num_classes": 1000,
            "dim": 768,
            "depth": 12,
            "heads": 12,
            "mlp_dim": 3072,
        }
    if arch == "simple_vit_l16_in1k":
        return {
            "image_size": 224,
            "patch_size": 16,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 24,
            "heads": 16,
            "mlp_dim": 4096,
        }
    if arch == "simple_vit_h14_in1k":
        return {
            "image_size": 224,
            "patch_size": 14,
            "num_classes": 1000,
            "dim": 1280,
            "depth": 32,
            "heads": 16,
            "mlp_dim": 5120,
        }
    raise NotImplementedError


def parse_granularity(granularity_str, vit_config):
    """
    This is the function that defines the convention that in linear_x_y_bias, x and y are the output and input dimensions of the linear layer, respectively. It must be read linear_dout_din_bias corresponding to a matrix dout x din.
    """
    if "bias" in granularity_str:
        bias = True
    else:
        bias = False

    if "down" in granularity_str:
        in_features = 4 * vit_config["dim"]
        assert in_features == vit_config["mlp_dim"]
    else:
        in_features = vit_config["dim"]

    if "up" in granularity_str:
        out_features = 4 * vit_config["dim"]
        assert out_features == vit_config["mlp_dim"]
    else:
        out_features = vit_config["dim"]

    return in_features, out_features, bias


def get_patterns_with_granularity(
    patterns_attention, patterns_down, patterns_up, granularity_str
):
    if "up" in granularity_str:
        patterns = patterns_up
    elif "down" in granularity_str:
        patterns = patterns_down
    else:
        patterns = patterns_attention
    return patterns

def get_input_shape(arch, granularity, bs_last, batch_size):
    vit_config = get_vit_config(arch)

    seq_length = (vit_config["image_size"] // vit_config["patch_size"]) ** 2
    if granularity in [
        "gelu",
        "linear",
        "linear_up",
        "linear_bias",
        "linear_up_bias",
        "layernorm",
        "custom_layernorm",
        "ffn",
        "ffn_residual",
        "attention",
        "attention_residual",
        "block",
        "only_linear_in_ffn",
        "only_linear_in_attn",
        "only_linear_in_block",
        "only_linear_in_vit",
        "vit_only_linear_in_ffn",
    ]:
        if bs_last:
            return (vit_config["dim"], seq_length, batch_size)
        else:
            return (batch_size, seq_length, vit_config["dim"])
    elif granularity in ["linear_down", "linear_down_bias"]:
        assert 4 * vit_config["dim"] == vit_config["mlp_dim"]
        if bs_last:
            return (4 * vit_config["dim"], seq_length, batch_size)
        else:
            return (batch_size, seq_length, 4 * vit_config["dim"])

    elif granularity in ["vit", "vit_surgery_ffn_only", "vit_"]:
        if bs_last:
            return (
                vit_config["image_size"],
                vit_config["image_size"],
                3,
                batch_size,
            )
        else:
            return (
                batch_size,
                3,
                vit_config["image_size"],
                vit_config["image_size"],
            )
    else:
        raise NotImplementedError

def get_submodel_vit(
    arch,
    bs_last,
    granularity,
    patterns_attention,
    patterns_down,
    patterns_up,
    algo,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    sdpa_version=None,
    split_qkv=None,
):
    # GET VIT CONFIG
    vit_config = get_vit_config(arch)

    # GET MODEL

    if granularity in [
        "linear",
        "linear_up",
        "linear_down",
        "linear_bias",
        "linear_up_bias",
        "linear_down_bias",
    ]:
        in_features, out_features, bias = parse_granularity(
            granularity, vit_config
        )
        patterns = get_patterns_with_granularity(
            patterns_attention, patterns_down, patterns_up, granularity
        )
        assert in_features, out_features == get_in_size_out_size(patterns)
        model = get_linear_layer(
            patterns=patterns,
            algo=algo,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
            bias=bias,
        )
    elif granularity in ["ffn", "ffn_residual"]:
        residual = granularity == "ffn_residual"
        model = get_ffn(
            vit_config["dim"],
            vit_config["mlp_dim"],
            residual,
            algo,
            patterns_down,
            patterns_up,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
        )
    elif granularity in ["attention", "attention_residual"]:
        residual = granularity == "attention_residual"
        model = get_attention(
            vit_config["dim"],
            vit_config["heads"],
            residual,
            algo,
            patterns_attention,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
            sdpa_version=sdpa_version,
            split_qkv=split_qkv,
        )
    elif granularity == "block":
        model = get_block(
            vit_config["dim"],
            vit_config["heads"],
            vit_config["mlp_dim"],
            algo,
            patterns_attention,
            patterns_down,
            patterns_up,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
            sdpa_version=sdpa_version,
            split_qkv=split_qkv,
        )
    elif granularity in ["vit", "vit_surgery_ffn_only"]:
        if (
            patterns_attention is not None
            or patterns_down is not None
            or patterns_up is not None
        ):
            if granularity == "vit":
                surgery_part = "all"
            elif granularity == "vit_surgery_ffn_only":
                surgery_part = "ffn_only"
        else:
            surgery_part = None
        model = get_vit(
            arch,
            algo,
            patterns_attention,
            patterns_down,
            patterns_up,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
            surgery_part=surgery_part,
            sdpa_version=sdpa_version,
            split_qkv=split_qkv,
        )
    elif granularity == "only_linear_in_ffn":
        model = get_only_linear_in_ffn(
            vit_config["dim"],
            vit_config["mlp_dim"],
            algo,
            patterns_down,
            patterns_up,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
        )
    elif granularity == "only_linear_in_attn":
        model = get_only_linear_in_attention(
            vit_config["dim"],
            vit_config["heads"],
            algo,
            patterns_attention,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
        )
    elif granularity == "only_linear_in_block":
        model = get_only_linear_in_block(
            vit_config["dim"],
            vit_config["heads"],
            vit_config["mlp_dim"],
            algo,
            patterns_attention,
            patterns_down,
            patterns_up,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
        )
    elif granularity == "only_linear_in_vit":
        model = get_only_linear_in_vit(
            vit_config["depth"],
            vit_config["dim"],
            vit_config["heads"],
            vit_config["mlp_dim"],
            algo,
            patterns_attention,
            patterns_down,
            patterns_up,
            dtype=dtype,
            bs_last=bs_last,
            device=device,
        )
    elif granularity == "vit_only_linear_in_ffn":
        list_linear = []
        for _ in range(vit_config["depth"]):
            if bs_last:
                list_linear.append(
                    LinearBsl(
                        vit_config["dim"], vit_config["mlp_dim"], bias=False
                    )
                )
                list_linear.append(
                    LinearBsl(
                        vit_config["mlp_dim"], vit_config["dim"], bias=False
                    )
                )
            else:
                list_linear.append(
                    torch.nn.Linear(
                        vit_config["dim"], vit_config["mlp_dim"], bias=False
                    )
                )
                list_linear.append(
                    torch.nn.Linear(
                        vit_config["mlp_dim"], vit_config["dim"], bias=False
                    )
                )
        model = torch.nn.Sequential(*list_linear).to(device).to(dtype)
    elif granularity == "gelu":
        model = torch.nn.GELU()
    elif granularity == "layernorm":
        if bs_last:
            raise NotImplementedError
        else:
            model = torch.nn.LayerNorm(vit_config["dim"], device=device, dtype=dtype)
    elif granularity == "custom_layernorm":
        if bs_last:
            model = CustomLayerNormBsl(vit_config["dim"], device=device, dtype=dtype)
        else:
            model = CustomLayerNorm(vit_config["dim"], device=device, dtype=dtype)

    else:
        raise NotImplementedError
    return model