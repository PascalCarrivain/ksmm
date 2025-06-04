import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
import math

from ksmm_py.benchmark.utils import pattern_is_dense

from ksmm_py.layer.kronecker_sparse.interface import KSLinear


class LinearBsl(nn.Module):
    """like nn.Linear but it assumes that the input is of dimension 2 and the last dimension is the batch size"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        version="torch_matmul",
        weights=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weights is not None:
            assert weights.shape == (out_features, in_features)
            self.weight = weights
        else:
            self.weight = torch.empty(
                (out_features, in_features), **factory_kwargs
            )
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight = Parameter(self.weight)
        if bias:
            self.bias = Parameter(
                torch.empty(out_features, 1, **factory_kwargs)
            )
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)
        self.version = version
        assert self.version in ["torch_matmul", "einsum"]

    def forward(self, input: Tensor) -> Tensor:
        if self.version == "torch_matmul":
            in_size, batch_shape = input.shape[0], input.shape[1:]
            out = input.view(in_size, -1)
            out = torch.matmul(self.weight, out)
            if self.bias is not None:
                out = out + self.bias
            out = out.view(self.out_features, *batch_shape)
            return out
        if self.version == "einsum":
            raise NotImplementedError

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class UnfusedLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weights=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(UnfusedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weights is not None:
            assert weights.shape == (out_features, in_features)
            self.weight = weights
        else:
            self.weight = torch.empty(
                (out_features, in_features), **factory_kwargs
            )
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight = Parameter(self.weight)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is not None:
            return F.linear(input, self.weight) + self.bias
        return F.linear(input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


def get_linear_layer(
    patterns=None,
    algo=None,
    dtype: torch.dtype = torch.float16,
    bs_last=False,
    vsize=4,
    bias=False,
    weights=None,
    device: str = "cpu",
):
    assert algo in [
        "sparse",
        "dense",
        "einsum",
        "kernel",
        "bmm",
        "bsr",
        "nn_linear",
        "unfused_linear",
    ]

    for pattern in patterns:
        assert len(pattern) == 4
    in_size = patterns[0][0] * patterns[0][2] * patterns[0][3]  # acd
    out_size = patterns[-1][0] * patterns[-1][1] * patterns[-1][3]  # abd

    if algo in ["nn_linear", "unfused_linear"]:
        assert len(patterns) == 1
        assert pattern_is_dense(patterns[0])
        if weights is not None:
            assert len(weights) == 1
        if algo == "nn_linear":
            assert not bs_last
            layer = torch.nn.Linear(
                in_size, out_size, bias=bias, dtype=dtype, device=device
            )
            if weights is not None:
                assert weights[0].shape == (out_size, in_size)
                layer.weight.data = weights[0]
        elif algo == "unfused_linear":
            if not bs_last:
                layer = UnfusedLinear(
                    in_size,
                    out_size,
                    bias=bias,
                    weights=weights,
                    dtype=dtype,
                    device=device,
                )
            elif bs_last:
                layer = LinearBsl(
                    in_size,
                    out_size,
                    bias=bias,
                    weights=weights,
                    dtype=dtype,
                    device=device,
                )
            else:
                raise NotImplementedError

    elif algo in [
        "dense",
        "sparse",
        "bmm",
        "kernel",
        "einsum",
        "bsr",
    ]:
        layer = KSLinear(
            patterns,
            weights=weights,
            bias=bias,
            algo=algo,
            dtype=dtype,
            bs_last=bs_last,
            vsize=vsize,
            device=device,
        )

    else:
        raise ValueError(f"algo {algo} not recognized")
    return layer
