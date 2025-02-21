import math

from einops import einsum, rearrange

import torch
from torch import nn
import torch.nn.functional as F

from ksmm_py.layer.bmm.forward_bmm import (
    bmm_bs_first,
    bmm_bs_last,
)



def generate_factor(a, b, c, d, algo, weights=None,
                    dtype: torch.dtype = torch.float16, device: str='cpu'):
    """
    Generate a Kronecker-sparse matrix with sparsity pattern (a,b,c,d), either initialized randomly or with the prescribed weights assumed to be stored as a 4D tensor of shape (a,b,c,d).
    The chosen format is adapted to the algorithm used for the forward pass.
    """
    assert algo in [
        "sparse",
        "bsr",
        "dense",
        "einsum",
        "kernel",
        "bmm",
    ]
    scaling = 1.0 / math.sqrt(c)
    if algo == "bmm":
        if weights is None:
            factor = torch.Tensor(a, d, b, c).uniform_(-scaling, scaling)
        else:
            factor = weights.permute(0,3,1,2)
        factor = factor.view(a * d, b, c).contiguous()
    elif algo == "bsr":
        if weights is None:
            factor = from_abcd_to_dense(
                torch.empty((a * d, b, c, 1)).uniform_(-scaling, scaling)
            )
        else:
            factor = weights.permute(0, 3, 1, 2).reshape(a * d, b, c, 1)
        # For now, the PyTorch BSR format only supports square blocks
        blocksize = math.gcd(b, c)
        factor = factor.to_sparse_bsr(blocksize=(blocksize, blocksize))
    elif algo == "einsum":
        if weights is None:
            factor = torch.Tensor(a, b, c, d).uniform_(-scaling, scaling)
        else:
            factor = weights
    elif algo == "sparse":
        if weights is None:
            factor = from_abcd_to_dense(
                torch.empty((a, b, c, d)).uniform_(-scaling, scaling)
            )
        else:
            factor = from_abcd_to_dense(weights)
        factor = factor.to_sparse_csr()
    elif algo == "dense":
        if weights is None:
            factor = torch.Tensor(a * b * d, a * c * d).uniform_(-scaling, scaling)
        else:
            factor = from_abcd_to_dense(weights)
    elif algo == "kernel":
        if weights is None:
            factor = torch.Tensor(a * d * c, b).uniform_(-scaling, scaling)
        else:
            factor = weights.permute(0, 3, 2, 1).reshape(a * d * c, b)
    else:
        raise NotImplementedError
    return factor.to(dtype).to(device)


def from_kernel_format_to_dense(factor, pattern):
    """Convert to a dense 2D-tensor a Kronecker-sparse matrix stored in the format adapted to the kernel algorithm."""
    a, b, c, d = pattern
    f = factor.view(a, d, c, b).permute(0, 3, 2, 1)
    dense = torch.zeros(a * b * d, a * c * d, dtype=factor.dtype)
    for i in range(a):
        sub_blocks = [
            [torch.diag(f[i, j, k, :]) for j in range(b)] for k in range(c)
        ]
        sub_intermediate_blocks = [torch.cat(sb, dim=0) for sb in sub_blocks]
        dense[i * b * d : (i + 1) * b * d, i * c * d : (i + 1) * c * d] = torch.cat(sub_intermediate_blocks, dim=1)
    return dense.to(factor.device)


def from_bmm_format_to_dense(factor, pattern):
    """Convert to a dense 2D-tensor a Kronecker-sparse matrix stored in the format adapted to the bmm algorithm."""
    a, b, c, d = pattern
    assert factor.shape == (a, d, b, c)
    blocks = []
    for block in factor:
        sub_blocks = [
            [torch.diag(block[:, i, j]) for i in range(b)] for j in range(c)
        ]
        sub_intermediate_blocks = [torch.cat(sb, dim=0) for sb in sub_blocks]
        blocks.append(torch.cat(sub_intermediate_blocks, dim=1))
    result = torch.zeros(a * d * b, a * d * c, dtype=factor.dtype)
    for i in range(a):
        result[d * b * i : d * b * (i + 1), d * c * i : d * c * (i + 1)] = (
            blocks[i]
        )
    return result.to(factor.device)


def from_abcd_to_dense(factor):
    """Convert to a dense 2D-tensor a Kronecker-sparse matrix stored in the format (a,b,c,d)."""
    a, b, c, d = factor.shape
    device = factor.device
    factor = torch.diag_embed(factor).to(device).permute(0, 1, 3, 2, 4).reshape(a, b * d, c * d)
    return torch.block_diag(*torch.unbind(factor, dim=0)).to(device)


class KSLinear(nn.Module):
    r"""Kronecker-sparse linear (KSLinear) layer that can be used as a drop-in replacement for torch.nn.Linear.
    The linear layer is described by a product of Kronecker-sparse matrices, described by
    sparsity patterns, and optional prescribed weights.
    Each sparsity pattern is given in format (a,b,c,d), with corresponding support
    $I_a \otimes 1_{b, c} \otimes I_d$. The optional weights are assumed to be stored as
    4D-tensors of shape (a,b,c,d).
    """

    def __init__(
        self,
        patterns,
        weights=None,
        bias=True,
        algo="dense",
        dtype: torch.dtype = torch.float16,
        bs_last: bool = False,
        device: str = 'cpu'
    ):
        """
        patterns: list of pattern (a,b,c,d), from rightmost factor to leftmost factor.
        weights: list of weights for each factor, given as a list of 4D-tensors of shape (a,b,c,d).
        """
        super().__init__()
        assert algo in [
            "sparse",
            "bsr",
            "dense",
            "einsum",
            "kernel",
            "bmm",
        ]
        for pattern in patterns:
            assert len(pattern) == 4
        self.in_size = patterns[0][0] * patterns[0][2] * patterns[0][3]  # acd
        self.out_size = (
            patterns[-1][0] * patterns[-1][1] * patterns[-1][3]
        )  # abd
        self.algo = algo
        self.num_mat = len(patterns)
        self.patterns = patterns
        self.dtype = dtype
        self.bs_last = bs_last
        self.weights=weights
        self.device = device
        if self.device == 'cpu' and self.algo == 'kernel':
            raise Exception("kernel expects device='cuda'.")
        parameters = []

        if self.weights is None:
            self.weights = [None] * len(self.patterns)

        assert len(self.weights) == len(self.patterns)

        for p, w in zip(self.patterns, self.weights):
            factor = generate_factor(
                *p, self.algo, w, dtype, device
            )
            parameters.append(torch.nn.Parameter(factor))

        self.factors = torch.nn.ParameterList(parameters)

        if bias:
            bound = 1 / math.sqrt(self.in_size)
            if self.bs_last is False:
                # bs first
                bias_shape = (self.out_size,)
                self.bias = nn.Parameter(torch.empty(*bias_shape, dtype=dtype, device=device).uniform_(-bound, bound))
            else:
                # bs last
                bias_shape = (self.out_size, 1)
                self.bias = nn.Parameter(torch.empty(*bias_shape, dtype=dtype, device=device).uniform_(-bound, bound))
        else:
            self.register_parameter("bias", None)

    def forward_bs_first(self, output):
            for i, factor in enumerate(self.factors):
                if self.algo == "bmm":
                    output = bmm_bs_first(output, self.factors[i], self.patterns[i])
                elif self.algo == "kernel":
                    from .kernel import Kernel
                    a, b, c, d = self.patterns[i]
                    output = Kernel.forward(
                        output,
                        factor,
                        self.bs_last,
                        a,
                        b,
                        c,
                        d,
                        fp16=self.dtype == torch.float16 or self.dtype == torch.half,
                    )
                elif self.algo == "sparse":
                    output = F.linear(output, factor)
                elif self.algo == "bsr":
                    a, b, c, d = self.patterns[i]
                    if d == 1:
                        output = F.linear(
                            output, factor
                        )  # recall that the shape of factor is (a, b, c)
                    elif a == 1:
                        batch_shape = output.shape[:-1]
                        output = (
                            output.view(*batch_shape, c, d)
                            .transpose(-1, -2)
                            .reshape(*batch_shape, c * d)
                        )

                        output = F.linear(
                            output, factor
                        )  # the shape of factor is (d, b, c)
                        output = (
                            output.view(*batch_shape, d, b)
                            .transpose(-1, -2)
                            .reshape(*batch_shape, b * d)
                        )
                    else:
                        # general case a > 1 and d > 1
                        batch_shape = output.shape[:-1]
                        output = (
                            output.view(*batch_shape, a, c, d)
                            .transpose(-1, -2)
                            .reshape(*batch_shape, a * c * d)
                        )
                        output = F.linear(
                            output, factor
                        )
                        output = (
                            output.view(*batch_shape, a, d, b)
                            .transpose(-1, -2)
                            .reshape(*batch_shape, a * b * d)
                        )
                elif self.algo == "dense":
                    output = F.linear(output, factor)
                elif self.algo == "einsum":
                    a, b, c, d = self.patterns[i]
                    output = rearrange(
                        output, "... (a c d) -> ... a c d", a=a, c=c, d=d
                    )
                    output = einsum(
                        output,
                        factor,
                        "... a c d, a b c d -> ... a b d",
                    )
                    output = rearrange(output, "... a b d-> ... (a b d)")
                else:
                    raise NotImplementedError
            return output

    def forward_bs_last(self, output):

        for i, factor in enumerate(self.factors):
            if self.algo == "bmm":
                output = bmm_bs_last(
                        output, self.factors[i], self.patterns[i]
                    )
            elif self.algo == "kernel":
                from .kernel import Kernel

                a, b, c, d = self.patterns[i]
                output = Kernel.forward(
                    output,
                    factor,
                    self.bs_last,
                    a,
                    b,
                    c,
                    d,
                    fp16=self.dtype == torch.float16 or self.dtype == torch.half,
                )
            elif self.algo == "sparse":
                output = torch.matmul(factor, output)
            elif self.algo == "bsr":
                a, b, c, d = self.patterns[i]
                if d == 1:
                    output = torch.matmul(factor, output)
                elif a == 1:
                    batch_shape = output.shape[1:]
                    output = (
                        output.view(c, d, *batch_shape)
                        .transpose(0, 1)
                        .reshape(c * d, *batch_shape)
                    )
                    output = torch.matmul(
                        factor, output
                    )  # the shape of factor is (d, b, c)
                    output = (
                        output.view(d, b, *batch_shape)
                        .transpose(0, 1)
                        .reshape(b * d, *batch_shape)
                    )
                else:
                    # general case a > 1 and d > 1
                    batch_shape = output.shape[1:]
                    output = (
                        output.view(a, c, d, *batch_shape)
                        .transpose(1, 2)
                        .reshape(a * c * d, *batch_shape)
                    )
                    output = torch.matmul(
                        factor, output
                    )  # the shape of factor is (d, b, c)
                    output = (
                        output.view(a, d, b, *batch_shape)
                        .transpose(1, 2)
                        .reshape(a * b * d, *batch_shape)
                    )

            elif self.algo == "dense":
                output = torch.matmul(factor, output)
            elif self.algo == "einsum":
                a, b, c, d = self.patterns[i]
                output = rearrange(
                    output, "(a c d) ... -> a c d ...", a=a, c=c, d=d
                )
                output = einsum(
                    output,
                    factor,
                    "a c d ..., a b c d -> a b d ...",
                )
                output = rearrange(output, "a b d ...-> (a b d) ...")
            else:
                raise NotImplementedError
        return output

    def forward(self, tensor):
        """
        Parameters:
            tensor: (..., in_size) if real
        Return:
            output: (..., out_size) if real
        """
        output = self.pre_process(tensor)

        if self.bs_last:
            output = self.forward_bs_last(output)
        else:
            output = self.forward_bs_first(output)

        return self.post_process(tensor, output)

    def get_dense_product(
        self,
    ):
        """
        Return the corresponding dense weight matrix with current parameters.
        Convert each factor to dense format to be sure.
        """
        list_factors = []  # from right to left

        for factor, pattern in zip(self.factors, self.patterns):
            a, b, c, d = pattern
            if self.algo == "bmm":
                list_factors.append(
                    from_bmm_format_to_dense(factor.view(a, d, b, c), pattern)
                )
            elif self.algo == "sparse":
                list_factors.append(factor.to_dense())
            elif self.algo == "bsr":
                if d == 1:
                    list_factors.append(factor.to_dense())
                elif a == 1:
                    tmp = factor.to_dense()
                    intermediate = torch.zeros(
                        a,
                        d,
                        b,
                        c,
                        dtype=tmp.dtype,
                        layout=tmp.layout,
                        device=tmp.device,
                    )
                    for i in range(d):
                        intermediate[0, i, :, :] = tmp[
                            i * b : (i + 1) * b, i * c : (i + 1) * c
                        ]
                    list_factors.append(
                        from_bmm_format_to_dense(intermediate, pattern)
                    )
                else:
                    tmp = factor.to_dense()
                    intermediate = torch.zeros(
                        a,
                        d,
                        b,
                        c,
                        dtype=tmp.dtype,
                        layout=tmp.layout,
                        device=tmp.device,
                    )
                    for j in range(a):
                        for i in range(d):
                            idx = j * d + i
                            intermediate[j, i, :, :] = tmp[
                                idx * b : (idx + 1) * b, idx * c : (idx + 1) * c
                            ]
                    list_factors.append(
                        from_bmm_format_to_dense(intermediate, pattern)
                    )
            elif self.algo == "dense":
                list_factors.append(factor)
            elif self.algo in ["kernel"]:
                list_factors.append(
                    from_kernel_format_to_dense(
                        factor, pattern,
                    )
                )
            elif self.algo == "einsum":
                list_factors.append(
                    from_abcd_to_dense(factor)
                )

        dense = list_factors[0]
        for factor in list_factors[1:]:
            dense = torch.matmul(factor, dense)
        return dense

    def pre_process(self, tensor):
        """
        Pre-processing to handle input that are not 2D tensors with shape (batch_dim, feature_dim)
        """
        if self.bs_last:
            output = tensor.view(tensor.size(0), -1)  # Reshape to (in_size, N)
        else:
            output = tensor.view(
                -1, tensor.size(-1)
            )  # Reshape to (N, in_size)
        return output

    def post_process(self, tensor, output):
        """
        Post-processing to handle input that are not 2D tensors with shape (batch_dim, feature_dim)
        """
        if self.bs_last:
            sizes = tensor.shape[1:]
            if self.bias is not None:
                output = (
                    output + self.bias
                )  # here output is of shape (out_size, batch) and bias is of shape (out_size, 1)
            return output.view(self.out_size, *sizes)

        else:
            sizes = tensor.shape[:-1]
            if self.bias is not None:
                output = output + self.bias
            return output.view(*sizes, self.out_size)

    def extra_repr(self):
        s = "in_size={}, out_size={}, num_mat={}, patterns={}, algo={}, bias={}".format(
            self.in_size,
            self.out_size,
            self.num_mat,
            self.patterns,
            self.algo,
            self.bias is not None,
        )
        return s
