import numpy as np
import torch

class BmmBsFirst(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, x, w_kron, pattern):
        out = forward_bmm_bs_first(w_kron, x, pattern)
        ctx.save_for_backward(x, w_kron)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        pass


bmm_bs_first = BmmBsFirst.apply


class BmmBsLast(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, x, w_kron, pattern):
        out = forward_bmm_bs_last(w_kron, x, pattern)
        ctx.save_for_backward(x, w_kron)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dout):
        pass


bmm_bs_last = BmmBsLast.apply


def forward_bmm_bs_first(w_kron, x, pattern):
    """
    Generalization to arbitrary (a,b,c,d) of the code of Dao et al., see the NOTICE file.
    """
    a, b, c, d = pattern
    assert w_kron.shape == (a * d, b, c)
    assert x.shape[-1] == a * c * d
    batch_shape = x.shape[:-1]
    n = np.prod(batch_shape)

    x = x.reshape(n, a, c, d).transpose(-1, -2).reshape(n, a * d, c).contiguous().transpose(0, 1)  # (a * d, n, c)
    out = torch.empty(n, a * d, b, device=x.device, dtype=x.dtype).transpose(0, 1)  # (a * d, n, b)
    torch.bmm(x, w_kron.transpose(-1, -2), out=out)  # (a * d, n, b)
    out = out.transpose(0, 1).reshape(n, a, d, b).transpose(-1, -2).reshape(*batch_shape, a * b * d)
    return out


def forward_bmm_bs_last(w_kron, x, pattern):
    """
    Generalization to arbitrary (a,b,c,d) of the code of Dao et al. (see the NOTICE file), adapted to batch-size-last.
    """
    a, b, c, d = pattern
    assert w_kron.shape == (a * d, b, c)
    assert x.shape[0] == a * c * d
    batch_shape = x.shape[1:]
    n = np.prod(batch_shape)

    x = x.reshape(a, c, d, n).transpose(1, 2).reshape(a * d, c, n).contiguous()  # (a * d, c, n)
    out = torch.empty(a * d, b, n, device=x.device, dtype=x.dtype)  # (a * d, b, n)
    torch.bmm(w_kron, x, out=out)
    out = out.reshape(a, d, b, n).transpose(1, 2).reshape(a * b * d, *batch_shape)
    return out
