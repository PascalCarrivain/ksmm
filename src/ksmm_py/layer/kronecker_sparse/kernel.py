import torch
import os
import inspect
from torch.utils.cpp_extension import load
from typing import Any, Tuple
import warnings
from warnings import warn
warnings.simplefilter(action="always")

filename = inspect.getframeinfo(inspect.currentframe()).filename
directory = os.path.dirname(os.path.abspath(filename))
path = directory + "/kernel"

if not os.path.isdir("build"):
    warn("Did not find build folder." +
         "\nPlease create build folder at the root of fast-butterfly folder.")

kernel = load(
    name="kernel",
    sources=[path + ".cpp", path + ".cu"],
    verbose=True,
    extra_cflags=["-g"],
    build_directory="build/",
)


class Kernel(torch.autograd.Function):
    @staticmethod
    def forward(
        input,
        factor,
        bs_last,
        a,
        b,
        c,
        d,
        fp16=False,
    ):
        output = kernel.forward(
            input,
            factor,
            bs_last,
            a,
            b,
            c,
            d,
            fp16,
        )

        return output

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        pass

    @staticmethod
    def backward(ctx, grad_output):
        input, matrix, output = ctx.saved_tensors
        grad_input, grad_rest = kernel.backward(
            grad_output, input, matrix
        )
        return grad_input, *grad_rest