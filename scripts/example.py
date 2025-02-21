from ksmm_py.layer.kronecker_sparse.interface import KSLinear
import torch

# Two factors W = K_1 K_2
patterns = [(6, 64, 64, 1), (1, 768, 192, 2)] # [(a_2, b_2, c_2, d_2), (a_1, b_1, c_1, d_1)], and in general for W = K_1 \dots K_L, patterns[0] should correspond to K_L (right-most factor), patterns[1] to K_{L-1}, ..., patterns[-1] to K_1 (left-most factor)
dim_in = 6 * 64 * 1 # a_2 * b_2 * d_2
batch_size = 25088 # support for other batch-sizes is in progress; for now we finetuned the hyperparameters of our kernel only for 25'088= 196 (ctx length) * 128 (batch size) = effective ViT batch size
batch_size_last = True # True or False, whether the batch size is the last or the first dimension of the input and output tensors
device='cpu' # 'cuda' or 'cpu', the kernel is only available on GPU

x = torch.randn((dim_in, batch_size) if batch_size_last else (batch_size, dim_in), 
                dtype=torch.float32, # torch.float16 is fine too
                device=device)

# either provide no weights (or set weights = None), and let KSLinear initialize each factor with uniform(-1/sqrt(c), 1/sqrt(c)), or provide a list of tensor of shape (a, b, c, d) for each pattern (a, b, c, d)
weights = [torch.ones(*pattern) for pattern in patterns]

ksl = KSLinear(patterns=patterns, weights=weights, algo='kernel', dtype=x.dtype, bs_last=batch_size_last, device=device)
y = ksl(x)
ksl = KSLinear(patterns=patterns, weights=weights, algo='dense', dtype=x.dtype, bs_last=batch_size_last, device=device)
z = ksl(x)
print("Relative error between kernel and dense implementation: ",
    torch.linalg.norm(y - z) / torch.linalg.norm(z))