import torch
import torch.backends.cudnn
import torch.backends.cuda


def forward_pass(model, x):
    with torch.no_grad():
        return model(x)


def pattern_is_dense(pattern):
    return pattern[0] == pattern[3] == 1


def set_device_and_get_device_name(device_id, device):
    if device_id >= torch.cuda.device_count():
        raise ValueError(f"args.device_id {device_id} is out of range.")
    if device == "cuda":
        torch.cuda.set_device(device_id)
        device_name = torch.cuda.get_device_name(device_id)
        print(f"using GPU = {device_name}")
    elif device == "cpu":
        device_name = "cpu"
        print("using CPU")
    device_name = device_name.replace(" ", "_")
    return device_name


def get_dtype(precision):
    if precision == "fp32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            False
        )
    elif precision == "fp16":
        dtype = torch.float16
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            True
        )
    elif precision == "tf32":
        dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        raise NotImplementedError
    return dtype


def save_measurements(save_dir, results_df):
    if save_dir is not None:
        results_df.to_csv(save_dir / "measurements.csv")


def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_path_submodules_vit(
    arch,
    granularity,
    algo,
    bs_position,
    precision,
    batch_size,
    device_name,
    patterns_d_d,
    patterns_d_4d,
    patterns_4d_d,
    sdpa_version,
    split_qkv,
):
    assert bs_position in ["bs_first", "bs_last"]
    assert algo is None or algo in [
        "nn_linear",
        "unfused_linear",
        "sparse",
        "dense",
        "bmm",
        "kernel",
        "bsr",
        "einsum",
    ]
    assert precision in ["fp32", "fp16"]
    assert (sdpa_version is None) or sdpa_version in [
        "flash",
        "equivalent",
        "default",
    ]
    assert (split_qkv is None) or isinstance(split_qkv, bool)
    # writing convention coming from the parsing of patterns
    if isinstance(patterns_d_d, tuple):
        patterns_d_d = [list(patterns_d_d)]
    if isinstance(patterns_d_4d, tuple):
        patterns_d_4d = [list(patterns_d_4d)]
    if isinstance(patterns_4d_d, tuple):
        patterns_4d_d = [list(patterns_4d_d)]

    path = ""
    assert arch is not None
    path += f"{arch}/"
    assert granularity is not None
    path += f"{granularity}/"
    if algo is not None:
        path += f"{algo}/"
    assert bs_position is not None
    path += f"{bs_position}/"
    assert precision is not None
    path += f"{precision}/"
    assert batch_size is not None
    path += f"batch_size_{batch_size}/"
    assert device_name is not None
    path += f"{device_name}/"
    if patterns_d_d is not None:
        patterns_d_d = str(patterns_d_d)
        patterns_d_d = patterns_d_d.replace(" ", "")
        path += f"patterns_d_d_{patterns_d_d}/"
    if patterns_d_4d is not None:
        patterns_d_4d = str(patterns_d_4d)
        patterns_d_4d = patterns_d_4d.replace(" ", "")
        path += f"patterns_d_4d_{patterns_d_4d}/"
    if patterns_4d_d is not None:
        patterns_4d_d = str(patterns_4d_d)
        patterns_4d_d = patterns_4d_d.replace(" ", "")
        path += f"patterns_4d_d_{patterns_4d_d}/"
    if sdpa_version is not None:
        path += f"sdpa_version_{sdpa_version}/"
    if split_qkv is not None:
        path += f"split_qkv_{split_qkv}/"
    path += "measurements.csv"
    return path


def parse_patterns(patterns):
    """
    Parse patterns from string "${A},${B},${C},${D}" to tuple of integers (A, B, C, D)
    """
    if patterns is None:
        return None
    for i, r in enumerate(patterns):
        rs = r.split(",")
        tmp = []
        for s in rs:
            tmp.append(int(s))
        assert len(tmp) == 4
        patterns[i] = tmp
    return patterns


def get_in_size_out_size(patterns):
    first_pattern = patterns[0]
    last_pattern = patterns[-1]

    in_size = first_pattern[0] * first_pattern[2] * first_pattern[3]
    out_size = last_pattern[0] * last_pattern[1] * last_pattern[3]

    return in_size, out_size
