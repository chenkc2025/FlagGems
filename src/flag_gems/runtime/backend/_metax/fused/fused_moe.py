import contextlib
import threading
from typing import Any

from flag_gems.fused import fused_moe as generic_fused_moe

_PATCH_LOCK = threading.RLock()
_GENERIC_GET_DEFAULT_CONFIG = generic_fused_moe.get_default_config
_PLAIN_HALF_CONFIG_DTYPES = ("fp16", "bf16")
_DIRECT_SUM_DISABLED_MIN_TOKENS = 1 << 60


def _is_qwen3_5_moe_shape(
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    gemm_stage: str,
) -> bool:
    if dtype not in _PLAIN_HALF_CONFIG_DTYPES or E != 512 or topk != 10:
        return False

    if gemm_stage == "gemm1":
        return (N, K) == (2048, 4096)
    if gemm_stage == "gemm2":
        return (N, K) == (4096, 1024)
    return False


def _metax_get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
    gemm_stage: str = "gemm1",
    enable_gemm_fast_path: bool = False,
) -> dict[str, Any]:
    if not _is_qwen3_5_moe_shape(E, N, K, topk, dtype, gemm_stage):
        return _GENERIC_GET_DEFAULT_CONFIG(
            M,
            E,
            N,
            K,
            topk,
            dtype,
            block_shape,
            gemm_stage,
            enable_gemm_fast_path,
        )

    if M <= 1024:
        block_m, block_n, block_k = 16, 128, 64
        group_m, num_warps, num_stages = 1, 4, 2
    elif M <= 2048:
        block_m, block_n, block_k = 64, 128, 64
        group_m, num_warps, num_stages = 1, 8, 2
    elif M <= 4096:
        block_m, block_n, block_k = 64, 256, 32
        group_m, num_warps, num_stages = 1, 8, 2
    elif M <= 8192:
        block_m, block_k = 128, 32
        block_n = 256 if gemm_stage == "gemm2" else 128
        group_m, num_warps = 1, 8
        num_stages = 2 if gemm_stage == "gemm2" else 3
    elif M <= 16384:
        block_m, block_k = 128, 32
        block_n = 256 if gemm_stage == "gemm2" else 128
        group_m, num_warps = 1, 8
        num_stages = 2 if gemm_stage == "gemm2" else 3
    else:
        block_m, block_k = 128, 32
        block_n = 256 if gemm_stage == "gemm2" else 128
        group_m, num_warps = 8, 8
        num_stages = 2 if gemm_stage == "gemm2" else 3

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def _is_qwen3_5_plain_half_call(args, kwargs) -> bool:
    try:
        hidden_states = args[0] if len(args) > 0 else kwargs["hidden_states"]
        w1 = args[1] if len(args) > 1 else kwargs["w1"]
        w2 = args[2] if len(args) > 2 else kwargs["w2"]
        topk_ids = args[4] if len(args) > 4 else kwargs["topk_ids"]
    except (KeyError, IndexError):
        return False

    return (
        str(hidden_states.dtype) in ("torch.float16", "torch.bfloat16")
        and tuple(w1.shape) == (512, 2048, 4096)
        and tuple(w2.shape) == (512, 4096, 1024)
        and topk_ids.ndim == 2
        and topk_ids.size(1) == 10
    )


@contextlib.contextmanager
def _metax_moe_config_patch(disable_direct_sum: bool):
    with _PATCH_LOCK:
        original_get_default_config = generic_fused_moe.get_default_config
        original_direct_sum_min_tokens = generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS
        generic_fused_moe.get_default_config = _metax_get_default_config
        if disable_direct_sum:
            generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS = _DIRECT_SUM_DISABLED_MIN_TOKENS
        try:
            yield
        finally:
            generic_fused_moe.get_default_config = original_get_default_config
            generic_fused_moe.MOE_DIRECT_SUM_MIN_TOKENS = original_direct_sum_min_tokens


def fused_experts_impl(*args, **kwargs):
    with _metax_moe_config_patch(_is_qwen3_5_plain_half_call(args, kwargs)):
        return generic_fused_moe.fused_experts_impl(*args, **kwargs)


def inplace_fused_experts(*args, **kwargs):
    with _metax_moe_config_patch(_is_qwen3_5_plain_half_call(args, kwargs)):
        return generic_fused_moe.inplace_fused_experts(*args, **kwargs)


def outplace_fused_experts(*args, **kwargs):
    with _metax_moe_config_patch(_is_qwen3_5_plain_half_call(args, kwargs)):
        return generic_fused_moe.outplace_fused_experts(*args, **kwargs)
