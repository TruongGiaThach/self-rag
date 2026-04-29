"""
vLLM Compatibility Shim for SELF-RAG
=====================================
Mục đích:
  1. **Low-VRAM**: Force INT8 quantization (load_format="bitsandbytes") → model 7B
     chỉ cần ~7 GB VRAM thay vì ~14 GB FP16.
  2. **API compat**: vLLM >= 0.3.0 đổi kiểu trả về của logprobs từ
       Dict[int, float]  →  Dict[int, Logprob(logprob, rank, decoded_token)]
     Scripts gốc của tác giả dùng  float(prob)  trực tiếp, sẽ fail với API mới.
     Shim này bọc dict logprobs để mọi truy cập trả về float như cũ.

Cách dùng:
  Thay  `from vllm import LLM, SamplingParams`
  bằng  `from vllm_compat import LLM, SamplingParams`
  trong các script inference. Không cần thay đổi gì khác.
"""

import vllm as _vllm
from vllm import SamplingParams  # re-export không đổi


# ─────────────────────────────────────────────────────────────────
# Logprobs compatibility wrapper
# ─────────────────────────────────────────────────────────────────

class _FloatLogprobsDict(dict):
    """
    Bọc Dict[int, Logprob] để trả về float logprob khi truy cập theo key,
    giúp code viết cho vLLM 0.2.x chạy đúng trên vLLM >= 0.3.0.
    """

    def __getitem__(self, key):
        val = super().__getitem__(key)
        # vLLM >= 0.3.0: Logprob object có thuộc tính .logprob (float)
        if hasattr(val, "logprob"):
            return val.logprob
        return float(val)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return super().__contains__(key)


def _patch_completion_output(comp_output):
    """Patch in-place một CompletionOutput object."""
    # Patch logprobs list
    if comp_output.logprobs is not None:
        patched = [
            _FloatLogprobsDict(step) if step is not None else None
            for step in comp_output.logprobs
        ]
        try:
            comp_output.logprobs = patched
        except AttributeError:
            # Nếu dataclass frozen, dùng object.__setattr__
            object.__setattr__(comp_output, "logprobs", patched)

    # Patch cumulative_logprob: vLLM mới có thể trả về None nếu không yêu cầu
    if getattr(comp_output, "cumulative_logprob", None) is None:
        try:
            comp_output.cumulative_logprob = 0.0
        except AttributeError:
            object.__setattr__(comp_output, "cumulative_logprob", 0.0)

    return comp_output


def _patch_request_output(req_output):
    """Patch in-place một RequestOutput object (toàn bộ outputs)."""
    for comp_output in req_output.outputs:
        _patch_completion_output(comp_output)
    return req_output


# ─────────────────────────────────────────────────────────────────
# Drop-in LLM replacement
# ─────────────────────────────────────────────────────────────────

class LLM(_vllm.LLM):
    """
    Drop-in replacement cho vllm.LLM với hai tính năng bổ sung:
      - Tự động inject load_format="bitsandbytes" + gpu_memory_utilization=0.85
        để chạy trên GPU VRAM < 12 GB (INT8 ~7 GB cho Llama-2-7B).
      - Patch kết quả generate() để logprobs luôn trả về float.
    Tất cả args/kwargs khác được truyền thẳng xuống vllm.LLM.
    """

    def __init__(self, *args, **kwargs):
        # INT8 quantization qua bitsandbytes — chỉ set nếu caller chưa chỉ định
        kwargs.setdefault("load_format", "bitsandbytes")
        # Giới hạn VRAM allocation: trừ buffer cho KV-cache và overhead
        kwargs.setdefault("gpu_memory_utilization", 0.85)
        super().__init__(*args, **kwargs)

    def generate(self, *args, **kwargs):
        results = super().generate(*args, **kwargs)
        return [_patch_request_output(r) for r in results]
