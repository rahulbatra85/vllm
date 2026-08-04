# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 MoE backend selection for the AITER triton a16w4 kernel on gfx950.

GPU-free: mocks the platform as gfx950 and enables AITER, then exercises the
oracle. gfx950 keeps the CK kernel (``AiterExperts``) by default; the triton
a16w4 kernel is reachable only via ``--moe-backend aiter_triton``, which maps
to its own backend enum member so that weight prep picks the triton layout.
"""

import dataclasses
from unittest.mock import patch

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle  # noqa: E402
from tests.kernels.moe.utils import make_dummy_moe_config  # noqa: E402
from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a8_moe import (  # noqa: E402
    AiterW4A16ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (  # noqa: E402
    AiterExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (  # noqa: E402
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    Mxfp4MoeBackend,
    backend_to_kernel_cls,
    map_mxfp4_backend,
    select_mxfp4_moe_backend,
    uses_triton_mxfp4_weight_format,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kMxfp4Static,
)


def _config(moe_backend: str = "auto", ep_size: int = 1):
    """A gpt-oss-shaped config: SwiGLU-OAI activation, Renormalize routing."""
    cfg = make_dummy_moe_config(
        num_experts=32,
        experts_per_token=4,
        hidden_dim=2880,
        intermediate_size=2880,
        activation=MoEActivation.SWIGLUOAI,
    )
    cfg = dataclasses.replace(
        cfg,
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )
    if ep_size != 1:
        cfg = dataclasses.replace(
            cfg,
            moe_parallel_config=dataclasses.replace(
                cfg.moe_parallel_config, ep_size=ep_size, use_ep=True
            ),
        )
    return cfg


def _gfx950(monkeypatch: pytest.MonkeyPatch):
    """Present as gfx950 with AITER MoE enabled.

    Both expert classes import ``on_gfx950``/``on_gfx1250`` lazily from
    ``vllm.platforms.rocm`` inside their support checks, so patching there is
    enough. The activation override reads the current vLLM config, which this
    test does not build.
    """
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)
    monkeypatch.setattr(rocm_aiter_ops, "_FMOE_ENABLED", True)
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    return patch.multiple(
        "vllm.platforms.rocm",
        on_gfx950=lambda: True,
        on_gfx1250=lambda: False,
    )


def test_aiter_triton_registered():
    """--moe-backend aiter_triton resolves to the monolithic kernel alone."""
    assert map_mxfp4_backend("aiter_triton") == [
        Mxfp4MoeBackend.AITER_MXFP4_BF16_TRITON
    ]
    assert backend_to_kernel_cls(Mxfp4MoeBackend.AITER_MXFP4_BF16_TRITON) == [
        AiterW4A16ExpertsMonolithic
    ]


def test_gfx950_default_is_ck(monkeypatch: pytest.MonkeyPatch):
    """The opt-in must not change what gfx950 picks by default."""
    with _gfx950(monkeypatch):
        assert select_mxfp4_moe_backend(_config("auto")) == (
            Mxfp4MoeBackend.AITER_MXFP4_BF16,
            AiterExperts,
        )
        assert select_mxfp4_moe_backend(_config("aiter")) == (
            Mxfp4MoeBackend.AITER_MXFP4_BF16,
            AiterExperts,
        )


def test_ck_pinned_at_index_zero():
    """The Kimi-K3 SiTU and Quark paths pin backend_to_kernel_cls(...)[0]."""
    assert backend_to_kernel_cls(Mxfp4MoeBackend.AITER_MXFP4_BF16)[0] is AiterExperts


def test_explicit_aiter_triton_selects_monolithic(monkeypatch: pytest.MonkeyPatch):
    """Opting in reaches the triton a16w4 kernel on gfx950."""
    with _gfx950(monkeypatch):
        assert select_mxfp4_moe_backend(_config("aiter_triton")) == (
            Mxfp4MoeBackend.AITER_MXFP4_BF16_TRITON,
            AiterW4A16ExpertsMonolithic,
        )


def test_weight_format_follows_backend(monkeypatch: pytest.MonkeyPatch):
    """The two gfx950 backends must disagree on weight format: the CK kernel
    reads shuffled plain tensors, the triton kernel a PrecisionConfig."""
    with _gfx950(monkeypatch):
        assert uses_triton_mxfp4_weight_format(Mxfp4MoeBackend.AITER_MXFP4_BF16_TRITON)
        assert not uses_triton_mxfp4_weight_format(Mxfp4MoeBackend.AITER_MXFP4_BF16)


def test_expert_parallel_rejected(monkeypatch: pytest.MonkeyPatch):
    """apply() routes over all global experts and ignores expert_map, so EP
    would read weights for experts this rank does not hold."""
    with _gfx950(monkeypatch):
        supported, reason = AiterW4A16ExpertsMonolithic.is_supported_config(
            AiterW4A16ExpertsMonolithic,
            _config("aiter_triton", ep_size=2),
            kMxfp4Static,
            None,
            FusedMoEActivationFormat.Standard,
        )
    assert supported is False
    assert "parallel config" in reason
