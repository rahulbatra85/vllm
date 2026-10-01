# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT-built HIP version of the four Lamport P2P kernels (csrc/lamport_p2p.hip)."""

from __future__ import annotations

import functools
import os

import torch


@functools.cache
def lamport_hip_ops():
    """Build (first use, cached under TORCH_EXTENSIONS_DIR) and load the extension."""
    from torch.utils.cpp_extension import load

    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    src = os.path.join(os.path.dirname(__file__), "csrc", "lamport_p2p.hip")
    return load(
        name="inkling_lamport_p2p",
        sources=[src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"--offload-arch={arch.split(':')[0]}"],
        verbose=False,
    )
