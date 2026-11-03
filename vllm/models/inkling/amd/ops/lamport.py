# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deadlock-free fused RS + short-conv + AG + residual + RMSNorm.

The public integration surface is ``LamportRSConv.rs_sconv_ag_add_norm``.

Liveness
--------
Large grids are deliberately split at the two communication dependencies:

  1. ``_publish_input_kernel`` only publishes rank partials.
  2. ``_reduce_insert_kernel`` waits, reduces, and inserts the local shard.
  3. ``_sconv_publish_kernel`` only computes and publishes the local result.
  4. ``_gather_norm_kernel`` waits, gathers, and normalizes.

Without PDL, CUDA stream order completes a producer before its consumer.  With
PDL, every producer CTA posts all peer stores before triggering its dependent,
and the consumer executes ``gdc_wait`` before polling.  A consumer therefore
waits only for stores from a producer that is already running (or complete) on
another GPU.  It never waits for another block in its own grid.  Consequently
spinning consumers cannot occupy resources needed by any producer, and the
wait-for graph has no cycle.  The proof is independent of grid size, block
dispatch order, and occupancy.

For one token, the first three phases use eight independent channel slices to
expose enough CTA parallelism for decode latency.  Each slice has exclusive
ownership of its cache and Lamport columns; the gather/RMSNorm phase retains
one CTA per token so no cross-CTA reduction or completion counter is needed.

Immediate buffer reuse (including replay of a captured CUDA graph) is also
safe.  Rank R cannot republish input for call n+1 until its gather for n has
finished; that gather waited for owner O's output, which O publishes only
after consuming R's input for n.  Likewise, R cannot republish output for n+1
until its reduction for n+1 has observed destination D's input; D publishes
that input only after its gather consumed R's output for n.  Thus every prior
read happens-before a same-slot rewrite.  Three generations reduce incidental
coupling for ordinary launches, but correctness does not depend on rotation.

The payload itself is the Lamport flag.  A 32-bit store publishes two bf16s
atomically; 0x80008000 (two negative zeroes) denotes an empty pair.  Real
negative zeroes are changed to positive zero before publication.  Consumers
use volatile 32-bit loads and restore the sentinel after consuming a slot.
"""

from __future__ import annotations

import functools
import os

import torch

from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_MAX_TOKENS = 16384
# ops/csrc/lamport_p2p.cu:ipc_alloc modes
_IPC_MODES = {"uncached": 0, "finegrained": 1, "coarse": 2}


@functools.cache
def lamport_hip_ops():
    """JIT-build (cached under TORCH_EXTENSIONS_DIR) and load csrc/lamport_p2p.cu."""
    from torch.utils.cpp_extension import load

    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    src = os.path.join(os.path.dirname(__file__), "csrc", "lamport_p2p.cu")
    return load(
        name="inkling_lamport_p2p",
        sources=[src],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"--offload-arch={arch.split(':')[0]}"],
        verbose=False,
    )

class LamportRSConv:
    """Persistent symmetric buffers for one TP group."""

    def _initialize_lamport_buffers(self) -> None:
        """Arm every slot and collectively verify the exact sentinel bits.

        Do not replace this with a zero-fill: Lamport readiness distinguishes
        the bf16 bit pattern for -0.0 (0x8000) from every published payload.
        The validation is deliberately collective so one rank cannot enter the
        first polling kernel while another rank still has an unarmed buffer.
        """
        if not self.buf_in.is_contiguous() or not self.buf_out.is_contiguous():
            raise RuntimeError("Lamport symmetric buffers must be contiguous")
        if self.buf_in.numel() % 2 or self.buf_out.numel() % 2:
            raise RuntimeError("Lamport buffers must contain whole uint32 pairs")

        # Fill through int16 so each bf16 lane receives the exact -0.0 bits.
        # This covers all three generations and all max-token slots, including
        # slots that a smaller first invocation does not touch.
        self.buf_in.view(torch.int16).fill_(-0x8000)
        self.buf_out.view(torch.int16).fill_(-0x8000)
        torch.accelerator.synchronize(self.device)
        self.tp.barrier()

        # Scan the complete local allocations.  Since every rank performs the
        # scan and participates in MAX, success proves every symmetric backing
        # allocation was armed before any rank is allowed to use generation 0.
        bad = torch.logical_or(
            self.buf_in.view(torch.int16).ne(-0x8000).any(),
            self.buf_out.view(torch.int16).ne(-0x8000).any(),
        ).to(dtype=torch.float32)
        bad = self.tp.all_reduce(bad)
        if int(bad.item()) != 0:
            raise RuntimeError("Lamport sentinel initialization failed on a TP rank")
        self.tp.barrier()

    def __init__(
        self, hidden_size: int, window_size: int, max_tokens: int = _MAX_TOKENS
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        tp = get_tp_group()
        self.tp = tp
        self.group = tp.device_group
        self.world_size = tp.world_size
        self.rank = tp.rank_in_group
        self.device = torch.device(tp.device)
        if self.world_size not in (2, 4, 8):
            raise ValueError(f"TP world size must be 2, 4, or 8, got {self.world_size}")
        if hidden_size % (2 * self.world_size) != 0:
            raise ValueError("hidden size must produce an even shard on every rank")
        if window_size != 4:
            raise ValueError(f"short-conv window size must be 4, got {window_size}")
        if max_tokens < 1 or max_tokens > _MAX_TOKENS:
            raise ValueError(f"max_tokens must be in [1, {_MAX_TOKENS}]")

        is_cross_node = not all(in_the_same_node_as(tp.cpu_group))
        assert is_cross_node is False, "Cannot use LamportRSConv on cross_node on ROCm"
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.shard_size = hidden_size // self.world_size
        # Three generations follow FlashInfer's Lamport layout.  A generation
        # is reused only after two intervening collective calls.
        self.num_buffers = 3
        self.input_generation_bytes = max_tokens * hidden_size * 2
        self.output_generation_bytes = max_tokens * hidden_size * 2

        # LAMPORT_MEM=ipc (default) shares hipIpc buffers like vLLM's / aiter's
        # custom all-reduce; =symm_mem uses PyTorch symmetric memory.
        self.mem = os.environ.get("LAMPORT_MEM", "ipc")
        self.ipc_mode = os.environ.get("LAMPORT_IPC_MODE", "uncached")
        if self.mem not in ("ipc", "symm_mem"):
            raise ValueError(f"LAMPORT_MEM must be ipc or symm_mem: {self.mem}")
        if self.ipc_mode not in _IPC_MODES:
            raise ValueError(f"LAMPORT_IPC_MODE must be one of {list(_IPC_MODES)}")
        in_shape = (self.num_buffers, max_tokens, self.world_size, self.shard_size)
        out_shape = (self.num_buffers, max_tokens, hidden_size)
        self._ipc_local: list[int] = []
        self._ipc_peers: list[int] = []
        self._ipc_tables: list[torch.Tensor] = []
        if self.mem == "ipc":
            self._ext = lamport_hip_ops()
            self.buf_in, self.input_peer_ptrs = self._ipc_buffer(in_shape)
            self.buf_out, self.output_peer_ptrs = self._ipc_buffer(out_shape)
        else:
            self.buf_in = symm_mem.empty(
                *in_shape, dtype=torch.bfloat16, device=self.device
            )
            self.buf_out = symm_mem.empty(
                *out_shape, dtype=torch.bfloat16, device=self.device
            )
            group_name = self.group.group_name
            input_handle = symm_mem.rendezvous(self.buf_in, group_name)
            output_handle = symm_mem.rendezvous(self.buf_out, group_name)
            self.input_peer_ptrs = input_handle.buffer_ptrs_dev
            self.output_peer_ptrs = output_handle.buffer_ptrs_dev
            self._input_handle = input_handle
            self._output_handle = output_handle
        self._initialize_lamport_buffers()
        self.generation = 0

        # Diagnostic: LAMPORT_MAX_SPINS>0 bounds the Lamport spin so a starved
        # waiter reports which peer's partial never arrived, instead of hanging
        # the stream and stalling every later collective on it. 0 = production
        # behaviour (unbounded).
        self.max_spins = int(os.environ.get("LAMPORT_MAX_SPINS", "0"))
        self._spin_limit = self.max_spins if self.max_spins else (1 << 40)
        # [0]=input stalls, [1]=output stalls, [2]=max pending pairs,
        # [3]=first token, [4]=max split, [8+i]=pending pairs from source i.
        self.stall = torch.zeros(8 + self.world_size, dtype=torch.int32, device=self.device)
        self.stall[3] = 2**30

        # LAMPORT_KERNELS=hip (default) runs the four kernels from
        # csrc/lamport_p2p.cu on the same buffers; =triton uses the Triton ones.
        # Built here so no compile happens during capture.
        #self.kernels = os.environ.get("LAMPORT_KERNELS", "hip")
        #if self.kernels not in ("triton", "hip"):
        #    raise ValueError(f"LAMPORT_KERNELS must be triton or hip: {self.kernels}")
        self.kernels = "hip"
        self._hip = lamport_hip_ops()
        logger.info(
            "Lamport P2P fused RS/sconv/AG enabled (%s kernels, mem=%s, tp=%d, "
            "max_tokens=%d)",
            self.kernels,
            f"ipc/{self.ipc_mode}" if self.mem == "ipc" else self.mem,
            self.world_size,
            max_tokens,
        )

    def _ipc_buffer(self, shape: tuple[int, ...]) -> tuple[torch.Tensor, int]:
        """Allocate a hipIpc-shared bf16 buffer.

        Returns the local view and the device address of a uint64 peer-pointer
        table, the same form as a symm_mem handle's ``buffer_ptrs_dev``.
        """
        nbytes = 2
        for s in shape:
            nbytes *= s
        ptr, handle = self._ext.ipc_alloc(nbytes, _IPC_MODES[self.ipc_mode])
        self._ipc_local.append(ptr)
        handles: list[torch.Tensor | None] = [None] * self.world_size
        torch.distributed.all_gather_object(handles, handle, group=self.tp.cpu_group)
        ptrs = []
        for r, h in enumerate(handles):
            if r == self.rank:
                ptrs.append(ptr)
            else:
                ptrs.append(self._ext.ipc_open(h))
                self._ipc_peers.append(ptrs[-1])
        local = self._ext.as_tensor(ptr, list(shape), self.device.index)
        table = torch.tensor(ptrs, dtype=torch.uint64, device=self.device)
        self._ipc_tables.append(table)
        return local, table.data_ptr()

    def close(self) -> None:
        """Release hipIpc buffers (no-op for symmetric memory)."""
        if not self._ipc_local and not self._ipc_peers:
            return
        torch.accelerator.synchronize(self.device)
        self.buf_in = self.buf_out = None
        for p in self._ipc_peers:
            self._ext.ipc_close(p)
        for p in self._ipc_local:
            self._ext.ipc_free(p)
        self._ipc_local, self._ipc_peers = [], []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


    def usable(self, num_tokens: int) -> bool:
        return 0 < num_tokens <= self.max_tokens

    def rs_sconv_ag_add_norm(
        self,
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        conv_weight: torch.Tensor,
        norm_weight: torch.Tensor | None,
        eps: float,
        cache: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        seq_idx: torch.Tensor,
        slot_mapping: torch.Tensor,
        off_s: int,
        ws: int,
        block_size: int,
        shared_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return ``(normed | None, new_residual)``, both shaped ``[T, 6144]``."""
        tokens, hidden_size = residual.shape
        if not self.usable(tokens):
            raise ValueError(f"num_tokens must be in [1, {self.max_tokens}]")
        if hidden_size != self.hidden_size or residual.dtype != torch.bfloat16:
            raise ValueError("residual must be bf16 [T, 6144]")
        if (
            input_tensor.shape != residual.shape
            or input_tensor.dtype != torch.bfloat16
            or input_tensor.stride(1) != 1
        ):
            raise ValueError("input_tensor must be channel-contiguous bf16 [T, 6144]")
        if shared_tensor is not None and (
            shared_tensor.shape != input_tensor.shape
            or shared_tensor.dtype != input_tensor.dtype
            or shared_tensor.device != input_tensor.device
            or shared_tensor.stride(1) != 1
        ):
            raise ValueError("shared_tensor must match input_tensor")
        shard_size = hidden_size // self.world_size
        if conv_weight.shape != (shard_size, self.window_size):
            raise ValueError(
                f"conv_weight must have shape [{shard_size}, {self.window_size}]"
            )
        if (
            conv_weight.dtype != torch.bfloat16
            or conv_weight.stride(0) != self.window_size
        ):
            raise ValueError("conv_weight must be contiguous bf16")
        if norm_weight is not None and (
            norm_weight.shape != (hidden_size,) or norm_weight.dtype != torch.bfloat16
        ):
            raise ValueError("norm_weight must be bf16 [6144] or None")
        if cache.dtype != torch.bfloat16 or cache.ndim != 4:
            raise ValueError("cache must be a 4-D bf16 tensor")
        if shard_size % ws != 0 or cache.shape[1] != shard_size // ws:
            raise ValueError("cache head layout is inconsistent with ws")
        if off_s < 0 or off_s + ws > cache.shape[3]:
            raise ValueError("cache channel offset is out of bounds")
        if block_size != cache.shape[2]:
            # slot // block_size indexes the block dimension, so a block_size
            # that disagrees with the cache's own page size walks off the
            # allocation and faults the GPU instead of raising here. Pass
            # owner.cache_block_size, not owner.block_size (the conv window).
            raise ValueError(
                f"block_size {block_size} does not match the cache page size "
                f"{cache.shape[2]}; slot indexing would leave the allocation"
            )

        # The NVIDIA publish kernel folds shared_tensor in via HAS_SHARED,
        # accumulating in fp32 before the bf16 store. The ROCm kernel has no
        # such path, so materialize the same sum up front -- identical numerics,
        # at the cost of one extra elementwise pass.
        if shared_tensor is not None:
            input_tensor = (
                input_tensor.to(torch.float32) + shared_tensor.to(torch.float32)
            ).to(torch.bfloat16)

        index = self.generation
        input_offset = index * self.input_generation_bytes // 4
        output_offset = index * self.output_generation_bytes // 4
        normed = torch.empty_like(residual) if norm_weight is not None else None
        residual_out = torch.empty_like(residual)
        phase_splits = 8 if tokens == 1 and shard_size % 8 == 0 else 1
        phase_tile_p2 = triton.next_power_of_2(shard_size // phase_splits)
        phase_grid = (tokens, phase_splits)
        # Wide CTAs win before the grid saturates; smaller CTAs reduce register
        # pressure once high-throughput batches provide enough parallelism.
        if 128 <= tokens <= 2048:
            phase_warps = 16
        elif tokens > 2048:
            phase_warps = 8
        else:
            phase_warps = 4
        gather_warps = 4 if tokens >= 256 else 8
        hip = self._hip
        hip.publish_input(
            input_tensor, self.input_peer_ptrs, input_offset, shard_size,
            phase_splits, self.rank, self.world_size,
        )  # fmt: skip
        hip.reduce_insert(
            self.input_peer_ptrs, input_offset, self._spin_limit, self.stall, cache,
            slot_mapping, tokens, block_size, shard_size, phase_splits, ws, off_s,
            self.rank, self.world_size,
        )  # fmt: skip
        hip.sconv_publish(
            self.output_peer_ptrs, output_offset, residual, conv_weight, cache,
            positions, seq_idx, slot_mapping, block_table, block_size, shard_size,
            phase_splits, ws, off_s, self.rank, self.world_size,
        )  # fmt: skip
        hip.gather_norm(
            self.output_peer_ptrs, output_offset, self._spin_limit, self.stall,
            norm_weight if norm_weight is not None else residual,
            normed if normed is not None else residual_out, residual_out, eps,
            self.rank, norm_weight is not None,
        )  # fmt: skip
        self.generation = (index + 1) % self.num_buffers
        return self._check_stalls(tokens, phase_splits, index, normed, residual_out)
    


    def _check_stalls(self, tokens, phase_splits, index, normed, residual_out):
        """Raise (after a sync) if a bounded spin gave up; diagnostic only.

        Skipped while a CUDA graph is being captured (a sync there is illegal);
        the eager warmup runs before each capture are still checked.
        """
        if self.max_spins and not torch.cuda.is_current_stream_capturing():
            # Forces a sync, so this is diagnostic-only. Without it a starved
            # waiter is invisible: the stream never drains and the failure
            # surfaces much later as an unrelated NCCL collective timeout.
            counts = self.stall.tolist()
            if counts[0] or counts[1]:
                per_source = ", ".join(
                    f"rank{i}:{n}" for i, n in enumerate(counts[8:]) if n
                )
                self.stall.zero_()
                self.stall[3] = 2**30
                raise RuntimeError(
                    f"Lamport spin-wait starved on rank {self.rank} after "
                    f"{self.max_spins} spins: {counts[0]} input-phase and "
                    f"{counts[1]} output-phase CTAs gave up; worst CTA still "
                    f"missing {counts[2]} pairs at token {counts[3]} split "
                    f"{counts[4]}; unpublished partials from [{per_source}] "
                    f"(tokens={tokens}, splits={phase_splits}, gen={index})"
                )

        return normed, residual_out


_STATE: LamportRSConv | None = None
_STATE_FAILED = False


def initialize_lamport_rs_conv(
    hidden_size: int, window_size: int, max_num_batched_tokens: int
) -> None:
    """Collectively initialize the TP-group state during model construction."""
    global _STATE, _STATE_FAILED
    if _STATE is not None:
        if _STATE.hidden_size != hidden_size or _STATE.window_size != window_size:
            raise RuntimeError("all Lamport users must share hidden and window sizes")
        return
    if _STATE_FAILED or os.environ.get("LAMPORT_RS_SCONV", "1") == "0":
        return
    try:
        max_tokens = min(_MAX_TOKENS, max_num_batched_tokens)
        _STATE = LamportRSConv(hidden_size, window_size, max_tokens=max_tokens)
    except Exception:
        _STATE_FAILED = True
        logger.exception("fused collective unavailable; use the NCCL fallback")


def get_lamport_rs_conv(hidden_size: int, window_size: int) -> LamportRSConv | None:
    """Return the state initialized with the model, or ``None`` for fallback."""
    if _STATE is not None and (
        _STATE.hidden_size != hidden_size or _STATE.window_size != window_size
    ):
        raise RuntimeError("all Lamport users must share hidden and window sizes")
    return _STATE
