# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton collectives over PyTorch symmetric memory (ROCm).

Each rank allocates a buffer with ``symm_mem.empty``; ``symm_mem.rendezvous``
maps every peer's buffer and signal pad, and Triton kernels access peers
directly through ``buffer_ptrs_dev`` / ``signal_pad_ptrs_dev``.

Synchronization is a blockwise epoch barrier built from Triton atomics, following
vLLM's ROCm custom all-reduce (``csrc/custom_collective_common.cuh``): each block
keeps a monotonically increasing epoch, pushes it into every peer's pad with a
system-scope release exchange, then polls its own pad until every peer caught up.

    signal pad of rank r (uint32):
        [b * WORLD + p]            epoch that rank p last reached in block b
        [epoch_base + b]           epoch rank r's block b has completed

A 0 -> 1 / 1 -> 0 compare-and-swap handshake on the peer's pad (what
``ptx_utils.symm_mem_sync`` does) hangs on MI355X: a remote CAS can report
success while the peer never observes the flag. Remote writes are therefore
plain exchanges and the only read-modify-write polling is on local memory.

Every rank launches the same grid for a given call, so block b only ever
synchronizes with block b on each peer: all data a block reads from a peer must
have been written by the same block id on that peer.

``TritonSymmMemRSAG`` implements the hidden-dim (``dim=-1``) reduce-scatter and
all-gather around Inkling's short conv without the transposes the generic
``dim=-1`` RCCL path performs:

    reduce_scatter       copy-in (+ shared) -> barrier -> pull-reduce own shard
    all_gather_add_norm  push shard to all  -> barrier -> residual add + rmsnorm

RS and AG strictly alternate and each opens with a full-grid barrier, so no
trailing barrier is needed before a buffer is rewritten: RS(n+1) runs only after
this rank's AG(n) passed its barrier, which every peer reached only after its
RS(n) (and thus all its reads of our RS buffer) completed; symmetrically for AG.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _blockwise_barrier(signal_pad_ptrs, epoch, RANK: tl.constexpr, WORLD: tl.constexpr):
    """Rendezvous this block with the same block id on every peer at ``epoch``.

    Release orders this rank's prior writes before peers can observe the
    epoch; acquire makes peers' writes visible to the loads that follow.
    """
    block_id = tl.program_id(0)
    pads = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    local_pad = tl.load(pads + RANK).to(tl.pointer_type(tl.uint32))
    # All threads of this block must have finished their writes before one
    # thread publishes the epoch on their behalf. s_barrier does not drain
    # vector-memory stores and the release emitted for the atomic below only
    # covers its own wave, so every wave drains its stores first. Callers
    # write data meant for peers with .wt so it never sits dirty in L2; with
    # plain stores peers observed the epoch before the data on MI355X.
    tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()
    for peer in tl.static_range(WORLD):
        remote_pad = tl.load(pads + peer).to(tl.pointer_type(tl.uint32))
        send = remote_pad + block_id * WORLD + RANK
        tl.atomic_xchg(send, epoch, sem="release", scope="sys")
    for peer in tl.static_range(WORLD):
        wait = local_pad + block_id * WORLD + peer
        # >=, not ==: a fast peer may already have published the next epoch.
        while tl.atomic_add(wait, 0, sem="acquire", scope="sys") < epoch:
            pass
    tl.debug_barrier()


@triton.jit
def _peer_buffer(bufs, peer, dtype):
    # symm_mem buffers are page aligned, but Triton cannot see that through a
    # pointer loaded at runtime and would otherwise emit 2-byte accesses.
    return tl.multiple_of(tl.load(bufs + peer).to(tl.pointer_type(dtype)), 16)


@triton.jit
def _load_epoch(signal_pad_ptrs, EPOCH_BASE: tl.constexpr, RANK: tl.constexpr):
    pads = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    local_pad = tl.load(pads + RANK).to(tl.pointer_type(tl.uint32))
    epoch_ptr = local_pad + EPOCH_BASE + tl.program_id(0)
    return epoch_ptr, tl.load(epoch_ptr)


@triton.jit
def _rs_lastdim_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    inp_ptr,
    shared_ptr,
    out_ptr,
    num_tokens,
    stride_inp,
    stride_shared,
    stride_out,
    C: tl.constexpr,
    CS: tl.constexpr,
    SPLITS: tl.constexpr,
    CSS_P2: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    EPOCH_BASE: tl.constexpr,
    TRAILING_BARRIER: tl.constexpr,
):
    """``out[T, CS] = sum_p (inp + shared)_p[:, RANK*CS:(RANK+1)*CS]``.

    Work item ``(token, split)`` owns channels ``split*CSS:(split+1)*CSS`` of
    every rank's shard, so a block reads back from peers exactly what the same
    block id wrote there.
    """
    CSS: tl.constexpr = CS // SPLITS
    dtype = out_ptr.dtype.element_ty
    bufs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    local_buf = _peer_buffer(bufs, RANK, dtype)
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    ch = tl.arange(0, CSS_P2)
    ch_mask = ch < CSS
    epoch_ptr, epoch = _load_epoch(signal_pad_ptrs, EPOCH_BASE, RANK)

    for item in range(pid, num_tokens * SPLITS, nprog):
        token = (item // SPLITS).to(tl.int64)
        split = item % SPLITS
        for owner in tl.static_range(WORLD):
            col = owner * CS + split * CSS + ch
            x = tl.load(inp_ptr + token * stride_inp + col, mask=ch_mask)
            if HAS_SHARED:
                s = tl.load(shared_ptr + token * stride_shared + col, mask=ch_mask)
                x = (x.to(tl.float32) + s.to(tl.float32)).to(dtype)
            tl.store(local_buf + token * C + col, x, mask=ch_mask, cache_modifier=".wt")

    _blockwise_barrier(signal_pad_ptrs, epoch + 1, RANK, WORLD)

    for item in range(pid, num_tokens * SPLITS, nprog):
        token = (item // SPLITS).to(tl.int64)
        split = item % SPLITS
        col = RANK * CS + split * CSS + ch
        acc = tl.zeros([CSS_P2], dtype=tl.float32)
        for peer in tl.static_range(WORLD):
            buf = _peer_buffer(bufs, peer, dtype)
            acc += tl.load(
                buf + token * C + col, mask=ch_mask, cache_modifier=".cv"
            ).to(tl.float32)
        tl.store(
            out_ptr + token * stride_out + split * CSS + ch,
            acc.to(dtype),
            mask=ch_mask,
        )

    if TRAILING_BARRIER:
        _blockwise_barrier(signal_pad_ptrs, epoch + 2, RANK, WORLD)
        tl.store(epoch_ptr, epoch + 2)
    else:
        tl.store(epoch_ptr, epoch + 1)


@triton.jit
def _ag_lastdim_add_norm_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    shard_ptr,
    res_ptr,
    weight_ptr,
    normed_ptr,
    res_out_ptr,
    eps,
    num_tokens,
    buf_offset,
    stride_shard,
    stride_res,
    stride_normed,
    stride_res_out,
    C: tl.constexpr,
    CS: tl.constexpr,
    C_P2: tl.constexpr,
    CS_P2: tl.constexpr,
    HAS_NORM: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    EPOCH_BASE: tl.constexpr,
    TRAILING_BARRIER: tl.constexpr,
):
    """``res_out = res + all_gather(shard, dim=-1); normed = rmsnorm(res_out)``.

    Numerics match ``_add_rmsnorm_fwd_kernel``: the sum is rounded to the
    residual dtype before the fp32 rmsnorm.
    """
    dtype = res_out_ptr.dtype.element_ty
    bufs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    local_buf = _peer_buffer(bufs, RANK, dtype) + buf_offset
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    sch = tl.arange(0, CS_P2)
    sch_mask = sch < CS
    epoch_ptr, epoch = _load_epoch(signal_pad_ptrs, EPOCH_BASE, RANK)

    for t in range(pid, num_tokens, nprog):
        token = t.to(tl.int64)
        x = tl.load(shard_ptr + token * stride_shard + sch, mask=sch_mask)
        for peer in tl.static_range(WORLD):
            buf = _peer_buffer(bufs, peer, dtype) + buf_offset
            tl.store(
                buf + token * C + RANK * CS + sch,
                x,
                mask=sch_mask,
                cache_modifier=".wt",
            )

    _blockwise_barrier(signal_pad_ptrs, epoch + 1, RANK, WORLD)

    ch = tl.arange(0, C_P2)
    ch_mask = ch < C
    if HAS_NORM:
        weight = tl.load(weight_ptr + ch, mask=ch_mask, other=0.0).to(tl.float32)
    for t in range(pid, num_tokens, nprog):
        token = t.to(tl.int64)
        full = tl.load(
            local_buf + token * C + ch, mask=ch_mask, other=0.0, cache_modifier=".cv"
        )
        r = tl.load(res_ptr + token * stride_res + ch, mask=ch_mask, other=0.0)
        s = (r.to(tl.float32) + full.to(tl.float32)).to(dtype)
        tl.store(res_out_ptr + token * stride_res_out + ch, s, mask=ch_mask)
        if HAS_NORM:
            xf = s.to(tl.float32)
            rstd = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / C + eps)
            tl.store(
                normed_ptr + token * stride_normed + ch,
                xf * rstd * weight,
                mask=ch_mask,
            )

    if TRAILING_BARRIER:
        _blockwise_barrier(signal_pad_ptrs, epoch + 2, RANK, WORLD)
        tl.store(epoch_ptr, epoch + 2)
    else:
        tl.store(epoch_ptr, epoch + 1)


class TritonSymmMemRSAG:
    """Hidden-dim reduce-scatter / all-gather(+add+rmsnorm) for ``[T, C]``.

    Calls must alternate ``reduce_scatter`` -> ``all_gather_add_norm`` on the
    same stream (see the module docstring for why that makes trailing barriers
    unnecessary). ``trailing_barrier=True`` adds them back for debugging.
    """

    def __init__(
        self,
        hidden_size: int,
        max_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup,
        trailing_barrier: bool = False,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        # symm_mem buffers are [2, max_tokens, C]: RS partials, then AG rows.
        self.buffer = symm_mem.empty(
            2, max_tokens, hidden_size, dtype=dtype, device=device
        )
        self.handle = symm_mem.rendezvous(self.buffer, group.group_name)
        self.rank = self.handle.rank
        self.world_size = self.handle.world_size
        if hidden_size % self.world_size:
            raise ValueError("hidden size must divide evenly across ranks")
        self.hidden_size = hidden_size
        self.shard_size = hidden_size // self.world_size
        self.max_tokens = max_tokens
        self.trailing_barrier = trailing_barrier

        # Each block owns WORLD peer epochs plus its own epoch per pad; the
        # grid must also fit on the device at once, since block b spins on its
        # peers' block b.
        pad_slots = self.handle.signal_pad_size // 4
        num_cus = torch.cuda.get_device_properties(device).multi_processor_count
        self.max_blocks = min(pad_slots // (self.world_size + 1), num_cus)
        self.epoch_base = self.max_blocks * self.world_size
        # Epochs must start at 0 on every rank before the first launch. Not
        # handle.barrier(): that synchronizes through these same pads.
        self.handle.get_signal_pad(self.rank).zero_()
        torch.cuda.synchronize(device)
        dist.barrier(group)

    def usable(self, num_tokens: int) -> bool:
        return 0 < num_tokens <= self.max_tokens

    def _common(self) -> dict:
        return dict(
            RANK=self.rank,
            WORLD=self.world_size,
            EPOCH_BASE=self.epoch_base,
            TRAILING_BARRIER=self.trailing_barrier,
        )

    def reduce_scatter(
        self, inp: torch.Tensor, shared: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sum ``inp (+ shared)`` over ranks; return this rank's ``[T, CS]``."""
        tokens, hidden = inp.shape
        assert hidden == self.hidden_size and self.usable(tokens)
        assert inp.stride(1) == 1
        assert shared is None or (shared.shape == inp.shape and shared.stride(1) == 1)
        out = torch.empty(tokens, self.shard_size, dtype=inp.dtype, device=inp.device)
        # Small batches split each shard across blocks so the grid has enough
        # parallelism; every split keeps whole 16-element vectors.
        splits = 1
        while (
            tokens * splits * 2 <= self.max_blocks
            and splits < 8
            and self.shard_size % (splits * 2 * 16) == 0
        ):
            splits *= 2
        css_p2 = triton.next_power_of_2(self.shard_size // splits)
        grid = (min(tokens * splits, self.max_blocks),)
        _rs_lastdim_kernel[grid](
            self.handle.buffer_ptrs_dev,
            self.handle.signal_pad_ptrs_dev,
            inp,
            shared if shared is not None else inp,
            out,
            tokens,
            inp.stride(0),
            shared.stride(0) if shared is not None else 0,
            out.stride(0),
            C=hidden,
            CS=self.shard_size,
            SPLITS=splits,
            CSS_P2=css_p2,
            HAS_SHARED=shared is not None,
            num_warps=4 if css_p2 <= 512 else 8,
            **self._common(),
        )
        return out

    def all_gather_add_norm(
        self,
        shard: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor | None,
        eps: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return ``(rmsnorm(residual + AG(shard)) | None, residual + AG(shard))``."""
        tokens, hidden = residual.shape
        assert hidden == self.hidden_size and self.usable(tokens)
        assert shard.shape == (tokens, self.shard_size) and shard.stride(1) == 1
        assert residual.stride(1) == 1
        res_out = torch.empty_like(residual)
        normed = torch.empty_like(residual) if norm_weight is not None else None
        c_p2 = triton.next_power_of_2(hidden)
        grid = (min(tokens, self.max_blocks),)
        _ag_lastdim_add_norm_kernel[grid](
            self.handle.buffer_ptrs_dev,
            self.handle.signal_pad_ptrs_dev,
            shard,
            residual,
            norm_weight if norm_weight is not None else residual,
            normed if normed is not None else res_out,
            res_out,
            eps,
            tokens,
            self.max_tokens * hidden,
            shard.stride(0),
            residual.stride(0),
            normed.stride(0) if normed is not None else 0,
            res_out.stride(0),
            C=hidden,
            CS=self.shard_size,
            C_P2=c_p2,
            CS_P2=triton.next_power_of_2(self.shard_size),
            HAS_NORM=norm_weight is not None,
            num_warps=8 if c_p2 >= 4096 else 4,
            **self._common(),
        )
        return normed, res_out


_STATE: TritonSymmMemRSAG | None = None
_STATE_FAILED = False


def initialize_triton_rs_ag(hidden_size: int, max_num_batched_tokens: int) -> None:
    """Collectively initialize the TP-group state (opt-in: INKLING_TRITON_RS_AG=1).

    Batches above ``INKLING_TRITON_RS_AG_MAX_TOKENS`` stay on RCCL, which also
    bounds the ``2 * max_tokens * hidden_size`` symmetric allocation.
    """
    global _STATE, _STATE_FAILED
    if _STATE is not None or _STATE_FAILED:
        return
    if os.environ.get("INKLING_TRITON_RS_AG", "0") != "1":
        return
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import in_the_same_node_as

    tp = get_tp_group()
    if tp.world_size == 1 or not all(in_the_same_node_as(tp.cpu_group)):
        return
    try:
        max_tokens = min(
            max_num_batched_tokens,
            int(os.environ.get("INKLING_TRITON_RS_AG_MAX_TOKENS", "2048")),
        )
        _STATE = TritonSymmMemRSAG(
            hidden_size,
            max_tokens,
            torch.bfloat16,
            torch.device(tp.device),
            tp.device_group,
            trailing_barrier=os.environ.get("INKLING_TRITON_RS_AG_TRAILING", "0")
            == "1",
        )
        logger.info(
            "Using Triton symm-mem RS/AG for sconv add-norm "
            "(tp=%d, max_tokens=%d, trailing_barrier=%s)",
            tp.world_size,
            max_tokens,
            _STATE.trailing_barrier,
        )
    except Exception:
        _STATE_FAILED = True
        logger.exception("Triton symm-mem RS/AG unavailable; use the RCCL fallback")


def get_triton_rs_ag() -> TritonSymmMemRSAG | None:
    return _STATE
