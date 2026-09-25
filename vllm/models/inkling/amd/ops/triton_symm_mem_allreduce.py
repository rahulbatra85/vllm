"""Triton all-reduce over PyTorch symmetric memory, plus a test/benchmark
harness for ``TritonSymmMemRSAG`` (Inkling's hidden-dim RS / AG+add+rmsnorm).

The blockwise barrier and the symm_mem conventions are documented in
``symm_mem_collectives.py``.

Two all-reduce algorithms:

    one_shot  barrier -> every rank sums all WORLD buffers -> barrier
              one round of WORLD-1 remote reads per element; best when small
    two_shot  barrier -> rank r reduces shard r and writes it into every
              peer's buffer -> barrier -> copy full buffer out
              moves 2(W-1)/W of the data per rank; best when large

Run (MODE=allreduce|rsag|all, default all):
    HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
        vllm/models/inkling/amd/ops/triton_symm_mem_allreduce.py
"""

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from vllm.models.inkling.amd.ops.norm import add_rmsnorm
from vllm.models.inkling.amd.ops.symm_mem_collectives import (
    TritonSymmMemRSAG,
    _blockwise_barrier,
    _load_epoch,
    _peer_buffer,
)


@triton.jit
def one_shot_all_reduce_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    out_ptr,
    numel,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    EPOCH_BASE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    epoch_ptr, epoch = _load_epoch(signal_pad_ptrs, EPOCH_BASE, RANK)
    _blockwise_barrier(signal_pad_ptrs, epoch + 1, RANK, WORLD)

    bufs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    pid = tl.program_id(0)
    for start in range(pid * BLOCK_SIZE, numel, tl.num_programs(0) * BLOCK_SIZE):
        offs = tl.multiple_of(start, BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for peer in tl.static_range(WORLD):
            buf = _peer_buffer(bufs, peer, out_ptr.dtype.element_ty)
            acc += tl.load(buf + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)

    # Peers are still reading this rank's buffer; it must not be overwritten
    # by the next call until they are done.
    _blockwise_barrier(signal_pad_ptrs, epoch + 2, RANK, WORLD)
    tl.store(epoch_ptr, epoch + 2)


@triton.jit
def two_shot_all_reduce_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    out_ptr,
    numel,
    shard_numel,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    EPOCH_BASE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    dtype = out_ptr.dtype.element_ty
    bufs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    local_buf = _peer_buffer(bufs, RANK, dtype)
    pid = tl.program_id(0)
    stride = tl.num_programs(0) * BLOCK_SIZE
    epoch_ptr, epoch = _load_epoch(signal_pad_ptrs, EPOCH_BASE, RANK)

    _blockwise_barrier(signal_pad_ptrs, epoch + 1, RANK, WORLD)

    # Reduce-scatter + push: block b owns the same tiles of every shard on
    # every rank, so peers only ever touch disjoint shards of each buffer.
    shard_start = RANK * shard_numel
    for start in range(pid * BLOCK_SIZE, shard_numel, stride):
        offs = shard_start + tl.multiple_of(start, BLOCK_SIZE)
        offs = tl.multiple_of(offs, 64) + tl.arange(0, BLOCK_SIZE)
        mask = (offs < numel) & (offs < shard_start + shard_numel)
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for peer in tl.static_range(WORLD):
            buf = _peer_buffer(bufs, peer, dtype)
            acc += tl.load(buf + offs, mask=mask).to(tl.float32)
        acc = acc.to(dtype)
        for peer in tl.static_range(WORLD):
            buf = _peer_buffer(bufs, peer, dtype)
            tl.store(buf + offs, acc, mask=mask)

    _blockwise_barrier(signal_pad_ptrs, epoch + 2, RANK, WORLD)
    tl.store(epoch_ptr, epoch + 2)

    # All-gather is now local. No trailing barrier: once block b clears the
    # barrier above, no peer touches this block's tiles of our buffer again.
    for shard in tl.static_range(WORLD):
        for start in range(pid * BLOCK_SIZE, shard_numel, stride):
            offs = shard * shard_numel + tl.multiple_of(start, BLOCK_SIZE)
            offs = tl.multiple_of(offs, 64) + tl.arange(0, BLOCK_SIZE)
            mask = (offs < numel) & (offs < (shard + 1) * shard_numel)
            tl.store(out_ptr + offs, tl.load(local_buf + offs, mask=mask), mask=mask)


class TritonSymmMemAllReduce:
    """Out-of-place all-reduce for tensors up to ``max_numel`` elements.

    Owns its buffer's signal pad: torch.ops.symm_mem collectives and
    ``handle.barrier()`` synchronize through the same pad and must not be run
    on ``self.buffer``.
    """

    def __init__(
        self,
        max_numel: int,
        dtype: torch.dtype,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
        one_shot_max_bytes: int = 512 * 1024,
        block_size: int = 4096,
        num_warps: int = 8,
        max_blocks: int = 32,
    ):
        group = group or dist.group.WORLD
        self.buffer = symm_mem.empty(max_numel, dtype=dtype, device=device)
        self.handle = symm_mem.rendezvous(self.buffer, group)
        self.rank = self.handle.rank
        self.world_size = self.handle.world_size
        self.one_shot_max_bytes = one_shot_max_bytes
        self.block_size = block_size
        self.num_warps = num_warps

        # Each block owns WORLD peer epochs plus its own epoch per pad; the
        # grid must also fit on the device at once, since block b spins on its
        # peers' block b. On MI355X symm_mem buffers are uncached and saturate
        # at ~8 blocks, while barrier cost grows with the block count, hence
        # the small default.
        pad_slots = self.handle.signal_pad_size // 4
        num_cus = torch.cuda.get_device_properties(device).multi_processor_count
        self.max_blocks = min(max_blocks, pad_slots // (self.world_size + 1), num_cus)
        self.epoch_base = self.max_blocks * self.world_size
        # Epochs must start at 0 on every rank before the first launch. Not
        # handle.barrier(): that synchronizes through these same pads.
        self.handle.get_signal_pad(self.rank).zero_()
        torch.cuda.synchronize(device)
        dist.barrier(group)

    def __call__(self, inp: torch.Tensor, algo: str | None = None) -> torch.Tensor:
        assert inp.is_contiguous() and inp.numel() <= self.buffer.numel()
        numel = inp.numel()
        if algo is None:
            small = numel * inp.element_size() <= self.one_shot_max_bytes
            algo = "one_shot" if small else "two_shot"

        self.buffer[:numel].copy_(inp.view(-1))
        out = torch.empty_like(inp)
        common = dict(
            RANK=self.rank,
            WORLD=self.world_size,
            EPOCH_BASE=self.epoch_base,
            BLOCK_SIZE=self.block_size,
            num_warps=self.num_warps,
        )
        if algo == "one_shot":
            grid = (min(triton.cdiv(numel, self.block_size), self.max_blocks),)
            one_shot_all_reduce_kernel[grid](
                self.handle.buffer_ptrs_dev,
                self.handle.signal_pad_ptrs_dev,
                out,
                numel,
                **common,
            )
        elif algo == "two_shot":
            # 64-element shards keep every shard 128-byte aligned.
            shard_numel = triton.cdiv(triton.cdiv(numel, self.world_size), 64) * 64
            grid = (min(triton.cdiv(shard_numel, self.block_size), self.max_blocks),)
            two_shot_all_reduce_kernel[grid](
                self.handle.buffer_ptrs_dev,
                self.handle.signal_pad_ptrs_dev,
                out,
                numel,
                shard_numel,
                **common,
            )
        else:
            raise ValueError(f"unknown algo {algo!r}")
        return out


def _bench_us(fn, iters: int = 50, warmup: int = 10) -> float:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def _run_allreduce(rank: int, world: int, device: torch.device) -> None:
    dtype = torch.bfloat16
    sizes = [2**k for k in range(10, 27, 2)]  # 1 Ki .. 64 Mi elements
    allreduce = TritonSymmMemAllReduce(max(sizes), dtype, device)

    # Integer-valued inputs keep the bf16 sums exact, so any mismatch against
    # RCCL is a synchronization bug, not rounding.
    for numel in sizes + [4097, 123457]:
        for algo in ("one_shot", "two_shot"):
            for it in range(3):
                gen = torch.Generator(device).manual_seed(1000 * rank + it)
                inp = torch.randint(-8, 8, (numel,), generator=gen, device=device)
                inp = inp.to(dtype)
                ref = inp.clone()
                dist.all_reduce(ref)
                out = allreduce(inp, algo)
                torch.testing.assert_close(out, ref, atol=0, rtol=0)
    if rank == 0:
        print(f"all-reduce correctness OK on {world} ranks", flush=True)

    if rank == 0:
        print(f"{'bytes':>10} {'one_shot':>10} {'two_shot':>10} {'rccl':>10}  (us)")
    for numel in sizes:
        inp = torch.randn(numel, dtype=dtype, device=device)
        t1 = _bench_us(lambda: allreduce(inp, "one_shot"))
        t2 = _bench_us(lambda: allreduce(inp, "two_shot"))
        tr = _bench_us(lambda: dist.all_reduce(inp))
        if rank == 0:
            nbytes = numel * inp.element_size()
            print(f"{nbytes:>10} {t1:>10.1f} {t2:>10.1f} {tr:>10.1f}", flush=True)


def _rccl_rs_lastdim(x: torch.Tensor, world: int) -> torch.Tensor:
    """What ``tensor_model_parallel_reduce_scatter(x, dim=-1)`` does on ROCm."""
    xt = x.movedim(0, -1).contiguous()
    out = torch.empty(xt.shape[0] // world, xt.shape[1], dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, xt)
    return out.movedim(0, -1).contiguous()


def _rccl_ag_lastdim(x: torch.Tensor, world: int) -> torch.Tensor:
    """What ``tensor_model_parallel_all_gather(x, dim=-1)`` does on ROCm."""
    tokens, shard = x.shape
    out = torch.empty(world * tokens, shard, dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x)
    return out.reshape(world, tokens, shard).movedim(0, 1).reshape(tokens, -1)


def _rsag_inputs(tokens, hidden, shard, rank, seed, device):
    gen = torch.Generator(device).manual_seed(1000 * rank + seed)

    def ints(*shape):
        return torch.randint(-8, 8, shape, generator=gen, device=device).to(
            torch.bfloat16
        )

    return (
        ints(tokens, hidden),
        ints(tokens, hidden),
        ints(tokens, shard),
        ints(tokens, hidden),
    )


def _run_rsag(rank: int, world: int, device: torch.device) -> None:
    hidden, max_tokens, eps = 6144, 2048, 1e-6
    shard = hidden // world
    rsag = TritonSymmMemRSAG(
        hidden,
        max_tokens,
        torch.bfloat16,
        device,
        dist.group.WORLD,
        trailing_barrier=os.environ.get("TRAILING", "0") == "1",
    )
    weight = torch.randn(hidden, dtype=torch.bfloat16, device=device)

    if os.environ.get("SKIP_CHECK") != "1":
        _check_rsag(rsag, rank, world, device, hidden, shard, weight, eps)
    _bench_rsag(rsag, rank, world, device, hidden, shard, weight, eps)


def _check_rsag(rsag, rank, world, device, hidden, shard, weight, eps) -> None:
    # Exactness: integer-valued inputs keep the sums exact in bf16.
    token_counts = [1, 2, 3, 7, 64, 127, 256, 1024, 2047, 2048]
    for tokens in token_counts:
        for it, (use_shared, norm_w) in enumerate(
            [(False, weight), (True, weight), (False, None), (True, None)]
        ):
            delta, shared, sh, res = _rsag_inputs(
                tokens, hidden, shard, rank, it, device
            )
            shared = shared if use_shared else None
            ref = _rccl_rs_lastdim(delta + shared if use_shared else delta, world)
            full = _rccl_ag_lastdim(sh, world)
            if norm_w is None:
                ref_n, ref_r = None, res + full
            else:
                ref_n, ref_r = add_rmsnorm(res, full, norm_w, eps)
            out = rsag.reduce_scatter(delta, shared)
            n, r = rsag.all_gather_add_norm(sh, res, norm_w, eps)
            tag = f"tokens={tokens} shared={use_shared} norm={norm_w is not None}"

            def msg(m, what, tag=tag):
                return f"{what} {tag}: {m}"

            torch.testing.assert_close(
                out, ref, atol=0, rtol=0, msg=lambda m: msg(m, "RS")
            )
            torch.testing.assert_close(
                r, ref_r, atol=0, rtol=0, msg=lambda m: msg(m, "AG")
            )
            if norm_w is not None:
                torch.testing.assert_close(n, ref_n, msg=lambda m: msg(m, "norm"))
    if rank == 0:
        print(f"rs/ag correctness OK on {world} ranks", flush=True)

    # Back-to-back buffer reuse without trailing barriers, replayed as a graph
    # with varying batch sizes.
    seq = [1, 2048, 3, 512, 1, 1, 777, 64, 2048, 2] * 5
    cases = [
        _rsag_inputs(t, hidden, shard, rank, 100 + i, device) for i, t in enumerate(seq)
    ]
    refs = []
    for delta, shared, sh, res in cases:
        refs.append(
            (
                _rccl_rs_lastdim(delta + shared, world),
                add_rmsnorm(res, _rccl_ag_lastdim(sh, world), weight, eps)[1],
            )
        )
    outs = []
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.graph(graph):
        for delta, shared, sh, res in cases:
            rs = rsag.reduce_scatter(delta, shared)
            outs.append((rs, rsag.all_gather_add_norm(sh, res, weight, eps)[1]))
    for _ in range(20):
        for rs, r in outs:
            rs.zero_()
            r.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for (rs, r), (ref_rs, ref_r) in zip(outs, refs):
            torch.testing.assert_close(rs, ref_rs, atol=0, rtol=0)
            torch.testing.assert_close(r, ref_r, atol=0, rtol=0)
    if rank == 0:
        print(f"rs/ag graph-replay reuse OK ({len(seq)} pairs x 20)", flush=True)


def _bench_rsag(rsag, rank, world, device, hidden, shard, weight, eps) -> None:
    if rank == 0:
        print(
            f"{'tokens':>7} {'triton':>10} {'rccl_rs':>8} {'rccl_ag':>8} "
            f"{'rccl':>8} {'speedup':>8}  (us; RS+AG+add+rmsnorm, no sconv)"
        )
    for tokens in [1, 4, 16, 64, 128, 256, 512, 1024, 2048]:
        delta, shared, sh, res = _rsag_inputs(tokens, hidden, shard, rank, 7, device)

        def tri_pair(delta=delta, shared=shared, sh=sh, res=res):
            rsag.reduce_scatter(delta, shared)
            rsag.all_gather_add_norm(sh, res, weight, eps)

        def rccl_rs(delta=delta, shared=shared):
            return _rccl_rs_lastdim(delta + shared, world)

        def rccl_ag(sh=sh, res=res):
            return add_rmsnorm(res, _rccl_ag_lastdim(sh, world), weight, eps)

        # The Triton ops must alternate, so only the pair is timed.
        t_tri = _bench_us(tri_pair)
        t_rccl_rs = _bench_us(rccl_rs)
        t_rccl_ag = _bench_us(rccl_ag)
        if rank == 0:
            t_rccl = t_rccl_rs + t_rccl_ag
            print(
                f"{tokens:>7} {t_tri:>10.1f} {t_rccl_rs:>8.1f} {t_rccl_ag:>8.1f} "
                f"{t_rccl:>8.1f} {t_rccl / t_tri:>7.2f}x",
                flush=True,
            )


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    mode = os.environ.get("MODE", "all")
    if mode in ("allreduce", "all"):
        _run_allreduce(rank, world, device)
    if mode in ("rsag", "all"):
        _run_rsag(rank, world, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
