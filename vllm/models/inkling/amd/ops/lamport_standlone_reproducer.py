"""Lamport publish + spin-wait over PyTorch symmetric memory.

Reproduces the first half of Inkling's fused collective from
``vllm/models/inkling/amd/ops/lamport.py``, at its real geometry:

    kernel 1  publish_kernel       -- packs bf16 pairs into u32 and stores them
                                      into every peer's buf_in slot [token][rank]
    kernel 2  wait_kernel          -- spins on this rank's slots until every
                                      peer's payload has replaced the -0.0
                                      sentinel, reduces, writes the KV cache
    kernel 3  sconv_publish_kernel -- gathers a 4-tap window out of the KV cache
                                      through the block table, then publishes
                                      the result into every peer's buf_out row
    kernel 4  gather_kernel        -- waits for a complete row, unpacks, re-arms

Kernels 2 and 3 carry the KV-cache addressing: the slot/head/dim math for the
write, and the block-table indirection for the windowed read. That addressing is
the only memory the real kernels touch that the symmetric buffers do not cover,
so SLOT_MODE exercises it under both decode-like and profiling-run conditions
(``pad`` fills slot_mapping with -1, the way vLLM's dummy run does).

Both are launched back to back on one stream, with **no barrier between them**,
exactly like the real code: Lamport's whole point is that the payload is its own
ready-flag, so a consumer never round-trips to the host. That makes kernel 2 a
direct test of cross-GPU write visibility -- the part a `dist.barrier()` would
otherwise paper over.

Unlike the real ``_wait_pairs``, the spin here is **bounded**. An unbounded spin
just reproduces the hang seen under CUDA graph capture and teaches nothing; with
a budget, a rank that never observes a peer's write reports which tokens were
still empty and why.

The loop runs many collectives back to back, rotating through the generations
and re-arming each slot as it is consumed -- the same lifecycle a real forward
pass drives 66 times over. No barrier anywhere inside the loop, so ranks drift
against each other exactly as they do in production; the 3-generation design is
what is supposed to absorb that drift.

    buf_in  = [GENERATIONS, MAX_TOKENS, WORLD, SHARD]   ~576 MiB at defaults
    buf_out = [GENERATIONS, MAX_TOKENS, HIDDEN]         ~576 MiB at defaults

Tunables (env): MAX_TOKENS, TOKENS, HIDDEN, GENERATIONS, ITERS, MAX_SPINS,
SLOT_MODE (valid | pad | mixed).

Run: torchrun --nproc-per-node 4 symm_mem_tutorial.py
"""

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

# Two bf16 -0.0 lanes: Lamport's "not published yet" sentinel. Must be a
# tl.constexpr -- Triton kernels cannot read plain Python globals.
EMPTY_PAIR = tl.constexpr(0x80008000)

# Lamport's defaults from LamportRSConv.__init__.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 16384))
HIDDEN = int(os.environ.get("HIDDEN", 6144))
GENERATIONS = int(os.environ.get("GENERATIONS", 3))
# Tokens actually published; 2048 is where the real server faulted.
TOKENS = int(os.environ.get("TOKENS", 2048))
# Collectives to run, rotating through the generations. A real forward pass
# makes ~66 of these (one per layer), so the default covers several passes.
ITERS = int(os.environ.get("ITERS", 200))
# Mock KV-cache geometry: [NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM].
# HEAD_SIZE/CACHE_OFFSET mirror the ws / off_s that stream_ranges supplies, and
# the shard must divide into whole heads exactly as LamportRSConv validates.
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 128))
HEAD_SIZE = int(os.environ.get("HEAD_SIZE", 64))
CACHE_OFFSET = int(os.environ.get("CACHE_OFFSET", 0))
HEAD_DIM = CACHE_OFFSET + HEAD_SIZE
WINDOW = 4
NUM_SEQS = int(os.environ.get("NUM_SEQS", 8))
# How slot_mapping is filled: "valid" = real slots, "pad" = all -1 like the
# profiling dummy run, "mixed" = every other token padded.
SLOT_MODE = os.environ.get("SLOT_MODE", "valid")

# Block size handed to the kernels. Defaults to the cache's real page size.
# Setting it lower reproduces the production bug: Inkling passed the conv
# window (owner.block_size = 4) where the cache page size (32) was required, so
# `slot // block_size` overshot the block dimension by 8x and ran ~12 GiB off
# the end of the allocation. KERNEL_BLOCK_SIZE=4 reproduces that fault here.
KERNEL_BLOCK_SIZE = int(os.environ.get("KERNEL_BLOCK_SIZE", BLOCK_SIZE))

# Channel splits per shard. The real code uses 8 at decode (tokens == 1) and 1
# otherwise, so SPLITS>1 is the decode-path geometry: more CTAs, each owning a
# narrower channel slice. Auto mirrors that rule.
SPLITS = os.environ.get("SPLITS", "auto")

# Spin budget per waiter program. Generous, but finite.
MAX_SPINS = int(os.environ.get("MAX_SPINS", 2_000_000))


@triton.jit
def _pack_bf16_pairs(values):
    """Pack bf16 values into u32 pairs, reserving -0.0 for the sentinel."""
    lo, hi = tl.split(values.reshape([values.shape[0] // 2, 2]))
    lo = lo.to(tl.uint16, bitcast=True)
    hi = hi.to(tl.uint16, bitcast=True)
    lo = tl.where(lo == 0x8000, 0, lo).to(tl.uint32)
    hi = tl.where(hi == 0x8000, 0, hi).to(tl.uint32)
    return lo | (hi << 16)


@triton.jit
def _unpack_bf16_pairs(values):
    """Inverse of _pack_bf16_pairs."""
    lo = (values & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    hi = (values >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    return tl.interleave(lo, hi)


@triton.jit
def publish_kernel(
    src_ptr,  # this rank's rows, [TOKENS, HIDDEN] bf16
    peer_ptrs,  # uint64[WORLD] device array of peer base pointers
    stride_src_t,
    gen_offset_u32,  # u32-element offset of this generation
    SHARD: tl.constexpr,  # CS: this rank's shard width
    BLOCK: tl.constexpr,  # CS_P2: next_pow2(SHARD // SPLITS), a per-split tile
    SPLITS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
):
    """Publish this rank's row into every shard owner's slot for one token.

    With SPLITS > 1 the shard is cut into SPLITS channel slices and each gets
    its own CTA -- what the real code does at decode (tokens == 1), where one
    CTA per token would leave the GPU idle.
    """
    # int64 throughout: at Lamport sizes offsets reach ~1.5e8 u32 words, and
    # intermediate products overflow int32 before that.
    token = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    CSS: tl.constexpr = SHARD // SPLITS
    elem = tl.arange(0, BLOCK).to(tl.int64)
    pair = tl.arange(0, BLOCK // 2).to(tl.int64)
    elem_mask = elem < CSS
    pair_mask = pair < CSS // 2

    ptrs = peer_ptrs.to(tl.pointer_type(tl.uint64))
    for owner in tl.static_range(WORLD):
        values = tl.load(
            src_ptr + token * stride_src_t + owner * SHARD + split * CSS + elem,
            mask=elem_mask,
            other=0.0,
        )
        packed = _pack_bf16_pairs(values)
        # Dereferencing a *remote* base pointer. Layout is
        # [GEN, MAX_TOKENS, WORLD, SHARD]; this rank owns slot RANK.
        base = tl.load(ptrs + owner).to(tl.pointer_type(tl.uint32))
        dst = (
            gen_offset_u32
            + (token * WORLD + RANK) * (SHARD // 2)
            + split * (CSS // 2)
            + pair
        )
        tl.store(base + dst, packed, mask=pair_mask)


@triton.jit
def wait_kernel(
    peer_ptrs,
    stage_ptr,  # uint32[TOKENS, WORLD, SHARD//2]: what this rank observed
    status_ptr,  # int32[TOKENS]: pairs still empty when the budget ran out
    spins_ptr,  # int32[TOKENS]: spins actually taken
    cache_ptr,
    slot_ptr,
    stride_cache_block,
    stride_cache_head,
    stride_cache_token,
    stride_cache_dim,
    block_size,
    gen_offset_u32,
    max_spins,
    SHARD: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,  # CS_P2 // 2
    SPLITS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    CACHE_OFF: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
):
    """Spin until every peer has published this rank's slots, then re-arm."""
    token = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    CSS: tl.constexpr = SHARD // SPLITS
    source = tl.arange(0, WORLD)
    pair = tl.arange(0, PAIR_BLOCK).to(tl.int64)
    pair_mask = pair < CSS // 2
    # Wait on all WORLD publishers at once, like _reduce_insert_kernel. Each
    # split owns its own channel slice of every publisher's slot.
    offsets = (
        (token * WORLD + source)[:, None] * (SHARD // 2)
        + split * (CSS // 2)
        + pair[None, :]
    )
    mask = tl.full([WORLD], True, tl.int1)[:, None] & pair_mask[None, :]

    # Read through this rank's own entry in the pointer table, as the real
    # kernel does -- peers publish *into* our buffer.
    ptrs = peer_ptrs.to(tl.pointer_type(tl.uint64))
    base = tl.load(ptrs + RANK).to(tl.pointer_type(tl.uint32))
    base += gen_offset_u32

    # volatile: the value must be re-read from memory every iteration, or the
    # compiler is free to hoist the load out of the loop and spin forever.
    values = tl.load(base + offsets, mask=mask, other=0, volatile=True)
    spins = 0
    while (
        tl.max(tl.where(mask & (values == EMPTY_PAIR), 1, 0)) != 0
        and spins < max_spins
    ):
        values = tl.load(base + offsets, mask=mask, other=0, volatile=True)
        spins += 1

    pending = tl.sum(tl.where(mask & (values == EMPTY_PAIR), 1, 0))
    # Keep what we observed before re-arming destroys it.
    tl.store(stage_ptr + offsets, values, mask=mask)

    # Reduce across publishers and insert into the KV cache, as the real
    # _reduce_insert_kernel does. This is the cache-write addressing under test.
    lo = (values & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    hi = (values >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    reduced = tl.interleave(
        tl.sum(lo.to(tl.float32), axis=0).to(tl.bfloat16),
        tl.sum(hi.to(tl.float32), axis=0).to(tl.bfloat16),
    )
    slot = tl.load(slot_ptr + token)
    valid = slot >= 0
    # max(slot, 0) keeps the address in range for padded tokens; the store is
    # masked off anyway. Dropping either guard is a fault waiting to happen.
    safe_slot = tl.maximum(slot, 0).to(tl.int64)
    channel = tl.arange(0, 2 * PAIR_BLOCK).to(tl.int64)
    channel_mask = channel < CSS
    global_channel = split * CSS + channel
    head = tl.minimum(global_channel // HEAD_SIZE, SHARD // HEAD_SIZE - 1)
    dim = CACHE_OFF + global_channel % HEAD_SIZE
    dst = (
        cache_ptr
        + (safe_slot // block_size) * stride_cache_block
        + head * stride_cache_head
        + (safe_slot % block_size) * stride_cache_token
        + dim * stride_cache_dim
    )
    tl.store(dst, reduced, mask=valid & channel_mask)
    # Re-arm the slots we just consumed, exactly as _reduce_insert_kernel does.
    # This is what lets a generation be reused after two intervening calls.
    tl.store(
        base + offsets,
        tl.full([WORLD, PAIR_BLOCK], EMPTY_PAIR, tl.uint32),
        mask=mask,
    )
    tl.store(status_ptr + token * SPLITS + split, pending)
    tl.store(spins_ptr + token * SPLITS + split, spins)


@triton.jit
def sconv_publish_kernel(
    src_ptr,  # this rank's residual, [TOKENS, SHARD] bf16
    peer_ptrs,
    gathered_ptr,  # [TOKENS, SHARD] bf16: what the window gather produced
    cache_ptr,
    slot_ptr,
    position_ptr,
    sequence_ptr,
    block_table_ptr,
    stride_src_t,
    stride_cache_block,
    stride_cache_head,
    stride_cache_token,
    stride_cache_dim,
    stride_block_table_r,
    max_blocks,
    block_size,
    gen_offset_u32,
    HIDDEN: tl.constexpr,
    SHARD: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLITS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    CACHE_OFF: tl.constexpr,
    WINDOW: tl.constexpr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
):
    """Kernel 3: publish this rank's shard into every rank's buf_out row.

    buf_out is [GEN, MAX_TOKENS, HIDDEN] -- no per-publisher slot dimension, so
    each rank writes its own channel range and the row assembles in place.
    """
    token = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    CSS: tl.constexpr = SHARD // SPLITS
    elem = tl.arange(0, BLOCK).to(tl.int64)
    pair = tl.arange(0, BLOCK // 2).to(tl.int64)
    elem_mask = elem < CSS
    pair_mask = pair < CSS // 2
    global_channel = split * CSS + elem

    residual = tl.load(
        src_ptr + token * stride_src_t + global_channel, mask=elem_mask, other=0.0
    )

    # --- windowed gather out of the KV cache, via the block table ---
    slot = tl.load(slot_ptr + token)
    valid = slot >= 0
    position = tl.load(position_ptr + token)
    sequence = tl.load(sequence_ptr + token)
    safe_slot = tl.maximum(slot, 0).to(tl.int64)
    head = tl.minimum(global_channel // HEAD_SIZE, SHARD // HEAD_SIZE - 1)
    dim = CACHE_OFF + global_channel % HEAD_SIZE
    own_ptr = (
        cache_ptr
        + (safe_slot // block_size) * stride_cache_block
        + head * stride_cache_head
        + (safe_slot % block_size) * stride_cache_token
        + dim * stride_cache_dim
    )
    current = tl.load(own_ptr, mask=valid & elem_mask, other=0.0)

    conv = tl.zeros([BLOCK], tl.float32)
    for tap_idx in tl.static_range(WINDOW):
        source_position = position - (WINDOW - 1) + tap_idx
        take = valid & (source_position >= 0)
        if tap_idx == WINDOW - 1:
            value = tl.where(take, current.to(tl.float32), 0.0)
        else:
            safe_position = tl.maximum(source_position, 0)
            # Clamping to max_blocks-1 keeps the block-table read in range even
            # when the position runs past what the table covers.
            logical_block = tl.minimum(safe_position // block_size, max_blocks - 1)
            physical_block = tl.load(
                block_table_ptr + sequence * stride_block_table_r + logical_block,
                mask=take,
                other=0,
            ).to(tl.int64)
            source_ptr = (
                cache_ptr
                + physical_block * stride_cache_block
                + head * stride_cache_head
                + (safe_position % block_size) * stride_cache_token
                + dim * stride_cache_dim
            )
            cached = tl.load(source_ptr, mask=take & elem_mask, other=0.0)
            value = tl.where(take, cached.to(tl.float32), 0.0)
        conv += value

    # Unit taps: the addressing is what is under test, not the convolution.
    output = (residual.to(tl.float32) + conv).to(tl.bfloat16)
    tl.store(gathered_ptr + token * SHARD + global_channel, output, mask=elem_mask)
    packed = _pack_bf16_pairs(output)

    ptrs = peer_ptrs.to(tl.pointer_type(tl.uint64))
    row_offset = (
        gen_offset_u32
        + token * (HIDDEN // 2)
        + RANK * (SHARD // 2)
        + split * (CSS // 2)
        + pair
    )
    for destination in tl.static_range(WORLD):
        base = tl.load(ptrs + destination).to(tl.pointer_type(tl.uint32))
        tl.store(base + row_offset, packed, mask=pair_mask)


@triton.jit
def gather_kernel(
    peer_ptrs,
    out_ptr,  # [TOKENS, HIDDEN] bf16: the assembled row
    status_ptr,
    spins_ptr,
    gen_offset_u32,
    max_spins,
    HIDDEN: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    RANK: tl.constexpr,
):
    """Kernel 4: one CTA waits for a complete row, unpacks it, then re-arms.

    Unlike kernel 2 this waits on the *whole* hidden row at once, so every
    rank's shard must have landed before it can proceed.
    """
    token = tl.program_id(0).to(tl.int64)
    pair = tl.arange(0, ROW_BLOCK // 2).to(tl.int64)
    pair_mask = pair < HIDDEN // 2
    channel = tl.arange(0, ROW_BLOCK).to(tl.int64)
    channel_mask = channel < HIDDEN

    ptrs = peer_ptrs.to(tl.pointer_type(tl.uint64))
    base = tl.load(ptrs + RANK).to(tl.pointer_type(tl.uint32))
    base += gen_offset_u32
    offsets = token * (HIDDEN // 2) + pair

    values = tl.load(base + offsets, mask=pair_mask, other=0, volatile=True)
    spins = 0
    while (
        tl.max(tl.where(pair_mask & (values == EMPTY_PAIR), 1, 0)) != 0
        and spins < max_spins
    ):
        values = tl.load(base + offsets, mask=pair_mask, other=0, volatile=True)
        spins += 1

    pending = tl.sum(tl.where(pair_mask & (values == EMPTY_PAIR), 1, 0))
    row = _unpack_bf16_pairs(values)
    tl.store(out_ptr + token * HIDDEN + channel, row, mask=channel_mask)
    # Re-arm, as _gather_norm_kernel does.
    tl.store(
        base + offsets,
        tl.full([ROW_BLOCK // 2], EMPTY_PAIR, tl.uint32),
        mask=pair_mask,
    )
    tl.store(status_ptr + token, pending)
    tl.store(spins_ptr + token, spins)


def rows_for(rank_id, tokens, hidden, device, it=0):
    """Per-rank, per-iteration payload. All values are small dyadic rationals,
    so they are exact in bf16 and comparisons can be exact. Never -0.0, which
    would collide with the sentinel."""
    c = torch.arange(hidden, device=device, dtype=torch.float32)
    t = torch.arange(tokens, device=device, dtype=torch.float32).unsqueeze(1)
    return (
        (rank_id + 1) + (c % 8) * 0.5 + (t % 4) * 0.125 + (it % 8) * 0.015625
    ).to(torch.bfloat16)


def shard_for(rank_id, tokens, shard, device, it=0):
    """Kernel 3's per-rank shard output. Offset by 0.25 so it can never be
    confused with a buf_in payload that leaked into the wrong buffer."""
    c = torch.arange(shard, device=device, dtype=torch.float32)
    t = torch.arange(tokens, device=device, dtype=torch.float32).unsqueeze(1)
    return (
        0.25 + (rank_id + 1) + (c % 8) * 0.5 + (t % 4) * 0.125 + (it % 8) * 0.015625
    ).to(torch.bfloat16)


dist.init_process_group()
rank = dist.get_rank()
world_sz = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)

shard_sz = HIDDEN // world_sz

splits = (
    (8 if TOKENS == 1 and shard_sz % 8 == 0 else 1) if SPLITS == "auto" else int(SPLITS)
)
if shard_sz % splits:
    raise SystemExit(f"shard {shard_sz} is not divisible by SPLITS={splits}")
# CS_P2 in the real code: next_pow2 of the PER-SPLIT tile, not the full shard.
BLOCK = triton.next_power_of_2(shard_sz // splits)
PAIR_BLOCK = BLOCK // 2
phase_grid = (TOKENS, splits)
device = torch.device("cuda", local_rank)
dtype = torch.bfloat16

if rank == 0:
    mib = GENERATIONS * MAX_TOKENS * HIDDEN * 2 / 2**20
    print(
        f"buf_in  [{GENERATIONS}, {MAX_TOKENS}, {world_sz}, {shard_sz}] = {mib:.0f} MiB\n"
        f"buf_out [{GENERATIONS}, {MAX_TOKENS}, {HIDDEN}] = {mib:.0f} MiB\n"
        f"{ITERS} iterations x 4 kernels, {TOKENS} tokens each, rotating "
        f"through {GENERATIONS} generations (spin budget {MAX_SPINS:,})",
        flush=True,
    )

# Lamport's exact geometry. buf_out is allocated but unused here; it keeps the
# symmetric-memory footprint equal to the real thing.
buf_in = symm_mem.empty(
    GENERATIONS, MAX_TOKENS, world_sz, shard_sz, dtype=dtype, device=device
)
buf_out = symm_mem.empty(GENERATIONS, MAX_TOKENS, HIDDEN, dtype=dtype, device=device)
buf_in.view(torch.int16).fill_(-0x8000)
buf_out.view(torch.int16).fill_(-0x8000)

in_hdl = symm_mem.rendezvous(buf_in, dist.group.WORLD)
out_hdl = symm_mem.rendezvous(buf_out, dist.group.WORLD)
if rank == 0:
    print("rendezvous OK for both buffers", flush=True)

status = torch.zeros(TOKENS * 8, dtype=torch.int32, device=device)
spins = torch.zeros(TOKENS * 8, dtype=torch.int32, device=device)
# What the waiters observed, before re-arming wipes it.
stage = torch.zeros(TOKENS, world_sz, shard_sz // 2, dtype=torch.int32, device=device)
row_stage = torch.zeros(TOKENS, HIDDEN, dtype=dtype, device=device)
gathered = torch.zeros(TOKENS, shard_sz, dtype=dtype, device=device)

# --- mock KV cache and the metadata the real kernels index it with ---
num_heads = shard_sz // HEAD_SIZE
max_blocks = (TOKENS // NUM_SEQS + BLOCK_SIZE) // BLOCK_SIZE + 1
num_blocks = NUM_SEQS * max_blocks
kv_cache = torch.zeros(
    num_blocks, num_heads, BLOCK_SIZE, HEAD_DIM, dtype=dtype, device=device
)
# Distinct physical blocks per sequence, so a wrong row in the table lands
# somewhere obviously wrong rather than aliasing onto the right data.
block_table = (
    torch.arange(NUM_SEQS * max_blocks, dtype=torch.int32, device=device)
    .view(NUM_SEQS, max_blocks)
)
seq_idx = (torch.arange(TOKENS, device=device) % NUM_SEQS).to(torch.int32)
positions = (torch.arange(TOKENS, device=device) // NUM_SEQS).to(torch.int32)
logical = positions // BLOCK_SIZE
physical = block_table[seq_idx.long(), logical.long()]
slot_mapping = (physical * BLOCK_SIZE + positions % BLOCK_SIZE).to(torch.int32)
if SLOT_MODE == "pad":
    # What vLLM's profiling dummy run passes: nothing is really cached.
    slot_mapping = torch.full_like(slot_mapping, -1)
elif SLOT_MODE == "mixed":
    slot_mapping[::2] = -1
elif SLOT_MODE != "valid":
    raise SystemExit(f"SLOT_MODE must be valid|pad|mixed, got {SLOT_MODE!r}")

# Channel -> (head, dim) exactly as the kernels compute it.
_ch = torch.arange(shard_sz, device=device)
head_idx = torch.clamp(_ch // HEAD_SIZE, max=num_heads - 1)
dim_idx = CACHE_OFFSET + _ch % HEAD_SIZE

if rank == 0:
    print(
        f"kv_cache [{num_blocks}, {num_heads}, {BLOCK_SIZE}, {HEAD_DIM}] = "
        f"{kv_cache.numel() * 2 / 2**20:.0f} MiB, slot_mode={SLOT_MODE}, "
        f"{int((slot_mapping >= 0).sum())}/{TOKENS} slots valid, "
        f"kernel block_size={KERNEL_BLOCK_SIZE}, SPLITS={splits} "
        f"(tile {shard_sz // splits} of shard {shard_sz})"
        + (
            f"  <-- MISMATCH, cache page size is {BLOCK_SIZE}; max slot "
            f"{int(slot_mapping.max())} -> block "
            f"{int(slot_mapping.max()) // KERNEL_BLOCK_SIZE} of {num_blocks}"
            if KERNEL_BLOCK_SIZE != BLOCK_SIZE
            else ""
        ),
        flush=True,
    )
status_out = torch.zeros(TOKENS, dtype=torch.int32, device=device)
spins_out = torch.zeros(TOKENS, dtype=torch.int32, device=device)

# Per-iteration tallies, kept on device so the loop never syncs to the host --
# a host sync each step would damp the rank drift this test is trying to create.
stalled_iters = torch.zeros(ITERS, dtype=torch.int32, device=device)
mismatch_iters = torch.zeros(ITERS, dtype=torch.int32, device=device)
max_spins_seen = torch.zeros(ITERS, dtype=torch.int32, device=device)
verify_iters = {0, ITERS - 1}

# Everyone must be armed before any rank writes into a peer. This is the only
# barrier: none between publish and wait, and none between iterations, so ranks
# are free to drift exactly as they do in a real forward pass.
torch.cuda.synchronize()
dist.barrier()

gen_in_u32 = MAX_TOKENS * world_sz * shard_sz // 2
gen_out_u32 = MAX_TOKENS * HIDDEN // 2
ROW_BLOCK = triton.next_power_of_2(HIDDEN)

for it in range(ITERS):
    gen = it % GENERATIONS
    gen_offset_u32 = gen * gen_in_u32
    # Payload varies per iteration, so data left over from a previous visit to
    # this generation shows up as a mismatch instead of passing silently.
    mine = rows_for(rank, TOKENS, HIDDEN, device, it)

    publish_kernel[phase_grid](
        mine,
        in_hdl.buffer_ptrs_dev,
        mine.stride(0),
        gen_offset_u32,
        SHARD=shard_sz,
        BLOCK=BLOCK,
        SPLITS=splits,
        RANK=rank,
        WORLD=world_sz,
    )
    wait_kernel[phase_grid](
        in_hdl.buffer_ptrs_dev,
        stage,
        status,
        spins,
        kv_cache,
        slot_mapping,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        KERNEL_BLOCK_SIZE,
        gen_offset_u32,
        MAX_SPINS,
        SHARD=shard_sz,
        PAIR_BLOCK=PAIR_BLOCK,
        SPLITS=splits,
        HEAD_SIZE=HEAD_SIZE,
        CACHE_OFF=CACHE_OFFSET,
        RANK=rank,
        WORLD=world_sz,
    )

    # --- second half: sconv publish into buf_out, then gather the full row ---
    my_shard = shard_for(rank, TOKENS, shard_sz, device, it)
    sconv_publish_kernel[phase_grid](
        my_shard,
        out_hdl.buffer_ptrs_dev,
        gathered,
        kv_cache,
        slot_mapping,
        positions,
        seq_idx,
        block_table,
        my_shard.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        block_table.stride(0),
        max_blocks,
        KERNEL_BLOCK_SIZE,
        gen * gen_out_u32,
        HIDDEN=HIDDEN,
        SHARD=shard_sz,
        BLOCK=BLOCK,
        SPLITS=splits,
        HEAD_SIZE=HEAD_SIZE,
        CACHE_OFF=CACHE_OFFSET,
        WINDOW=WINDOW,
        RANK=rank,
        WORLD=world_sz,
    )
    gather_kernel[(TOKENS,)](
        out_hdl.buffer_ptrs_dev,
        row_stage,
        status_out,
        spins_out,
        gen * gen_out_u32,
        MAX_SPINS,
        HIDDEN=HIDDEN,
        ROW_BLOCK=ROW_BLOCK,
        RANK=rank,
    )

    # Verify on device: stage[token, p] must hold publisher p's shard for us.
    observed = stage.view(torch.bfloat16)  # [TOKENS, WORLD, SHARD]
    bad = torch.zeros((), dtype=torch.int32, device=device)
    for p in range(world_sz):
        want = rows_for(p, TOKENS, HIDDEN, device, it)[
            :, rank * shard_sz : (rank + 1) * shard_sz
        ]
        bad += (observed[:, p, :] != want).sum().to(torch.int32)
    # This rank's slice of the assembled row must be exactly what it published.
    own = row_stage[:, rank * shard_sz : (rank + 1) * shard_sz]
    bad += (own != gathered).sum().to(torch.int32)

    # Checking the cache addressing means re-deriving it in torch, which is far
    # too slow to do every iteration; first and last is enough to catch a wrong
    # index, and every iteration still runs the kernels.
    if it in verify_iters:
        reduced = stage.view(torch.bfloat16).float().sum(dim=1).to(dtype)
        live = (slot_mapping >= 0).nonzero().flatten()
        if live.numel():
            sl = slot_mapping[live].long()
            blk, tk = sl // BLOCK_SIZE, sl % BLOCK_SIZE
            got = kv_cache[blk[:, None], head_idx[None, :], tk[:, None], dim_idx[None, :]]
            bad += (got != reduced[live]).sum().to(torch.int32)

        # Re-derive the 4-tap window gather straight from the cache contents.
        acc = torch.zeros(TOKENS, shard_sz, dtype=torch.float32, device=device)
        valid = slot_mapping >= 0
        for tap in range(WINDOW):
            sp = positions.long() - (WINDOW - 1) + tap
            take = valid & (sp >= 0)
            safe_sp = sp.clamp(min=0)
            if tap == WINDOW - 1:
                sl_all = slot_mapping.clamp(min=0).long()
                blk_t, tk_t = sl_all // BLOCK_SIZE, sl_all % BLOCK_SIZE
            else:
                lg = torch.minimum(
                    safe_sp // BLOCK_SIZE,
                    torch.full_like(safe_sp, max_blocks - 1),
                )
                blk_t = block_table[seq_idx.long(), lg].long()
                tk_t = safe_sp % BLOCK_SIZE
            vals = kv_cache[
                blk_t[:, None], head_idx[None, :], tk_t[:, None], dim_idx[None, :]
            ].float()
            acc += torch.where(take[:, None], vals, torch.zeros_like(vals))
        want_gathered = (my_shard.float() + acc).to(dtype)
        bad += (gathered != want_gathered).sum().to(torch.int32)

    mismatch_iters[it] = bad
    stalled_iters[it] = ((status != 0).sum() + (status_out != 0).sum()).to(torch.int32)
    max_spins_seen[it] = torch.maximum(spins.max(), spins_out.max())

torch.cuda.synchronize()

failures = []
stalled_total = int(stalled_iters.sum())
if stalled_total:
    first = int((stalled_iters != 0).nonzero()[0])
    failures.append(
        f"spin-wait gave up {stalled_total} time(s) after {MAX_SPINS:,} spins, "
        f"first at iteration {first} (generation {first % GENERATIONS}) -- peer "
        f"writes never became visible"
    )

mismatch_total = int(mismatch_iters.sum())
if mismatch_total:
    first = int((mismatch_iters != 0).nonzero()[0])
    failures.append(
        f"observed wrong payload on {int((mismatch_iters != 0).sum())} of {ITERS} "
        f"iterations ({mismatch_total} elements total), first at iteration {first} "
        f"(generation {first % GENERATIONS}) -- stale data from an earlier visit "
        f"to this generation, or a peer overwrote it"
    )

# Every generation must be back to the sentinel: the consumer re-arms what it
# reads, so nothing should be left behind once the loop drains.
dist.barrier()
for g in range(GENERATIONS):
    left = int((buf_in[g].view(torch.int16) != -0x8000).sum())
    if left:
        failures.append(f"generation {g} left {left} slots un-rearmed")
for g in range(GENERATIONS):
    left = int((buf_out[g].view(torch.int16) != -0x8000).sum())
    if left:
        failures.append(f"buf_out generation {g} left {left} slots un-rearmed")

if failures:
    print(f"[rank {rank}] FAIL", *failures, sep="\n  ", flush=True)
else:
    print(
        f"[rank {rank}] OK: {ITERS} iterations over {GENERATIONS} rotating "
        f"generations, {TOKENS} tokens each (max {int(max_spins_seen.max())} spins)",
        flush=True,
    )

# Let the exit status reflect every rank, not just this one.
bad = torch.tensor(float(len(failures)), device=device)
dist.all_reduce(bad)
dist.barrier()
dist.destroy_process_group()
raise SystemExit(1 if bad.item() else 0)
