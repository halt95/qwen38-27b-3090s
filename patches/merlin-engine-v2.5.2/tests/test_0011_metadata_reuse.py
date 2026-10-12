# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN RecoverSSM metadata: the later GDN groups of a batch reuse the first
group's metadata (`update_block_table`) instead of a full per-group build.

Every group's metadata must equal, value for value, what a full `build` of that
group produces: the batch-level fields, the per-group state indices (spec,
non-spec, prefill, the graph-staged buffers) and the per-group commit inputs
(state indices, query offsets, request rows, block table, computed counts,
block size, context binding). Runs the real MRV2 `build_attn_metadata` over
three groups with distinct block tables on randomized batches: mixed
prefill/decode, resumed one-token chunks, partial drafts, graph padding,
reordered rows, align and non-align modes, eager and FULL staging, capture."""

import random
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.triton_utils as _tu

if getattr(_tu.tl, "constexpr", None) is None:  # triton-less host (CPU tests)
    _tu.tl.constexpr = lambda x: x

from vllm.config.compilation import CUDAGraphMode  # noqa: E402
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata  # noqa: E402
from vllm.v1.attention.backends.gdn_recoverssm import (  # noqa: E402
    GDNRecoverSSMMetadata,
    GDNRecoverSSMMetadataBuilder,
    _promote,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata  # noqa: E402
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (  # noqa: E402
    MambaHybridAttnMetadata,
)

NUM_SPEC = 4
SPEC_QUERY_LEN = NUM_SPEC + 1
NUM_GROUPS = 3
LAYERS_PER_GROUP = 2
NUM_BLOCKS = 512
HV, V, K = 2, 4, 4
CONV_DIM = 8
MAX_REQS = 8


def _config(mode: str, full_cg: bool, fc: dict):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=(
                CUDAGraphMode.FULL_AND_PIECEWISE if full_cg else CUDAGraphMode.NONE
            ),
            max_cudagraph_capture_size=None,
            static_forward_context=fc,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=NUM_SPEC, parallel_drafting=False
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_REQS),
        cache_config=SimpleNamespace(mamba_cache_mode=mode),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        additional_config={},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=K)
        ),
    )


def _layers(device):
    fc = {}
    names = []
    for g in range(NUM_GROUPS):
        group = []
        for i in range(LAYERS_PER_GROUP):
            name = f"model.layers.{g * LAYERS_PER_GROUP + i}.linear_attn"
            conv = torch.zeros(NUM_BLOCKS, SPEC_QUERY_LEN + 2, CONV_DIM, device=device)
            state = torch.zeros(NUM_BLOCKS, HV, V, K, device=device).half()
            replay = torch.zeros(
                NUM_BLOCKS, HV, SPEC_QUERY_LEN, V + K + 1, device=device
            )
            fc[name] = SimpleNamespace(kv_cache=[conv, state, replay])
            group.append(name)
        names.append(group)
    return fc, names


def _builders(mode, full_cg, device, reuse: bool, block_size: int):
    fc, names = _layers(device)
    cfg = _config(mode, full_cg, fc)
    spec = MambaSpec(
        block_size=block_size,
        shapes=((SPEC_QUERY_LEN + 2, CONV_DIM), (HV, V, K)),
        dtypes=(torch.float32, torch.float16),
        mamba_cache_mode=mode,
    )
    groups, builders = [], []
    for g in range(NUM_GROUPS):
        b = GDNRecoverSSMMetadataBuilder(spec, names[g], cfg, device)
        # The production condition needs CUDA; force both arms on any device.
        b.supports_update_block_table = reuse
        builders.append(b)
        groups.append(
            [
                SimpleNamespace(
                    get_metadata_builder=(lambda idx, b=b: b),
                    kv_cache_spec=spec,
                    layer_names=names[g],
                )
            ]
        )
    kv_cfg = SimpleNamespace(kv_cache_groups=[None] * NUM_GROUPS)
    return groups, builders, kv_cfg, names


KINDS = ("spec", "spec_part", "decode0", "prefill", "prefill_cont", "prefill1")


def _batch(rng: random.Random, mode: str, block_size: int, device, uniform: bool):
    num_real = rng.randint(1, 5)
    if uniform:
        kinds = ["spec"] * num_real
    else:
        kinds = [rng.choice(KINDS) for _ in range(num_real)]
    num_pad = rng.randint(0, MAX_REQS - num_real)
    n = num_real + num_pad
    qlens, comp, drafts, prefilling = [], [], [], []
    for k in kinds:
        if k == "spec":
            q, d, c, p = SPEC_QUERY_LEN, NUM_SPEC, rng.randint(1, 200), False
        elif k == "spec_part":
            d = rng.randint(1, NUM_SPEC - 1)
            q, c, p = d + 1, rng.randint(1, 200), False
        elif k == "decode0":
            q, d, c, p = 1, 0, rng.randint(1, 200), False
        elif k == "prefill":
            q, d, c, p = rng.randint(2, 40), 0, 0, True
        elif k == "prefill_cont":
            q, d, c, p = rng.randint(2, 40), 0, rng.randint(1, 120), True
        else:  # resumed one-token chunk
            q, d, c, p = 1, 0, rng.randint(1, 120), True
        qlens.append(q)
        comp.append(c)
        drafts.append(d)
        prefilling.append(p)
    qlens += [0] * num_pad
    comp += [0] * num_pad
    seq = [c + q for c, q in zip(comp, qlens)]
    qsl = np.zeros(n + 1, dtype=np.int32)
    qsl[1:] = np.cumsum(qlens)
    if mode == "align":
        max_blocks = -(-max(max(seq), 1) // block_size) + 1
    else:  # "none": one checkpoint slot per spec position
        max_blocks = SPEC_QUERY_LEN
    tables = []
    for _ in range(NUM_GROUPS):
        ids = rng.sample(range(1, NUM_BLOCKS), n * max_blocks)
        tables.append(
            torch.tensor(ids, dtype=torch.int32, device=device).view(n, max_blocks)
        )
    is_prefilling = torch.zeros(n, dtype=torch.bool)
    is_prefilling[:num_real] = torch.tensor(prefilling)
    # As MambaHybridModelState.prepare_attn: -1 unless a whole-prompt decode row
    # with drafts.
    ndd = np.full(n, -1, dtype=np.int32)
    for i, (k, d, q) in enumerate(zip(kinds, drafts, qlens)):
        if d > 0 and q == d + 1 and not prefilling[i]:
            ndd[i] = d
    accepted = torch.ones(n, dtype=torch.int32, device=device)
    accepted[:num_real] = torch.tensor(
        [rng.randint(1, SPEC_QUERY_LEN) for _ in range(num_real)], dtype=torch.int32
    )
    seq_t = torch.tensor(seq, dtype=torch.int32, device=device)
    return SimpleNamespace(
        n=n,
        num_tokens=int(qsl[-1]),
        qsl_gpu=torch.from_numpy(qsl).to(device),
        qsl_cpu=torch.from_numpy(qsl),
        max_query_len=max(qlens),
        seq_lens=seq_t,
        seq_cpu=torch.tensor(seq, dtype=torch.int32),
        tables=tables,
        mm=MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=accepted,
            num_decode_draft_tokens_cpu=torch.from_numpy(ndd),
        ),
    )


def _build(groups, kv_cfg, b, device, capture: bool):
    slots = torch.zeros(
        NUM_GROUPS, max(b.num_tokens, 1), dtype=torch.int64, device=device
    )
    return build_attn_metadata(
        attn_groups=groups,
        num_reqs=b.n,
        num_tokens=b.num_tokens,
        query_start_loc_gpu=b.qsl_gpu,
        query_start_loc_cpu=b.qsl_cpu,
        max_query_len=b.max_query_len,
        seq_lens=b.seq_lens,
        max_seq_len=int(b.seq_cpu.max()),
        block_tables=b.tables,
        slot_mappings=slots,
        kv_cache_config=kv_cfg,
        seq_lens_cpu_upper_bound=b.seq_cpu,
        model_specific_attn_metadata=None if capture else b.mm,
        is_prefilling=b.mm.is_prefilling if capture else None,
        for_cudagraph_capture=capture,
    )


def _same(a, b, where):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), where
        assert a.dtype == b.dtype and a.shape == b.shape, where
        assert torch.equal(a.cpu(), b.cpu()), where
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), where
        for k in a:
            _same(a[k], b[k], f"{where}.{k}")
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"


def _compare(ref, new, names):
    for g, group in enumerate(names):
        r, x = ref[group[0]], new[group[0]]
        assert type(r) is type(x) is GDNRecoverSSMMetadata
        for f in fields(GDNAttentionMetadata):
            _same(getattr(r, f.name), getattr(x, f.name), f"group {g} {f.name}")
        rc, xc = r.recoverssm_commit, x.recoverssm_commit
        assert (rc is None) == (xc is None), f"group {g} commit"
        if rc is None:
            assert r.recoverssm_context is None and x.recoverssm_context is None
            continue
        for f in fields(rc):
            _same(
                getattr(rc, f.name), getattr(xc, f.name), f"group {g} commit.{f.name}"
            )
        # Bound to this group's layers (not the first group's).
        assert [t.data_ptr() for t in x.recoverssm_context.checkpoints] == [
            t.data_ptr() for t in r.recoverssm_context.checkpoints
        ]
        # The commit's state indices are the metadata's checkpoint column.
        assert xc.state_indices is x.spec_state_indices_tensor


def _devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


@pytest.mark.parametrize("device", _devices(), ids=str)
@pytest.mark.parametrize("mode", ["align", "none"])
@pytest.mark.parametrize("full_cg", [True, False], ids=["full", "eager"])
@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "mixed"])
def test_reuse_matches_full_build(device, mode, full_cg, uniform):
    block_size = 16 if mode == "align" else 4096
    ref_groups, ref_b, kv_cfg, names = _builders(
        mode, full_cg, device, False, block_size
    )
    new_groups, new_b, _, _ = _builders(mode, full_cg, device, True, block_size)
    # Both arms bind the same layer objects for the context check.
    for rb, nb in zip(ref_b, new_b):
        nb.vllm_config.compilation_config.static_forward_context = (
            rb.vllm_config.compilation_config.static_forward_context
        )
    rng = random.Random(f"{device}-{mode}-{full_cg}-{uniform}")
    for _ in range(150):
        b = _batch(rng, mode, block_size, device, uniform)
        ref = _build(ref_groups, kv_cfg, b, device, capture=False)
        new = _build(new_groups, kv_cfg, b, device, capture=False)
        _compare(ref, new, names)


@pytest.mark.parametrize("device", _devices(), ids=str)
@pytest.mark.parametrize("mode", ["align", "none"])
def test_reuse_matches_full_build_at_capture(device, mode):
    block_size = 16 if mode == "align" else 4096
    ref_groups, ref_b, kv_cfg, names = _builders(mode, True, device, False, block_size)
    new_groups, new_b, _, _ = _builders(mode, True, device, True, block_size)
    for rb, nb in zip(ref_b, new_b):
        nb.vllm_config.compilation_config.static_forward_context = (
            rb.vllm_config.compilation_config.static_forward_context
        )
    rng = random.Random(f"capture-{device}-{mode}")
    for _ in range(40):
        b = _batch(rng, mode, block_size, device, uniform=True)
        b.mm.is_prefilling.zero_()
        ref = _build(ref_groups, kv_cfg, b, device, capture=True)
        new = _build(new_groups, kv_cfg, b, device, capture=True)
        _compare(ref, new, names)


def test_reused_staging_writes_own_state_buffers():
    """FULL staging: each group's staged state indices live in its own builder's
    buffer (the verify graph of that group reads them), batch-level buffers are
    the first group's."""
    device = torch.device("cpu")
    groups, builders, kv_cfg, names = _builders("align", True, device, True, 16)
    b = _batch(random.Random(7), "align", 16, device, uniform=True)
    md = _build(groups, kv_cfg, b, device, capture=False)
    for g, group in enumerate(names):
        m = md[group[0]]
        assert m.spec_state_indices_tensor is not None
        own = builders[g].spec_state_indices_tensor
        assert m.spec_state_indices_tensor.data_ptr() == own.data_ptr()
        assert m.spec_query_start_loc.data_ptr() == (
            builders[0].spec_query_start_loc.data_ptr()
        )
        assert m.recoverssm_commit.block_table is b.tables[g]


def test_promote_equals_field_copy():
    rng = random.Random(3)
    device = torch.device("cpu")
    groups, builders, kv_cfg, names = _builders("align", True, device, False, 16)
    for _ in range(30):
        b = _batch(rng, "align", 16, device, uniform=False)
        md = _build(groups, kv_cfg, b, device, capture=False)
        m = md[names[0][0]]
        extra = {
            "recoverssm_commit": m.recoverssm_commit,
            "recoverssm_context": m.recoverssm_context,
            "recoverssm_seq_lens": m.recoverssm_seq_lens,
        }
        base = {f.name: getattr(m, f.name) for f in fields(GDNAttentionMetadata)}
        ref = GDNRecoverSSMMetadata(**base, **extra)
        out = _promote(GDNAttentionMetadata(**base), **extra)
        assert type(out) is GDNRecoverSSMMetadata
        assert out.__dict__.keys() == ref.__dict__.keys()
        for k in ref.__dict__:
            assert out.__dict__[k] is ref.__dict__[k], k


def test_context_binding_still_fails_after_rebind():
    device = torch.device("cpu")
    groups, builders, kv_cfg, names = _builders("align", True, device, True, 16)
    b = _batch(random.Random(11), "align", 16, device, uniform=True)
    _build(groups, kv_cfg, b, device, capture=False)
    fc = builders[0].vllm_config.compilation_config.static_forward_context
    # A KV cache re-allocation binds new tensors to every layer.
    for name in fc:
        fc[name].kv_cache = [t.clone() for t in fc[name].kv_cache]
    with pytest.raises(RuntimeError, match="stale context"):
        _build(groups, kv_cfg, b, device, capture=False)


def _spec_rows_torch_reference(is_prefilling, qsl, drafts_in, num_reqs):
    # The pre-NumPy formulation of recoverssm_spec_rows.
    query_lens = qsl[: num_reqs + 1].diff()
    active_decode = (~is_prefilling[:num_reqs]) & (query_lens > 0)
    spec_mask = (drafts_in[:num_reqs] >= 0) | active_decode
    drafts = torch.full_like(drafts_in, -1)
    drafts[:num_reqs] = torch.where(spec_mask, 1, -1).to(drafts.dtype)
    return drafts, spec_mask.nonzero().flatten()


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_spec_rows_numpy_matches_torch(dtype):
    from vllm.v1.attention.backends.recoverssm_metadata import recoverssm_spec_rows

    rng = random.Random(str(dtype))
    with torch.inference_mode():
        for _ in range(2000):
            n_pad = rng.randint(1, 12)
            num_reqs = rng.randint(0, n_pad)
            qlens = [rng.choice([0, 1, 2, 5, 9]) for _ in range(num_reqs)]
            qlens += [0] * (n_pad - num_reqs)
            qsl = torch.zeros(n_pad + 1, dtype=torch.int32)
            qsl[1:] = torch.tensor(qlens, dtype=torch.int32).cumsum(0)
            prefilling = torch.tensor(
                [rng.random() < 0.4 for _ in range(n_pad)], dtype=torch.bool
            )
            drafts = torch.tensor(
                [rng.choice([-1, -1, 0, 1, 4]) for _ in range(n_pad)], dtype=dtype
            )
            ref = _spec_rows_torch_reference(prefilling, qsl, drafts, num_reqs)
            out = recoverssm_spec_rows(prefilling, qsl, drafts, num_reqs)
            for r, o in zip(ref, out):
                assert r.dtype == o.dtype and r.shape == o.shape
                assert torch.equal(r, o)


@pytest.mark.parametrize("layer", [1, LAYERS_PER_GROUP - 1])
def test_context_binding_fails_after_non_first_layer_rebind(layer):
    """Rebinding any one layer's checkpoint (not just the first) must reach the
    full address check."""
    device = torch.device("cpu")
    groups, builders, kv_cfg, names = _builders("align", True, device, True, 16)
    b = _batch(random.Random(13), "align", 16, device, uniform=True)
    _build(groups, kv_cfg, b, device, capture=False)
    fc = builders[0].vllm_config.compilation_config.static_forward_context
    kv = fc[names[0][layer]].kv_cache
    fc[names[0][layer]].kv_cache = [kv[0], kv[1].clone(), kv[2]]
    with pytest.raises(RuntimeError, match="stale context"):
        _build(groups, kv_cfg, b, device, capture=False)


def _built_after_one_step(seed):
    device = torch.device("cpu")
    groups, builders, kv_cfg, names = _builders("align", True, device, True, 16)
    b = _batch(random.Random(seed), "align", 16, device, uniform=True)
    _build(groups, kv_cfg, b, device, capture=False)
    fc = builders[0].vllm_config.compilation_config.static_forward_context
    return groups, kv_cfg, names, b, device, fc


@pytest.mark.parametrize("group", [0, NUM_GROUPS - 1])
@pytest.mark.parametrize(
    "change", ["set_storage", "set_shape", "set_stride"], ids=lambda c: c
)
def test_context_binding_fails_after_in_place_storage_change(group, change):
    """The same checkpoint tensor OBJECT with new storage, shape or stride on a
    non-first layer (the KV cache re-initialised through set_()) must reach the
    full address check and fail: the commit reaches that layer only through the
    address table the context was built from, so nothing downstream would see
    the stale pointer. Identity alone (the pre-fix fast path) passes these."""
    groups, kv_cfg, names, b, device, fc = _built_after_one_step(17)
    layer = LAYERS_PER_GROUP - 1
    ckpt = fc[names[group][layer]].kv_cache[1]
    if change == "set_storage":
        ckpt.set_(ckpt.clone())  # same object, same shape/stride, new storage
    elif change == "set_shape":
        # same storage and data_ptr, one block fewer
        ckpt.set_(
            ckpt.untyped_storage(),
            ckpt.storage_offset(),
            (ckpt.shape[0] - 1,) + tuple(ckpt.shape[1:]),
            ckpt.stride(),
        )
    else:
        # same storage, data_ptr and shape, a different block stride
        st = list(ckpt.stride())
        st[0] = st[0] - 1
        ckpt.set_(ckpt.untyped_storage(), ckpt.storage_offset(), ckpt.shape, st)
    with pytest.raises(RuntimeError, match="stale context"):
        _build(groups, kv_cfg, b, device, capture=False)


def test_context_binding_accepts_a_new_object_over_the_same_storage():
    """Control: a different tensor object over the very same storage (same
    data_ptr, shape and stride) is the same binding, as before the fix."""
    groups, kv_cfg, names, b, device, fc = _built_after_one_step(19)
    kv = fc[names[1][LAYERS_PER_GROUP - 1]].kv_cache
    view = kv[1].view(kv[1].shape)
    assert view is not kv[1] and view.data_ptr() == kv[1].data_ptr()
    fc[names[1][LAYERS_PER_GROUP - 1]].kv_cache = [kv[0], view, kv[2]]
    _build(groups, kv_cfg, b, device, capture=False)


def _graph_read(md, names, outs):
    """What a captured GDN verify reads from each group's metadata (persistent
    pointers), copied into fixed output buffers."""
    for g, group in enumerate(names):
        m = md[group[0]]
        for k, name in enumerate(
            (
                "spec_state_indices_tensor",
                "spec_query_start_loc",
                "spec_sequence_masks",
                "num_accepted_tokens",
            )
        ):
            src = getattr(m, name)
            outs[g][k][: src.numel()].copy_(src.reshape(-1).to(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("reuse", [False, True], ids=["full-build", "reuse"])
def test_cuda_graph_replay_reads_fresh_metadata(reuse):
    """Persistent-pointer claim: capture a graph over the capture-time metadata,
    then rebuild metadata for new uniform batches of the same padded shape and
    replay. The graph must read exactly what an eager read of the freshly built
    metadata (reference arm, full builds) sees."""
    device = torch.device("cuda")
    groups, builders, kv_cfg, names = _builders("align", True, device, reuse, 16)
    ref_groups, ref_b, _, _ = _builders("align", True, device, False, 16)
    fc = builders[0].vllm_config.compilation_config.static_forward_context
    for rb in ref_b:
        rb.vllm_config.compilation_config.static_forward_context = fc
    rng = random.Random(f"graph-{reuse}")

    def uniform(n):
        # Uniform spec-decode batches (1-5 real rows) padded to n requests.
        while True:
            b = _batch(rng, "align", 16, device, uniform=True)
            if b.n == n:
                return b

    n = 6
    cap = uniform(n)
    cap.mm.is_prefilling.zero_()
    md = _build(groups, kv_cfg, cap, device, capture=True)
    outs = [
        [torch.zeros(64, dtype=torch.int32, device=device) for _ in range(4)]
        for _ in names
    ]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _graph_read(md, names, outs)
    for _ in range(20):
        b = uniform(n)
        _build(groups, kv_cfg, b, device, capture=False)
        for o in outs:
            for t in o:
                t.fill_(-7)
        graph.replay()
        ref = _build(ref_groups, kv_cfg, b, device, capture=False)
        expect = [[torch.full_like(t, -7) for t in o] for o in outs]
        _graph_read(ref, names, expect)
        torch.cuda.synchronize()
        for o, e in zip(outs, expect):
            for t, u in zip(o, e):
                assert torch.equal(t, u)
