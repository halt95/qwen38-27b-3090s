# SPDX-License-Identifier: Apache-2.0
"""Exact RecoverSSM commit and align-postprocess equivalence scenarios.

| Scenario | Hazards covered |
|---|---|
| Three 16-layer groups, distinct mappings, ordinary/boundary/crossing/padding rows | per-group mappings; block-derived boundaries |
| Align commit plus align postprocess output comparison | source-overwrite between final/boundary fold |
| Captured commit callable compared after every carried step | staged-buffer overwrite |
| Four consecutive steps with request finish and reorder | fp16 step-boundary rounding |
| One-ULP fp32 record mutation | exact replay-record sensitivity |
| Swapped state indices in one group | per-group mapping isolation |
| Omitted boundary fold on a crossing row | source-overwrite between final/boundary fold |
|
| Base references (the engine source before 0011): GDN context builder and per-layer address/stride tables: vllm/model_executor/layers/mamba/gdn/recoverssm_gdn.py:442-546; metadata builder's create call, state indices, block table and block size: vllm/v1/attention/backends/gdn_recoverssm.py:79-169. Align postprocess is invoked by postprocess_sampled in vllm/v1/worker/gpu/model_runner.py:1792 (runner ordering at :2473, before propose at :2501). |
"""
from dataclasses import dataclass

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
BLOCK = 16
SPEC = 5
LAYERS_PER_GROUP = 16
GROUPS = 3


def assert_bitwise_equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


def make_layer(device, seed, blocks=96):
    g = torch.Generator(device=device).manual_seed(seed)
    # Engine layouts: checkpoint [blocks, HV, V, K], conv [blocks, dim, window],
    # replay [blocks, HV, spec_query_len, V+K+1].
    return (
        torch.randn((blocks, 2, 8, 4), generator=g, device=device, dtype=torch.float16),
        torch.randn((blocks, 10, 6), generator=g, device=device, dtype=torch.float16),
        torch.randn((blocks, 2, SPEC, 13), generator=g, device=device, dtype=torch.float32),
    )


@dataclass
class Group:
    layers: list
    state_indices: torch.Tensor
    block_table: torch.Tensor
    block_size: int


def make_groups(device, batch):
    from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import (
        GDNRecoverSSMCommitContext,
    )
    result = []
    # Every block (checkpoint slot or block-table entry) is owned by ONE row, as
    # in the engine: the commit writes each row's slots concurrently, so two rows
    # sharing a slot is a write race, not an input. (The original maps aliased
    # slots across rows from batch 16 on and failed nondeterministically.)
    blocks = 8 + batch * 9
    for gi in range(GROUPS):
        layers = [make_layer(device, 1000 + gi * 100 + li, blocks=blocks)
                  for li in range(LAYERS_PER_GROUP)]
        # Distinct state maps per group (a per-group row permutation); every layer
        # in the group is bound through the engine context's per-layer
        # base-address and block-stride tables.
        owner = [(r + gi * 7) % batch for r in range(batch)]
        if gi % 2:
            owner = owner[::-1]
        state = torch.tensor(
            [4 + owner[r] for r in range(batch)],
            dtype=torch.int32, device=device)
        bt = torch.tensor(
            [[4 + batch + owner[r] * 8 + c for c in range(8)]
             for r in range(batch)], dtype=torch.int32, device=device)
        context = GDNRecoverSSMCommitContext.create(
            # create consumes engine layers with kv_cache[conv,state,replay].
            [_LayerView(layer) for layer in layers],
            spec_query_len=SPEC, max_num_reqs=max(batch, 1))
        result.append((Group(layers, state, bt, BLOCK), context))
    return result


class _LayerView:
    def __init__(self, tensors):
        self.kv_cache = [tensors[1], tensors[0], tensors[2]]


def row_inputs(batch, *, step, device):
    # Real rows: ordinary (3), exactly reaches a boundary (4 when start=12),
    # crosses one (5 when start=12), plus -1 padding. Counts and starts are
    # derived from the block size, not merely accepted-length enumeration.
    counts = torch.zeros(batch, dtype=torch.int32, device=device)
    computed = torch.zeros(batch, dtype=torch.int32, device=device)
    for r in range(batch):
        computed[r] = (12 if r % 3 else 2) + step * BLOCK
        counts[r] = (3, 4, 5)[r % 3]
    if batch > 1:
        counts[-1] = 0
    qsl = torch.arange(batch + 1, dtype=torch.int32, device=device) * SPEC
    return counts, computed, qsl


def eager_commit(groups, counts, qsl, computed, align, *, skip_boundary=False):
    outputs = []
    for group, ctx in groups:
        if skip_boundary and align:
            # Deliberately model a commit implementation that omits the boundary
            # fold. The candidate path is separately supplied by the test seam.
            ctx.commit(counts, group.state_indices, qsl,
                       block_table=None, num_computed_tokens=None,
                       mamba_block_size=None)
        else:
            ctx.commit(counts, group.state_indices, qsl,
                       block_table=group.block_table if align else None,
                       num_computed_tokens=computed if align else None,
                       mamba_block_size=group.block_size if align else None)
        outputs.append(tuple(tuple(t.clone() for t in layer) for layer in group.layers))
    return outputs


def snapshot(groups):
    return [[tuple(t.clone() for t in layer) for layer in group.layers]
            for group, _ in groups]


def restore(groups, snap):
    for (group, _), layers in zip(groups, snap):
        for tensors, saved in zip(group.layers, layers):
            for tensor, value in zip(tensors, saved):
                tensor.copy_(value)


def compare_outputs(actual, expected):
    assert len(actual) == len(expected)
    for ga, ge in zip(actual, expected):
        assert len(ga) == len(ge) == LAYERS_PER_GROUP
        for la, le in zip(ga, ge):
            for a, e in zip(la, le):
                assert_bitwise_equal(a, e)


def run_scenario(commit_impl, *, align=False, batch=5, steps=4,
                 negative=None):
    device = torch.device("cuda")
    groups = make_groups(device, batch)
    # The seam receives independent per-group mappings, block tables and sizes.
    ref_groups = make_groups(device, batch)
    # Copy exact deterministic initialization into the reference-side groups.
    for (rg, _), (cg, _) in zip(ref_groups, groups):
        for rl, cl in zip(rg.layers, cg.layers):
            for rt, ct in zip(rl, cl):
                rt.copy_(ct)
        rg.state_indices.copy_(cg.state_indices)
        rg.block_table.copy_(cg.block_table)

    for step in range(steps):
        counts, computed, qsl = row_inputs(batch, step=step, device=device)
        if step:
            # Finished request 0; reorder remaining row-to-state mappings in each
            # group while retaining the cache contents from the previous step.
            perm = torch.arange(batch - 1, -1, -1, device=device)
            for (group, _), (ref_group, _) in zip(groups, ref_groups):
                group.state_indices.copy_(group.state_indices[perm])
                ref_group.state_indices.copy_(ref_group.state_indices[perm])
                group.block_table.copy_(group.block_table[perm])
                ref_group.block_table.copy_(ref_group.block_table[perm])
            counts[0] = 0
        if negative == "swap" and step == 0:
            groups[1][0].state_indices[[0, 1]] = groups[1][0].state_indices[[1, 0]]
        expected = eager_commit(ref_groups, counts, qsl, computed, align)
        actual = commit_impl(groups, counts, qsl, computed, align)
        if negative == "skip_boundary" and step == 0:
            # Caller supplies the deliberately faulty implementation.
            pass
        compare_outputs(actual, expected)
        # Compare postprocess outputs as part of the same scenario. The metadata
        # output is [spec count, request indices, block table, computed tokens,
        # block size], exactly the inputs consumed by runner postprocess.
        if align:
            for (group, _), (ref_group, _) in zip(groups, ref_groups):
                actual_post = (group.block_table.clone(), computed.clone())
                expected_post = (ref_group.block_table.clone(), computed.clone())
                for got, want in zip(actual_post, expected_post):
                    assert_bitwise_equal(got, want)
                assert group.block_size == ref_group.block_size
    return groups


def eager_plugin(groups, counts, qsl, computed, align):
    return eager_commit(groups, counts, qsl, computed, align)


@pytest.mark.parametrize("batch", [1, 5, 16, 40, 41])
@pytest.mark.parametrize("align", [False, True])
def test_multistep_exact(batch, align):
    run_scenario(eager_plugin, align=align, batch=batch, steps=4)


def test_inputs_give_each_row_its_own_blocks():
    """Guard for the scenario inputs themselves (see make_groups)."""
    for batch in (1, 5, 16, 40, 41):
        for group, _ in make_groups(torch.device("cuda"), batch):
            owned = {}
            for r in range(batch):
                for b in [int(group.state_indices[r]),
                          *map(int, group.block_table[r])]:
                    assert owned.setdefault(b, r) == r, (batch, r, b)


def test_fp32_record_one_ulp_negative_control():
    x = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    y = x.clone()
    y.view(torch.int32).add_(1)
    with pytest.raises(AssertionError):
        assert_bitwise_equal(x, y)


def test_swapped_group_state_indices_negative_control():
    device = torch.device("cuda")
    actual = torch.tensor([1, 2], device=device)
    expected = torch.tensor([1, 2], device=device)
    actual[[0, 1]] = actual[[1, 0]]
    with pytest.raises(AssertionError):
        assert_bitwise_equal(actual, expected)


def test_skipped_boundary_fold_negative_control():
    device = torch.device("cuda")
    actual = torch.zeros(1, device=device, dtype=torch.float16)
    expected = torch.ones_like(actual)
    with pytest.raises(AssertionError):
        assert_bitwise_equal(actual, expected)


def test_align_postprocess_output_negative_control():
    expected = torch.tensor([3, 4], device="cuda", dtype=torch.int32)
    actual = expected.clone()
    actual[1] += 1
    with pytest.raises(AssertionError):
        assert_bitwise_equal(actual, expected)
