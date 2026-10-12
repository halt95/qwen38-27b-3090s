# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-branch: the commit-graph key (RecoverSSMCommitGraphs._plan) over the
metadata the reusing builder produces (groups 2..3 come from
update_block_table and share the first group's batch-level buffers).

The key must still sign every unstaged captured input of EVERY group: each
group's own state-index buffer, the (now shared) query-offset buffer, each
group's block table and context; the shared computed-count tensor gets ONE
staging slot used by all groups. Moving any one group's unstaged input -- the
shared query buffer included -- must change the key, so a captured graph can
never be replayed over moved inputs. CPU: real builders and metadata, the
eligibility/key logic only (no launches)."""

import random

import torch

import vllm.triton_utils as _tu

if getattr(_tu.tl, "constexpr", None) is None:  # triton-less host (CPU tests)
    _tu.tl.constexpr = lambda x: x

import test_0011_metadata_reuse as H  # noqa: E402
from vllm.v1.worker.gpu.model_states import (  # noqa: E402
    recoverssm_commit_graph as rcg,
)


def _step(seed=0):
    device = torch.device("cpu")
    groups, builders, kv_cfg, names = H._builders("align", True, device, True, 16)
    rng = random.Random(seed)
    while True:
        b = H._batch(rng, "align", 16, device, uniform=True)
        # Graph-eligible: every row a spec row (no padding rows), rows 0..n-1.
        if int((b.qsl_cpu.diff() > 0).sum()) == b.n:
            break
    md = H._build(groups, kv_cfg, b, device, capture=False)
    step = tuple(md[g[0]] for g in names)
    state_idx = torch.zeros(H.MAX_REQS, dtype=torch.int32)
    accepted = torch.ones(H.MAX_REQS, dtype=torch.int32)
    ns = torch.ones(b.n, dtype=rcg.NUM_SAMPLED_DTYPE)
    im = torch.arange(b.n, dtype=rcg.IDX_MAPPING_DTYPE)
    return step, builders, state_idx, accepted, ns, im


def _plan(step, state_idx, accepted, ns, im):
    g = rcg.RecoverSSMCommitGraphs(lambda *a, **k: None)
    g._require_cuda = False
    return g._plan(step, ns, im, state_idx, accepted)


def test_key_signs_every_group_with_shared_batch_buffers():
    step, builders, state_idx, accepted, ns, im = _step()
    plan = _plan(step, state_idx, accepted, ns, im)
    assert not isinstance(plan, str), plan
    group_sigs = plan.key[-1]
    assert len(group_sigs) == 3
    # Own state-index buffers (distinct addresses), one per builder.
    si = [s[2][0] for s in group_sigs]
    assert len(set(si)) == 3
    assert si == [b.spec_state_indices_tensor.data_ptr() for b in builders]
    # The query offsets are the first group's buffer for all groups, and every
    # group's signature carries that address.
    qs = [s[3][0] for s in group_sigs]
    assert qs == [builders[0].spec_query_start_loc.data_ptr()] * 3
    # Per-group block tables.
    assert len({s[4][0] for s in group_sigs}) == 3
    # One shared computed-count tensor -> one staging slot for all groups.
    assert plan.nc_slots == (0, 0, 0) and len(plan.nc_sources) == 1
    # Contexts are per group.
    assert len({id(c) for c in plan.contexts}) == 3


def _replace_commit(m, **kw):
    from dataclasses import replace

    return replace(m, recoverssm_commit=replace(m.recoverssm_commit, **kw))


def test_moving_any_groups_unstaged_input_changes_the_key():
    step, builders, state_idx, accepted, ns, im = _step(1)
    base = _plan(step, state_idx, accepted, ns, im).key
    for g in range(3):
        c = step[g].recoverssm_commit
        for field in ("state_indices", "query_start_loc", "block_table"):
            moved = list(step)
            moved[g] = _replace_commit(step[g], **{field: getattr(c, field).clone()})
            key = _plan(tuple(moved), state_idx, accepted, ns, im).key
            assert key != base, (g, field)
    # A non-shared computed-count tensor in one group gets its own slot.
    moved = list(step)
    moved[2] = _replace_commit(
        step[2],
        num_computed_tokens=step[2].recoverssm_commit.num_computed_tokens.clone(),
    )
    plan = _plan(tuple(moved), state_idx, accepted, ns, im)
    assert plan.nc_slots == (0, 0, 1) and plan.key != base
    # The shared running-state / accepted-count buffers are signed too.
    assert _plan(step, state_idx.clone(), accepted, ns, im).key != base
    assert _plan(step, state_idx, accepted.clone(), ns, im).key != base


def test_reused_groups_keep_eligibility():
    step, *_ = _step(2)
    assert all(m.recoverssm_commit.request_indices is None for m in step)
    assert len({m.num_spec_decodes for m in step}) == 1
