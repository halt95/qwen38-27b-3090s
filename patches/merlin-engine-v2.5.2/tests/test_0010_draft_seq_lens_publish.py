# SPDX-License-Identifier: Apache-2.0
"""0010 (v1.3.0 code audit, finding B2): a failed metadata build must not poison the next
no-sync draft derivation.

CPU only: drives the real FlashInferMetadataBuilder.build() on a decode-only batch with
the FlashInfer planning call mocked; a CPU int32 tensor behind DeviceBuf stands in for the
device seq_lens buffer (same storage = same data_ptr, which is what the handoff tag compares). The draft
step reproduces the speculator's handoff verbatim (speculator.py, 0005 + 0007): derive
next = last + 1 (tag next_buf = last_buf), build, clear next/next_buf in `finally`.

Run from the patched source root (0001-0010 applied), in a venv that imports vllm:
    python -m pytest <repo>/patches/merlin-engine-v2.5.2/tests -q
Before 0010: test_failed_build_retry_takes_device_lengths derives n+2 and
test_failed_build_leaves_no_published_state finds last = n+1.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.v1.attention.backends import flashinfer as fi

N = 1000
MAX_MODEL_LEN = 262144


class Injected(RuntimeError):
    pass


class DeviceBuf:
    """Stand-in for the device seq_lens buffer: .cpu() copies, as a device-to-host copy
    does (a CPU tensor's .cpu() would alias the buffer and hide the defect)."""

    def __init__(self, t):
        self.t = t
        self.shape = t.shape

    def cpu(self):
        return self.t.clone()

    def data_ptr(self):
        return self.t.data_ptr()


class Harness:
    def __init__(self, num_reqs=2):
        self.num_reqs = num_reqs
        self.fail_plan = False
        self.seen_seq_lens = []  # CPU seq_lens each build consumed
        # the "device" seq_lens buffer of the persistent input batch
        self.gpu_seq_lens = torch.full((num_reqs,), N, dtype=torch.int32)
        b = fi.FlashInferMetadataBuilder.__new__(fi.FlashInferMetadataBuilder)
        attrs = dict(
            reorder_batch_threshold=2, use_xqa=False, page_size=16,
            attention_config=SimpleNamespace(use_trtllm_attention=False),
            num_qo_heads=6, num_kv_heads=1, head_dim=256, dcp_world_size=1, use_dcp=False,
            cache_dtype="fp8", kv_cache_dtype=torch.float8_e4m3fn,
            q_data_type_prefill=torch.bfloat16, q_data_type_decode=torch.bfloat16,
            has_sinks=False, use_trtllm_decode_attention=False,
            global_hyperparameters=SimpleNamespace(
                has_same_window_lefts=True, has_same_all_params=True),
            enable_cuda_graph=False, _decode_cudagraph_max_bs=0, is_kvcache_nvfp4=False,
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            paged_kv_indptr=SimpleNamespace(cpu=torch.zeros(num_reqs + 1, dtype=torch.int32)),
            paged_kv_last_page_len=SimpleNamespace(cpu=torch.ones(num_reqs, dtype=torch.int32)),
            sm_scale=0.0625, window_left=-1, logits_soft_cap=None,
            decode_fixed_split_size=-1, disable_split_kv=False, _qwen35_series=True,
        )
        for k, v in attrs.items():
            setattr(b, k, v)
        b._get_decode_wrapper = mock.MagicMock(name="decode_wrapper")
        b._compute_flashinfer_kv_metadata = self._kv_metadata
        self.builder = b

    def _kv_metadata(self, num_blocks_np, seq_lens_np, *args):
        self.seen_seq_lens.append(seq_lens_np.copy())
        return torch.zeros(1, dtype=torch.int32)

    def _plan(self, *args, **kwargs):
        if self.fail_plan:
            raise Injected("planning failed")

    def cam(self):
        qsl = torch.arange(self.num_reqs + 1, dtype=torch.int32)
        return SimpleNamespace(
            num_reqs=self.num_reqs, num_actual_tokens=self.num_reqs, causal=True,
            max_seq_len=int(self.gpu_seq_lens.max()), seq_lens=DeviceBuf(self.gpu_seq_lens),
            block_table_tensor=torch.zeros((self.num_reqs, 8), dtype=torch.int32),
            query_start_loc=qsl, query_start_loc_cpu=qsl,
            slot_mapping=torch.zeros(self.num_reqs, dtype=torch.int64),
            dcp_local_seq_lens=None)

    def build(self):
        n = self.num_reqs
        with mock.patch.object(fi, "split_decodes_and_prefills", return_value=(n, 0, n, 0)), \
                mock.patch.object(fi, "use_trtllm_attention", return_value=False), \
                mock.patch.object(fi, "fast_plan_decode", self._plan), \
                mock.patch.object(fi, "get_seq_lens", lambda *a: torch.ones(n, dtype=torch.int32)), \
                mock.patch.object(fi, "PIN_MEMORY", False), \
                mock.patch.object(fi, "FIDecode", lambda wrapper: SimpleNamespace(wrapper=wrapper)):
            return self.builder.build(common_prefix_len=0, common_attn_metadata=self.cam())

    def draft_step(self, step):
        """speculator.py draft-step handoff (0005 + 0007), then the metadata build."""
        dsl = fi._DRAFT_SEQ_LENS
        if step >= 2 and dsl["last"] is not None:
            dsl["next"] = torch.clamp(dsl["last"] + 1, max=MAX_MODEL_LEN)
            dsl["next_buf"] = dsl.get("last_buf")
        try:
            return self.build()
        finally:
            dsl["next"] = None
            dsl["next_buf"] = None


@pytest.fixture(autouse=True)
def fresh_handoff():
    saved = dict(fi._DRAFT_SEQ_LENS)
    fi._DRAFT_SEQ_LENS.update(enabled=True, verify=False, next=None, last=None, next_buf=None,
                              last_buf=None, bypassed=0, synced=0, verified=0, mismatch=0)
    yield
    fi._DRAFT_SEQ_LENS.clear()
    fi._DRAFT_SEQ_LENS.update(saved)


def _start(h):
    """Draft step 1 (synced) succeeds at n; the device then advances once to n+1."""
    h.draft_step(1)
    assert fi._DRAFT_SEQ_LENS["synced"] == 1
    assert torch.equal(fi._DRAFT_SEQ_LENS["last"], torch.full((h.num_reqs,), N, dtype=torch.int32))
    h.gpu_seq_lens += 1  # in place: same buffer, same data_ptr


def test_successful_draft_step_derives_without_sync():
    h = Harness()
    _start(h)
    h.draft_step(2)
    d = fi._DRAFT_SEQ_LENS
    assert d["bypassed"] == 1 and d["synced"] == 1
    assert (h.seen_seq_lens[-1] == N + 1).all()
    assert torch.equal(d["last"], torch.full((h.num_reqs,), N + 1, dtype=torch.int32))
    assert d["last_buf"] == h.gpu_seq_lens.data_ptr()


def test_failed_build_leaves_no_published_state():
    h = Harness()
    _start(h)
    h.fail_plan = True
    with pytest.raises(Injected):
        h.draft_step(2)
    d = fi._DRAFT_SEQ_LENS
    assert (h.seen_seq_lens[-1] == N + 1).all()  # the failed build had derived n+1
    assert d["next"] is None and d["next_buf"] is None  # pending handoff cleared
    assert d["last"] is None and d["last_buf"] is None  # nothing published by a failed build


def test_failed_build_retry_takes_device_lengths():
    h = Harness()
    _start(h)
    h.fail_plan = True
    with pytest.raises(Injected):
        h.draft_step(2)
    h.fail_plan = False
    h.draft_step(2)  # retry, device lengths unchanged at n+1
    assert (h.seen_seq_lens[-1] == N + 1).all(), h.seen_seq_lens[-1]
    d = fi._DRAFT_SEQ_LENS
    assert d["synced"] == 2  # the retry synchronised instead of deriving
    assert torch.equal(d["last"], torch.full((h.num_reqs,), N + 1, dtype=torch.int32))


def test_failed_synced_build_publishes_nothing():
    h = Harness()
    h.fail_plan = True
    with pytest.raises(Injected):
        h.draft_step(1)
    d = fi._DRAFT_SEQ_LENS
    assert d["last"] is None and d["last_buf"] is None and d["next"] is None


def test_gate_off_never_publishes():
    h = Harness()
    h.builder._qwen35_series = False
    with contextlib.suppress(Injected):
        h.draft_step(1)
    assert fi._DRAFT_SEQ_LENS["last"] is None
