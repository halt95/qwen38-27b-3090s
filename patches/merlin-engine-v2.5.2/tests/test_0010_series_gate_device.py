# SPDX-License-Identifier: Apache-2.0
"""0010 (v1.3.0 code audit, finding B1): the family gate fails closed on an unresolved device.

CPU only, no GPU: torch.cuda and the platform capability lookup are mocked with the
mixed host of the audit, capabilities {0: sm_86, 1: sm_89}, and the reverse.

Run from the patched source root (0001-0010 applied), in a venv that imports vllm:
    python -m pytest <repo>/patches/merlin-engine-v2.5.2/tests -q
Before 0010 the unresolved cases return True (device 0 substituted) and the builder is
not asked about its own device: test_uninitialised_cuda_fails_closed,
test_current_device_raising_fails_closed, test_builder_targets_its_own_device fail.
"""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import vllm
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1 import qwen35_series

DENSE = SimpleNamespace(model_config=SimpleNamespace(architecture="Qwen3_5ForCausalLM"))
FLASH_NEXT = SimpleNamespace(model_config=SimpleNamespace(architecture="Qwen4ExpForCausalLM"))
MIXED = {0: (8, 6), 1: (8, 9)}
REVERSED = {0: (8, 9), 1: (8, 6)}


def _caps(table, asked):
    def get_device_capability(device_id=0):
        asked.append(device_id)
        major, minor = table[device_id]
        return DeviceCapability(major=major, minor=minor)

    return get_device_capability


def _host(table, initialised, current=0, current_raises=False):
    """Patch torch.cuda + the platform capability lookup; returns (patchers, asked)."""
    asked = []
    cur = mock.Mock(side_effect=RuntimeError("current_device failed")) if current_raises \
        else mock.Mock(return_value=current)
    return [
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "is_initialized", return_value=initialised),
        mock.patch.object(torch.cuda, "current_device", cur),
        mock.patch.object(current_platform, "get_device_capability", _caps(table, asked)),
    ], asked


def _active(table, *, initialised, current=0, current_raises=False, device=None, cfg=DENSE):
    patchers, asked = _host(table, initialised, current, current_raises)
    for p in patchers:
        p.start()
    try:
        if device is None:  # the pre-0010 signature: also runs (and fails) on 0001-0009
            return qwen35_series.series_active(cfg), asked
        return qwen35_series.series_active(cfg, device=device), asked
    finally:
        for p in reversed(patchers):
            p.stop()


@pytest.mark.parametrize("table", [MIXED, REVERSED], ids=["0:86,1:89", "0:89,1:86"])
def test_uninitialised_cuda_fails_closed(table):
    active, asked = _active(table, initialised=False)
    assert active is False
    assert asked == []  # no capability is read for a substituted device 0


@pytest.mark.parametrize("table", [MIXED, REVERSED], ids=["0:86,1:89", "0:89,1:86"])
def test_current_device_raising_fails_closed(table):
    active, asked = _active(table, initialised=True, current_raises=True)
    assert active is False
    assert asked == []


@pytest.mark.parametrize(
    "table,current,expect",
    [(MIXED, 0, True), (MIXED, 1, False), (REVERSED, 0, False), (REVERSED, 1, True)],
)
def test_initialised_current_device_decides(table, current, expect):
    active, asked = _active(table, initialised=True, current=current)
    assert active is expect
    assert asked == [current]


@pytest.mark.parametrize(
    "table,device,expect",
    [
        (MIXED, torch.device("cuda", 1), False),
        (MIXED, torch.device("cuda", 0), True),
        (MIXED, "cuda:1", False),
        (MIXED, 1, False),
        (REVERSED, torch.device("cuda", 1), True),
        (REVERSED, torch.device("cuda", 0), False),
    ],
)
def test_explicit_device_wins_over_current_device(table, device, expect):
    # current device 0 (initialised): the explicit execution device decides, not device 0
    active, asked = _active(table, initialised=True, current=0, device=device)
    assert active is expect
    assert asked == [torch.device(device).index if not isinstance(device, int) else device]


@pytest.mark.parametrize("initialised", [False, True])
def test_explicit_device_needs_no_cuda_init(initialised):
    active, _ = _active(MIXED, initialised=initialised, current_raises=True,
                        device=torch.device("cuda", 0))
    assert active is True


@pytest.mark.parametrize("device", [torch.device("cpu"), "cpu", -1, True])
def test_non_cuda_or_invalid_device_fails_closed(device):
    active, asked = _active(MIXED, initialised=True, current=0, device=device)
    assert active is False
    assert asked == []


def test_indexless_cuda_device_uses_resolved_current_device():
    assert _active(MIXED, initialised=True, current=1, device="cuda")[0] is False
    assert _active(MIXED, initialised=False, device="cuda")[0] is False
    assert _active(REVERSED, initialised=True, current=1, device="cuda")[0] is True


def test_capability_lookup_raising_fails_closed():
    def boom(device_id=0):
        raise RuntimeError("no such device")

    with mock.patch.object(current_platform, "get_device_capability", boom):
        assert qwen35_series.series_active(DENSE, device=torch.device("cuda", 3)) is False


@pytest.mark.parametrize("table", [MIXED, REVERSED])
def test_other_families_never_active(table):
    for dev in (None, torch.device("cuda", 0), torch.device("cuda", 1)):
        assert _active(table, initialised=True, current=0, device=dev, cfg=FLASH_NEXT)[0] is False


class _Stop(Exception):
    pass


def test_builder_targets_its_own_device():
    """The FlashInfer metadata builder asks the gate about ITS device (cuda:1), not the
    process's current device (0)."""
    from vllm.v1.attention.backend import AttentionMetadataBuilder
    from vllm.v1.attention.backends import flashinfer as fi

    seen = []

    def spy(vllm_config, device=None):
        seen.append(device)
        raise _Stop  # stop __init__ right after the gate (the rest needs a real config)

    b = fi.FlashInferMetadataBuilder.__new__(fi.FlashInferMetadataBuilder)
    with mock.patch.object(fi, "series_active", spy), \
            mock.patch.object(AttentionMetadataBuilder, "__init__", lambda self, *a, **k: None):
        with pytest.raises(_Stop):
            fi.FlashInferMetadataBuilder.__init__(
                b, mock.MagicMock(), [], DENSE, torch.device("cuda", 1))
    assert seen == [torch.device("cuda", 1)]


def test_speculator_passes_its_device():
    """Source check (the V2 speculator needs a CUDA device to construct)."""
    src = (Path(vllm.__file__).parent / "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
           ).read_text()
    assert re.search(r"series_active\(\s*vllm_config,\s*device=device\s*\)", src)
    assert "series_active(vllm_config)" not in src
