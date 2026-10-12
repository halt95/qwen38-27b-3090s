# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounds-guard reader idle quiesce (series patch 0012).

The reader used to copy the counters every VLLM_E2_GUARD_INTERVAL forever, which
keeps idle GPUs in P2. With the gate armed it fences the engine stream once no
engine call is open, drains every counter after the fence completes, and parks
with no device work until the next engine call.

CPU tests (no GPU): counters on the CPU device; the fence is a controllable fake
event, so "GPU work still in flight after the last call returned" is simulated
exactly. GPU tests (skipped without CUDA): the real fence on the real stream,
with a delayed kernel that writes the counter after the call has returned.

The planted-hit tests carry FAILING CONTROLS: with the final drain (CPU) or the
fence (GPU) removed, the same scenario parks with the hit unreported, so the
tests discriminate the protocol rather than pass by construction.

Run from the patched source root:  python -m pytest <this file> -q
"""

import importlib
import sys
import threading
import time

import pytest
import torch

MOD = "vllm.v1.worker.guard_telemetry"
CPU = torch.device("cpu")


class _Log:
    """Stand-in for the module logger: records (level, message, parked, parks)."""

    def __init__(self, gt):
        self.gt = gt
        self.records = []

    def _rec(self, level, msg, *args, **_kw):
        text = msg % args if args else msg
        self.records.append(
            (level, text, self.gt._reader_parked, self.gt._stats["parks"])
        )

    def info(self, msg, *a, **k):
        self._rec("info", msg, *a, **k)

    def warning(self, msg, *a, **k):
        self._rec("warning", msg, *a, **k)

    def error(self, msg, *a, **k):
        self._rec("error", msg, *a, **k)

    def exception(self, msg, *a, **k):
        self._rec("exception", msg, *a, **k)

    def debug(self, msg, *a, **k):
        pass

    def warnings(self):
        return [r for r in self.records if r[0] == "warning"]


class _FakeEvent:
    """A fence event whose completion the test controls (GPU work in flight)."""

    def __init__(self):
        self.done = threading.Event()

    def query(self):
        return self.done.is_set()


def _until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.002)
    return pred()


def _fresh(monkeypatch, **env):
    for k in ("VLLM_E2_GUARD_MODE", "VLLM_E2_GUARD_INTERVAL", "VLLM_E2_GUARD_IDLE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop(MOD, None)
    return importlib.import_module(MOD)


@pytest.fixture
def make(monkeypatch):
    """Fresh module, CPU counters, recorded log; the reader is always stopped."""
    made = []

    def _make(gate=True, **env):
        env.setdefault("VLLM_E2_GUARD_MODE", "warn")
        env.setdefault("VLLM_E2_GUARD_INTERVAL", "0.01")
        gt = _fresh(monkeypatch, **env)
        log = _Log(gt)
        monkeypatch.setattr(gt, "logger", log)
        gt.prealloc(CPU)
        if gate:
            gt.enable_idle_gate(CPU, [])
        made.append(gt)

        @gt.engine_call
        def step(body=None):
            gt.check_pending()  # what the V2 runner does first
            if body is not None:
                body()

        return gt, log, step

    yield _make
    for gt in made:
        assert gt.stop_reader(timeout=5.0), "reader did not join"


def _fence_with(gt, monkeypatch):
    """Replace the CUDA fence with controllable fake events; returns the list
    of fences the reader records."""
    fences = []

    def record():
        ev = _FakeEvent()
        fences.append(ev)
        gt._stats["fences"] += 1
        return [ev]

    monkeypatch.setattr(gt, "_record_fence", record)
    return fences


def _engine_thread_call(gt, fn):
    """Run fn as an engine call on a helper thread that plays the engine
    thread (engine calls from any other thread disable the gate)."""

    def run():
        gt._engine_thread = threading.get_ident()
        fn()

    return threading.Thread(target=run)


def _device_ops(gt):
    s = gt._stats
    return s["polls"] + s["fences"] + s["drains"]


# --- environment ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected", [(None, True), ("", True), ("1", True), (" 1 ", True), ("0", False)]
)
def test_idle_switch_parsing(monkeypatch, text, expected):
    env = {} if text is None else {"VLLM_E2_GUARD_IDLE": text}
    gt = _fresh(monkeypatch, **env)
    assert gt._IDLE_GATE is expected


@pytest.mark.parametrize("bad", ["2", "yes", "true", "on", "-1", "inf"])
def test_idle_switch_rejects_garbage_at_import(monkeypatch, bad):
    with pytest.raises(ValueError, match="VLLM_E2_GUARD_IDLE"):
        _fresh(monkeypatch, VLLM_E2_GUARD_IDLE=bad)


def test_idle_switch_is_registered_and_not_a_compile_factor():
    import vllm.envs as envs

    assert "VLLM_E2_GUARD_IDLE" in envs.environment_variables
    src = open(envs.__file__, encoding="utf-8").read()
    ignored = src[src.index('"VLLM_E2_GUARD_INTERVAL",\n', src.index("ignored")) :]
    assert '"VLLM_E2_GUARD_IDLE",' in ignored[:200]


# --- idle: zero device work ---------------------------------------------------


def test_idle_parks_with_zero_device_ops(make):
    gt, log, step = make()
    gt.start_reader()
    for _ in range(5):
        step()
    assert _until(lambda: gt._reader_parked), "reader never parked"
    ops = _device_ops(gt)
    time.sleep(0.5)  # 50 intervals
    assert gt._reader_parked
    assert _device_ops(gt) == ops, "the parked reader issued device work"
    assert gt._stats["parks"] == 1 and gt._stats["drains"] >= 1
    assert any("parked with no device work" in r[1] for r in log.records)


def test_no_park_while_a_call_is_open(make, monkeypatch):
    gt, _log, _step = make()
    fences = _fence_with(gt, monkeypatch)
    gt.start_reader()
    entered, release = threading.Event(), threading.Event()

    @gt.engine_call
    def long_prefill():
        entered.set()
        release.wait(5)

    t = _engine_thread_call(gt, long_prefill)
    t.start()
    entered.wait(5)
    time.sleep(0.3)  # 30 intervals inside one call
    assert not fences and not gt._reader_parked
    polls = gt._stats["polls"]
    time.sleep(0.05)
    assert gt._stats["polls"] > polls, "an open call must keep the cadence"
    release.set()
    t.join()
    assert _until(lambda: fences)
    fences[-1].done.set()
    assert _until(lambda: gt._reader_parked)


def test_a_call_during_the_fence_aborts_the_quiesce(make, monkeypatch):
    gt, _log, step = make()
    fences = _fence_with(gt, monkeypatch)
    gt.start_reader()
    step()
    assert _until(lambda: fences)
    entered, release = threading.Event(), threading.Event()

    @gt.engine_call
    def call():
        entered.set()
        release.wait(5)

    t = _engine_thread_call(gt, call)
    t.start()
    entered.wait(5)
    fences[0].done.set()  # the stale fence completes while a call is open
    time.sleep(0.1)
    assert not gt._reader_parked and gt._stats["drains"] == 0
    release.set()
    t.join()
    assert _until(lambda: len(fences) >= 2), "no fresh fence after the call"
    time.sleep(0.05)
    assert not gt._reader_parked  # the fresh fence is still pending
    fences[-1].done.set()
    assert _until(lambda: gt._reader_parked)


# --- planted hits ---------------------------------------------------------------


def _plant_late_hit(gt, monkeypatch, site, start=True):
    """The last call returns; its GPU work (fence pending) then writes a hit."""
    fences = _fence_with(gt, monkeypatch)
    drains0 = gt._stats["drains"]
    if start:
        gt.start_reader()
    gt._call_enter()
    gt._call_exit()
    assert _until(lambda: fences), "no fence recorded after the last call"
    # The fence must hold the drain back (review finding: without this wait the
    # planted-hit tests also passed with no fence wait at all).
    time.sleep(5 * gt._INTERVAL)
    assert gt._stats["drains"] == drains0 and not gt._reader_parked
    buf = gt._buffers[CPU]
    buf[site, 0] += 1  # the kernel lands after the call returned
    buf[site, 1] = gt.encode_slot(7)
    buf[site, 2] = gt.encode_slot(4096)
    fences[0].done.set()
    return fences


def test_planted_hit_in_the_last_step_is_read_and_attributed(make, monkeypatch):
    gt, log, _step = make()
    site = gt.SITE_GDN_RECURRENT_STATE  # a GDN site the 27B uses
    _plant_late_hit(gt, monkeypatch, site)
    assert _until(lambda: gt._reader_parked)
    warns = log.warnings()
    assert len(warns) == 1, warns
    _lvl, text, parked_at_report, parks_at_report = warns[0]
    assert gt.SITE_NAMES[site] in text and "cpu" in text
    assert "index-field 4096" in text and "row-field 7" in text
    assert not parked_at_report and parks_at_report == 0, "reported after parking"
    assert gt._seen[(CPU, site)] == 1


def test_control_without_the_final_drain_the_hit_is_missed(make, monkeypatch):
    """FAILING CONTROL: a tail-style quiesce (no drain after the fence) parks
    with the planted hit unreported -- the test above can fail."""
    gt, log, _step = make()
    monkeypatch.setattr(gt, "_final_drain", lambda fence_seq: True)
    _plant_late_hit(gt, monkeypatch, gt.SITE_GDN_RECURRENT_STATE)
    assert _until(lambda: gt._reader_parked)
    time.sleep(0.1)
    assert log.warnings() == []
    assert (CPU, gt.SITE_GDN_RECURRENT_STATE) not in gt._seen


def test_always_fatal_site_still_raises_in_warn_across_idle(make, monkeypatch):
    gt, _log, step = make()
    assert gt._MODE == "warn"
    _plant_late_hit(gt, monkeypatch, gt.SITE_QSA_ROPE_POS)
    assert _until(lambda: gt._reader_parked)
    assert gt._pending is not None, "fatal site not armed before parking"
    with pytest.raises(RuntimeError, match="_ALWAYS_FATAL_SITES"):
        step()  # the first resumed step stops the rank


def test_plain_site_does_not_raise_in_warn(make, monkeypatch):
    gt, _log, step = make()
    _plant_late_hit(gt, monkeypatch, gt.SITE_BT_GATHER)
    assert _until(lambda: gt._reader_parked)
    step()  # warn: logged, not armed


def test_planted_hit_during_resume(make, monkeypatch):
    gt, log, step = make()
    gt.start_reader()
    step()
    assert _until(lambda: gt._reader_parked)
    site = gt.SITE_SLOT_MAPPING

    def body():
        gt._buffers[CPU][site, 0] += 2

    step(body)  # wakes the reader; the hit lands inside the resumed call
    assert _until(lambda: gt._stats["parks"] >= 2)
    assert gt._stats["resumes"] >= 1
    assert gt._seen[(CPU, site)] == 2
    assert len(log.warnings()) == 1


def test_unread_copy_at_the_idle_boundary_is_drained(make, monkeypatch):
    """Review finding: the last cadence poll consumed a stale copy and submitted the
    one carrying the hit. The drain must read fresh, not stop there."""
    gt, log, _step = make()
    buf = gt._buffers[CPU]

    class Pipelined:
        """poll_one returns the PREVIOUS copy and starts a new one."""

        def __init__(self):
            self.prev = None

        def registered(self, key):
            return True

        def register(self, key, tensor):
            pass

        def poll_one(self, key):
            rows, self.prev = self.prev, [list(r) for r in buf.tolist()]
            return rows

        def read_settled(self, key, timeout, cancel=None):
            return buf.tolist()

    monkeypatch.setattr(gt, "_nb_reader", Pipelined())
    fences = _fence_with(gt, monkeypatch)
    gt.start_reader()
    gt._call_enter()
    buf[gt.SITE_KV_ZEROER, 0] += 1
    gt._call_exit()
    assert _until(lambda: fences)
    fences[0].done.set()
    assert _until(lambda: gt._reader_parked)
    assert gt._seen[(CPU, gt.SITE_KV_ZEROER)] == 1
    assert len(log.warnings()) == 1
    assert CPU not in gt._read_disabled


def test_half_confirmed_increase_is_reported_by_the_drain(make):
    """A cadence copy showed an increase and is waiting for its confirming
    copy when the engine goes idle: the drain reports it once, coherently."""
    gt, log, _step = make()
    site = gt.SITE_MAMBA_COPY
    buf = gt._buffers[CPU]
    buf[site, 0] = 1
    seen_by_cadence = [list(r) for r in buf.tolist()]
    gt._confirm_pending[CPU] = seen_by_cadence
    buf[site, 1] = gt.encode_slot(3)  # the maxima landed after that copy
    buf[site, 2] = gt.encode_slot(9)
    assert gt._final_drain(gt._call_seq)
    assert gt._seen[(CPU, site)] == 1 and CPU not in gt._confirm_pending
    (warn,) = log.warnings()
    assert "row-field 3, index-field 9" in warn[1]


def test_a_call_racing_the_drain_copy_takes_the_confirm_path(make, monkeypatch):
    """Review finding: a call enters while the drain copy is in flight, so its kernels
    may run beside the copy (new count, stale maxima). The drain must not report
    that copy as coherent, must not park, and the confirming copy reports the
    hit with the maxima that landed after it."""
    gt, log, _step = make()
    site = gt.SITE_GDN_DECODE_STATE
    buf = gt._buffers[CPU]
    racing = {"armed": False}
    real = gt._nb_reader

    class Racing:
        def registered(self, key):
            return True

        def register(self, key, tensor):
            pass

        def poll_one(self, key):
            return real.poll_one(key)

        def read_settled(self, key, timeout, cancel=None):
            if racing["armed"]:
                racing["armed"] = False
                engine = gt._engine_thread
                gt._engine_thread = threading.get_ident()  # the engine's call
                gt._call_enter()  # the call enters mid-copy ...
                buf[site, 0] += 1  # ... its kernel bumps the count first
                rows = buf.tolist()  # the copy sees count without maxima
                buf[site, 1] = gt.encode_slot(11)
                buf[site, 2] = gt.encode_slot(22)
                gt._call_exit()
                gt._engine_thread = engine
                return rows
            return buf.tolist()

    monkeypatch.setattr(gt, "_nb_reader", Racing())
    fences = _fence_with(gt, monkeypatch)
    gt.start_reader()
    gt._call_enter()
    gt._call_exit()
    assert _until(lambda: fences)
    racing["armed"] = True
    fences[0].done.set()
    assert _until(lambda: not racing["armed"])
    assert _until(lambda: len(fences) >= 2), "the racing drain must not park"
    assert gt._stats["parks"] == 0
    fences[-1].done.set()
    assert _until(lambda: gt._reader_parked)
    (warn,) = log.warnings()
    assert "row-field 11, index-field 22" in warn[1], warn[1]
    assert CPU not in gt._read_disabled


def test_wait_idle_is_false_while_a_call_is_open(make):
    gt, _log, step = make()
    gt.start_reader()
    step()
    assert gt.wait_idle(timeout=5.0)
    gt._call_enter()
    try:
        assert gt.wait_idle(timeout=0.1) is False
    finally:
        gt._call_exit()
    assert gt.wait_idle(timeout=5.0)


def test_off_thread_guarded_launch_falls_back_to_polling(make):
    """Review finding: a launch from another thread while an engine call is open (its
    stream may not be fenced) disables the gate."""
    gt, log, _step = make()
    gt.start_reader()
    gt._call_enter()
    try:
        t = threading.Thread(target=gt.counter, args=(gt.SITE_KV_ZEROER, CPU))
        t.start()
        t.join()
    finally:
        gt._call_exit()
    assert gt._gate_broken is not None
    assert any("idle quiesce DISABLED" in r[1] for r in log.records)


def test_shutdown_drain_is_bounded_when_the_reader_holds_the_counters(make):
    gt, log, _step = make()
    assert gt._io_lock.acquire()
    try:
        t0 = time.monotonic()
        assert gt.drain(timeout=0.2) == {}
        assert time.monotonic() - t0 < 2.0
    finally:
        gt._io_lock.release()
    assert any("FINAL DRAIN skipped" in r[1] for r in log.records)


# --- fallbacks ------------------------------------------------------------------


def test_runner_without_the_step_signal_keeps_polling(make):
    gt, _log, _step = make(gate=False)
    gt.start_reader()
    time.sleep(0.1)
    polls = gt._stats["polls"]
    time.sleep(0.2)
    assert gt._stats["polls"] > polls and not gt._reader_parked


def test_idle_switch_off_keeps_polling(make):
    gt, _log, _step = make(VLLM_E2_GUARD_IDLE="0")
    assert not gt._gate_armed  # enable_idle_gate refused
    gt.start_reader()
    time.sleep(0.1)
    polls = gt._stats["polls"]
    time.sleep(0.2)
    assert gt._stats["polls"] > polls and not gt._reader_parked


def test_unscoped_guarded_launch_falls_back_to_polling(make):
    gt, log, step = make()
    gt.start_reader()
    step()
    assert _until(lambda: gt._reader_parked)
    gt.counter(gt.SITE_KV_ZEROER, CPU)  # an eager launch outside any engine call
    assert _until(lambda: not gt._reader_parked)
    polls = gt._stats["polls"]
    time.sleep(0.2)
    assert gt._stats["polls"] > polls
    assert any("idle quiesce DISABLED" in r[1] for r in log.records)


def test_counter_inside_a_call_does_not_break_the_gate(make):
    gt, _log, step = make()
    step(lambda: gt.counter(gt.SITE_KV_ZEROER, CPU))
    assert gt._gate_broken is None


def test_mode_off_starts_nothing(make):
    gt, _log, _step = make(VLLM_E2_GUARD_MODE="off")
    assert not gt._gate_armed
    gt.start_reader()
    assert gt._reader_thread is None


# --- engine thread, lifecycle -----------------------------------------------------


def test_engine_call_makes_no_cuda_call(make, monkeypatch):
    gt, _log, _step = make()

    def boom(*a, **k):
        raise AssertionError("CUDA call on the engine thread")

    for name in ("synchronize", "current_stream", "Event", "Stream"):
        monkeypatch.setattr(torch.cuda, name, boom)
    for _ in range(100):
        gt._call_enter()
        gt._call_exit()


def test_engine_call_preserves_result_and_exception(make):
    gt, _log, _step = make()

    @gt.engine_call
    def ok(x):
        return x + 1

    @gt.engine_call
    def bad():
        raise KeyError("x")

    assert ok(1) == 2
    with pytest.raises(KeyError):
        bad()
    assert gt._calls_open == 0


def test_wait_idle_and_shutdown_join_from_parked(make):
    gt, _log, step = make()
    gt.start_reader()
    step()
    assert gt.wait_idle(timeout=5.0)
    t = gt._reader_thread
    assert gt.stop_reader(timeout=5.0)
    assert not t.is_alive()


def test_shutdown_joins_from_active(make):
    gt, _log, _step = make(gate=False)
    gt.start_reader()
    time.sleep(0.05)
    t = gt._reader_thread
    assert gt.stop_reader(timeout=5.0)
    assert not t.is_alive()


def test_wait_idle_is_false_without_a_live_gate(make):
    gt, _log, _step = make(gate=False)
    gt.start_reader()
    assert gt.wait_idle(timeout=0.1) is False


# --- GPU ------------------------------------------------------------------------

def _gpu_runs() -> bool:
    """A CUDA device this torch build can run kernels on (skip, not fail, on
    e.g. an sm_50 desk card under a build without sm_50 kernels)."""
    if not torch.cuda.is_available():
        return False
    try:
        (torch.zeros(1, device="cuda") + 1).item()
        return True
    except Exception:  # noqa: BLE001
        return False


gpu = pytest.mark.skipif(not _gpu_runs(), reason="needs a CUDA device torch can run")


def _gpu_make(monkeypatch, fence=True):
    gt = _fresh(
        monkeypatch,
        VLLM_E2_GUARD_MODE="warn",
        VLLM_E2_GUARD_INTERVAL="0.01",
    )
    log = _Log(gt)
    monkeypatch.setattr(gt, "logger", log)
    dev = torch.device("cuda", torch.cuda.current_device())
    gt.prealloc(dev)
    gt.enable_idle_gate(dev, [torch.cuda.current_stream(dev)])
    if not fence:
        monkeypatch.setattr(gt, "_record_fence", lambda: [])
    return gt, log, dev


def _gpu_late_hit(gt, dev, site):
    """Enqueue ~0.5 s of spinning, then the counter write, on the engine stream;
    the call returns at once, so the write lands long after it."""
    buf = gt._buffers[dev]

    @gt.engine_call
    def step():
        torch.cuda._sleep(int(1e9))
        buf[site, 0] += 1

    gt.start_reader()
    step()


@gpu
def test_gpu_fence_covers_work_still_running_after_the_call(monkeypatch):
    gt, log, dev = _gpu_make(monkeypatch)
    try:
        _gpu_late_hit(gt, dev, gt.SITE_QSA_ROPE_POS)
        assert _until(lambda: gt._reader_parked, timeout=20)
        assert gt._seen.get((dev, gt.SITE_QSA_ROPE_POS)) == 1
        assert gt._pending is not None  # always-fatal, armed under warn
        ops = _device_ops(gt)
        time.sleep(0.5)
        assert _device_ops(gt) == ops
    finally:
        assert gt.stop_reader(timeout=10.0)
        torch.cuda.synchronize()


@gpu
def test_gpu_call_on_an_unfenced_stream_disables_the_gate(monkeypatch):
    gt, _log, dev = _gpu_make(monkeypatch)
    try:
        other = torch.cuda.Stream(dev)

        @gt.engine_call
        def step():
            pass

        with torch.cuda.stream(other):
            step()
        assert gt._gate_broken is not None
    finally:
        assert gt.stop_reader(timeout=10.0)


@gpu
def test_gpu_guarded_write_on_a_side_stream_is_never_parked_over(monkeypatch):
    """Review finding: a guarded kernel launched on a side stream inside
    an engine call, still running after the call returned. FAILS on the r1 code
    (fence on the engine stream completes, the drain reads 0, the reader parks
    with the hit unread); r2 falls back to polling and reports it."""
    gt, log, dev = _gpu_make(monkeypatch)
    side = torch.cuda.Stream(dev)
    site = gt.SITE_GDN_RECURRENT_STATE
    try:
        @gt.engine_call
        def step():
            with torch.cuda.stream(side):
                torch.cuda._sleep(int(1e9))
                gt.counter(site, dev)[0] += 1

        gt.start_reader()
        step()
        assert _until(lambda: gt._seen.get((dev, site)) == 1, timeout=20), (
            "hit on the side stream never reported"
        )
        assert gt._stats["parks"] == 0
        assert "stream the fence does not cover" in (gt._gate_broken or "")
    finally:
        assert gt.stop_reader(timeout=10.0)
        torch.cuda.synchronize()


@gpu
def test_gpu_control_without_the_fence_the_hit_is_missed(monkeypatch):
    """FAILING CONTROL: without the fence the drain copy runs beside the
    spinning kernel and the reader parks before the write lands."""
    gt, log, dev = _gpu_make(monkeypatch, fence=False)
    try:
        _gpu_late_hit(gt, dev, gt.SITE_QSA_ROPE_POS)
        assert _until(lambda: gt._reader_parked, timeout=20)
        assert (dev, gt.SITE_QSA_ROPE_POS) not in gt._seen
    finally:
        assert gt.stop_reader(timeout=10.0)
        torch.cuda.synchronize()


# --- r2: single-stream invariant -----------------------------------------------


def _fake_streams(gt, monkeypatch, current):
    """Pretend CPU has a fence stream "S"; ``current["key"]`` is the calling
    thread's current stream."""
    gt._fence_keys[CPU] = frozenset({("S",)})
    gt._engine_device = CPU
    monkeypatch.setattr(gt, "_current_stream_key", lambda device: current["key"])


def test_eager_launch_on_an_unfenced_stream_falls_back_loudly(make, monkeypatch):
    gt, log, step = make()
    cur = {"key": ("S",)}
    _fake_streams(gt, monkeypatch, cur)
    step()
    assert gt._gate_broken is None

    def side_launch():
        cur["key"] = ("T",)  # with torch.cuda.stream(side): <guarded kernel>
        try:
            gt.counter(gt.SITE_GDN_RECURRENT_STATE, CPU)
        finally:
            cur["key"] = ("S",)

    step(side_launch)
    assert "stream the fence does not cover" in (gt._gate_broken or "")
    errors = [r for r in log.records if r[0] == "error"]
    assert any("idle quiesce DISABLED" in r[1] for r in errors)


def test_every_engine_call_checks_its_stream_not_only_the_first(make, monkeypatch):
    gt, _log, step = make()
    cur = {"key": ("S",)}
    _fake_streams(gt, monkeypatch, cur)
    for _ in range(3):
        step()
    assert gt._gate_broken is None
    cur["key"] = ("T",)  # a later call (and the graphs it replays) on another stream
    step()
    assert "stream the fence does not cover" in (gt._gate_broken or "")


def test_failing_stream_lookup_never_leaks_an_open_call(make, monkeypatch):
    gt, _log, step = make()
    gt._fence_keys[CPU] = frozenset({("S",)})
    gt._engine_device = CPU

    def boom(device):
        raise RuntimeError("no stream")

    monkeypatch.setattr(gt, "_current_stream_key", boom)
    step()
    assert gt._calls_open == 0 and gt._gate_broken is not None


# --- r2: fatal counts arm on first observation --------------------------------


def test_fatal_count_arms_on_first_observation(make):
    gt, log, _step = make()
    buf = gt._buffers[CPU]
    gt._call_enter()
    try:
        buf[gt.SITE_QSA_RING_UNCOMMITTED, 0] += 1  # count first, maxima later
        gt._poll_once()  # first copy that sees the count
        assert gt._pending is not None, "fatal count not armed on first observation"
        assert (CPU, gt.SITE_QSA_RING_UNCOMMITTED) not in gt._seen  # report waits
        (first,) = log.warnings()  # the evidence is logged before arming
        assert "first observation" in first[1] and "(count 1)" in first[1]
        gt._poll_once()  # confirming copy: the WARNING with the record
        assert gt._seen[(CPU, gt.SITE_QSA_RING_UNCOMMITTED)] == 1
        assert len(log.warnings()) == 2
    finally:
        gt._call_exit()


def test_plain_count_does_not_arm_in_warn(make):
    gt, _log, _step = make()
    gt._buffers[CPU][gt.SITE_BT_GATHER, 0] += 1
    gt._poll_once()
    gt._poll_once()
    assert gt._pending is None and gt._seen[(CPU, gt.SITE_BT_GATHER)] == 1


# --- r2: a resumed call / stop cancels a stale drain wait ------------------------


class _NeverEvent:
    def query(self):
        return False

    def synchronize(self):
        raise AssertionError("blocking wait")


def _stale_drain(gt, monkeypatch):
    """Fence completes, then the drain waits on a copy that never lands."""
    fences = _fence_with(gt, monkeypatch)
    gt.start_reader()
    gt._call_enter()
    gt._call_exit()
    assert _until(lambda: fences)
    job = gt._nb_reader._jobs[CPU]
    with gt._io_lock:  # a copy in flight that will not land
        job[1], job[2], job[3] = object(), _NeverEvent(), True
    fences[0].done.set()
    assert _until(lambda: gt._stats["drains"] == 1)
    return job


def test_a_resumed_call_cancels_a_stale_drain_wait(make, monkeypatch):
    gt, log, _step = make()
    job = _stale_drain(gt, monkeypatch)
    time.sleep(0.05)
    polls = gt._stats["polls"]
    gt._call_enter()  # serving resumes while the drain copy is still in flight
    try:
        # back on the non-blocking cadence well before the 2 s drain timeout
        assert _until(lambda: gt._stats["polls"] > polls + 2, timeout=0.5)
        assert job[3] is True, "the in-flight copy must be kept for the cadence"
        assert not gt._reader_parked
        assert not any("did not complete" in r[1] for r in log.records)
    finally:
        gt._call_exit()


def test_stop_cancels_a_stale_drain_wait(make, monkeypatch):
    gt, _log, _step = make()
    _stale_drain(gt, monkeypatch)
    t0 = time.monotonic()
    assert gt.stop_reader(timeout=5.0)
    assert time.monotonic() - t0 < 1.0, "stop waited out the drain timeout"


# --- r2: lifecycle ------------------------------------------------------------


def test_restart_after_stop_reads_and_arms_again_without_reimport(make, monkeypatch):
    gt, log, step = make()
    gt.start_reader()
    step()
    assert _until(lambda: gt._reader_parked)
    first = gt._reader_thread
    assert gt.stop_reader(timeout=5.0) and not first.is_alive()
    assert not gt._gate_armed  # the next worker arms its own gate
    # a second worker in the same process
    gt.start_reader()
    assert gt._reader_thread is not None and gt._reader_thread is not first
    assert gt._reader_thread.is_alive()
    gt.enable_idle_gate(CPU, [])
    _plant_late_hit(gt, monkeypatch, gt.SITE_QSA_ROPE_POS, start=False)
    assert _until(lambda: gt._reader_parked)
    assert gt._pending is not None and gt._seen[(CPU, gt.SITE_QSA_ROPE_POS)] == 1


def test_suspend_stops_every_reader_cuda_call_even_with_the_idle_switch_off(make):
    gt, log, _step = make(VLLM_E2_GUARD_IDLE="0")
    gt.start_reader()
    assert _until(lambda: gt._stats["polls"] > 3)
    gt._buffers[CPU][gt.SITE_SLOT_MAPPING, 0] += 1  # last write before sleep
    assert gt.suspend_reader(timeout=2.0)
    assert gt._seen.get((CPU, gt.SITE_SLOT_MAPPING)) == 1, "not read before suspension"
    ops = _device_ops(gt)
    time.sleep(0.3)
    assert _device_ops(gt) == ops, "the suspended reader issued device work"
    gt.resume_reader()
    assert _until(lambda: _device_ops(gt) > ops)


def test_suspend_wakes_a_parked_reader_and_resume_parks_again(make):
    gt, _log, step = make()
    gt.start_reader()
    step()
    assert _until(lambda: gt._reader_parked)
    assert gt.suspend_reader(timeout=2.0) and gt._reader_suspended
    gt.resume_reader()
    step()
    assert _until(lambda: gt._reader_parked)


def test_suspend_without_a_reader_is_quiet_and_true(make):
    gt, log, _step = make(VLLM_E2_GUARD_MODE="off")
    gt.start_reader()
    assert gt.suspend_reader(timeout=0.1) is True
    assert log.records == []


# --- r2: PP patterns ------------------------------------------------------------


def test_pp_a_call_between_the_sample_and_the_fence_record_discards_it(make, monkeypatch):
    gt, _log, _step = make()
    fences = []

    def record():
        if not fences:  # a call slips in between the _cv sample and the record
            engine = gt._engine_thread
            gt._engine_thread = threading.get_ident()  # it is the engine's call
            try:
                gt._call_enter()
                gt._call_exit()
            finally:
                gt._engine_thread = engine
        ev = _FakeEvent()
        ev.done.set()  # even an instantly complete fence must not be trusted
        fences.append(ev)
        gt._stats["fences"] += 1
        return [ev]

    monkeypatch.setattr(gt, "_record_fence", record)
    gt.start_reader()
    gt._call_enter()
    gt._call_exit()
    assert _until(lambda: gt._reader_parked)
    assert len(fences) == 2 and gt._stats["drains"] == 1


def test_pp_stage0_pattern_parks_over_an_unfenced_non_producer(make, monkeypatch):
    """A non-last PP stage returns with only a recv pending on its broadcast
    stream (not a producer, not fenced): the reader parks, and the next call
    resumes it before any counter read."""
    gt, _log, _step = make()
    pending_recv = _FakeEvent()  # never completes while we look
    gt.start_reader()
    gt._call_enter()
    gt._call_exit()
    assert _until(lambda: gt._reader_parked)
    assert not pending_recv.query()
    polls = gt._stats["polls"]
    time.sleep(0.1)
    assert gt._stats["polls"] == polls
    gt._call_enter()
    try:
        assert _until(lambda: gt._stats["polls"] > polls)
        assert gt._stats["resumes"] >= 1
    finally:
        gt._call_exit()


# --- r2: randomized interleavings ----------------------------------------------


class _SimGPU:
    """In-order 'stream': (delay, fn) items run one after another."""

    def __init__(self):
        import queue

        self.q = queue.Queue()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            time.sleep(item[0])
            item[1]()

    def launch(self, delay, fn):
        self.q.put((delay, fn))


def _stress(gt, monkeypatch, seed, instant_fence=False, bursts=25):
    """Returns the list of invariant violations seen at park time."""
    import random

    rnd = random.Random(seed)
    sim = _SimGPU()

    def rec():
        ev = _FakeEvent()
        if instant_fence:
            ev.done.set()  # CONTROL: a fence that does not wait for the queue
        else:
            sim.launch(0.0, ev.done.set)
        gt._stats["fences"] += 1
        return [ev]

    monkeypatch.setattr(gt, "_record_fence", rec)
    buf = gt._buffers[CPU]
    site = gt.SITE_GDN_RECURRENT_STATE

    def hit():
        buf[site, 0] += 1
        buf[site, 1] = gt.encode_slot(1)
        buf[site, 2] = gt.encode_slot(2)

    @gt.engine_call
    def step():  # async-scheduling style: work that runs after the call returns
        for _ in range(rnd.randint(0, 3)):
            sim.launch(rnd.uniform(0, 0.03), hit if rnd.random() < 0.3 else (lambda: None))

    bad = []
    gt.start_reader()
    try:
        for burst in range(bursts):
            for _ in range(rnd.randint(1, 8)):
                step()
                time.sleep(rnd.choice([0, 0.001, 0.004, 0.006, 0.012]))
            assert _until(lambda: gt._reader_parked, timeout=5), "never parked"
            n = int(buf[site, 0])
            if not sim.q.empty() or gt._seen.get((CPU, site), 0) != n:
                bad.append((burst, n, gt._seen.get((CPU, site), 0)))
            if rnd.random() < 0.5:
                time.sleep(rnd.uniform(0, 0.02))
    finally:
        sim.q.put(None)
    return bad


@pytest.mark.parametrize("seed", range(6))
def test_random_interleavings_never_park_with_an_unreported_hit(make, monkeypatch, seed):
    gt, _log, _step = make(VLLM_E2_GUARD_INTERVAL="0.005")
    assert _stress(gt, monkeypatch, seed) == []


def test_stress_control_an_instant_fence_parks_over_queued_hits(make, monkeypatch):
    """FAILING CONTROL for the stress test: a fence that does not wait for the
    queued work parks with hits unread (the stress test can fail)."""
    gt, _log, _step = make(VLLM_E2_GUARD_INTERVAL="0.005")
    assert _stress(gt, monkeypatch, 0, instant_fence=True) != []


# --- r2 (review): versioned suspend, asleep drain, lifecycle generations ----


def _never_landing_copy(gt):
    job = gt._nb_reader._jobs[CPU]
    with gt._io_lock:
        job[1], job[2], job[3] = object(), _NeverEvent(), True
    return job


def test_suspend_timeout_withdraws_the_request_and_the_reader_works_on(make):
    gt, _log, _step = make(VLLM_E2_GUARD_IDLE="0")
    gt.start_reader()
    assert _until(lambda: gt._stats["polls"] > 2)
    job = _never_landing_copy(gt)  # the settled read can never complete
    assert gt.suspend_reader(timeout=0.3) is False
    assert not gt._suspend_req and not gt._reader_suspended and not gt._asleep
    assert job[3] is True  # never acknowledged with a copy outstanding
    polls = gt._stats["polls"]
    assert _until(lambda: gt._stats["polls"] > polls), "reader did not go back to work"


def test_rapid_resume_and_suspend_reads_again(make):
    gt, _log, _step = make(VLLM_E2_GUARD_IDLE="0")
    gt.start_reader()
    assert gt.suspend_reader(timeout=2.0)
    gt.resume_reader()
    gt._buffers[CPU][gt.SITE_KV_ZEROER, 0] += 1  # written between the two sleeps
    assert gt.suspend_reader(timeout=2.0)  # a NEW request: a new settled read
    assert gt._seen.get((CPU, gt.SITE_KV_ZEROER)) == 1
    gt.resume_reader()


def test_drain_is_refused_while_the_worker_is_asleep(make):
    gt, log, _step = make()
    gt.start_reader()
    assert gt.suspend_reader(timeout=2.0)
    assert gt.drain(timeout=1.0) == {}
    assert any("FINAL DRAIN skipped: the worker is asleep" in r[1] for r in log.records)
    gt.resume_reader()
    gt.drain(timeout=1.0)
    assert any("FINAL DRAIN: no fatal-site hits" in r[1] for r in log.records)


def test_a_dead_reader_whose_stop_timed_out_is_reaped_on_start(make):
    gt, log, _step = make()
    lingering = threading.Thread(target=time.sleep, args=(0.3,))
    lingering.start()
    with gt._life_lock:  # what a timed-out stop_reader() leaves behind
        gt._reader_started = True
        gt._reader_thread = lingering
        gt._reader_stop.set()
    gt.start_reader()  # refused, loudly, while it lives
    assert gt._reader_thread is lingering
    assert any("reader NOT started" in r[1] for r in log.records)
    lingering.join()
    gt.start_reader()  # reaped: a fresh reader runs
    assert gt._reader_thread is not lingering and gt._reader_thread.is_alive()
    assert not gt._reader_stop.is_set()


def test_a_violation_seen_under_an_old_gate_does_not_break_a_rearmed_one(make):
    gt, _log, _step = make()
    old = gt._gate_gen
    gt.enable_idle_gate(CPU, [])  # re-armed (a later worker)
    gt._break_gate("seen under the old gate", old)
    assert gt._gate_broken is None
    gt._break_gate("seen under the current gate", gt._gate_gen)
    assert gt._gate_broken is not None


def test_a_reader_restarted_while_asleep_stays_suspended(make):
    """Review finding: sleep -> stop -> start must not poll the sleeping counters."""
    gt, _log, _step = make(VLLM_E2_GUARD_IDLE="0")
    gt.start_reader()
    assert gt.suspend_reader(timeout=2.0)
    assert gt.stop_reader(timeout=5.0)
    gt.start_reader()
    assert _until(lambda: gt._reader_suspended)
    ops = _device_ops(gt)
    time.sleep(0.2)
    assert _device_ops(gt) == ops, "a reader started while asleep read the device"
    gt.resume_reader()
    assert _until(lambda: _device_ops(gt) > ops)


def test_suspend_while_asleep_is_idempotent(make):
    gt, _log, _step = make(VLLM_E2_GUARD_IDLE="0")
    gt.start_reader()
    assert gt.suspend_reader(timeout=2.0)
    ops, gen = _device_ops(gt), gt._suspend_gen
    assert gt.suspend_reader(timeout=2.0)  # no second request, no second read
    assert gt._suspend_gen == gen and _device_ops(gt) == ops
    gt.resume_reader()
