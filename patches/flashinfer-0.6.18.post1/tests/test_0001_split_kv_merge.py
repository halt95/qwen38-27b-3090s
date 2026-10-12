"""Split-KV merge regression for patch 0001 (the sm_86 FP8 32-row verify tile).

Compares the PATCHED tile (FLASHINFER_SM8X_FP8_SMALLQ=cta32) with stock FlashInfer (the same patched install with the knob
off, "0": the stock CTA64 planner and kernels) on the MERGED attention output of the MTP verify plan with split-KV enabled:
q_len 4 and 5, E4M3 KV, the 27B's per-rank heads (24 q / 4 kv heads x 256: TP2 = 12/2, TP4 = 6/1) and the serving attention
block sizes (TP2 960 with RecoverSSM, TP4 848). The knob is frozen per process, so each arm runs in its own subprocess (this
file with --worker) on identical inputs; the comparison runs here.

Cells
  masked   kv = 2 blocks + r keys, one block per KV chunk (fixed split 1): for r < q_len the last chunk is FULLY masked for
           query rows p < q_len - r (the chunk boundary sits right after their causal limit); r >= q_len leaves it partially
           masked. Random K/V with outlier channels, calibrated-style static scales.            (audit cases a, b)
  peaked   as masked with q x 6: logits spread to about +-90 (sharp rows).                       (case c)
  long     32K / 128K keys, CUDA-graph plan, the planner's own split.                             (case c)
  subsent  the delta audit's analytic case with finite logits below FlashInfer's -5e4 mask sentinel: chunk A (block 0) all keys at
           raw q.k = -59984, V = 0; chunk B (block 1) all keys at -49998, V = 1; sm_scale_log2 = 1 (sm_scale = ln 2,
           k_scale 1). Exact answer 1.0. Stock and the tile are expected to DIFFER here (report, asserted as a difference).
  subsent27b the same logits at the 27B's softmax scale (sm_scale 1/16 x k_scale 0.045 -> sm_scale_log2 0.0041): the
           sentinel-clamped chunk's weight underflows in both arms. Gated on the tile's error vs fp64 (within one bf16 ulp
           of the exact 1.0, or within the budget relative to stock), not on |tile - stock|: stock is the arm that is off here.

Gates (every cell except the sm_scale_log2 = 1 subsent cells): the cta32 arm planned CTA 32 and the stock arm CTA 64; split-KV was on for every fixed-split
cell; the tile meets the error budget that accepted it (microbench v3/v4 rule, both vs an fp64 reference computed from the
stored fp8 bytes): rel_l2(tile) <= 1.10 * rel_l2(stock) + 1e-5 and max_abs(tile) <= 2 * max_abs(stock) + 1e-6. For the
fully masked chunks the per-chunk LSE is read back from the workspace: in both arms it must sit >= 126 (base 2) below the
row's largest chunk LSE, i.e. the chunk's merge weight is below 2^-126 (zero after the merge's flush-to-zero exp2).
The cta32 worker must print the proof line once per q_len with device=0 (planning-time evidence, delta audit finding 2).

Run (one GPU, sm_86, the release venv with the patch applied; pytest optional):
  CUDA_VISIBLE_DEVICES=0 <venv>/bin/python patches/flashinfer-0.6.18.post1/tests/test_0001_split_kv_merge.py
Exit 0 = pass, 1 = fail, 77 = skipped (no CUDA / not sm_86). JIT modules build under FLASHINFER_WORKSPACE_BASE (or ~).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile

KNOB = "FLASHINFER_SM8X_FP8_SMALLQ"
URI_SUFFIX = "_sm8x_smallq_v5"
HEAD_DIM = 256
SHAPES = {"TP2": (12, 2, 960), "TP4": (6, 1, 848)}  # per-rank (H_q, H_kv), serving attention block size
SM_SCALE = 1.0 / 16.0
K_SCALE_27B = 0.045  # calibrated E4M3 k_scale of the 27B: 0.030 .. 0.063 (quant/kv_scales-philbert-e4m3.json)
SENTINEL = -5e4  # FlashInfer's finite mask value (-math::inf) for fp32 accumulation
ARMS = {"stock": "0", "cta32": "cta32"}
PROOF_RE = r"^\[flashinfer-sm8x-tile\] smallq=cta32 q_len=%d rows=%d cta_tile_q=32 device=0$"


# ------------------------------------------------------------------------------------------------ cells
def cell_specs():
    cells = []
    for shape in ("TP2", "TP4"):
        for q_len in (4, 5):
            for r in range(1, q_len + 2):
                cells.append(dict(name=f"masked-{shape}-q{q_len}-r{r}", kind="random", shape=shape, q_len=q_len,
                                  blocks=2, tail=r, split="fixed1", q_mult=1.0, seed=1000 + 37 * r + q_len))
            for r in (1, q_len - 1, q_len + 1):
                cells.append(dict(name=f"peaked-{shape}-q{q_len}-r{r}", kind="random", shape=shape, q_len=q_len,
                                  blocks=2, tail=r, split="fixed1", q_mult=6.0, seed=2000 + 37 * r + q_len))
            cells.append(dict(name=f"long32k-{shape}-q{q_len}", kind="random", shape=shape, q_len=q_len,
                              kv_len=32768 + q_len - 3, split="auto", q_mult=1.0, seed=3000 + q_len))
            cells.append(dict(name=f"subsent-{shape}-q{q_len}", kind="subsent", shape=shape, q_len=q_len, blocks=2, tail=0,
                              split="fixed1", sm_scale=math.log(2.0)))
            cells.append(dict(name=f"subsent27b-{shape}-q{q_len}", kind="subsent", shape=shape, q_len=q_len, blocks=2, tail=0,
                              split="fixed1", sm_scale=SM_SCALE * K_SCALE_27B))
        cells.append(dict(name=f"long128k-{shape}-q5", kind="random", shape=shape, q_len=5, kv_len=131072 + 3,
                          split="auto", q_mult=1.0, seed=3100))
    return cells


def kv_len_of(c):
    page = SHAPES[c["shape"]][2]
    return c["kv_len"] if "kv_len" in c else c["blocks"] * page + c["tail"]


def make_cell(torch, c, dev="cuda"):
    """Inputs (deterministic per spec): paged E4M3 KV [pages, 2, page, hkv, 256], q [q_len, hq, 256] bf16, scales."""
    hq, hkv, page = SHAPES[c["shape"]]
    q_len, n = c["q_len"], kv_len_of(c)
    pages = (n + page - 1) // page
    total = pages + 2
    fp8 = torch.float8_e4m3fn
    if c["kind"] == "subsent":
        u8 = torch.zeros((total, 2, page, hkv, HEAD_DIM), dtype=torch.uint8, device=dev)
        code = lambda v: int(torch.tensor([v], dtype=torch.float32).to(fp8).view(torch.uint8).item())  # noqa: E731
        u8[0, 0, :, :, 0], u8[0, 0, :, :, 1] = code(384.0), code(64.0)          # chunk A: -156*384 - 1.25*64 = -59984
        u8[1, 0, :, :, 0], u8[1, 0, :, :, 1], u8[1, 0, :, :, 2] = code(320.0), code(64.0), code(2.0)  # chunk B: -49998
        u8[1, 1] = code(1.0)                                                      # V: chunk A 0, chunk B 1
        kv = u8.view(fp8)
        q = torch.zeros((q_len, hq, HEAD_DIM), dtype=torch.float32, device=dev)
        q[:, :, 0], q[:, :, 1], q[:, :, 2] = -156.0, -1.25, 1.0
        q = q.to(torch.bfloat16)
        ks = vs = 1.0
        indices = torch.arange(pages, dtype=torch.int32, device=dev)
    else:
        g = torch.Generator(device=dev).manual_seed(c["seed"])
        K = torch.randn((total, page, hkv, HEAD_DIM), generator=g, device=dev)
        V = torch.randn((total, page, hkv, HEAD_DIM), generator=g, device=dev)
        for X in (K, V):  # outlier channels and near-zero entries, as the microbench cells
            dims = torch.randperm(HEAD_DIM, generator=g, device=dev)[:8]
            X[..., dims] *= 20.0
            X[torch.rand(X.shape, generator=g, device=dev) < 0.05] *= 1e-3
        ks = float(K.abs().max().item() / 448.0)
        vs = float(V.abs().max().item() / 448.0)
        kv = torch.empty((total, 2, page, hkv, HEAD_DIM), dtype=fp8, device=dev)
        kv[:, 0] = (K / ks).clamp(-448, 448).to(fp8)
        kv[:, 1] = (V / vs).clamp(-448, 448).to(fp8)
        del K, V
        indices = torch.randperm(total, generator=g, device=dev)[:pages].to(torch.int32)
        q = (torch.randn((q_len, hq, HEAD_DIM), generator=g, device=dev) * c["q_mult"]).to(torch.bfloat16)
    indptr = torch.tensor([0, pages], dtype=torch.int32, device=dev)
    lpl = torch.tensor([n - (pages - 1) * page], dtype=torch.int32, device=dev)
    return dict(kv=kv, q=q, ks=ks, vs=vs, indices=indices, indptr=indptr, last_page_len=lpl, n=n, page=page,
                hq=hq, hkv=hkv, q_len=q_len, pages=pages, sm_scale=c.get("sm_scale", SM_SCALE))


def reference_f64(torch, cell):
    """fp64 attention from the STORED fp8 bytes: row p of the verify block sees keys [0, n - q_len + p]."""
    kv, idx = cell["kv"], cell["indices"].long()
    q, q_len, hq, hkv, n = cell["q"], cell["q_len"], cell["hq"], cell["hkv"], cell["n"]
    group = hq // hkv
    o = torch.zeros((q_len, hq, HEAD_DIM), dtype=torch.float64, device=q.device)
    for h in range(hkv):
        Kd = kv[idx, 0, :, h, :].reshape(-1, HEAD_DIM)[:n].to(torch.float64) * cell["ks"]
        Vd = kv[idx, 1, :, h, :].reshape(-1, HEAD_DIM)[:n].to(torch.float64) * cell["vs"]
        rows = q[:, h * group:(h + 1) * group, :].to(torch.float64).reshape(q_len * group, HEAD_DIM)
        logits = (rows @ Kd.T) * cell["sm_scale"]
        pos = torch.arange(q_len, device=q.device).repeat_interleave(group)
        keys = torch.arange(n, device=q.device).unsqueeze(0)
        logits = logits.masked_fill(keys > (n - q_len + pos).unsqueeze(1), float("-inf"))
        w = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
        o[:, h * group:(h + 1) * group, :] = ((w @ Vd) / w.sum(dim=1, keepdim=True)).reshape(q_len, group, HEAD_DIM)
    return o


# ------------------------------------------------------------------------------------------------ worker (one arm)
def worker(arm, out_path):
    import torch
    import flashinfer
    from flashinfer.jit.attention.modules import get_batch_prefill_uri

    if os.environ.get(KNOB) != ARMS[arm]:
        raise SystemExit(f"worker {arm}: {KNOB} must be {ARMS[arm]!r}, got {os.environ.get(KNOB)!r}")
    bf16, e4m3 = torch.bfloat16, torch.float8_e4m3fn
    uri = get_batch_prefill_uri("fa2", bf16, e4m3, bf16, torch.int32, HEAD_DIM, HEAD_DIM, 0, False, False, False)
    res = dict(arm=arm, uri=uri, flashinfer=flashinfer.__version__, device=torch.cuda.get_device_name(),
               cc=list(torch.cuda.get_device_capability()), cells={})
    ws = torch.zeros(512 << 20, dtype=torch.uint8, device="cuda")
    for c in cell_specs():
        cell = make_cell(torch, c)
        graph = c["split"] == "auto"
        if graph:
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                ws, "NHD", use_cuda_graph=True, use_tensor_cores=True,
                paged_kv_indptr_buffer=torch.zeros(2, dtype=torch.int32, device="cuda"),
                paged_kv_indices_buffer=torch.zeros(cell["kv"].shape[0], dtype=torch.int32, device="cuda"),
                paged_kv_last_page_len_buffer=torch.zeros(1, dtype=torch.int32, device="cuda"))
        else:
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
        w.plan(cell["indptr"], cell["indices"], cell["last_page_len"], cell["hq"], cell["hkv"], HEAD_DIM, cell["page"],
               pos_encoding_mode="NONE", q_data_type=bf16, kv_data_type=e4m3, o_data_type=bf16,
               sm_scale=cell["sm_scale"], q_len_per_req=cell["q_len"],
               fixed_split_size=(1 if c["split"] == "fixed1" else None), disable_split_kv=False)
        info = [int(x) for x in w._plan_info]  # PrefillPlanInfo::ToVector
        out = torch.empty_like(cell["q"])
        lse = torch.empty(cell["q"].shape[:2], dtype=torch.float32, device="cuda")
        w.run(cell["q"], cell["kv"], k_scale=cell["ks"], v_scale=cell["vs"], out=out, lse=lse, return_lse=True)
        torch.cuda.synchronize()
        rec = dict(out=out.float().cpu(), lse=lse.cpu(), cta_tile_q=info[3], split_kv=bool(info[14]),
                   q_sha=hash(tuple(cell["q"].float().flatten()[:4096].tolist())), n=cell["n"])
        if c["split"] == "fixed1" and info[14]:
            # per-chunk partial LSE (base 2) from the workspace: tmp_s[(qo_idx * chunks + chunk) * H + h]
            chunks = cell["pages"]
            s_off = info[11]
            cnt = cell["q_len"] * chunks * cell["hq"]
            rec["chunk_lse"] = ws[s_off:s_off + 4 * cnt].view(torch.float32).reshape(
                cell["q_len"], chunks, cell["hq"]).cpu().clone()
        if arm == "stock":
            rec["ref"] = reference_f64(torch, cell).cpu()
        res["cells"][c["name"]] = rec
        del cell, w
        torch.cuda.empty_cache()
    torch.save(res, out_path)
    return 0


# ------------------------------------------------------------------------------------------------ comparison
def run_all():
    """Returns (status, report lines). status: 'pass' | 'fail' | 'skip'."""
    try:
        import torch
    except ImportError:
        return "skip", ["torch not importable"]
    if not torch.cuda.is_available():
        return "skip", ["no CUDA device"]
    if tuple(torch.cuda.get_device_capability()) != (8, 6):
        return "skip", [f"device is sm_{''.join(map(str, torch.cuda.get_device_capability()))}, the tile is sm_86 only"]
    rep, fails = [], []
    with tempfile.TemporaryDirectory(prefix="fi-splitkv-") as td:
        data, errs = {}, {}
        for arm, knob in ARMS.items():
            env = dict(os.environ, **{KNOB: knob})
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", arm, os.path.join(td, arm + ".pt")],
                               env=env, capture_output=True, text=True)
            errs[arm] = p.stderr
            if p.returncode != 0:
                return "fail", [f"worker {arm} exited {p.returncode}", p.stdout[-3000:], p.stderr[-6000:]]
            data[arm] = torch.load(os.path.join(td, arm + ".pt"), weights_only=True)
    S, T = data["stock"], data["cta32"]
    rep.append(f"flashinfer {T['flashinfer']} on {T['device']} (sm_{T['cc'][0]}{T['cc'][1]}); uri {T['uri']}")
    if not T["uri"].endswith(URI_SUFFIX):
        fails.append(f"not the patched FlashInfer (JIT URI {T['uri']} lacks {URI_SUFFIX}): run release/apply-flashinfer-patch.sh")
    # proof line (planning-time) from the cta32 worker only, device 0, once per q_len
    for q_len in (4, 5):
        rx = re.compile(PROOF_RE % (q_len, 6 * q_len))
        nt = sum(1 for ln in errs["cta32"].splitlines() if rx.match(ln))
        ns = sum(1 for ln in errs["stock"].splitlines() if "[flashinfer-sm8x-tile]" in ln)
        rep.append(f"proof line q_len={q_len}: cta32 worker x{nt} (want 1, device=0), stock worker any-tile-line x{ns} (want 0)")
        if nt != 1 or ns != 0:
            fails.append(f"proof line q_len={q_len}: cta32 x{nt}, stock x{ns}")

    worst = dict(diff=0.0, cell="")
    rep.append("cell                       n       split cta S/T  max|T-S|   maxabs S / T          rel_l2 S / T          budget")
    for c in cell_specs():
        nm = c["name"]
        s, t = S["cells"][nm], T["cells"][nm]
        if s["q_sha"] != t["q_sha"] or s["n"] != t["n"]:
            fails.append(f"{nm}: arms saw different inputs")
            continue
        ref = s["ref"]
        diff = float((t["out"] - s["out"]).abs().max())
        def err(o):
            d = o.double() - ref
            return float(d.abs().max()), float(d.norm() / ref.norm().clamp_min(1e-300))
        ms, rs = err(s["out"])
        mt, rt = err(t["out"])
        tiles_ok = s["cta_tile_q"] == 64 and t["cta_tile_q"] == 32
        if not tiles_ok:
            fails.append(f"{nm}: planned CTA stock {s['cta_tile_q']} / cta32 {t['cta_tile_q']} (want 64 / 32)")
        if c["split"] == "fixed1" and not (s["split_kv"] and t["split_kv"]):
            fails.append(f"{nm}: split-KV not on (stock {s['split_kv']}, cta32 {t['split_kv']})")
        if c["kind"] == "subsent" and c["sm_scale"] > 0.5:
            verdict = "differs (expected)" if diff > 0.05 else "NO DIFFERENCE"
            if diff <= 0.05:
                fails.append(f"{nm}: the sm_scale_log2=1 case shows no stock/tile difference (max {diff:.4g})")
            rep.append(f"{nm:26s} {s['n']:7d} {c['split']:6s} {s['cta_tile_q']}/{t['cta_tile_q']}  {diff:9.3g}  "
                       f"stock out {float(s['out'][..., 0].min()):.5f}..{float(s['out'][..., 0].max()):.5f}, "
                       f"tile {float(t['out'][..., 0].min()):.5f}..{float(t['out'][..., 0].max()):.5f}, exact 1.0: {verdict}")
            continue
        budget = (rt <= 1.10 * rs + 1e-5) and (mt <= 2.0 * ms + 1e-6)
        if c["kind"] == "subsent":
            # 27B scale: gate the tile on its error vs fp64 (exact answer 1.0), never on |T-S|: stock CTA64 can be the
            # one that is off here (its d = 1 / m = -5e4 init state adds mass when every logit is far below zero).
            budget = mt <= 0.0079 or budget
        if not budget:
            fails.append(f"{nm}: outside the error budget (max_abs S {ms:.3g} T {mt:.3g}; rel_l2 S {rs:.3g} T {rt:.3g}; |T-S| {diff:.3g})")
        if c["kind"] != "subsent" and diff > worst["diff"]:
            worst = dict(diff=diff, cell=nm)
        line = (f"{nm:26s} {s['n']:7d} {c['split']:6s} {s['cta_tile_q']}/{t['cta_tile_q']}  {diff:9.3g}  "
                f"{ms:9.3g} / {mt:9.3g}  {rs:9.3g} / {rt:9.3g}  {'ok' if budget else 'FAIL'}")
        # fully masked last chunk: its partial LSE must weigh nothing in the merge, in both arms
        r = c.get("tail", 0)
        if c["split"] == "fixed1" and c["kind"] == "random" and 0 < r < c["q_len"] and "chunk_lse" in s and "chunk_lse" in t:
            rows = list(range(c["q_len"] - r))  # rows whose causal limit ends at the last full block
            gaps, vals = [], {}
            for arm, x in (("S", s), ("T", t)):
                cl = x["chunk_lse"]  # [q_len, chunks, H]
                last = cl[rows, -1, :]
                others = cl[rows, :-1, :].max(dim=1).values
                gaps.append(float((others - last).min()))
                vals[arm] = (float(last.min()), float(last.max()))
            line += (f"  masked-chunk LSE S {vals['S'][0]:.1f}..{vals['S'][1]:.1f} T {vals['T'][0]:.1f}..{vals['T'][1]:.1f},"
                     f" min gap S {gaps[0]:.0f} T {gaps[1]:.0f}")
            if min(gaps) < 126:
                fails.append(f"{nm}: a fully masked chunk's LSE is within {min(gaps):.0f} of the row's other chunks")
        rep.append(line)
    rep.append(f"largest |tile - stock| over the budget-gated cells: {worst['diff']:.3g} ({worst['cell']}); "
               "budget: rel_l2(T) <= 1.10 rel_l2(S) + 1e-5 and max_abs(T) <= 2 max_abs(S) + 1e-6 vs fp64")
    rep.extend("FAIL: " + f for f in fails)
    return ("fail" if fails else "pass"), rep


def test_split_kv_merge_tile_vs_stock():
    status, rep = run_all()
    print("\n".join(rep))
    if status == "skip":
        import pytest
        pytest.skip(rep[0])
    assert status == "pass", "\n".join(r for r in rep if r.startswith("FAIL") or "exited" in r)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        sys.exit(worker(sys.argv[2], sys.argv[3]))
    status, rep = run_all()
    print("\n".join(rep))
    print(f"test_0001_split_kv_merge: {status.upper()}")
    sys.exit({"pass": 0, "fail": 1, "skip": 77}[status])
