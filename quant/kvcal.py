"""vLLM general plugin: record per-attention-layer |K| / |V| absmax during a BF16-KV calibration run.
Loaded in every vLLM process via the `vllm.general_plugins` entry point (PYTHONPATH=/path/to/kvcal, VLLM_PLUGINS=kvcal).
Writes KVCAL_OUT/absmax.<rank>.<pid>.json (running max, flushed every KVCAL_FLUSH forwards and at exit)."""
import atexit, json, os, tempfile
import torch

OUT = os.environ.get("KVCAL_OUT", "./kvcal-absmax")
FLUSH = int(os.environ.get("KVCAL_FLUSH", "500"))
_state: dict[str, list[torch.Tensor]] = {}
_calls = [0]


def _rank() -> str:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return f"r{dist.get_rank()}"
    except Exception:
        pass
    return "rX"


def flush() -> None:
    if not _state:
        return
    os.makedirs(OUT, exist_ok=True)
    data = {n: {"k_absmax": float(s[0].item()), "v_absmax": float(s[1].item()), "calls": int(s[2])} for n, s in _state.items()}
    path = os.path.join(OUT, f"absmax.{_rank()}.{os.getpid()}.json")
    fd, tmp = tempfile.mkstemp(dir=OUT, prefix=".tmp-")
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def register() -> None:
    from vllm.model_executor.layers.attention import attention as A
    if getattr(A.Attention, "_kvcal_patched", False):
        return
    orig = A.Attention.forward

    def forward(self, query, key, value, *args, **kwargs):
        if key is not None and value is not None and key.numel() > 0:
            km = key.detach().abs().amax().float()
            vm = value.detach().abs().amax().float()
            st = _state.get(self.layer_name)
            if st is None:
                _state[self.layer_name] = [km.clone(), vm.clone(), 1]
            else:
                torch.maximum(st[0], km, out=st[0]); torch.maximum(st[1], vm, out=st[1]); st[2] += 1
            _calls[0] += 1
            if _calls[0] % FLUSH == 0:
                flush()
        return orig(self, query, key, value, *args, **kwargs)

    A.Attention.forward = forward
    A.Attention._kvcal_patched = True
    atexit.register(flush)
