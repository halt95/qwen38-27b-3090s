#!/usr/bin/env python3
"""Write release/PACKAGE-MANIFEST.md and release/SHA256SUMS from the files staged in Git.

    git add -A && python3 release/make_manifest.py && git add release/PACKAGE-MANIFEST.md release/SHA256SUMS

Both are computed from the staged blobs (`git ls-files -s` + `git cat-file blob`), so they
describe exactly the bytes and modes that get committed, whatever the checkout's line endings.

  * PACKAGE-MANIFEST.md lists every staged file except itself and SHA256SUMS, with its Git mode,
    size in bytes and sha256.
  * SHA256SUMS covers every staged file except itself, including PACKAGE-MANIFEST.md, so
    `sha256sum -c release/SHA256SUMS` from the repository root checks the whole tree.
  * Checkpoint files are not listed here; they are in release/checkpoint.sha256 and are checked
    inside the download directory: `(cd ckpt && sha256sum -c ../release/checkpoint.sha256)`.

A manifest cannot name the commit that contains it; the commit is identified by the repository
tag or commit ID you cloned.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "release/PACKAGE-MANIFEST.md"
SUMS = "release/SHA256SUMS"


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE).stdout


def staged() -> list[tuple[str, str, str]]:
    """(mode, blob id, path) for every staged file, sorted by path."""
    rows = []
    for rec in git("ls-files", "-s", "-z").split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, blob, stage = meta.decode().split()
        if stage != "0":
            raise SystemExit(f"unmerged entry: {path.decode()}")
        rows.append((mode, blob, path.decode("utf-8")))
    return sorted(rows, key=lambda r: r[2])


def main() -> int:
    rows = []
    for mode, blob, path in staged():
        if path in (MANIFEST, SUMS):
            continue
        data = git("cat-file", "blob", blob)
        rows.append((path, mode, len(data), hashlib.sha256(data).hexdigest()))

    manifest = [
        "# Package manifest",
        "",
        "Every file in this repository except this manifest and `release/SHA256SUMS`, with its Git",
        "mode, size in bytes and SHA-256 of the committed bytes. `release/SHA256SUMS` covers this",
        "manifest too. Checkpoint files are listed in `release/checkpoint.sha256`.",
        "",
        "| file | mode | bytes | sha256 |",
        "|---|---|---:|---|",
        *[f"| `{p}` | {m} | {n} | `{h}` |" for p, m, n, h in rows],
        "",
    ]
    manifest_bytes = "\n".join(manifest).encode("utf-8")
    (ROOT / MANIFEST).write_bytes(manifest_bytes)

    sums = sorted([(p, h) for p, _, _, h in rows] + [(MANIFEST, hashlib.sha256(manifest_bytes).hexdigest())])
    (ROOT / SUMS).write_bytes(("".join(f"{h}  {p}\n" for p, h in sums)).encode("utf-8"))
    print(f"manifest: {len(rows)} files; SHA256SUMS: {len(sums)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
