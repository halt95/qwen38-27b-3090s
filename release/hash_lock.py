#!/usr/bin/env python3
"""Rebuild release/requirements-pinned.txt as a hash-locked file.

Method (network: PyPI JSON API + PEP 658 metadata files only, nothing is installed):

1. Read the `name==version` pins from the input (the pip freeze of the gated env).
2. For each pin, list its files on PyPI (/pypi/<name>/<version>/json) and keep the wheels
   installable on CPython 3.13, Linux x86_64 (cp313 / abi3 / none-any tags; manylinux up to
   glibc 2.40). If no such wheel exists, keep the sdist. Yanked files are refused.
3. Read each kept wheel's own METADATA (PEP 658 `<wheel-url>.metadata`) and walk
   `Requires-Dist` from the roots (vllm, torch, flashinfer-python) with the environment
   markers evaluated for that target, following requested extras.
4. Write every pin in that closure with one `--hash=sha256:` per kept file. Pins outside the
   closure are dropped and listed on stderr. A closure package with no pin, or a pin that does
   not satisfy a dependent's specifier, is an error (pip's hash-checking mode needs every
   dependency pinned and hashed).

Usage: python3 release/hash_lock.py <frozen-requirements> > release/requirements-pinned.txt
"""
import json
import sys
import urllib.request

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

ROOTS = ["vllm", "torch", "flashinfer-python"]
PLATFORMS = (
    [f"manylinux_2_{m}_x86_64" for m in range(40, 4, -1)]
    + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
)
TAGS = set(cpython_tags((3, 13), abis=["cp313", "abi3", "none"], platforms=PLATFORMS))
TAGS |= set(compatible_tags((3, 13), interpreter="cp313", platforms=PLATFORMS))
ENV = default_environment()
ENV.update(
    python_version="3.13", python_full_version="3.13.0", sys_platform="linux",
    platform_system="Linux", platform_machine="x86_64", os_name="posix",
    implementation_name="cpython", platform_python_implementation="CPython",
    platform_release="", platform_version="", implementation_version="3.13.0",
)


def get(url, accept=None):
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def files_for(name, version):
    d = json.loads(get(f"https://pypi.org/pypi/{name}/{version}/json"))
    wheels, sdists = [], []
    for u in d["urls"]:
        if u.get("yanked"):
            raise SystemExit(f"yanked file for {name}=={version}: {u['filename']}")
        if u["packagetype"] == "bdist_wheel":
            _, _, _, tags = parse_wheel_filename(u["filename"])
            if tags & TAGS:
                wheels.append(u)
        elif u["packagetype"] == "sdist":
            sdists.append(u)
    return wheels, sdists, d["info"].get("requires_dist") or []


def requires_of(wheel, info_requires):
    if wheel is None:
        return info_requires
    meta = get(wheel["url"] + ".metadata").decode("utf-8", "replace")
    out = []
    for line in meta.split("\n"):
        if line == "":
            break  # end of headers
        if line.startswith("Requires-Dist:"):
            out.append(line.split(":", 1)[1].strip())
    return out


def main():
    sys.stdout.reconfigure(newline="\n")  # LF output on every OS
    pins = {}
    for line in open(sys.argv[1], encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, ver = line.split("==")
        pins[canonicalize_name(name)] = (name, ver)

    info, errors = {}, []
    for key, (name, ver) in pins.items():
        wheels, sdists, req = files_for(name, ver)
        kept = wheels or sdists
        if not kept:
            errors.append(f"no installable file for {name}=={ver}")
            continue
        info[key] = {"files": kept, "requires": requires_of(wheels[0] if wheels else None, req)}

    need, why, queue = {}, {}, [(canonicalize_name(r), frozenset()) for r in ROOTS]
    seen = set()
    while queue:
        key, extras = queue.pop()
        if (key, extras) in seen:
            continue
        seen.add((key, extras))
        if key not in pins:
            errors.append(f"dependency {key} is not pinned (required by {why.get(key)})")
            continue
        need.setdefault(key, set()).update(extras)
        for spec in info[key]["requires"]:
            r = Requirement(spec)
            ok = any(r.marker is None or r.marker.evaluate(dict(ENV, extra=e))
                     for e in (list(extras) or [""]))
            if not ok:
                continue
            dep = canonicalize_name(r.name)
            why.setdefault(dep, pins[key][0])
            if dep in pins and not r.specifier.contains(Version(pins[dep][1]), prereleases=True):
                errors.append(f"{pins[dep][0]}=={pins[dep][1]} does not satisfy {spec} ({pins[key][0]})")
            queue.append((dep, frozenset(r.extras)))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)

    print("# Pins: the pip freeze of the gated serving env (2026-09-25), reduced to the dependency")
    print("# closure of vllm, torch and flashinfer-python; hashes from PyPI by release/hash_lock.py.")
    print("# Target: CPython 3.13, Linux x86_64. Install with pip --require-hashes (release/install-env.sh).")
    for key in sorted(need, key=lambda k: pins[k][0].lower()):
        name, ver = pins[key]
        hashes = sorted(f["digests"]["sha256"] for f in info[key]["files"])
        print(f"{name}=={ver} \\")
        print(" \\\n".join(f"    --hash=sha256:{h}" for h in hashes))
    for key in sorted(set(pins) - set(need)):
        print(f"dropped (not required by {', '.join(ROOTS)}): {pins[key][0]}=={pins[key][1]}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
