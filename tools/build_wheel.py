#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Build the platform wheel of the Python SDK with the latai core inside (task 16.5.1).

    python3 sdk/tools/build_wheel.py --core dist/core/latai-core-<v>-<target>.tar.gz --out dist/wheels

The wheel holds the package `latai` (sdk/python/latai) and, in `latai/_core/`, the core archive's
files: liblatai, the `latai` stage child, `lib/latai/` (the llama libraries), `LICENSE-BINARY`,
`THIRD_PARTY_LICENSES`, `NOTICE`, the SBOM and `core.json`. The loader looks in `_core/` first, so
`pip install` gives a working runtime with no other file. Every file of the archive is checked
against its `core.json` before it goes in.

The platform tag comes from the binaries: `manylinux_2_<N>_<arch>` with N the highest `GLIBC_2.N`
any of them needs (objdump), `macosx_<min>_<arch>` from `--macos-min` (default 11.0 for arm64,
10.15 for x86_64), `win_amd64` / `win_arm64`. The wheel is deterministic: sorted entries, the
1980-01-01 time zip allows as its earliest, modes 0644 or 0755.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

# In latai the SDK is sdk/python and this script sdk/tools; in the SDK's own public repository the
# SDK is the repository and this script tools/.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1] if (_HERE.parents[1] / "sdk" / "python").is_dir() else _HERE.parent
SDK_DIR = ROOT / "sdk" / "python" if (ROOT / "sdk" / "python").is_dir() else ROOT
PACKAGE = SDK_DIR / "latai"
PYPROJECT = SDK_DIR / "pyproject.toml"
README = SDK_DIR / "README.md"
LICENCES = [ROOT / "LICENSE-APACHE", ROOT / "LICENSE-MIT"]
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
# Files of the archive that do not go into the wheel (the header is for C hosts).
SKIP_PREFIXES = ("include/",)


def read_core(archive: Path) -> tuple[dict, dict[str, tuple[bytes, bool]]]:
    """core.json and every file of the archive (path inside the top folder -> bytes, executable),
    each checked against core.json."""
    files: dict[str, tuple[bytes, bool]] = {}
    with tarfile.open(archive, "r:gz") as tar:
        tops = set()
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if not parts or m.name.startswith("/") or ".." in parts:
                raise SystemExit(f"{archive}: entry {m.name!r} is outside the archive's folder")
            tops.add(parts[0])
            if m.isdir():
                continue
            if not m.isfile():
                raise SystemExit(f"{archive}: entry {m.name!r} is not a file or a folder")
            rel = "/".join(parts[1:])
            files[rel] = (tar.extractfile(m).read(), bool(m.mode & 0o111))
        if len(tops) != 1:
            raise SystemExit(f"{archive}: more than one top folder: {sorted(tops)}")
    manifest = json.loads(files["core.json"][0])
    listed = {f["path"]: f for f in manifest["files"]}
    for rel, (data, _) in files.items():
        if rel == "core.json":
            continue
        f = listed.get(rel)
        if f is None or f["bytes"] != len(data) or f["sha256"] != hashlib.sha256(data).hexdigest():
            raise SystemExit(f"{archive}: {rel} is not the file core.json lists")
    missing = set(listed) - set(files)
    if missing:
        raise SystemExit(f"{archive}: core.json lists files the archive lacks: {sorted(missing)}")
    return manifest, files


def glibc_needed(files: dict[str, tuple[bytes, bool]]) -> int:
    """The highest GLIBC_2.N symbol version any ELF file of the core needs."""
    best = 17  # manylinux2014's floor
    with tempfile.TemporaryDirectory() as tmp:
        for rel, (data, _) in files.items():
            if not data.startswith(b"\x7fELF"):
                continue
            path = Path(tmp) / Path(rel).name
            path.write_bytes(data)
            out = subprocess.run(["objdump", "-T", str(path)], capture_output=True, text=True, check=True).stdout
            for n in re.findall(r"GLIBC_2\.(\d+)", out):
                best = max(best, int(n))
    return best


def platform_tag(target: str, files: dict, macos_min: str | None) -> str:
    arch = target.split("-")[0]
    if "linux" in target and "android" not in target:
        if "musl" in target:
            return f"musllinux_1_2_{arch}"
        return f"manylinux_2_{glibc_needed(files)}_{arch}"
    if "apple-darwin" in target:
        mac_arch = "arm64" if arch == "aarch64" else "x86_64"
        minimum = macos_min or ("11.0" if mac_arch == "arm64" else "10.15")
        return f"macosx_{minimum.replace('.', '_')}_{mac_arch}"
    if "windows" in target:
        return "win_arm64" if arch == "aarch64" else "win_amd64"
    raise SystemExit(f"no wheel platform tag for {target}")


def version() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.M)
    if not m:
        raise SystemExit(f"{PYPROJECT}: no version")
    return m.group(1)


def record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def build(core: Path, out: Path, macos_min: str | None) -> Path:
    manifest, core_files = read_core(core)
    ver = version()
    if manifest["version"] != ver.split("+")[0].split(".post")[0]:
        print(f"note: SDK {ver} carries core {manifest['version']}", file=sys.stderr)
    tag = platform_tag(manifest["target"], core_files, macos_min)
    dist_info = f"latai-{ver}.dist-info"
    entries: dict[str, tuple[bytes, bool]] = {}
    for p in sorted(PACKAGE.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            entries[f"latai/{p.relative_to(PACKAGE).as_posix()}"] = (p.read_bytes(), False)
    for rel, (data, exe) in core_files.items():
        if not rel.startswith(SKIP_PREFIXES):
            entries[f"latai/_core/{rel}"] = (data, exe)
    for lic in LICENCES:
        entries[f"{dist_info}/licenses/{lic.name}"] = (lic.read_bytes(), False)
    readme = README.read_text() if README.exists() else ""
    metadata = "\n".join([
        "Metadata-Version: 2.4",
        "Name: latai",
        f"Version: {ver}",
        "Summary: Run small open GGUF models on the device: text, chat, JSON, tool calls, speech and images (latai runtime included).",
        "License-Expression: Apache-2.0 OR MIT",
        "License-File: LICENSE-APACHE",
        "License-File: LICENSE-MIT",
        "Project-URL: Repository, https://github.com/hitl-systems/latai-py",
        "Requires-Python: >=3.9",
        "Classifier: Programming Language :: Python :: 3",
        "Classifier: Typing :: Typed",
        "Description-Content-Type: text/markdown",
        "",
        "The Python source of this package is Apache-2.0 OR MIT. The prebuilt latai core in",
        "`latai/_core/` is licensed separately: see `latai/_core/LICENSE-BINARY`,",
        "`THIRD_PARTY_LICENSES` and `NOTICE` there.",
        "",
        readme,
    ]).rstrip() + "\n"
    entries[f"{dist_info}/METADATA"] = (metadata.encode(), False)
    wheel = f"Wheel-Version: 1.0\nGenerator: latai build_wheel.py\nRoot-Is-Purelib: false\nTag: py3-none-{tag}\n"
    entries[f"{dist_info}/WHEEL"] = (wheel.encode(), False)
    entries[f"{dist_info}/top_level.txt"] = (b"latai\n", False)
    record = [f"{name},{record_hash(data)},{len(data)}" for name, (data, _) in sorted(entries.items())]
    record.append(f"{dist_info}/RECORD,,")
    entries[f"{dist_info}/RECORD"] = (("\n".join(record) + "\n").encode(), False)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"latai-{ver}-py3-none-{tag}.whl"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        # RECORD last, as the wheel spec recommends; everything else sorted.
        names = sorted(n for n in entries if not n.endswith("/RECORD")) + [f"{dist_info}/RECORD"]
        for name in names:
            data, exe = entries[name]
            info = zipfile.ZipInfo(name, ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = ((0o100755 if exe else 0o100644) & 0xFFFF) << 16
            z.writestr(info, data)
    path.write_bytes(buf.getvalue())
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--core", type=Path, required=True, help="a core archive of `cargo xtask core-archive`")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "wheels")
    ap.add_argument("--macos-min", help="the macOS deployment target the core was built for")
    args = ap.parse_args()
    path = build(args.core, args.out, args.macos_min)
    data = path.read_bytes()
    print(f"wrote {path} ({len(data)} B, sha256 {hashlib.sha256(data).hexdigest()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
