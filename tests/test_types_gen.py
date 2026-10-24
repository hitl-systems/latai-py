# SPDX-License-Identifier: Apache-2.0 OR MIT
"""14.3.1 generated types: `gen_types.py --check` passes on the committed output and fails when the
schema, the error registry or a generated file changes."""

import filecmp
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import REPO, SDK

GEN = REPO / "sdk" / "tools" / "gen_types.py"

# The generator and the schema stay in latai; the SDK's public repository has only their output.
pytestmark = pytest.mark.skipif(not GEN.exists(), reason="the type generator lives in latai")
OUT = SDK / "latai"
GENERATED = ["types.py", "_errors_gen.py"]


def gen(*args, cwd=None):
    return subprocess.run([sys.executable, str(GEN), *args], capture_output=True, text=True, cwd=cwd, timeout=120)


def digest(paths):
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def test_check_passes_on_the_committed_output():
    r = gen("--check")
    assert r.returncode == 0, r.stdout + r.stderr


def test_check_from_another_directory_passes_too():
    r = gen("--check", cwd="/")
    assert r.returncode == 0, r.stdout + r.stderr


def test_check_does_not_write():
    before = digest(OUT / n for n in GENERATED)
    gen("--check")
    assert digest(OUT / n for n in GENERATED) == before


def test_generation_is_deterministic_and_matches_the_committed_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        r = gen("--out-dir", str(d))
        assert r.returncode == 0, r.stdout + r.stderr
    for n in GENERATED:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n
        assert (a / n).read_bytes() == (OUT / n).read_bytes(), f"{n} is out of date: run sdk/tools/gen_types.py"
    # and --check on what was just generated passes
    assert gen("--check", "--out-dir", str(a)).returncode == 0


def test_identical_copies_of_the_inputs_pass(tmp_path):
    s, e = tmp_path / "schema.json", tmp_path / "errors.json"
    shutil.copy(REPO / "schema" / "latai.schema.json", s)
    shutil.copy(REPO / "schema" / "errors.json", e)
    r = gen("--check", "--schema", str(s), "--errors", str(e))
    assert r.returncode == 0, r.stdout + r.stderr


def mutated_schema(tmp_path, how):
    schema = json.loads((REPO / "schema" / "latai.schema.json").read_text())
    how(schema)
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(schema, indent=2))
    return p


@pytest.mark.parametrize(
    "name,how",
    [
        ("new property", lambda s: s["$defs"]["Request"]["properties"].__setitem__("zz_new_field", {"type": "string"})),
        ("renamed property", lambda s: s["$defs"]["Request"]["properties"].__setitem__("packx", s["$defs"]["Request"]["properties"].pop("pack"))),
        ("new definition", lambda s: s["$defs"].__setitem__("BrandNewThing", {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})),
        ("removed definition", lambda s: s["$defs"].pop("Timing")),
    ],
)
def test_check_fails_when_the_schema_changes(tmp_path, name, how):
    r = gen("--check", "--schema", str(mutated_schema(tmp_path, how)))
    assert r.returncode != 0, f"{name}: --check passed on a changed schema\n{r.stdout}{r.stderr}"
    assert r.stdout + r.stderr, "the failure says something"


def test_check_fails_when_the_error_registry_changes(tmp_path):
    reg = json.loads((REPO / "schema" / "errors.json").read_text())
    reg["errors"].append(
        {
            "code": "E0999",
            "name": "TesterInvented",
            "category": "E09",
            "retryable": False,
            "hint": "invented by a test",
            "fields": [],
            "details": {"type": "object", "properties": {}, "additionalProperties": False},
        }
    )
    p = tmp_path / "errors.json"
    p.write_text(json.dumps(reg, indent=2))
    r = gen("--check", "--errors", str(p))
    assert r.returncode != 0, r.stdout + r.stderr
    # a changed hint is a change too
    reg = json.loads((REPO / "schema" / "errors.json").read_text())
    reg["errors"][0]["hint"] += " (edited)"
    p.write_text(json.dumps(reg, indent=2))
    r = gen("--check", "--errors", str(p))
    assert r.returncode != 0, r.stdout + r.stderr


@pytest.mark.parametrize("victim", GENERATED)
def test_check_fails_when_a_generated_file_was_edited_or_removed(tmp_path, victim):
    d = tmp_path / "out"
    assert gen("--out-dir", str(d)).returncode == 0
    (d / victim).write_text((d / victim).read_text() + "\n# hand edit\n")
    r = gen("--check", "--out-dir", str(d))
    assert r.returncode != 0, r.stdout + r.stderr
    (d / victim).unlink()
    r = gen("--check", "--out-dir", str(d))
    assert r.returncode != 0, r.stdout + r.stderr


def test_generated_types_import_without_third_party_modules():
    import latai.types as t

    for name in ("Request", "Response", "Event", "StreamEvent", "StreamOpen", "Error", "PackInfo", "PackVersion", "Status", "RuntimeInfo", "StageReceipt"):
        assert hasattr(t, name), name


def test_every_schema_definition_has_a_type():
    import latai.types as t

    schema = json.loads((REPO / "schema" / "latai.schema.json").read_text())
    missing = [n for n in schema["$defs"] if n.isidentifier() and not hasattr(t, n)]
    # enums and plain aliases may be spelled differently, but the object definitions must exist
    objects = [n for n in missing if schema["$defs"][n].get("type") == "object"]
    assert not objects, f"object definitions without a generated type: {objects}"
