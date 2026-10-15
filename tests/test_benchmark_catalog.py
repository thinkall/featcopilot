"""Admission and provider regressions; real network is disabled by default."""

import ast
import copy
import hashlib
import io
import json
import uuid
from collections import Counter
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd
import pytest

from benchmarks import catalog

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "benchmarks" / "suites"
REAL_FETCH = catalog._fetch_bytes


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must mock network acquisition explicitly")

    monkeypatch.setattr(catalog, "_fetch_bytes", blocked)
    monkeypatch.setattr(catalog, "urlopen", blocked)


@pytest.fixture
def workspace():
    path = ROOT / "benchmarks" / ".cache" / "catalog-tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path
    for file in path.rglob("*"):
        if file.is_file():
            file.unlink()


def registry_before_regression_stress_reuse():
    """Reverse only the latest audited admission delta, preserving exact historic bytes."""
    audit = json.loads((SUITES / "regression_exposed_stress_replay_audit_20261008.json").read_bytes())
    delta = audit["registry_delta"]
    raw = (ROOT / "benchmarks" / "families.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == delta["after_file_sha256"]
    registry = json.loads(raw)
    by_id = {record["family_id"]: record for record in registry["families"]}
    text = raw.decode("utf-8")
    for fid in delta["changed_family_ids"]:
        original = copy.deepcopy(by_id[fid])
        assert (
            hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == delta["record_fingerprints"][fid]["after"]
        )
        for field, values in delta["changed_existing_fields"].items():
            assert original[field] == values["after"]
            original[field] = values["before"]
        for field in delta["added_fields"]:
            assert original.pop(field) == delta["added_audit_value"]
        assert (
            hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == delta["record_fingerprints"][fid]["before"]
        )
        line = next(line for line in text.splitlines() if f'"family_id":"{fid}"' in line)
        replacement = "    " + json.dumps(original, separators=(",", ":"), ensure_ascii=True)
        text = text.replace(line, replacement + ("," if line.endswith(",") else ""), 1)
    restored = text.encode("utf-8")
    assert hashlib.sha256(restored).hexdigest() == delta["before_file_sha256"]
    return restored


@pytest.fixture
def pre_regression_stress_registry(workspace, monkeypatch):
    """Validate old confirmation definitions against their recorded pre-reuse registry."""
    snapshot = workspace / "pre-regression-stress-families.json"
    snapshot.write_bytes(registry_before_regression_stress_reuse())
    monkeypatch.setattr(catalog, "_REGISTRY", snapshot)
    original_reader = catalog._read_json

    def historical_registry(path):
        if path.resolve() == (ROOT / "benchmarks" / "families.json").resolve():
            return original_reader(snapshot)
        return original_reader(path)

    monkeypatch.setattr(catalog, "_read_json", historical_registry)


def registry_before_qualified_binary_append():
    """Recover the prior immutable registry snapshot without editing the live registry."""
    audit = json.loads((SUITES / "qualified_binary_reservation_audit_20261004.json").read_bytes())
    ids = audit["registry_append_only"]["appended_family_ids"]
    lines = registry_before_regression_stress_reuse().decode("utf-8").splitlines(keepends=True)
    lines = [line for line in lines if not any(f'"family_id":"{fid}"' in line for fid in ids)]
    for index in range(len(lines) - 1, -1, -1):
        if '"family_id":' in lines[index]:
            ending = "\r\n" if lines[index].endswith("\r\n") else "\n"
            lines[index] = lines[index].rstrip("\r\n").removesuffix(",") + ending
            break
    restored = "".join(lines).encode("utf-8")
    assert hashlib.sha256(restored).hexdigest() == audit["registry_append_only"]["before_sha256"]
    return restored


def registry_before_standard_iid_append():
    """Recover prior registry bytes while leaving the live separate-tier records intact."""
    audit = json.loads((SUITES / "standard_iid_entity_reservation_audit_20261003.json").read_bytes())
    appended = audit["registry_append_only"]["appended_family_ids"]
    lines = registry_before_qualified_binary_append().decode("utf-8").splitlines(keepends=True)
    lines = [line for line in lines if not any(f'"family_id":"{fid}"' in line for fid in appended)]
    for index in range(len(lines) - 1, -1, -1):
        if '"family_id":' in lines[index]:
            ending = "\r\n" if lines[index].endswith("\r\n") else "\n"
            lines[index] = lines[index].rstrip("\r\n").removesuffix(",") + ending
            break
    restored = "".join(lines).encode("utf-8")
    assert hashlib.sha256(restored).hexdigest() == audit["registry_append_only"]["before_sha256"]
    return restored


def registry_before_cycle2_append():
    """Recover the immutable first-cycle registry layout from append-only metadata."""
    audit = json.loads((SUITES / "cycle2_origin_reservation_audit_20260929.json").read_bytes())
    appended = audit["registry_append_only"]["appended_family_ids"]
    lines = registry_before_standard_iid_append().decode("utf-8").splitlines(keepends=True)
    lines = [line for line in lines if not any(f'"family_id":"{fid}"' in line for fid in appended)]
    for index in range(len(lines) - 1, -1, -1):
        if '"family_id":' in lines[index]:
            ending = "\r\n" if lines[index].endswith("\r\n") else "\n"
            lines[index] = lines[index].rstrip("\r\n").removesuffix(",") + ending
            break
    restored = "".join(lines).encode("utf-8")
    assert hashlib.sha256(restored).hexdigest() == audit["exposure_audit"]["registry_before_sha256"]
    return restored


def registry_before_fresh_origin_append():
    """Recover the prior registry byte layout without removing any live records."""
    audit = json.loads((SUITES / "fresh_origin_reservation_audit_20260924.json").read_bytes())
    appended = audit["registry_append_only"]["appended_family_ids"]
    raw = registry_before_cycle2_append()
    lines = raw.decode("utf-8").splitlines(keepends=True)
    lines = [line for line in lines if not any(f'"family_id":"{fid}"' in line for fid in appended)]
    for index in range(len(lines) - 1, -1, -1):
        if '"family_id":' in lines[index]:
            ending = "\r\n" if lines[index].endswith("\r\n") else "\n"
            lines[index] = lines[index].rstrip("\r\n").removesuffix(",") + ending
            break
    restored = "".join(lines).encode("utf-8")
    assert hashlib.sha256(restored).hexdigest() == audit["exposure_audit"]["registry_before_sha256"]
    return restored


@pytest.fixture
def pre_release_registry(workspace, monkeypatch):
    """Replay historical reservation contracts against the recorded ACTIVE registry."""
    release = json.loads((SUITES / "final_reserved_confirmation_release_20260924.json").read_bytes())
    transition = release["registry_transition"]
    text = registry_before_fresh_origin_append().decode("utf-8")
    for fid, family_record in transition["before"].items():
        line = next(line for line in text.splitlines() if f'"family_id":"{fid}"' in line)
        historical = "    " + json.dumps(family_record, separators=(",", ":"), ensure_ascii=True)
        if line.endswith(","):
            historical += ","
        text = text.replace(line, historical, 1)
    snapshot = workspace / "pre-release-families.json"
    snapshot.write_bytes(text.encode("utf-8"))
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == transition["before_sha256"]
    monkeypatch.setattr(catalog, "_REGISTRY", snapshot)
    read_json = catalog._read_json

    def read_historical_registry(path):
        if path.resolve() == (ROOT / "benchmarks" / "families.json").resolve():
            return read_json(snapshot)
        return read_json(path)

    monkeypatch.setattr(catalog, "_read_json", read_historical_registry)


@pytest.fixture
def pre_iid_release_registry(workspace, monkeypatch):
    """Replay the ten candidates' ACTIVE state from the immutable IID release audit."""
    release = json.loads((SUITES / "frequency_fresh_confirmation_iid_release_20260928.json").read_bytes())
    transition = release["registry_transition"]
    text = registry_before_cycle2_append().decode("utf-8")
    for fid, family_record in transition["before"].items():
        line = next(line for line in text.splitlines() if f'"family_id":"{fid}"' in line)
        original = "    " + json.dumps(family_record, separators=(",", ":"), ensure_ascii=True)
        text = text.replace(line, original + ("," if line.endswith(",") else ""), 1)
    snapshot = workspace / "pre-iid-release-families.json"
    snapshot.write_bytes(text.encode("utf-8"))
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == transition["before_sha256"]
    monkeypatch.setattr(catalog, "_REGISTRY", snapshot)
    read_json = catalog._read_json

    def historical_registry(path):
        if path.resolve() == (ROOT / "benchmarks" / "families.json").resolve():
            return read_json(snapshot)
        return read_json(path)

    monkeypatch.setattr(catalog, "_read_json", historical_registry)


@pytest.fixture
def pre_cycle2_robust_release_registry(workspace, monkeypatch):
    """Replay the four-candidate reservation against its captured pre-release state."""
    audit = json.loads((SUITES / "cycle2_robust_confirmation_release_20260930.json").read_bytes())
    transition = audit["registry_transition"]
    text = registry_before_standard_iid_append().decode("utf-8")
    for fid, original in transition["before"].items():
        line = next(line for line in text.splitlines() if f'"family_id":"{fid}"' in line)
        replacement = "    " + json.dumps(original, separators=(",", ":"), ensure_ascii=True)
        text = text.replace(line, replacement + ("," if line.endswith(",") else ""), 1)
    snapshot = workspace / "pre-cycle2-robust-release-families.json"
    snapshot.write_bytes(text.encode("utf-8"))
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == transition["before_sha256"]
    monkeypatch.setattr(catalog, "_REGISTRY", snapshot)
    read_json = catalog._read_json

    def historical_registry(path):
        if path.resolve() == (ROOT / "benchmarks" / "families.json").resolve():
            return read_json(snapshot)
        return read_json(path)

    monkeypatch.setattr(catalog, "_read_json", historical_registry)


@pytest.fixture
def pre_standard_iid_policy_release_registry(workspace, monkeypatch):
    """Replay active qualified-tier reservations without weakening live release gates."""
    release = json.loads((SUITES / "standard_iid_policy_confirmation_release_20261003.json").read_bytes())
    transition = release["registry_transition"]
    text = registry_before_qualified_binary_append().decode("utf-8")
    for fid, original in transition["before"].items():
        line = next(line for line in text.splitlines() if f'"family_id":"{fid}"' in line)
        replacement = "    " + json.dumps(original, separators=(",", ":"), ensure_ascii=True)
        text = text.replace(line, replacement + ("," if line.endswith(",") else ""), 1)
    snapshot = workspace / "pre-standard-iid-policy-families.json"
    snapshot.write_bytes(text.encode("utf-8"))
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == transition["before_sha256"]
    monkeypatch.setattr(catalog, "_REGISTRY", snapshot)
    read_json = catalog._read_json

    def historical_registry(path):
        if path.resolve() == (ROOT / "benchmarks" / "families.json").resolve():
            return read_json(snapshot)
        return read_json(path)

    monkeypatch.setattr(catalog, "_read_json", historical_registry)


@pytest.fixture
def pre_thoracic_release_registry(workspace, monkeypatch):
    """Keep earlier Thoracic inventory checks on their recorded ACTIVE snapshot."""
    release = json.loads((SUITES / "thoracic_fixed_average_confirmation_release_20261004.json").read_bytes())
    transition = release["registry_transition"]
    text = registry_before_regression_stress_reuse().decode("utf-8")
    for fid, original in transition["before"].items():
        line = next(line for line in text.splitlines() if f'"family_id":"{fid}"' in line)
        replacement = "    " + json.dumps(original, separators=(",", ":"), ensure_ascii=True)
        text = text.replace(line, replacement + ("," if line.endswith(",") else ""), 1)
    snapshot = workspace / "pre-thoracic-release-families.json"
    snapshot.write_bytes(text.encode("utf-8"))
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == transition["before_sha256"]
    monkeypatch.setattr(catalog, "_REGISTRY", snapshot)
    original_reader = catalog._read_json

    def historical_registry(path):
        if path.resolve() == (ROOT / "benchmarks" / "families.json").resolve():
            return original_reader(snapshot)
        return original_reader(path)

    monkeypatch.setattr(catalog, "_read_json", historical_registry)


def family(exposure="metadata_only", role="development"):
    result = {
        "family_id": "test-family",
        "aliases": ["fixture-alias"],
        "source_ids": [],
        "upstream_origin": "Synthetic unit-test fixture, not a benchmark candidate",
        "prior_exposure": exposure,
        "role": role,
    }
    if exposure == "audited_unexposed":
        result["exposure_audit"] = {"status": "complete", "evidence": ["unit-test-only exposure audit"]}
    return result


def local_spec(workspace, data=b"measurement,kind,y\n1,A,no\n2,B,yes\n3,A,no\n4,B,yes\n"):
    path = workspace / "input.csv"
    path.write_bytes(data)
    return {
        "name": "test-data",
        "family_id": "test-family",
        "provider": "local_csv",
        "task": "classification",
        "role": "development",
        "prior_exposure": "metadata_only",
        "origin_kind": "synthetic_diagnostic",
        "target": "y",
        "categorical": ["kind"],
        "drop_columns": [],
        "split_policy": "iid",
        "prediction_time": "Unit-test measurements before synthetic fixture outcome.",
        "rights": {
            "status": "approved",
            "license": "CC0-1.0",
            "access": "local_only",
            "approval_basis": "Locally generated unit-test fixture",
            "evidence": ["tests/test_benchmark_catalog.py"],
        },
        "source": {
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "schema_reviewed": True,
            "columns": ["measurement", "kind", "y"],
            "target_columns": ["y"],
        },
    }


def normalize(spec, *, family_record=None, allow_confirmation=False):
    return catalog.validate_manifest(
        {"name": "fixture-suite", "families": [family_record or family()], "datasets": [spec]},
        allow_confirmation=allow_confirmation,
    )["datasets"][0]


def historical_manifest_snapshot(path):
    """Expand an immutable historical declaration without authorizing acquisition."""
    suite = json.loads(path.read_text(encoding="utf-8"))
    suite["datasets"] = [{**copy.deepcopy(suite.get("dataset_defaults", {})), **row} for row in suite["datasets"]]
    suite["family_count"] = len({row["family_id"] for row in suite["datasets"]})
    return suite


def load_local(workspace, spec):
    return catalog.load_dataset(normalize(spec), workspace / "raw-cache")


def test_shipped_manifests_are_metadata_only_until_load():
    development = catalog.load_suite(SUITES / "uci_development.json")
    assert len(development["datasets"]) == development["family_count"] == 7
    assert {spec["source"]["uci_id"] for spec in development["datasets"]} == {2, 144, 468, 602, 198, 165, 242}
    assert {spec["role"] for spec in development["datasets"]} == {"development"}
    assert all(spec["rights"]["status"] == "approved" for spec in development["datasets"])
    assert all(spec["prior_exposure"] != "audited_unexposed" for spec in development["datasets"])
    energy = next(spec for spec in development["datasets"] if spec["source"]["uci_id"] == 242)
    assert energy["origin_kind"] == "simulation"
    assert energy["target"] == "Y1" and energy["drop_columns"] == ["Y2"]
    adult = next(spec for spec in development["datasets"] if spec["source"]["uci_id"] == 2)
    assert adult["original_split"]["normalized_boundary_index"] == 32561
    assert not adult["original_split"]["evaluation_uses_boundary"]
    assert adult["drop_columns"] == ["fnlwgt"]
    seoul = catalog.load_suite(SUITES / "seoul_inventory.json")["datasets"][0]
    assert seoul["role"] == "inventory"
    assert seoul["target"] == "Rented Bike Count"
    assert seoul["source"]["target_columns"] == ["Functioning Day"]


def test_tabarena_pinned_inventory_and_known_overlap():
    suite = catalog.load_suite(SUITES / "tabarena_v01_inventory.json")
    assert len(suite["datasets"]) == 51
    assert len({item["source"]["task_id"] for item in suite["datasets"]}) == 51
    assert suite["source"]["revision"] == "f87f22d301b5985554c037c1bb1429ae6635dded"
    assert suite["source"]["suite_id"] == 457
    by_id = {spec["source"]["dataset_id"]: spec for spec in suite["datasets"]}
    assert by_id[46917]["family_id"] == "concrete-strength"
    assert by_id[46918]["family_id"] == "german-credit"
    assert by_id[46947]["family_id"] == "online-shoppers"
    assert by_id[46934]["prior_exposure"] == "exposed"
    assert by_id[46921]["family_id"] != by_id[46922]["family_id"]
    assert {spec["role"] for spec in suite["datasets"]} == {"inventory"}
    assert {spec["rights"]["status"] for spec in suite["datasets"]} == {"pending_review"}


def test_registry_covers_all_historical_variants_and_current_development():
    registry = catalog._read_json(ROOT / "benchmarks" / "families.json")
    tree = ast.parse((ROOT / "benchmarks" / "datasets.py").read_text(encoding="utf-8"))
    variants = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "INRIA_DATASETS" for target in node.targets)
    )
    assert len(variants) == 30
    assert set(registry["historical_inria_variants"]) == set(variants)
    aliases = {
        catalog._alias(alias): record
        for record in registry["families"]
        for alias in [record["family_id"], *record["aliases"]]
    }
    for name in [*variants, "fake_news", *registry["current_development_datasets"]]:
        assert aliases[catalog._alias(name)]["prior_exposure"] == "exposed"
    history = json.loads((ROOT / "benchmarks" / "simple_models" / "SIMPLE_MODELS_CACHE_LLM.json").read_text())
    assert {row["dataset"] for row in history if row["source"] == "real_world"} == {*variants, "fake_news"}
    assert "synthetic" in aliases[catalog._alias("synthetic:house_prices")]["upstream_origin"].lower()


def test_registry_canonical_ids_match_parent_exposure_ledger():
    registry = catalog._families([])
    aliases = {
        catalog._alias(alias): record["family_id"]
        for record in registry.values()
        for alias in [record["family_id"], *record["aliases"]]
    }
    expected = {
        "magic": "magic-telescope",
        "bank": "bank-marketing",
        "credit": "give-me-some-credit",
        "heloc": "heloc",
        "credit_default": "default-of-credit-card-clients",
        "eye_movements": "eye-movements",
        "cpu": "cpu-act",
        "wine": "wine-quality",
        "houses": "california-housing",
        "elevators": "elevators",
        "abalone": "abalone",
        "diamonds": "diamonds",
    }
    assert {name: aliases[catalog._alias(name)] for name in expected} == expected
    assert aliases[catalog._alias("cpu-activity")] == "cpu-act"
    assert aliases[catalog._alias("credit-card-default")] == "default-of-credit-card-clients"


@pytest.mark.parametrize("status", ["pending_review", "restricted"])
def test_pending_development_can_be_listed_but_not_acquired(workspace, status):
    spec = local_spec(workspace)
    spec["rights"]["status"] = status
    spec["source"].update(path="input.csv", schema_reviewed=False)
    manifest = workspace / "suite.json"
    manifest.write_text(json.dumps({"name": "pending-suite", "families": [family()], "datasets": [spec]}))
    before = manifest.read_bytes()
    normalized = catalog.load_suite(manifest)["datasets"][0]
    assert normalized["role"] == "development"
    assert normalized["rights"]["status"] == status
    assert normalized["source"]["path"] == str(workspace / "input.csv")
    with pytest.raises(catalog.AdmissionError, match="rights block admission"):
        catalog.load_dataset(normalized, workspace / "forbidden-cache")
    assert manifest.read_bytes() == before
    assert not (workspace / "forbidden-cache").exists()


@pytest.mark.parametrize(
    "name", ["../escape", "..\\escape", "C:\\escape", "/escape", "a/b", "a.b", "name ", "CON", "con", "lpt1", ""]
)
def test_unsafe_names_are_rejected_before_io(workspace, name):
    spec = local_spec(workspace)
    spec["name"] = name
    with pytest.raises(catalog.AdmissionError, match="safe lowercase identifier"):
        normalize(spec)


@pytest.mark.parametrize("role", ["fresh", "pending", None, "", 1, [], {}])
def test_invalid_roles(workspace, role):
    spec = local_spec(workspace)
    spec["role"] = role
    with pytest.raises(catalog.AdmissionError, match="invalid role"):
        normalize(spec)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target", "", "explicit target"),
        ("target", None, "explicit target"),
        ("target", "measurement", "reviewed native target"),
        ("categorical", ["y"], "must be predictors"),
        ("categorical", ["missing"], "must be predictors"),
        ("drop_columns", ["missing"], "unknown drop_columns"),
        ("target_transform", "guess", "unsupported target_transform"),
        ("task", "binary", "invalid task"),
        ("prior_exposure", "audited_unexposed", "prior_exposure mismatch"),
    ],
)
def test_invalid_dataset_schema(workspace, field, value, message):
    spec = local_spec(workspace)
    spec[field] = value
    with pytest.raises(catalog.AdmissionError, match=message):
        normalize(spec)


@pytest.mark.parametrize("status", ["restricted", "pending_review", "unknown"])
def test_unapproved_rights_never_acquire(workspace, status):
    spec = local_spec(workspace)
    spec["rights"]["status"] = status
    with pytest.raises(catalog.AdmissionError, match="rights"):
        catalog.load_dataset({**spec, "_family": family()}, workspace / "forbidden-cache")
    assert not (workspace / "forbidden-cache").exists()


@pytest.mark.parametrize("field", ["license", "approval_basis", "evidence"])
def test_approved_rights_need_declaration_and_evidence(workspace, field):
    spec = local_spec(workspace)
    del spec["rights"][field]
    with pytest.raises(catalog.AdmissionError):
        normalize(spec)


def test_confirmation_requires_all_three_gates(workspace):
    spec = local_spec(workspace)
    spec.update(role="confirmation", prior_exposure="audited_unexposed")
    audit = family("audited_unexposed", "confirmation")
    with pytest.raises(catalog.AdmissionError, match="confirmation is locked"):
        normalize(spec, family_record=audit)
    admitted = normalize(spec, family_record=audit, allow_confirmation=True)
    assert len(catalog.load_dataset(admitted, workspace / "raw-cache").X) == 4
    del audit["exposure_audit"]
    with pytest.raises(catalog.AdmissionError, match="completed exposure audit"):
        normalize(spec, family_record=audit, allow_confirmation=True)
    spec["rights"]["status"] = "pending_review"
    with pytest.raises(catalog.AdmissionError, match="rights"):
        normalize(spec, family_record=family("audited_unexposed", "confirmation"), allow_confirmation=True)


@pytest.mark.parametrize("exposure", ["exposed", "loader_only", "metadata_only", "unknown"])
def test_confirmation_not_unlocked_by_flag_alone(workspace, exposure):
    spec = local_spec(workspace)
    spec.update(role="confirmation", prior_exposure=exposure)
    with pytest.raises(catalog.AdmissionError, match="audited_unexposed"):
        normalize(spec, family_record=family(exposure, "confirmation"), allow_confirmation=True)


@pytest.mark.parametrize("alias", ["houses", "wine_quality", "covertype_cat", "fake_news", "california"])
def test_known_alias_cannot_be_relabeled_fresh(workspace, alias):
    spec = local_spec(workspace)
    spec["aliases"] = [alias]
    with pytest.raises(catalog.AdmissionError, match="belongs to another family"):
        normalize(spec)


def test_known_source_cannot_hide_exposure(workspace):
    spec = local_spec(workspace)
    spec["source"]["dataset_id"] = 46934
    with pytest.raises(catalog.AdmissionError, match="belongs to another family"):
        normalize(spec)


def test_hf_inria_path_and_mirror_cannot_hide_exposure(workspace):
    spec = local_spec(workspace)
    spec.update(provider="hf_csv")
    spec["rights"]["access"] = "public"
    spec["source"].update(repo_id="inria-soda/tabular-benchmark", revision="a" * 40, path="reg_num/houses.csv")
    with pytest.raises(catalog.AdmissionError, match="belongs to another family"):
        normalize(spec)
    spec["source"].update(repo_id="mstz/covertype", path="data.csv")
    with pytest.raises(catalog.AdmissionError, match="belongs to another family"):
        normalize(spec)


def test_confirmation_family_cannot_bypass_lock_by_diagnostic_relabeling(workspace):
    spec = local_spec(workspace)
    spec.update(role="diagnostic", prior_exposure="audited_unexposed")
    with pytest.raises(catalog.AdmissionError, match="family role mismatch"):
        normalize(spec, family_record=family("audited_unexposed", "confirmation"))


def test_local_only_rights_cannot_authorize_remote_provider(workspace):
    spec = local_spec(workspace)
    spec["provider"] = "hf_csv"
    spec["source"].update(repo_id="fixture/data", revision="a" * 40, path="data.csv")
    with pytest.raises(catalog.AdmissionError, match="public acquisition rights"):
        normalize(spec)


def test_cannot_override_exposed_registry_family():
    current = catalog._families([])["california-housing"]
    record = copy.deepcopy(current)
    record["prior_exposure"] = "metadata_only"
    resolved = catalog._families([record])["california-housing"]
    assert resolved["prior_exposure"] == "exposed"
    assert resolved["exposure_history"] == current["exposure_history"]
    record["upstream_origin"] = "A different, unverified origin"
    with pytest.raises(catalog.AdmissionError, match="cannot override registry"):
        catalog._families([record])


def test_duplicate_variants_and_names_rejected(workspace):
    spec = local_spec(workspace)
    suite = {"name": "test-suite", "families": [family()], "datasets": [spec, copy.deepcopy(spec)]}
    with pytest.raises(catalog.AdmissionError, match="Duplicate dataset name"):
        catalog.validate_manifest(suite)
    suite["datasets"][1]["name"] = "another-variant"
    with pytest.raises(catalog.AdmissionError, match="Duplicate primary family"):
        catalog.validate_manifest(suite)


@pytest.mark.parametrize("filename", ["seoul_inventory.json", "tabarena_v01_inventory.json"])
def test_inventory_never_downloads_even_with_allow_flag(workspace, filename):
    suite = catalog.load_suite(SUITES / filename, allow_confirmation=True)
    for spec in suite["datasets"]:
        with pytest.raises(catalog.AdmissionError, match="must never download"):
            catalog.load_dataset(spec, workspace / "forbidden-cache")
    assert not (workspace / "forbidden-cache").exists()


def test_relative_local_path_and_original_mapping(workspace):
    spec = local_spec(workspace)
    spec["source"]["path"] = "input.csv"
    manifest = workspace / "suite.json"
    manifest.write_text(json.dumps({"name": "test-suite", "families": [family()], "datasets": [spec]}))
    normalized = catalog.load_suite(manifest)["datasets"][0]
    original_manifest = manifest.read_bytes()
    original_source = (workspace / "input.csv").read_bytes()
    assert Path(normalized["source"]["path"]) == workspace / "input.csv"
    loaded = catalog.load_dataset(normalized, workspace / "raw-cache")
    assert loaded.X.columns.tolist() == ["x0", "x1"]
    assert loaded.metadata["column_mapping"] == {"x0": "measurement", "x1": "kind"}
    assert loaded.metadata["class_labels"] == ["no", "yes"]
    assert loaded.y.tolist() == [0, 1, 0, 1]
    assert loaded.metadata["original_profile"]["class_prevalence"] == {"0": 0.5, "1": 0.5}
    assert loaded.X.index.equals(loaded.y.index)
    assert loaded.official_splits is None
    assert loaded.label_end_times is None
    cached = catalog.load_dataset(normalized, workspace / "raw-cache")
    assert loaded.metadata == cached.metadata
    assert json.loads(json.dumps(loaded.metadata)) == loaded.metadata
    assert manifest.read_bytes() == original_manifest
    assert (workspace / "input.csv").read_bytes() == original_source


def test_checksums_reject_local_drift_and_raw_cache_corruption(workspace):
    spec = normalize(local_spec(workspace))
    cache = workspace / "raw-cache"
    loaded = catalog.load_dataset(spec, cache)
    assert loaded.metadata["sha256"] == spec["source"]["sha256"]
    payload = catalog._cache_directory(spec, cache) / "data.csv"
    original = payload.read_bytes()
    payload.write_bytes(b"corruption")
    with pytest.raises(catalog.AcquisitionError, match="corrupt cache"):
        catalog.load_dataset(spec, cache)
    payload.write_bytes(original)
    (workspace / "input.csv").write_bytes(b"changed local data")
    with pytest.raises(catalog.AcquisitionError, match="Local source checksum drift"):
        catalog.load_dataset(spec, cache)


def test_incomplete_pin_cache_is_not_silently_repaired(workspace):
    directory = workspace / "cache"
    directory.mkdir()
    (directory / "data.csv").write_bytes(b"partial")
    with pytest.raises(catalog.AcquisitionError, match="Incomplete raw cache"):
        catalog._cached_bytes(directory, "data.csv", {}, lambda: b"replacement")


def test_hf_revision_and_payload_checksum_both_required(workspace, monkeypatch):
    spec = local_spec(workspace)
    raw = (workspace / "input.csv").read_bytes()
    spec["provider"] = "hf_csv"
    spec["rights"]["access"] = "public"
    spec["source"].update(repo_id="fixture/data", revision="a" * 40, path="data/train.csv")
    urls = []

    def fetch(url):
        urls.append(url)
        return raw

    monkeypatch.setattr(catalog, "_fetch_bytes", fetch)
    loaded = load_local(workspace, spec)
    assert len(loaded.X) == 4
    assert urls == [f"https://huggingface.co/datasets/fixture/data/resolve/{'a' * 40}/data/train.csv"]
    load_local(workspace, spec)
    assert len(urls) == 1
    spec["source"]["revision"] = "main"
    with pytest.raises(catalog.AdmissionError, match="immutable"):
        normalize(spec)
    spec["source"]["revision"] = "a" * 40
    del spec["source"]["sha256"]
    with pytest.raises(catalog.AdmissionError, match="payload SHA-256"):
        normalize(spec)


def test_remote_checksum_mismatch_rejects_before_cache_write(workspace, monkeypatch):
    monkeypatch.setattr(catalog, "_fetch_bytes", lambda url: b"unexpected content")
    spec = local_spec(workspace)
    spec["provider"] = "hf_csv"
    spec["rights"]["access"] = "public"
    spec["source"].update(repo_id="fixture/data", revision="a" * 40, path="data.csv")
    with pytest.raises(catalog.AcquisitionError, match="SHA-256 mismatch"):
        load_local(workspace, spec)
    assert not (catalog._cache_directory(normalize(spec), workspace / "raw-cache") / "data.csv").exists()


def test_uci_metadata_and_payload_pins(workspace, monkeypatch):
    spec = local_spec(workspace)
    raw = (workspace / "input.csv").read_bytes()
    spec["provider"] = "uci"
    spec["rights"]["access"] = "public"
    spec["source"].update(uci_id=999001, data_url="https://archive.ics.uci.edu/static/public/999001/data.csv")
    del spec["source"]["sha256"]
    metadata = {
        "data": {
            "uci_id": 999001,
            "data_url": spec["source"]["data_url"],
            "variables": [
                {"name": col, "role": "Target" if col == "y" else "Feature"} for col in spec["source"]["columns"]
            ],
        }
    }
    urls = []

    def fetch(url):
        urls.append(url)
        return json.dumps(metadata).encode() if "api/dataset" in url else raw

    monkeypatch.setattr(catalog, "_fetch_bytes", fetch)
    result = load_local(workspace, spec)
    assert len(urls) == 2 and result.metadata["sha256"] == hashlib.sha256(raw).hexdigest()
    cached = load_local(workspace, spec)
    assert cached.metadata == result.metadata
    assert len(urls) == 2
    metadata["data"]["variables"][-1]["role"] = "Feature"
    with pytest.raises(catalog.AdmissionError, match="target-role metadata conflicts"):
        catalog.load_dataset(normalize(spec), workspace / "conflicting-raw-cache")


def test_multitarget_exclusion_is_mandatory(workspace):
    raw = b"measurement,kind,y,other_target\n1,A,10,100\n2,B,20,200\n"
    spec = local_spec(workspace, raw)
    spec.update(task="regression")
    spec["source"]["columns"].append("other_target")
    spec["source"]["target_columns"].append("other_target")
    with pytest.raises(catalog.AdmissionError, match="exclude all other native target"):
        normalize(spec)
    spec["drop_columns"] = ["other_target"]
    result = load_local(workspace, spec)
    assert result.X.shape == (2, 2)
    assert result.y.tolist() == [10.0, 20.0]
    assert "other_target" not in result.metadata["columns"]


@pytest.mark.parametrize("invalid_row", ["1,1,1", "1,0,0", "1,2,0", "1,,1"])
def test_one_hot_labels_require_exclusive_binary_membership(workspace, invalid_row):
    raw = f"measurement,a,b\n{invalid_row}\n2,0,1\n".encode()
    spec = local_spec(workspace, raw)
    spec.update(target="fault_type", target_transform="one_hot", categorical=[], drop_columns=["a", "b"])
    spec["source"].update(columns=["measurement", "a", "b"], target_columns=["a", "b"])
    with pytest.raises(catalog.AdmissionError, match="exactly one active binary"):
        load_local(workspace, spec)


def test_seven_one_hot_targets_are_all_removed(workspace):
    targets = ["Pastry", "Z_Scratch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    frame = pd.DataFrame(np.eye(7, dtype=int), columns=targets)
    frame.insert(0, "measurement", np.arange(7))
    spec = local_spec(workspace, frame.to_csv(index=False).encode())
    spec.update(target="fault_type", target_transform="one_hot", categorical=[], drop_columns=targets)
    spec["source"].update(columns=frame.columns.tolist(), target_columns=targets)
    result = load_local(workspace, spec)
    assert result.X.shape == (7, 1)
    assert set(result.metadata["class_labels"]) == set(targets)
    assert result.metadata["original_profile"]["n_classes"] == 7
    spec["drop_columns"] = targets[:-1]
    with pytest.raises(catalog.AdmissionError, match="drop all one-hot target"):
        normalize(spec)


def test_sampling_preserves_alignment_imbalance_and_determinism(workspace):
    frame = pd.DataFrame({"measurement": np.arange(100), "kind": ["a"] * 100, "y": ["no"] * 90 + ["yes"] * 10})
    spec = normalize(local_spec(workspace, frame.to_csv(index=False).encode()))
    first = catalog.load_dataset(spec, workspace / "raw-cache", max_rows=20, sample_seed=7)
    second = catalog.load_dataset(spec, workspace / "raw-cache", max_rows=20, sample_seed=7)
    pd.testing.assert_frame_equal(first.X, second.X)
    assert first.y.value_counts().to_dict() == {0: 18, 1: 2}
    np.testing.assert_array_equal(first.X.x0, first.metadata["sample_indices"])
    np.testing.assert_array_equal(first.y, (first.X.x0 >= 90).astype(int))
    assert first.metadata["original_rows"] == 100 and first.metadata["sampled_rows"] == 20
    assert first.metadata["sample_indices_sha256"] == second.metadata["sample_indices_sha256"]


def test_groups_and_timestamps_are_separate_and_aligned(workspace):
    raw = b"measurement,kind,y,entity,date\n1,A,no,g1,2025-01-01\n2,B,yes,g2,2025-01-02\n3,A,no,g1,2025-01-03\n4,B,yes,g2,2025-01-04\n"
    spec = local_spec(workspace, raw)
    spec["source"]["columns"] += ["entity", "date"]
    spec.update(group_column="entity", split_policy="grouped", drop_columns=["date"])
    result = load_local(workspace, spec)
    assert result.groups.tolist() == ["g1", "g2", "g1", "g2"]
    assert result.groups.index.equals(result.X.index)
    assert result.metadata["columns"] == ["measurement", "kind"]
    del spec["group_column"]
    spec.update(time_column="date", time_format="%Y-%m-%d", split_policy="temporal", drop_columns=["entity"])
    temporal = load_local(workspace, spec)
    assert temporal.timestamps.dt.day.tolist() == [1, 2, 3, 4]
    assert temporal.timestamps.index.equals(temporal.y.index)
    assert temporal.label_end_times is None
    with pytest.raises(catalog.AdmissionError, match="row caps are forbidden"):
        catalog.load_dataset(normalize(spec), workspace / "unused", max_rows=1)


def label_end_spec(workspace, end_values=None):
    frame = pd.DataFrame(
        {
            "measurement": [3, 1, 4, 2],
            "kind": ["A", "B", "A", "B"],
            "y": ["no", "yes", "no", "yes"],
            "entity": ["g1", "g2", "g1", "g2"],
            "date": ["2025-01-03", "2025-01-01", "2025-01-04", "2025-01-02"],
            "label_available": end_values or ["2025-01-05", "2025-01-09", "2025-01-04", "2025-01-08"],
        }
    )
    spec = local_spec(workspace, frame.to_csv(index=False).encode())
    spec["source"]["columns"] = frame.columns.tolist()
    spec.update(
        group_column="entity",
        time_column="date",
        label_end_column="label_available",
        time_format="%Y-%m-%d",
        split_policy="temporal",
    )
    return spec


def test_explicit_label_end_times_are_removed_and_aligned_without_sorting(workspace):
    spec = label_end_spec(workspace)
    result = load_local(workspace, spec)
    assert result.X.x0.tolist() == [3, 1, 4, 2]
    assert result.y.tolist() == [0, 1, 0, 1]
    assert result.timestamps.dt.day.tolist() == [3, 1, 4, 2]
    assert result.label_end_times.dt.day.tolist() == [5, 9, 4, 8]
    assert result.groups.tolist() == ["g1", "g2", "g1", "g2"]
    for series in (result.y, result.timestamps, result.label_end_times, result.groups):
        assert series.index.equals(result.X.index)
    assert result.metadata["sample_indices"] == [0, 1, 2, 3]
    assert result.metadata["columns"] == ["measurement", "kind"]
    assert result.metadata["label_end_column"] == "label_available"
    cached = load_local(workspace, spec)
    pd.testing.assert_series_equal(cached.label_end_times, result.label_end_times)
    assert cached.metadata == result.metadata
    with pytest.raises(catalog.AdmissionError, match="row caps are forbidden"):
        catalog.load_dataset(normalize(spec), workspace / "unused", max_rows=2)


@pytest.mark.parametrize("column", ["unknown_end", "y", "", ["label_available"]])
def test_label_end_column_must_be_a_reviewed_nontarget_column(workspace, column):
    spec = label_end_spec(workspace)
    spec["label_end_column"] = column
    with pytest.raises(catalog.AdmissionError, match="label.end|invalid label_end_column"):
        normalize(spec)


def test_label_end_column_requires_time_column_and_cannot_be_a_predictor(workspace):
    spec = label_end_spec(workspace)
    del spec["time_column"]
    with pytest.raises(catalog.AdmissionError, match="label_end_column requires time_column"):
        normalize(spec)
    spec["time_column"] = "date"
    spec["categorical"].append("label_available")
    with pytest.raises(catalog.AdmissionError, match="must be predictors"):
        normalize(spec)


@pytest.mark.parametrize("invalid_end", ["", "not-a-date", "01/05/2025"])
def test_label_end_timestamps_require_nonmissing_values_in_declared_format(workspace, invalid_end):
    spec = label_end_spec(workspace, [invalid_end, "2025-01-09", "2025-01-04", "2025-01-08"])
    with pytest.raises(ValueError):
        load_local(workspace, spec)


def test_label_end_times_are_never_inferred_from_a_horizon(workspace):
    spec = label_end_spec(workspace)
    del spec["label_end_column"]
    spec["drop_columns"] = ["label_available"]
    spec["horizon_days"] = 7
    result = load_local(workspace, spec)
    assert result.timestamps is not None
    assert result.label_end_times is None


def test_catalog_does_not_fit_statistics_or_drop_missing_predictors(workspace):
    raw = b"measurement,kind,y\n,A,no\n2,A,yes\n1000,B,no\n2000,B,yes\n"
    result = load_local(workspace, local_spec(workspace, raw))
    assert pd.isna(result.X.iloc[0, 0])
    assert result.X.x0.iloc[1:].tolist() == [2.0, 1000.0, 2000.0]
    assert result.metadata["original_profile"]["missingness"]["measurement"] == 0.25
    assert result.metadata["original_profile"]["cardinality"]["kind"] == 2


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"measurement,kind,y\n1,A,\n2,B,yes\n", "Missing targets"),
        (b"measurement,kind,y\n1,A,no\n2,B,no\n", "at least two"),
        (b"measurement,kind,y,extra\n1,A,no,2\n2,B,yes,3\n", "Payload column schema"),
        (b"measurement,measurement,y\n1,2,no\n2,3,yes\n", "Duplicate source column"),
        (b"measurement,kind,y\ntext,A,no\n2,B,yes\n", "explicitly declared categorical"),
        (b"measurement,kind,y\n1,2,A,no\n2,3,B,yes\n", "refusing implicit index/column shifts"),
    ],
)
def test_payload_schema_and_target_failures(workspace, raw, message):
    with pytest.raises(catalog.AdmissionError, match=message):
        load_local(workspace, local_spec(workspace, raw))


@pytest.mark.parametrize("value", ["inf", "-inf", "nan"])
def test_regression_nonfinite_targets_rejected(workspace, value):
    spec = local_spec(workspace, f"measurement,kind,y\n1,A,{value}\n2,B,1\n".encode())
    spec["task"] = "regression"
    with pytest.raises(catalog.AdmissionError, match="finite|Missing targets"):
        load_local(workspace, spec)


def test_declared_log_transform(workspace):
    spec = local_spec(workspace, b"measurement,kind,y\n1,A,0\n2,B,3\n")
    spec.update(task="regression", target_transform="log1p")
    result = load_local(workspace, spec)
    np.testing.assert_allclose(result.y, [0, np.log(4)])


def test_approved_kaggle_local_never_invokes_download(workspace):
    spec = local_spec(workspace)
    spec["provider"] = "kaggle_local"
    result = load_local(workspace, spec)
    assert result.metadata["provider"] == "kaggle_local"
    spec["source"]["path"] = str(workspace / "not-supplied.csv")
    with pytest.raises(catalog.AcquisitionError, match="Automatic Kaggle downloads are disabled"):
        load_local(workspace, spec)


def test_fetch_timeout_size_limit_and_helpful_failure(monkeypatch):
    seen = {}

    def response(request, *, timeout):
        seen["timeout"] = timeout
        return io.BytesIO(b"payload")

    monkeypatch.setattr(catalog, "urlopen", response)
    assert REAL_FETCH("https://example.org/fixture", timeout=7) == b"payload"
    assert seen["timeout"] == 7
    with pytest.raises(catalog.AcquisitionError, match="acquisition limit"):
        REAL_FETCH("https://example.org/fixture", max_bytes=2)

    def failure(*args, **kwargs):
        raise URLError("test timeout")

    monkeypatch.setattr(catalog, "urlopen", failure)
    with pytest.raises(catalog.AcquisitionError, match="No synthetic fallback"):
        REAL_FETCH("https://example.org/fixture")


@pytest.mark.parametrize(
    ("train", "test", "message"),
    [
        ([0, 1], [1, 2], "overlap"),
        ([0, 0], [2, 3], "duplicate"),
        ([0, -1], [2, 3], "out of range"),
        ([0, 8], [2, 3], "out of range"),
        ([0, 1.5], [2, 3], "nonintegral"),
        ([0, np.nan], [2, 3], "missing"),
    ],
)
def test_official_indices_validated(train, test, message):
    frame = pd.DataFrame(
        {"type": ["TRAIN"] * len(train) + ["TEST"] * len(test), "rowid": train + test, "repeat": 0, "fold": 0}
    )
    with pytest.raises(catalog.AdmissionError, match=message):
        catalog._official_splits(frame, 8)


def openml_fixture(workspace, monkeypatch):
    arff = pytest.importorskip("arff")
    pytest.importorskip("openml")
    spec = local_spec(workspace)
    spec.update(provider="openml_task", split_policy="official")
    spec["rights"]["access"] = "public"
    spec["source"] = {
        "task_id": 999001,
        "dataset_id": 999002,
        "dataset_version": 1,
        "schema_reviewed": True,
        "columns": ["measurement", "kind", "y", "other_target"],
        "target_columns": ["y", "other_target"],
    }
    spec["drop_columns"] = ["other_target"]
    data = arff.dumps(
        {
            "relation": "fixture",
            "attributes": [
                ("measurement", "NUMERIC"),
                ("kind", ["A", "B"]),
                ("y", ["no", "yes"]),
                ("other_target", "NUMERIC"),
            ],
            "data": [[i, "A" if i < 4 else "B", "yes" if i % 2 else "no", i * 100] for i in range(8)],
        }
    ).encode()
    rows = []
    for repeat in range(2):
        for fold in range(2):
            rows.extend([["TRAIN" if (i < 4) == (fold == 0) else "TEST", i, repeat, fold] for i in range(8)])
    splits = arff.dumps(
        {
            "relation": "fixture-splits",
            "attributes": [
                ("type", ["TRAIN", "TEST"]),
                ("rowid", "NUMERIC"),
                ("repeat", "NUMERIC"),
                ("fold", "NUMERIC"),
            ],
            "data": rows,
        }
    ).encode()
    task = {
        "task": {
            "task_id": "999001",
            "task_type_id": "1",
            "input": [
                {"name": "source_data", "data_set": {"data_set_id": "999002", "target_feature": "y"}},
                {
                    "name": "estimation_procedure",
                    "estimation_procedure": {"data_splits_url": "https://openml.org/fixture-splits.arff"},
                },
            ],
        }
    }
    dataset = {
        "data_set_description": {
            "id": "999002",
            "version": "1",
            "format": "arff",
            "url": "https://openml.org/fixture-data.arff",
            "md5_checksum": hashlib.md5(data).hexdigest(),
        }
    }
    payloads = {
        "https://www.openml.org/api/v1/json/task/999001": json.dumps(task).encode(),
        "https://www.openml.org/api/v1/json/data/999002": json.dumps(dataset).encode(),
        "https://openml.org/fixture-data.arff": data,
        "https://openml.org/fixture-splits.arff": splits,
    }
    monkeypatch.setattr(catalog, "_fetch_bytes", lambda url: payloads[url])
    return normalize(spec), payloads


def test_openml_task_preserves_official_repeat_fold_indices_and_raw_rows(workspace, monkeypatch):
    spec, payloads = openml_fixture(workspace, monkeypatch)
    loaded = catalog.load_dataset(spec, workspace / "raw-cache")
    assert loaded.X.x0.tolist() == list(range(8))
    assert loaded.y.tolist() == [0, 1] * 4
    assert loaded.metadata["columns"] == ["measurement", "kind"]
    assert len(loaded.official_splits) == 4
    assert loaded.metadata["official_split_ids"] == [
        {"repeat": repeat, "fold": fold, "sample": 0} for repeat in range(2) for fold in range(2)
    ]
    train, test = loaded.official_splits[0]
    assert loaded.X.iloc[train].x0.tolist() == [0, 1, 2, 3]
    assert loaded.X.iloc[test].x0.tolist() == [4, 5, 6, 7]
    assert loaded.X.iloc[train].x1.unique().tolist() == ["A"]
    assert loaded.X.iloc[test].x1.unique().tolist() == ["B"]
    assert loaded.metadata["version"]["dataset_version"] == 1
    assert (
        loaded.metadata["official_splits_sha256"]
        == hashlib.sha256(payloads["https://openml.org/fixture-splits.arff"]).hexdigest()
    )
    monkeypatch.setattr(catalog, "_fetch_bytes", lambda url: pytest.fail("Cache should prevent another download"))
    cached = catalog.load_dataset(spec, workspace / "raw-cache")
    pd.testing.assert_frame_equal(cached.X, loaded.X)
    assert cached.metadata == loaded.metadata
    assert not list((workspace / "raw-cache").rglob("*.pkl*"))


@pytest.mark.parametrize("cap", [1, 8, 1000])
def test_openml_rowcap_rejected_before_any_acquisition(workspace, monkeypatch, cap):
    spec, _ = openml_fixture(workspace, monkeypatch)
    monkeypatch.setattr(catalog, "_fetch_bytes", lambda url: pytest.fail("Row cap must fail before acquisition"))
    with pytest.raises(catalog.AdmissionError, match="row caps are forbidden"):
        catalog.load_dataset(spec, workspace / "raw-cache", max_rows=cap)


@pytest.mark.parametrize("field", ["dataset_version", "dataset_id", "target", "task"])
def test_openml_task_identity_and_version_cannot_drift(workspace, monkeypatch, field):
    spec, _ = openml_fixture(workspace, monkeypatch)
    if field in {"dataset_version", "dataset_id"}:
        spec["source"][field] += 1
    elif field == "target":
        spec["target"] = "other_target"
        spec["drop_columns"] = ["y"]
    else:
        spec["task"] = "regression"
    with pytest.raises(catalog.AdmissionError, match="mismatch"):
        catalog.load_dataset(spec, workspace / "raw-cache")


def test_official_split_cache_corruption_rejected(workspace, monkeypatch):
    spec, _ = openml_fixture(workspace, monkeypatch)
    cache = workspace / "raw-cache"
    catalog.load_dataset(spec, cache)
    (catalog._cache_directory(spec, cache) / "splits.arff").write_bytes(b"corrupted official splits")
    with pytest.raises(catalog.AcquisitionError, match="corrupt cache"):
        catalog.load_dataset(spec, cache)


def test_completed_confirmation_is_exposed_and_historical_preregistration_is_preserved(workspace):
    path = SUITES / "tabarena_confirmation_candidates_20260920.json"
    for allow in (False, True):
        with pytest.raises(catalog.AdmissionError, match="already exposed"):
            catalog.load_suite(path, allow_confirmation=allow)
    suite = historical_manifest_snapshot(path)
    assert len(suite["datasets"]) == suite["family_count"] == 6
    assert {spec["role"] for spec in suite["datasets"]} == {"confirmation"}
    assert {spec["prior_exposure"] for spec in suite["datasets"]} == {"audited_unexposed"}
    assert {spec["rights"]["status"] for spec in suite["datasets"]} == {"approved"}
    preregistration = suite["preregistration"]
    assert preregistration["selection_parameters"] == {
        "stability_repeats": 3,
        "screening": "auto",
        "max_candidates": 1000,
        "max_added_features": 50,
    }
    assert preregistration["max_rows"] == 0
    assert preregistration["expected_official_splits"] == 6 * 10 * 3
    assert preregistration["expected_cells"] == 6 * 10 * 3 * 2 * 4
    assert preregistration["total_active_seconds_ceiling"] <= 45.8 * 3600
    assert preregistration["threads"] == 8
    assert preregistration["family_ids"] == [spec["family_id"] for spec in suite["datasets"]]
    exposed = suite["exposure_audit"]["ledger_snapshot"]["exposed_family_ids"]
    assert not set(preregistration["family_ids"]).intersection(exposed)
    for spec in suite["datasets"]:
        assert spec["source"]["official_split_dimensions"] == {"repeats": 10, "folds": 3, "samples": 1}
        assert spec["source"]["schema_reviewed"]
        assert len(spec["source"]["md5_checksum"]) == 32
        assert any(url.startswith("https://archive.ics.uci.edu/dataset/") for url in spec["rights"]["evidence"])
        assert len(spec["rights"]["primary_page_sha256"]) == 64
        with pytest.raises(catalog.AdmissionError, match="already exposed"):
            catalog.load_dataset(spec, workspace / "forbidden-cache", max_rows=1)
    assert not (workspace / "forbidden-cache").exists()


def test_confirmation_task_schema_and_version_pins():
    suite = historical_manifest_snapshot(SUITES / "tabarena_confirmation_candidates_20260920.json")
    expected = {
        46904: (363612, 9, 1503, 5, 1),
        46916: (363624, 1, 9822, 85, 3),
        46937: (363681, 1, 12684, 24, 21),
        46954: (363698, 8, 907, 6, 0),
        46958: (363702, 4, 3190, 60, 60),
        46969: (363689, 3, 7491, 86, 86),
    }
    for spec in suite["datasets"]:
        source = spec["source"]
        actual = (
            source["task_id"],
            source["dataset_version"],
            source["expected_rows"],
            len(source["columns"]) - 1,
            len(spec["categorical"]),
        )
        assert actual == expected[source["dataset_id"]]
        assert source["target_columns"] == [spec["target"]]
        assert spec["target"] not in spec["categorical"]
        assert len(set(source["columns"])) == len(source["columns"])
        if spec["task"] == "classification":
            assert source["metadata_class_support"]["minority_class_size"] >= spec["class_support"]["min_total"]


def test_broader_audit_preserves_inventory_and_separates_clearance(workspace):
    original = catalog.load_suite(SUITES / "tabarena_v01_inventory.json")
    audited = catalog.load_suite(SUITES / "tabarena_v01_audit_20260920.json")
    assert len(audited["datasets"]) == audited["family_count"] == 51
    assert Counter(spec["review_status"] for spec in audited["datasets"]) == {
        "development_exposed": 14,
        "locked_confirmation": 6,
        "pending_review": 31,
    }
    assert audited["audit_summary"]["certified_independent_inventory_origins"] is None

    def identity(spec):
        return spec["source"]["dataset_id"], spec["source"]["task_id"], spec["task"], spec["target"]

    assert {identity(spec) for spec in original["datasets"]} == {identity(spec) for spec in audited["datasets"]}
    assert {spec["role"] for spec in audited["datasets"]} == {"inventory"}
    for spec in audited["datasets"]:
        with pytest.raises(catalog.AdmissionError, match="must never download"):
            catalog.load_dataset(spec, workspace / "forbidden-cache")
    pending = audited["audit_summary"]["pending_decisions"]
    assert "GPL" in pending["tabarena-46908"]
    assert "minority class size 8" in pending["tabarena-46906"]
    assert "cutoff" in pending["tabarena-46913"]
    assert "overlap" in pending["tabarena-46952"]
    assert "time/group" in pending["tabarena-46956"]
    assert "semester" in pending["tabarena-46960"]
    assert "Suspicious" in pending["tabarena-46963"]


def test_historical_source_manifests_are_unchanged_by_audit():
    expected = {
        "uci_development.json": "72b5257603de43b1653c00449f0dfd3279a7dea3fa53c5f036d70e5e3443ee7d",
        "tabarena_v01_inventory.json": "92dc0e6e58e09422104735cbe049dd42eca559acca3808a6abee55e9cf870243",
    }
    for name, checksum in expected.items():
        assert hashlib.sha256((SUITES / name).read_bytes()).hexdigest() == checksum


def test_historical_inventory_does_not_downgrade_later_family_audit(workspace):
    suite = catalog.load_suite(SUITES / "tabarena_v01_inventory.json")
    spec = next(spec for spec in suite["datasets"] if spec["source"]["dataset_id"] == 46904)
    assert spec["role"] == "inventory" and spec["rights"]["status"] == "pending_review"
    assert spec["declared_prior_exposure"] == "metadata_only"
    assert spec["prior_exposure"] == "exposed"
    assert spec["_family"]["role"] == "development"
    assert spec["_family"]["exposure_history"]
    with pytest.raises(catalog.AdmissionError, match="must never download"):
        catalog.load_dataset(spec, workspace / "forbidden-cache")
    malformed_snapshot = copy.deepcopy(spec["_family"])
    malformed_snapshot.update(role="inventory", prior_exposure="metadata_only", aliases=["fabricated-origin"])
    with pytest.raises(catalog.AdmissionError, match="cannot override registry"):
        catalog._families([malformed_snapshot])


def test_openml_pinned_metadata_checksum_rejects_drift_before_payload(workspace, monkeypatch):
    spec, payloads = openml_fixture(workspace, monkeypatch)
    spec["source"]["md5_checksum"] = "a" * 32
    visited = []

    def fetch(url):
        visited.append(url)
        assert url.endswith(("/task/999001", "/data/999002"))
        return payloads[url]

    monkeypatch.setattr(catalog, "_fetch_bytes", fetch)
    with pytest.raises(catalog.AdmissionError, match="checksum changed since metadata admission"):
        catalog.load_dataset(spec, workspace / "raw-cache")
    assert len(visited) == 2


def test_openml_protocol_dimensions_are_locked(workspace, monkeypatch):
    spec, payloads = openml_fixture(workspace, monkeypatch)
    spec["source"]["md5_checksum"] = hashlib.md5(payloads["https://openml.org/fixture-data.arff"]).hexdigest()
    spec["source"]["official_split_dimensions"] = {"repeats": 2, "folds": 2, "samples": 1}
    cache = workspace / "raw-cache"
    loaded = catalog.load_dataset(spec, cache)
    assert len(loaded.official_splits) == len(loaded.metadata["official_split_ids"]) == 4
    spec["source"]["official_split_dimensions"]["repeats"] = 3
    with pytest.raises(catalog.AdmissionError, match="locked task protocol"):
        catalog.load_dataset(spec, cache)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("labels", ["unreviewed", "yes"], "reviewed class schema"),
        ("min_total", 5, "Total class support"),
        ("min_outer_train", 3, "Official split class support"),
        ("min_outer_test", 3, "Official split class support"),
    ],
)
def test_predeclared_class_support_margin_is_checked_before_models(workspace, monkeypatch, field, value, message):
    spec, _ = openml_fixture(workspace, monkeypatch)
    spec["class_support"] = {"labels": ["no", "yes"], "min_total": 4, "min_outer_train": 2, "min_outer_test": 2}
    loaded = catalog.load_dataset(spec, workspace / "raw-cache")
    assert loaded.y.value_counts().to_dict() == {0: 4, 1: 4}
    spec["class_support"][field] = value
    with pytest.raises(catalog.AdmissionError, match=message):
        catalog.load_dataset(spec, workspace / "raw-cache")


def test_malformed_admission_pins_are_rejected_without_acquisition(workspace, monkeypatch):
    spec, _ = openml_fixture(workspace, monkeypatch)
    spec["source"]["md5_checksum"] = "not-an-md5"
    with pytest.raises(catalog.AdmissionError, match="invalid upstream payload MD5"):
        catalog.load_dataset(spec, workspace / "forbidden-cache")
    del spec["source"]["md5_checksum"]
    spec["source"]["official_split_dimensions"] = {"repeats": True, "folds": 2, "samples": 1}
    with pytest.raises(catalog.AdmissionError, match="positive integer"):
        catalog.load_dataset(spec, workspace / "forbidden-cache")
    assert not (workspace / "forbidden-cache").exists()


def test_seoul_development_is_explicitly_retrospective_and_history_is_preserved(workspace):
    suite = catalog.load_suite(SUITES / "seoul_development.json")
    spec = suite["datasets"][0]
    assert spec["role"] == "development" and spec["family_id"] == "seoul-bike"
    assert spec["family_id"] != catalog._families([])["capital-bikeshare"]["family_id"]
    assert spec["target"] == "Rented Bike Count"
    assert spec["drop_columns"] == ["Functioning Day"]
    assert spec["time_column"] == "Date"
    assert spec.get("group_column") is None
    assert spec.get("time_hour_column") is None
    assert spec["time_format"] == "%d/%m/%Y"
    assert spec["split_policy"] == "temporal"
    assert not spec["time_policy"]["ahead_of_time_forecasting_approved"]
    assert spec["time_policy"]["label_end_times"] is None
    assert spec["source"]["time_audit"]["calendar_days"] == 365
    assert spec["source"]["expected_rows"] == 365 * 24
    original_path = SUITES / "seoul_inventory.json"
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == suite["historical_inventory"]["sha256"]
    original = catalog.load_suite(original_path)["datasets"][0]
    assert original["role"] == "inventory" and original["rights"]["status"] == "pending_review"
    with pytest.raises(catalog.AdmissionError, match="must never download"):
        catalog.load_dataset(original, workspace / "forbidden-cache")
    with pytest.raises(catalog.AdmissionError, match="row caps are forbidden"):
        catalog.load_dataset(spec, workspace / "forbidden-cache", max_rows=100)


def seoul_fixture(monkeypatch, first_hour=0, *, full_days=None):
    spec = catalog.load_suite(SUITES / "seoul_development.json")["datasets"][0]
    frame = pd.DataFrame(
        {
            "Date": ["01/12/2017", "01/12/2017", "02/12/2017", "02/12/2017"],
            "Rented Bike Count": [7, 9, 8, 10],
            "Hour": [first_hour, 23, 0, 23],
            "Temperature": [0.2, 1.0, 0.5, 0.7],
            "Humidity": [30, 20, 25, 35],
            "Wind speed": [0.8] * 4,
            "Visibility": [2000] * 4,
            "Dew point temperature": [-2.0] * 4,
            "Solar Radiation": [0.0] * 4,
            "Rainfall": [0.0] * 4,
            "Snowfall": [0.0] * 4,
            "Seasons": ["Winter"] * 4,
            "Holiday": ["No Holiday"] * 4,
            "Functioning Day": ["Yes", "No", "Yes", "No"],
        }
    )
    if full_days is not None:
        frame = pd.concat([frame.iloc[[0]]] * (full_days * 24), ignore_index=True)
        frame["Date"] = np.repeat(pd.date_range("2017-12-01", periods=full_days).strftime("%d/%m/%Y"), 24)
        frame["Hour"] = np.tile(np.arange(24), full_days)
        frame["Rented Bike Count"] = np.arange(len(frame)) + 1
    raw = frame.to_csv(index=False).encode()
    spec["source"].update(expected_rows=len(frame), sha256=hashlib.sha256(raw).hexdigest())
    spec["source"].pop("time_audit")
    metadata = {
        "data": {
            "uci_id": 560,
            "data_url": spec["source"]["data_url"],
            "variables": [
                {"name": name, "role": "Target" if name == "Functioning Day" else "Feature"} for name in frame
            ],
        }
    }
    monkeypatch.setattr(
        catalog, "_fetch_bytes", lambda url: json.dumps(metadata).encode() if "api/dataset" in url else raw
    )
    return spec, metadata


def test_reviewed_seoul_target_override_and_whole_day_bucket_policy(workspace, monkeypatch):
    spec, _ = seoul_fixture(monkeypatch)
    result = catalog.load_dataset(spec, workspace / "raw-cache")
    assert result.X.shape == (4, 11)
    assert result.y.tolist() == [7.0, 9.0, 8.0, 10.0]
    assert result.metadata["columns"][0] == "Hour"
    assert not {"Date", "Functioning Day", "Rented Bike Count"}.intersection(result.metadata["columns"])
    assert result.X.x0.tolist() == [0, 23, 0, 23]
    assert result.timestamps.dt.day.tolist() == [1, 1, 2, 2]
    assert result.timestamps.dt.hour.tolist() == [0, 0, 0, 0]
    assert result.groups is None
    assert result.X.index.equals(result.timestamps.index)
    assert result.label_end_times is None
    assert result.metadata["target_role_override"]["metadata_target_columns"] == ["Functioning Day"]
    cached = catalog.load_dataset(spec, workspace / "raw-cache")
    assert result.metadata == cached.metadata
    pd.testing.assert_series_equal(result.timestamps, cached.timestamps)


@pytest.mark.parametrize("first_hour", [-1, 24, 0.5, None, np.inf])
def test_calendar_hour_policy_rejects_invalid_offsets(workspace, monkeypatch, first_hour):
    spec, _ = seoul_fixture(monkeypatch, first_hour)
    spec["time_hour_column"] = "Hour"
    with pytest.raises(catalog.AdmissionError, match="Calendar-hour offsets"):
        catalog.load_dataset(spec, workspace / "raw-cache")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "pending_review"),
        ("selected_target", "Functioning Day"),
        ("metadata_target_columns", ["Rented Bike Count"]),
        ("metadata_selected_target_role", "Target"),
        ("rationale", ""),
        ("evidence", []),
    ],
)
def test_target_role_override_requires_a_coherent_explicit_review(workspace, monkeypatch, field, value):
    spec, _ = seoul_fixture(monkeypatch)
    spec["source"]["target_role_override"][field] = value
    with pytest.raises(catalog.AdmissionError):
        catalog.load_dataset(spec, workspace / "forbidden-cache")
    assert not (workspace / "forbidden-cache").exists()


def test_role_override_cannot_keep_old_target_predictors(workspace, monkeypatch):
    spec, _ = seoul_fixture(monkeypatch)
    spec["drop_columns"] = []
    with pytest.raises(catalog.AdmissionError, match="exclude all other native target columns"):
        catalog.load_dataset(spec, workspace / "forbidden-cache")
    spec["drop_columns"] = ["Functioning Day"]
    spec["source"]["target_columns"] = ["Rented Bike Count"]
    with pytest.raises(catalog.AdmissionError, match="preserve all metadata targets"):
        catalog.load_dataset(spec, workspace / "forbidden-cache")


@pytest.mark.parametrize("role", ["Target", "ID"])
def test_role_override_does_not_ignore_subsequent_metadata_drift(workspace, monkeypatch, role):
    spec, metadata = seoul_fixture(monkeypatch)
    selected = next(item for item in metadata["data"]["variables"] if item["name"] == spec["target"])
    selected["role"] = role
    with pytest.raises(catalog.AdmissionError, match="target-role metadata|selected-column role changed"):
        catalog.load_dataset(spec, workspace / "raw-cache")


def test_calendar_hour_column_requires_reviewed_time_and_predictor_fields(workspace, monkeypatch):
    spec, _ = seoul_fixture(monkeypatch)
    spec["time_hour_column"] = "Hour"
    del spec["time_column"]
    with pytest.raises(catalog.AdmissionError, match="time_hour_column requires"):
        catalog.load_dataset(spec, workspace / "forbidden-cache")
    spec["time_column"] = "Date"
    spec["time_hour_column"] = "Rented Bike Count"
    with pytest.raises(catalog.AdmissionError, match="reviewed predictor"):
        catalog.load_dataset(spec, workspace / "forbidden-cache")


def test_seoul_provider_integrates_with_strict_whole_day_forward_splits(workspace, monkeypatch):
    from featcopilot.utils.splitting import make_cv_splits, make_holdout_splits

    spec, _ = seoul_fixture(monkeypatch, full_days=40)
    result = catalog.load_dataset(spec, workspace / "raw-cache")
    assert result.groups is None and result.label_end_times is None
    assert result.timestamps.nunique() == 40
    assert result.timestamps.value_counts().eq(24).all()
    assert result.X.x0.tolist() == np.tile(np.arange(24), 40).tolist()

    def assert_complete_days(timestamps, pairs):
        for train, validation in pairs:
            train_times = timestamps.iloc[train]
            validation_times = timestamps.iloc[validation]
            assert train_times.max() < validation_times.min()
            assert set(train_times).isdisjoint(validation_times)
            assert train_times.value_counts().eq(24).all()
            assert validation_times.value_counts().eq(24).all()

    outer = make_cv_splits(
        result.y,
        "regression",
        3,
        2026,
        groups=result.groups,
        timestamps=result.timestamps,
        label_end_times=result.label_end_times,
    )
    assert len(outer) == 3
    assert_complete_days(result.timestamps, outer)
    for train, _ in outer:
        y = result.y.iloc[train].reset_index(drop=True)
        timestamps = result.timestamps.iloc[train].reset_index(drop=True)
        for splitter in (make_cv_splits, make_holdout_splits):
            inner = splitter(y, "regression", 3, 2026, timestamps=timestamps)
            assert_complete_days(timestamps, inner)


def test_confirmation_completion_repair_changes_only_two_categorical_declarations():
    parent_path = SUITES / "tabarena_confirmation_candidates_20260920.json"
    repair_path = SUITES / "tabarena_confirmation_completion_repair_20260921.json"
    with pytest.raises(catalog.AdmissionError, match="already exposed"):
        catalog.load_suite(repair_path)
    parent = historical_manifest_snapshot(parent_path)
    repair = historical_manifest_snapshot(repair_path)
    before = {spec["name"]: spec for spec in parent["datasets"]}
    assert {spec["name"] for spec in repair["datasets"]} == {"tabarena-46916", "tabarena-46937"}
    assert repair["family_count"] == 2
    assert repair["dataset_defaults"] == parent["dataset_defaults"]
    additions = {
        "tabarena-46916": {"avgAge", "contributionPrivateThirdPartyInsurance"},
        "tabarena-46937": {"coupon"},
    }
    for spec in repair["datasets"]:
        original = before[spec["name"]]
        assert {key: value for key, value in spec.items() if key != "categorical"} == {
            key: value for key, value in original.items() if key != "categorical"
        }
        assert set(spec["categorical"]) - set(original["categorical"]) == additions[spec["name"]]
        assert set(original["categorical"]).issubset(spec["categorical"])
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == repair["repair"]["original_suite_sha256"]


def test_completion_repair_preserves_experiment_and_records_status_only_lineage():
    parent = historical_manifest_snapshot(SUITES / "tabarena_confirmation_candidates_20260920.json")
    repair = historical_manifest_snapshot(SUITES / "tabarena_confirmation_completion_repair_20260921.json")
    before, after = parent["preregistration"], repair["preregistration"]
    for field in (
        "methods",
        "learners",
        "learner_search",
        "selection_parameters",
        "seeds",
        "sample_seed",
        "max_rows",
        "threads",
        "case_timeout_seconds",
        "split_policy",
        "expected_official_split_dimensions_per_task",
        "classification_admission_margin",
        "primary_metrics",
        "promotion_gates",
    ):
        assert after[field] == before[field]
    assert after["expected_official_splits"] == 2 * 30
    assert after["expected_cells"] == 2 * 30 * 2 * 4 == 480
    assert after["total_active_seconds_ceiling"] == 20 * 3600
    assert after["case_timeout_seconds"] == 600
    assert after["seeds"] == [2026]
    assert after["family_ids"] == repair["repair"]["repair_family_ids"]
    assert repair["repair"]["parent_run_id"] == "9d595eba9cb0866b6c14e031627cd81661f136ddc51327884888662f60b1cf94"
    assert repair["repair"]["runtime_python_changes"] is False
    assert repair["exposure_audit"]["selected_family_model_admission_rows"] == 0
    assert repair["exposure_audit"]["other_original_family_admission_rows"] == 4
    excluded = set(repair["repair"]["already_exposed_family_ids_not_to_rerun"])
    assert excluded == {"tabarena-46904", "tabarena-46954", "tabarena-46958", "tabarena-46969"}
    assert not excluded.intersection(after["family_ids"])
    assert excluded | set(after["family_ids"]) == set(before["family_ids"])
    failures = repair["repair"]["failure_evidence"]
    assert sum(item["dataset_error_cells"] for item in failures) == 480
    assert all(item["worker_attempts"] == 0 and not item["present_in_parent_prepared_datasets"] for item in failures)


def test_completion_repair_audits_all_nominal_and_string_predictors():
    repair = historical_manifest_snapshot(SUITES / "tabarena_confirmation_completion_repair_20260921.json")
    expected_counts = {"tabarena-46916": (85, 5, 80), "tabarena-46937": (24, 22, 2)}
    corrections = {item["family_id"]: item for item in repair["repair"]["schema_corrections"]}
    for spec in repair["datasets"]:
        columns, categorical, numeric = expected_counts[spec["family_id"]]
        evidence = corrections[spec["family_id"]]
        assert len(spec["source"]["columns"]) - 1 == columns
        assert len(spec["categorical"]) == categorical
        assert evidence["corrected_categorical_count"] == categorical
        assert evidence["numeric_predictors_preserved"] == numeric
        assert set(evidence["official_string_fields"]) == set(evidence["added_categorical"])
        assert set(evidence["official_string_fields"]).issubset(spec["categorical"])
        assert evidence["official_feature_metadata_sha256"] == spec["source"]["openml_features_sha256"]
        assert evidence["all_arff_attribute_types_match_official_metadata"]
        assert evidence["all_numeric_predictor_columns_parse_numerically"]
        assert len(evidence["arff_sha256"]) == len(evidence["official_splits_sha256"]) == 64


@pytest.mark.parametrize("string_columns", [["avgAge", "contributionPrivateThirdPartyInsurance"], ["coupon"]])
def test_openml_string_fields_need_explicit_complete_categorical_declarations(workspace, monkeypatch, string_columns):
    arff = pytest.importorskip("arff")
    spec, payloads = openml_fixture(workspace, monkeypatch)
    data_url = "https://openml.org/fixture-data.arff"
    description_url = "https://www.openml.org/api/v1/json/data/999002"
    parsed = arff.loads(payloads[data_url].decode())
    parsed["attributes"].extend((column, "STRING") for column in string_columns)
    for index, row in enumerate(parsed["data"]):
        row.extend(f"band-{column_index}-{index % 2}" for column_index in range(len(string_columns)))
    payloads[data_url] = arff.dumps(parsed).encode()
    description = json.loads(payloads[description_url])
    description["data_set_description"]["md5_checksum"] = hashlib.md5(payloads[data_url]).hexdigest()
    payloads[description_url] = json.dumps(description).encode()
    spec["source"]["columns"].extend(string_columns)
    for column in string_columns:
        with pytest.raises(catalog.AdmissionError, match=f"{column}: nonnumeric predictor"):
            catalog.load_dataset(spec, workspace / "raw-cache")
        spec["categorical"].append(column)
    loaded = catalog.load_dataset(spec, workspace / "raw-cache")
    assert loaded.X.shape == (8, 2 + len(string_columns))
    assert loaded.y.tolist() == [0, 1] * 4
    assert len(loaded.official_splits) == 4
    for column in string_columns:
        machine_name = next(key for key, value in loaded.metadata["column_mapping"].items() if value == column)
        assert isinstance(loaded.X[machine_name].dtype, pd.CategoricalDtype)


@pytest.mark.usefixtures("pre_release_registry")
def test_development_expansion_has_fifty_validated_origins_and_no_fresh_confirmation():
    suite = catalog.load_suite(SUITES / "development_expansion_20260922.json")
    summary = suite["admission_summary"]
    assert len(suite["datasets"]) == suite["family_count"] == summary["admitted_unique_origins"] == 50
    assert summary["observational_origins"] == 48
    assert summary["domain_simulation_origins"] == 2
    assert summary["synthetic_diagnostic_substitutes"] == summary["fresh_confirmation_origins"] == 0
    assert {spec["role"] for spec in suite["datasets"]} == {"development"}
    assert {spec["rights"]["status"] for spec in suite["datasets"]} == {"approved"}
    assert summary["split_policy_counts"] == {"official": 24, "iid": 21, "grouped": 2, "temporal": 3}
    assert summary["task_counts"] == {"classification": 34, "regression": 16}
    assert summary["official_outer_splits_total"] == 24 * 30
    for spec in suite["datasets"]:
        source, profile = spec["source"], spec["admission_profile"]
        assert source["schema_reviewed"] is True
        assert len(source["sha256"]) == 64 and source["sha256"] == profile["raw_sha256"]
        assert source["expected_rows"] == profile["shape"][0] == profile["profile"]["rows"]
        excluded = set(source["target_columns"]) | set(spec["drop_columns"])
        excluded.update(spec[key] for key in ("group_column", "time_column", "label_end_column") if spec.get(key))
        assert len(set(source["columns"]) - excluded) == profile["shape"][1]
        assert set(source["target_columns"]) - {spec["target"]} <= set(spec["drop_columns"])
        assert not set(spec["categorical"]).intersection(excluded)
        assert profile["model_calls"] == 0
        if spec["task"] == "classification":
            assert sum(profile["observed_class_counts"].values()) == profile["shape"][0]
            assert len(profile["observed_class_counts"]) == profile["profile"]["n_classes"]
        if spec["provider"] == "openml_task":
            assert len(source["splits_sha256"]) == 64
            assert source["official_split_dimensions"] == {"repeats": 10, "folds": 3, "samples": 1}
            assert "nominal AND string" in source["categorical_schema_rule"]
    assert sum(spec["admission_profile"]["profile"].get("n_classes", 0) > 2 for spec in suite["datasets"]) >= 10
    assert sum(spec["admission_profile"]["shape"][1] >= 50 for spec in suite["datasets"]) >= 10


@pytest.mark.usefixtures("pre_release_registry")
def test_expansion_retires_all_six_consumed_confirmation_origins():
    suite = catalog.load_suite(SUITES / "development_expansion_20260922.json")
    by_family = {spec["family_id"]: spec for spec in suite["datasets"]}
    consumed = {
        "tabarena-46904",
        "tabarena-46916",
        "tabarena-46937",
        "tabarena-46954",
        "tabarena-46958",
        "tabarena-46969",
    }
    for fid in consumed:
        spec = by_family[fid]
        assert spec["prior_exposure"] == "exposed"
        assert spec["_family"]["role"] == "development"
        assert spec["_family"]["exposure_history"]
        attempted = copy.deepcopy(spec)
        attempted.update(role="confirmation", prior_exposure="audited_unexposed")
        with pytest.raises(catalog.AdmissionError, match="already exposed"):
            catalog.validate_manifest({"name": "forbidden-reuse", "datasets": [attempted]}, allow_confirmation=True)


@pytest.mark.usefixtures("pre_release_registry")
def test_development_stress_does_not_inherit_confirmation_class_margin():
    suite = catalog.load_suite(SUITES / "development_expansion_20260922.json")
    by_family = {spec["family_id"]: spec for spec in suite["datasets"]}
    anneal = by_family["tabarena-46906"]
    assert min(anneal["admission_profile"]["observed_class_counts"].values()) == 8
    assert anneal["role"] == "development" and anneal["evaluation_track"] == "rare_class_stress"
    assert "class_support" not in anneal
    assert anneal["admission_profile"]["split_support"]["outer_splits"] == 30
    musk = by_family["musk-molecules"]
    assert musk["group_column"] == "molecule_name"
    assert musk["admission_profile"]["groups"] == 102
    assert musk["evaluation_track"] == "nested_cv_stress"
    assert musk["admission_profile"]["split_support"]["nested_split_check"]["failing_outer_folds"] == 2
    assert musk["admission_profile"]["profile"]["rows"] == 6598


@pytest.mark.usefixtures("pre_release_registry")
def test_expansion_declares_posttarget_and_other_target_exclusions():
    by_family = {
        spec["family_id"]: spec
        for spec in catalog.load_suite(SUITES / "development_expansion_20260922.json")["datasets"]
    }
    assert {"time"} <= set(by_family["heart-failure-clinical"]["drop_columns"])
    assert {"casual", "registered", "instant"} <= set(by_family["capital-bikeshare"]["drop_columns"])
    assert {"G1", "G2"} <= set(by_family["student-performance"]["drop_columns"])
    assert {"total_UPDRS", "subject#"} <= set(by_family["parkinsons-telemonitoring"]["drop_columns"])
    assert by_family["parkinsons-telemonitoring"]["group_column"] == "subject#"
    assert {"lights", "rv1", "rv2"} <= set(by_family["appliances-energy"]["drop_columns"])
    assert by_family["appliances-energy"]["time_format"] == "%Y-%m-%d%H:%M:%S"
    mic = by_family["tabarena-46980"]
    assert mic["variant"] == "donor_documented_admission_time"
    assert len(mic["drop_columns"]) == 9
    assert mic["admission_profile"]["shape"][1] == 102
    students = by_family["tabarena-46960"]
    assert students["variant"] == "conservative_enrollment_record"
    assert sum(name.startswith("Curricular_units_") for name in students["drop_columns"]) == 12
    assert students["admission_profile"]["shape"][1] == 18


@pytest.mark.usefixtures("pre_release_registry")
def test_expansion_private_grants_and_pending_inventory_remain_distinct(workspace):
    suite = catalog.load_suite(SUITES / "development_expansion_20260922.json")
    by_family = {spec["family_id"]: spec for spec in suite["datasets"]}
    aps = by_family["tabarena-46908"]
    assert aps["rights"]["license"] == "GPL-3.0-or-later"
    assert aps["rights"]["redistribution_approved"] is False
    assert "https://www.gnu.org/licenses/gpl-3.0.html#section2" in aps["rights"]["evidence"]
    assert len(aps["rights"]["gpl_primary_text_sha256"]) == 64
    assert aps["admission_profile"]["shape"] == [76000, 170]
    microwave = by_family["tabarena-46930"]
    assert microwave["rights"]["redistribution_approved"] is False
    assert "45538" in microwave["rights"]["evidence"][0]
    fiat = by_family["tabarena-46907"]
    assert fiat["rights"]["license"] == "CC0-1.0"
    assert "api/v1/datasets/view/paolocons/" in fiat["rights"]["evidence"][1]
    audit = catalog.load_suite(SUITES / "development_expansion_audit_20260922.json")
    assert len(audit["datasets"]) == audit["family_count"] == 74
    assert Counter(spec["review_status"] for spec in audit["datasets"]) == {
        "admitted_origin": 50,
        "pending_review": 24,
    }
    assert audit["inventory_summary"]["confirmation_candidates_allocated"] == 0
    assert audit["inventory_summary"]["tabarena_origins_admitted"] == 27
    assert len(audit["native_resource_holds"]) == 3
    assert not any(spec["family_id"] == "tabarena-46952" for spec in suite["datasets"])
    pending = next(spec for spec in audit["datasets"] if spec["family_id"] == "tabarena-46952")
    assert pending["hold_category"] == "origin_independence"
    for spec in audit["datasets"]:
        with pytest.raises(catalog.AdmissionError, match="must never download"):
            catalog.load_dataset(spec, workspace / "inventory-must-not-download")
    assert not (workspace / "inventory-must-not-download").exists()


@pytest.mark.usefixtures("pre_release_registry")
def test_expansion_preserves_every_existing_manifest_byte_for_byte():
    audit = catalog.load_suite(SUITES / "development_expansion_audit_20260922.json")
    hashes = audit["preserved_historical_manifest_sha256"]
    assert len(hashes) == 7
    for name, expected in hashes.items():
        assert hashlib.sha256((SUITES / name).read_bytes()).hexdigest() == expected


def test_working_development_declarations_follow_later_exposure_without_rewriting_history():
    suite = catalog.load_suite(SUITES / "uci_development.json")
    assert all(spec["prior_exposure"] == "exposed" for spec in suite["datasets"])
    assert all(spec["declared_prior_exposure"] in {"metadata_only", "loader_only"} for spec in suite["datasets"])
    assert all(spec["role"] == "development" for spec in suite["datasets"])


def test_static_legacy_exposure_also_overrides_stale_inventory(workspace):
    record = copy.deepcopy(catalog._families([])["higgs"])
    record.update(role="inventory", prior_exposure="metadata_only")
    spec = local_spec(workspace)
    spec.update(name="higgs", family_id="higgs", role="inventory", prior_exposure="metadata_only")
    spec["rights"]["status"] = "pending_review"
    resolved = catalog.validate_manifest({"name": "legacy-inventory", "families": [record], "datasets": [spec]})
    assert resolved["datasets"][0]["prior_exposure"] == "exposed"
    assert resolved["datasets"][0]["declared_prior_exposure"] == "metadata_only"
    assert resolved["families"][0]["prior_exposure"] == "exposed"


def shared_cache_uci_fixture(workspace, monkeypatch):
    raw = b"measurement,kind,y\n1,NA,no\n2,B,yes\n3,A,no\n4,B,yes\n"
    spec = local_spec(workspace, raw)
    spec["provider"] = "uci"
    spec["rights"]["access"] = "public"
    spec["source"].pop("path")
    spec["source"].update(uci_id=999001, data_url="https://archive.ics.uci.edu/static/public/999001/data.csv")
    metadata_url = "https://archive.ics.uci.edu/api/dataset?id=999001"
    metadata = {
        "data": {
            "uci_id": 999001,
            "data_url": spec["source"]["data_url"],
            "variables": [
                {"name": name, "role": "Target" if name == "y" else "Feature"} for name in spec["source"]["columns"]
            ],
        }
    }
    payloads = {metadata_url: json.dumps(metadata).encode(), spec["source"]["data_url"]: raw}
    calls = []

    def fetch(url):
        calls.append(url)
        return payloads[url]

    monkeypatch.setattr(catalog, "_fetch_bytes", fetch)
    return normalize(spec), payloads, calls


def test_cache_namespace_normalizes_source_order_and_parser_defaults(workspace):
    first = normalize(local_spec(workspace))
    second = copy.deepcopy(first)
    second["source"] = dict(reversed(list(second["source"].items())))
    second["source"].update(
        encoding="utf-8-sig", separator=",", na_values=["?"], keep_default_na=True, skipinitialspace=False
    )
    cache = workspace / "cache"
    assert catalog._cache_directory(first, cache) == catalog._cache_directory(second, cache)
    assert catalog._source_cache_identity(first) == catalog._source_cache_identity(second)
    one = catalog.load_dataset(first, cache)
    two = catalog.load_dataset(second, cache)
    pd.testing.assert_frame_equal(one.X, two.X)
    assert one.metadata["source"] == first["source"]
    assert two.metadata["source"] == second["source"]
    assert one.metadata != two.metadata  # Explicit manifest provenance still participates in resume identity.
    second["source"]["keep_default_na"] = False
    assert catalog._cache_directory(first, cache) != catalog._cache_directory(second, cache)


def test_legacy_and_pinned_parser_versions_coexist_without_rewriting_pins(workspace, monkeypatch):
    old, payloads, calls = shared_cache_uci_fixture(workspace, monkeypatch)
    old["source"].pop("sha256")
    cache = workspace / "shared-cache"
    legacy = cache / old["name"]
    raw = payloads[old["source"]["data_url"]]
    metadata_url = "https://archive.ics.uci.edu/api/dataset?id=999001"
    catalog._cached_bytes(legacy, "metadata.json", {"url": metadata_url}, lambda: payloads[metadata_url])
    catalog._cached_bytes(legacy, "data.csv", {"provider": "uci", "source": old["source"]}, lambda: raw)
    preserved = {path.name: path.read_bytes() for path in legacy.iterdir() if path.is_file()}
    expanded = copy.deepcopy(old)
    expanded["source"].update(sha256=hashlib.sha256(raw).hexdigest(), keep_default_na=False, expected_rows=4)
    first = catalog.load_dataset(old, cache)
    second = catalog.load_dataset(expanded, cache)
    assert calls == []
    assert pd.isna(first.X.iloc[0, 1]) and second.X.iloc[0, 1] == "NA"
    assert catalog._cache_directory(old, cache) != catalog._cache_directory(expanded, cache)
    assert all((legacy / name).read_bytes() == content for name, content in preserved.items())
    pd.testing.assert_frame_equal(catalog.load_dataset(old, cache).X, first.X)
    assert catalog.load_dataset(expanded, cache).metadata == second.metadata
    (catalog._cache_directory(expanded, cache) / "data.csv").write_bytes(b"corrupt new namespace")
    with pytest.raises(catalog.AcquisitionError, match="corrupt cache"):
        catalog.load_dataset(expanded, cache)
    pd.testing.assert_frame_equal(catalog.load_dataset(old, cache).X, first.X)
    (catalog._cache_directory(old, cache) / "data.csv").write_bytes(b"corrupt old namespace")
    with pytest.raises(catalog.AcquisitionError, match="corrupt cache"):
        catalog.load_dataset(old, cache)
    assert calls == []
    assert all((legacy / name).read_bytes() == content for name, content in preserved.items())


def test_corrupt_legacy_bytes_are_not_silently_reused_or_refetched(workspace, monkeypatch):
    spec, payloads, calls = shared_cache_uci_fixture(workspace, monkeypatch)
    cache = workspace / "cache"
    legacy = cache / spec["name"]
    raw = payloads[spec["source"]["data_url"]]
    catalog._cached_bytes(legacy, "data.csv", {"provider": "uci", "source": spec["source"]}, lambda: raw)
    pin = (legacy / "data.csv.pin.json").read_bytes()
    (legacy / "data.csv").write_bytes(b"corrupted approved raw bytes")
    with pytest.raises(catalog.AcquisitionError, match="corrupt cache at reuse source"):
        catalog.load_dataset(spec, cache)
    assert spec["source"]["data_url"] not in calls
    assert (legacy / "data.csv.pin.json").read_bytes() == pin
    assert (legacy / "data.csv").read_bytes() == b"corrupted approved raw bytes"
    assert not (catalog._cache_directory(spec, cache) / "data.csv").exists()


def test_same_name_different_payload_pins_coexist_and_wrong_checksum_fails(workspace, monkeypatch):
    first, payloads, calls = shared_cache_uci_fixture(workspace, monkeypatch)
    cache = workspace / "cache"
    original = catalog.load_dataset(first, cache)
    second = copy.deepcopy(first)
    data_url = first["source"]["data_url"]
    payloads[data_url] = payloads[data_url].replace(b"\n1,NA,", b"\n101,NA,")
    second["source"]["sha256"] = hashlib.sha256(payloads[data_url]).hexdigest()
    updated = catalog.load_dataset(second, cache)
    assert original.X.x0.iloc[0] == 1 and updated.X.x0.iloc[0] == 101
    assert catalog._cache_directory(first, cache) != catalog._cache_directory(second, cache)
    assert calls.count(data_url) == 2
    pd.testing.assert_frame_equal(catalog.load_dataset(first, cache).X, original.X)
    before = catalog.load_dataset(second, cache).metadata
    invalid = copy.deepcopy(second)
    invalid["source"]["sha256"] = "0" * 64
    with pytest.raises(catalog.AcquisitionError, match="SHA-256 mismatch"):
        catalog.load_dataset(invalid, cache)
    assert not (catalog._cache_directory(invalid, cache) / "data.csv").exists()
    assert catalog.load_dataset(second, cache).metadata == before
    (catalog._cache_directory(first, cache) / "data.csv").write_bytes(b"corrupt first version")
    with pytest.raises(catalog.AcquisitionError, match="corrupt cache"):
        catalog.load_dataset(first, cache)
    pd.testing.assert_frame_equal(catalog.load_dataset(second, cache).X, updated.X)


def test_parser_namespace_reuses_verified_sibling_bytes_without_download(workspace, monkeypatch):
    spec, payloads, calls = shared_cache_uci_fixture(workspace, monkeypatch)
    cache = workspace / "cache"
    first = catalog.load_dataset(spec, cache)
    variant = copy.deepcopy(spec)
    variant["source"]["keep_default_na"] = False
    second = catalog.load_dataset(variant, cache)
    assert len(calls) == 2
    assert not (cache / spec["name"] / "data.csv").exists()
    assert pd.isna(first.X.iloc[0, 1]) and second.X.iloc[0, 1] == "NA"
    for item in (spec, variant):
        directory = catalog._cache_directory(item, cache)
        pin = json.loads((directory / "data.csv.pin.json").read_bytes())
        assert pin["identity"] == catalog._source_cache_identity(item)
        assert pin["sha256"] == hashlib.sha256((directory / "data.csv").read_bytes()).hexdigest()


def test_unpinned_changed_declarations_do_not_borrow_another_source_version(workspace, monkeypatch):
    spec, payloads, calls = shared_cache_uci_fixture(workspace, monkeypatch)
    spec["source"].pop("sha256")
    cache = workspace / "cache"
    catalog.load_dataset(spec, cache)
    changed = copy.deepcopy(spec)
    changed["source"]["keep_default_na"] = False
    catalog.load_dataset(changed, cache)
    assert calls.count(spec["source"]["data_url"]) == 2


@pytest.mark.usefixtures("pre_release_registry")
def test_candidate_reservation_is_inventory_only_and_has_no_algorithm_settings(workspace):
    suite = catalog.load_suite(SUITES / "confirmation_candidate_reservation_20260923.json")
    assert suite["family_count"] == len(suite["datasets"]) == 12
    assert suite["candidate_counts"] == {
        "total": 12,
        "primary_observational_candidates": 11,
        "secondary_domain_simulation_candidates": 1,
    }
    assert suite["model_execution_authorized"] is False
    assert suite["algorithm_settings_frozen"] is False
    assert not {"preregistration", "methods", "learners", "model_settings", "selection_parameters"}.intersection(suite)
    for spec in suite["datasets"]:
        assert spec["role"] == "inventory"
        assert spec["reservation_role"] == "confirmation_candidate"
        assert spec["prior_exposure"] not in {"exposed", "audited_unexposed"}
        assert spec["_family"]["reservation"]["status"] == "active"
        assert spec["split_policy"] in {"iid", "official"}
        assert spec["reservation_metadata"]["rows"] <= 50000
        assert spec["reservation_metadata"]["predictors"] <= 200
        if spec["task"] == "classification":
            assert min(spec["reservation_metadata"]["observed_class_counts"].values()) >= 30
        with pytest.raises(catalog.AdmissionError, match="must never download"):
            catalog.load_dataset(spec, workspace / "forbidden-reservation-cache")
    assert not (workspace / "forbidden-reservation-cache").exists()


@pytest.mark.usefixtures("pre_release_registry")
def test_candidate_reservation_recomputes_fixed_filters_and_selection_order():
    suite = catalog.load_suite(SUITES / "confirmation_candidate_reservation_20260923.json")
    audit = catalog.load_suite(SUITES / "confirmation_candidate_reservation_audit_20260923.json")
    pool = catalog.load_suite(SUITES / "development_expansion_20260922.json")
    proof, filters = audit["exposure_proof"], audit["metadata_filters"]
    used = set(proof["registry_exposed_family_ids_before"])
    used.update(row["family_id"] for row in proof["ledger"]["rows"])
    used.update(row["resolved_family_id"] for row in proof["historical_identifiers_only"] if row["resolved_family_id"])
    decisions = {row["family_id"]: row for row in audit["decisions"]}
    assert len(decisions) == len(audit["datasets"]) == 50
    order = ["binary", "multiclass", "regression"]
    eligible = {kind: [] for kind in order}
    for spec in pool["datasets"]:
        fid, profile = spec["family_id"], spec["admission_profile"]
        counts = profile.get("observed_class_counts", {})
        stratum = "regression" if spec["task"] == "regression" else "binary" if len(counts) == 2 else "multiclass"
        reasons = []
        if fid in used:
            reasons.append("recorded_family_exposure")
        if spec["split_policy"] not in {"iid", "official"}:
            reasons.append("not_iid_or_official")
        if spec["rights"]["status"] != "approved" or not spec["source"]["schema_reviewed"]:
            reasons.append("rights_or_schema_not_cleared")
        if profile["shape"][0] > 50000:
            reasons.append("rows_above_50000")
        if profile["shape"][1] > 200:
            reasons.append("predictors_above_200")
        if spec["task"] == "classification" and (not counts or min(counts.values()) < 30):
            reasons.append("class_count_below_30")
        if fid in filters["toy_image_panel_exclusions"]:
            reasons.append("toy_or_image_only_panel")
        assert decisions[fid]["reasons"] == reasons
        assert decisions[fid]["eligible"] == (not reasons)
        assert decisions[fid]["stratum"] == stratum
        if not reasons:
            eligible[stratum].append(fid)
    eligible = {kind: sorted(values) for kind, values in eligible.items()}
    positions = dict.fromkeys(order, 0)
    selected = []
    for _ in range(4):
        for kind in order:
            if positions[kind] < len(eligible[kind]):
                selected.append(eligible[kind][positions[kind]])
                positions[kind] += 1
    while len(selected) < 12:
        previous = len(selected)
        for kind in order:
            if len(selected) < 12 and positions[kind] < len(eligible[kind]):
                selected.append(eligible[kind][positions[kind]])
                positions[kind] += 1
        if len(selected) == previous:
            break
    policy = suite["selection_policy"]
    assert policy["eligible_family_ids_by_stratum"] == eligible
    assert policy["fixed_selected_order"] == selected == [spec["family_id"] for spec in suite["datasets"]]
    assert policy["actual_stratum_counts"] == positions == {"binary": 6, "multiclass": 3, "regression": 3}
    assert policy["initial_quotas"] == {"binary": 4, "multiclass": 4, "regression": 4}
    assert policy["redistributed_slots"] == {"binary": 2, "multiclass": 0, "regression": 0}
    assert audit["summary"]["eligible_origins"] == 12
    assert audit["summary"]["excluded_origins"] == 38
    assert audit["summary"]["eligible_unselected_origins"] == 0


@pytest.mark.usefixtures("pre_release_registry")
def test_candidate_exposure_proof_excludes_recent_development_and_old_confirmation():
    suite = catalog.load_suite(SUITES / "confirmation_candidate_reservation_20260923.json")
    selected = {spec["family_id"] for spec in suite["datasets"]}
    proof = suite["exposure_proof"]
    recent = set(proof["current_development_family_ids"])
    previous = set(proof["previous_confirmation_families_exposed"])
    assert len(recent) == 16 and len(previous) == 6
    assert not selected.intersection(recent | previous)
    assert proof["ledger"]["row_count"] == 39
    rows = [[row["family_id"], row["run_id"], row["role"], row["created_at"]] for row in proof["ledger"]["rows"]]
    assert (
        hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
        == proof["ledger"]["logical_rows_sha256"]
    )
    assert not selected.intersection(row[0] for row in rows)
    assert proof["ledger"]["selected_family_matches"] == []
    registry = catalog._families([])
    assert all(registry[fid]["prior_exposure"] == "exposed" for fid in recent | previous)
    for item in proof["historical_cache_inputs"]:
        assert hashlib.sha256((ROOT / Path(item["path"])).read_bytes()).hexdigest() == item["sha256"]


@pytest.mark.usefixtures("pre_release_registry")
def test_reservation_blocks_old_development_manifests_and_stale_family_snapshots(workspace):
    reservation = catalog.load_suite(SUITES / "confirmation_candidate_reservation_20260923.json")
    pool = catalog.load_suite(SUITES / "development_expansion_20260922.json")
    selected = {row["family_id"] for row in reservation["datasets"]}
    for spec in pool["datasets"]:
        if spec["family_id"] not in selected:
            continue
        assert spec["role"] == "development"  # Old source manifest is not rewritten.
        stale = copy.deepcopy(spec)
        stale["_family"].pop("reservation")
        stale["_allow_confirmation"] = True
        with pytest.raises(catalog.AdmissionError, match="reserved as a confirmation candidate"):
            catalog.load_dataset(stale, workspace / "forbidden-old-manifest-cache")
        tampered = copy.deepcopy(spec)
        tampered["_family"]["reservation"]["status"] = "released"
        with pytest.raises(catalog.AdmissionError, match="reserved as a confirmation candidate"):
            catalog.load_dataset(tampered, workspace / "forbidden-old-manifest-cache")
    assert not (workspace / "forbidden-old-manifest-cache").exists()


@pytest.mark.usefixtures("pre_release_registry")
def test_reservation_preserves_sources_and_all_prior_manifests():
    reservation = catalog.load_suite(SUITES / "confirmation_candidate_reservation_20260923.json")
    audit = catalog.load_suite(SUITES / "confirmation_candidate_reservation_audit_20260923.json")
    original = {
        row["family_id"]: row
        for row in historical_manifest_snapshot(SUITES / "development_expansion_20260922.json")["datasets"]
    }
    for candidate in reservation["datasets"]:
        source = original[candidate["family_id"]]
        for field in ("source", "rights", "target", "categorical", "drop_columns", "task", "split_policy"):
            assert candidate[field] == source[field]
    for name, expected in audit["preserved_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / name).read_bytes()).hexdigest() == expected
    assert audit["summary"]["model_calls"] == audit["summary"]["payload_reads"] == 0


@pytest.mark.usefixtures("pre_release_registry")
def test_final_reserved_draft_stays_nonexecutable_and_preserves_reserved_schemas(workspace):
    draft = catalog.load_suite(SUITES / "final_reserved_confirmation_draft_20260924.json")
    reservation = catalog.load_suite(SUITES / "confirmation_candidate_reservation_20260923.json")
    assert draft["release_state"]["status"] == "awaiting_explicit_parent_authorization"
    assert draft["release_state"]["model_execution_authorized"] is False
    assert draft["release_state"]["registry_modified_by_this_draft"] is False
    assert draft["family_count"] == reservation["family_count"] == 12
    assert draft["reservation_selection_unchanged"] == reservation["selection_policy"]
    assert draft["candidate_counts"] == reservation["candidate_counts"]
    for proposed, reserved in zip(draft["datasets"], reservation["datasets"], strict=True):
        assert proposed["family_id"] == reserved["family_id"]
        for key in (
            "source",
            "target",
            "target_transform",
            "task",
            "categorical",
            "drop_columns",
            "split_policy",
            "rights",
            "claim_track",
        ):
            assert proposed[key] == reserved[key]
        assert proposed["role"] == "inventory"
        assert proposed["intended_role_after_parent_release"] == "confirmation"
        assert proposed["_family"]["reservation"]["status"] == "active"
        with pytest.raises(catalog.AdmissionError, match="must never download"):
            catalog.load_dataset(proposed, workspace / "draft-must-not-acquire")
    assert not (workspace / "draft-must-not-acquire").exists()


@pytest.mark.usefixtures("pre_release_registry")
def test_final_reserved_draft_split_plan_has_exact_fixed_ids_and_128_cells():
    draft = catalog.load_suite(SUITES / "final_reserved_confirmation_draft_20260924.json")
    protocol = draft["preregistration"]
    assert protocol["family_ids"] == [s["family_id"] for s in draft["datasets"]]
    assert set(protocol["split_plan"]) == set(protocol["family_ids"])
    total = 0
    policies = Counter()
    for spec in draft["datasets"]:
        plan = protocol["split_plan"][spec["family_id"]]
        policies[plan["policy"]] += 1
        if spec["provider"] == "openml_task":
            assert plan == {"policy": "official", "official_split_ids": [{"repeat": 0, "fold": 0, "sample": 0}]}
            assert spec["source"]["official_split_dimensions"] == {"repeats": 10, "folds": 3, "samples": 1}
            total += len(plan["official_split_ids"])
        else:
            assert plan == {"policy": "iid", "folds": 3}
            total += plan["folds"]
    assert policies == {"official": 10, "iid": 2}
    assert total == protocol["expected_total_splits"] == 16
    assert protocol["expected_official_splits"] == 10
    assert protocol["expected_iid_splits"] == 6
    assert total * len(protocol["methods"]) * len(protocol["learners"]) * len(protocol["seeds"]) == 128
    assert protocol["expected_cells"] == 128
    assert protocol["expected_primary_observational_cells"] == 120
    assert protocol["expected_secondary_simulation_cells"] == 8


@pytest.mark.usefixtures("pre_release_registry")
def test_final_reserved_draft_locks_one_candidate_and_native_reference_policies():
    protocol = catalog.load_suite(SUITES / "final_reserved_confirmation_draft_20260924.json")["preregistration"]
    assert protocol["methods"] == [
        "baseline",
        "featcopilot-importance-paired",
        "openfe-inductive-native",
        "autofeat-native",
    ]
    assert protocol["learners"] == ["linear", "lightgbm"]
    assert protocol["selection_parameters"] == {
        "max_candidates": 1000,
        "max_added_features": 50,
        "stability_repeats": 3,
        "screening": "auto",
    }
    assert protocol["learner_search"]["linear"] == {
        "regularization": [0.1, 1.0, 10.0],
        "selection_estimator_regularization": 1.0,
    }
    assert protocol["learner_search"]["lightgbm"] == {
        "num_leaves": [7, 15, 31],
        "selection_estimator_num_leaves": 15,
    }
    assert protocol["seeds"] == [2026] and protocol["sample_seed"] == 2026
    assert protocol["folds"] == 3 and protocol["max_rows"] == 0
    assert protocol["threads"] == 8 and protocol["memory_limit_mb"] == 4096
    assert protocol["case_timeout_seconds"] == 900 and protocol["total_active_seconds_ceiling"] == 136800
    assert not protocol["synthetic"] and not protocol["live_llm"] and protocol["llm_ledger"] is None
    methods = protocol["method_semantics"]
    candidate = methods["featcopilot-importance-paired"]
    assert candidate["candidate_set"] == "extended"
    assert candidate["screening"] == "importance" and candidate["global_screening_config"] == "auto"
    assert candidate["selection_rule"] == "paired_gain" and candidate["stability_repeats"] == 3
    for reference in ("openfe-inductive-native", "autofeat-native"):
        assert methods[reference]["max_added_features"] is None
    assert methods["autofeat-native"]["max_gb"] is None
    assert methods["autofeat-native"]["feateng_steps"] == 2
    assert methods["autofeat-native"]["featsel_runs"] == 5
    assert protocol["budget_accounting"]["maximum_cumulative_hours"] <= 72
    assert protocol["budget_accounting"]["new_ceiling_hours"] == 38


@pytest.mark.usefixtures("pre_release_registry")
def test_final_reserved_draft_class_constraints_and_string_schema_audit():
    draft = catalog.load_suite(SUITES / "final_reserved_confirmation_draft_20260924.json")
    audit = catalog.load_suite(SUITES / "final_reserved_confirmation_audit_20260924.json")
    assert audit["summary"]["smallest_total_class_count"] == 96
    assert audit["summary"]["smallest_previously_verified_outer_train_class_count"] == 64
    assert audit["summary"]["smallest_previously_verified_outer_test_class_count"] == 32
    assert draft["preregistration"]["classification_admission_margin"] == {
        "min_total": 30,
        "min_outer_train": 20,
        "min_outer_test": 10,
    }
    by_family = {spec["family_id"]: spec for spec in draft["datasets"]}
    for reviewed in audit["per_family"]:
        spec = by_family[reviewed["family_id"]]
        assert reviewed["categorical_columns"] == spec["categorical"]
        assert set(reviewed["string_predictors_included_as_categorical"]) <= set(spec["categorical"])
        assert reviewed["reservation_status"] == "active" and reviewed["ledger_matches"] == []
        assert reviewed["planned_cells"] == reviewed["planned_split_count"] * 4 * 2
        if reviewed["stratum"] != "regression":
            assert reviewed["minimum_total_class_count"] >= 30
            assert reviewed["minimum_previously_verified_outer_train_class_count"] >= 20
            assert reviewed["minimum_previously_verified_outer_test_class_count"] >= 10
    seismic = next(row for row in audit["per_family"] if row["family_id"] == "tabarena-46956")
    assert seismic["string_predictors_included_as_categorical"] == ["ghazard"]
    assert audit["summary"]["payload_reads"] == audit["summary"]["model_calls"] == 0


@pytest.mark.usefixtures("pre_release_registry")
def test_final_reserved_draft_has_explicit_nonautomatic_complete_pair_claim_gate():
    draft = catalog.load_suite(SUITES / "final_reserved_confirmation_draft_20260924.json")
    claims = draft["preregistration"]["decision_and_reporting_plan"]
    assert len(claims["primary_family_ids"]) == 11
    assert claims["secondary_simulation_family_ids"] == ["tabarena-46949"]
    assert set(claims["primary_family_ids"]).isdisjoint(claims["secondary_simulation_family_ids"])
    assert set(claims["primary_gate_strata"]) == {
        "classification/linear",
        "classification/lightgbm",
        "regression/linear",
        "regression/lightgbm",
    }
    assert len(claims["primary_gate_strata"]["classification/linear"]) == 9
    assert len(claims["primary_gate_strata"]["regression/linear"]) == 2
    assert claims["win_tie_loss"]["win"] == "G > 0.01"
    assert claims["win_tie_loss"]["tie"] == "-0.01 <= G <= 0.01"
    assert claims["family_harm"]["threshold_fraction"] == 0.05
    gate = claims["decision_gate"]
    assert gate["all_conditions_required"] is True
    assert gate["numeric_criteria"] == {
        "minimum_mean_relative_improvement_vs_raw": 0.0,
        "raw_mean_comparison": "strictly_greater",
        "maximum_family_relative_harm_vs_raw": 0.05,
        "family_harm_comparison": "less_than_or_equal",
        "minimum_mean_relative_improvement_vs_each_strong_reference": 0.0,
        "reference_mean_comparison": "greater_than_or_equal",
        "required_successful_finite_cell_fraction": 1.0,
        "required_planned_cells": 128,
    }
    assert "strictly > 0" in gate["mean_vs_raw"]
    assert ">5%" in gate["family_harm"]
    assert "EACH" in gate["strong_reference_noninferiority"] and ">= 0" in gate["strong_reference_noninferiority"]
    assert "All 128" in gate["completion"] and "blocks broad" in gate["completion"]
    assert gate["automatic_promotion"] is False and gate["automatic_sota_claim"] is False
    assert "Descriptive only" in claims["intervals"]["interpretation"]
    assert claims["intervals"]["seed"] == 2026
    assert any("NOT the full" in limit for limit in claims["scope_limits"])


@pytest.mark.usefixtures("pre_release_registry")
def test_final_reserved_draft_preserves_prior_manifests_registry_and_exposure_evidence():
    audit = catalog.load_suite(SUITES / "final_reserved_confirmation_audit_20260924.json")
    for name, checksum in audit["preserved_prior_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / name).read_bytes()).hexdigest() == checksum
    assert hashlib.sha256(catalog._REGISTRY.read_bytes()).hexdigest() == audit["evidence"]["registry_sha256"]
    ledger = audit["evidence"]["current_ledger"]
    rows = [[row["family_id"], row["run_id"], row["role"], row["created_at"]] for row in ledger["rows"]]
    assert len(rows) == ledger["row_count"] == 71
    assert hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == ledger["logical_rows_sha256"]
    assert ledger["selected_family_matches"] == []
    selected = {spec["family_id"] for spec in audit["datasets"]}
    assert not selected.intersection(row[0] for row in rows)


@pytest.mark.usefixtures("pre_release_registry")
def test_final_draft_explicitly_discloses_heart_failure_and_student_endpoint_timing():
    draft = catalog.load_suite(SUITES / "final_reserved_confirmation_draft_20260924.json")
    audit = catalog.load_suite(SUITES / "final_reserved_confirmation_audit_20260924.json")
    rows = {spec["family_id"]: spec for spec in draft["datasets"]}
    audited = {spec["family_id"]: spec for spec in audit["datasets"]}
    heart = rows["heart-failure-clinical"]
    assert heart["target"] == "death_event" and heart["drop_columns"] == ["time"]
    timing = heart["target_timing_review"]
    assert timing["schema_changed"] is False and timing["excluded_followup_column"] == "time"
    assert timing["excluded_followup_units"] == "days"
    assert "Retrospective" in timing["task"] and "NOT" in timing["task"]
    assert "common baseline" in timing["predictor_timing_limit"]
    assert "No fixed prediction horizon" in timing["survival_analysis_limit"]
    assert "not indefinite survival" in timing["endpoint"]
    student = rows["student-performance"]
    assert student["target"] == "G3" and student["drop_columns"] == ["G1", "G2"]
    timing = student["target_timing_review"]
    assert timing["schema_changed"] is False and timing["prior_grade_predictors_included"] is False
    assert timing["excluded_prior_grade_columns"] == ["G1", "G2"]
    assert timing["prior_grade_definitions"] == {"G1": "first-period grade", "G2": "second-period grade"}
    assert "third period" in timing["endpoint"] and "0-20" in timing["endpoint"]
    assert "absences" in timing["predictor_timing_limit"]
    assert "No enrollment-time" in timing["predictor_timing_limit"]
    for fid in ("heart-failure-clinical", "student-performance"):
        assert rows[fid]["target_timing_review"] == audited[fid]["target_timing_review"]
    review = audit["additional_preacquisition_timing_review"]
    assert not review["schema_changes"] and not review["method_or_split_changes"] and not review["runtime_code_changes"]


@pytest.mark.usefixtures("pre_release_registry")
def test_final_draft_matches_executor_schema_without_releasing_any_reservation():
    from benchmarks.execution import bind_preregistration

    draft = catalog.load_suite(SUITES / "final_reserved_confirmation_draft_20260924.json")
    protocol = draft["preregistration"]
    config = {
        **protocol["selection_parameters"],
        **{
            key: protocol[key]
            for key in ("methods", "learners", "seeds", "folds", "max_rows", "threads", "memory_limit_mb")
        },
        "case_timeout": protocol["case_timeout_seconds"],
        "synthetic": False,
        "live_llm": False,
        "llm_ledger": None,
        "allow_confirmation": True,
    }
    wrapper = {"manifests": [draft]}
    assert bind_preregistration(wrapper, draft["datasets"], config, 136800) is None
    # Validate only the proposed confirmation schema, not catalog admission or acquisition.
    projected = [{**spec, "role": "confirmation"} for spec in draft["datasets"]]
    assert bind_preregistration(wrapper, projected, config, 136800) == protocol
    with pytest.raises(ValueError, match="memory_limit_mb"):
        bind_preregistration(wrapper, projected, {**config, "memory_limit_mb": 2048}, 136800)
    with pytest.raises(ValueError, match="budget exceeds"):
        bind_preregistration(wrapper, projected, config, 136801)
    assert all(
        spec["role"] == "inventory" and spec["_family"]["reservation"]["status"] == "active"
        for spec in draft["datasets"]
    )
    audit = catalog.load_suite(SUITES / "final_reserved_confirmation_audit_20260924.json")
    assert audit["execution_binding_review"]["method_semantics_field_enforced"] is False
    assert audit["execution_binding_review"]["class_support_thresholds"] == protocol["classification_admission_margin"]


def test_final_release_admits_exactly_the_authorized_families_with_explicit_allowance():
    path = SUITES / "final_reserved_confirmation_20260924.json"
    with pytest.raises(catalog.AdmissionError, match="confirmation is locked"):
        catalog.load_suite(path)
    suite = catalog.load_suite(path, allow_confirmation=True)
    draft = historical_manifest_snapshot(SUITES / "final_reserved_confirmation_draft_20260924.json")
    assert suite["family_count"] == len(suite["datasets"]) == 12
    assert suite["preregistration"] == draft["preregistration"]
    assert suite["release_state"]["status"] == "authorized_released"
    assert suite["release_state"]["model_execution_authorized"] is True
    for released, original in zip(suite["datasets"], draft["datasets"], strict=True):
        assert released["family_id"] == original["family_id"]
        changed = {key for key in original if released[key] != original[key]}
        assert changed == {"role", "prior_exposure", "protocol_state"}
        assert released["role"] == released["_family"]["role"] == "confirmation"
        assert released["prior_exposure"] == released["_family"]["prior_exposure"] == "audited_unexposed"
        assert released["_family"]["reservation"]["status"] == "released"
        assert released["_family"]["exposure_audit"]["status"] == "complete"
        assert released["_family"]["exposure_audit"]["evidence"]


def test_final_release_registry_change_is_limited_and_prior_files_are_preserved():
    release = catalog.load_suite(SUITES / "final_reserved_confirmation_release_20260924.json")
    transition = release["registry_transition"]
    assert len(transition["changed_family_ids"]) == 12
    assert transition["nonselected_family_records_changed"] is False
    assert transition["registry_top_level_fields_changed"] is False
    assert hashlib.sha256(registry_before_fresh_origin_append()).hexdigest() == transition["after_sha256"]
    registry = catalog._families([])
    for fid in transition["changed_family_ids"]:
        before, after = transition["before"][fid], transition["after"][fid]
        assert registry[fid] == after
        changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
        assert changed == {"role", "prior_exposure", "reservation", "exposure_audit"}
        assert before["reservation"]["status"] == "active"
        assert after["reservation"]["status"] == "released"
        assert after["source_ids"] == before["source_ids"] and after["aliases"] == before["aliases"]
    for name, checksum in release["preserved_prior_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / name).read_bytes()).hexdigest() == checksum
    assert release["protocol_preservation"]["preregistration_unchanged"] is True
    assert release["protocol_preservation"]["algorithm_or_runtime_code_changes"] is False
    assert (
        release["no_payload_calls"]
        and release["no_model_calls"]
        and release["no_llm_calls"]
        and release["no_ledger_writes"]
    )


def test_final_release_binds_exact_intended_configuration_and_128_cells_without_data():
    from benchmarks.execution import bind_preregistration

    suite = catalog.load_suite(SUITES / "final_reserved_confirmation_20260924.json", allow_confirmation=True)
    registration = suite["preregistration"]
    config = {
        **registration["selection_parameters"],
        **{
            key: registration[key]
            for key in ("methods", "learners", "seeds", "folds", "max_rows", "threads", "memory_limit_mb")
        },
        "case_timeout": registration["case_timeout_seconds"],
        "synthetic": False,
        "live_llm": False,
        "llm_ledger": None,
        "allow_confirmation": True,
    }
    assert bind_preregistration({"manifests": [suite]}, suite["datasets"], config, 136800) == registration
    splits = sum(
        len(plan["official_split_ids"]) if plan["policy"] == "official" else plan["folds"]
        for plan in registration["split_plan"].values()
    )
    assert splits == 16
    assert (
        splits * len(config["methods"]) * len(config["learners"]) * len(config["seeds"])
        == registration["expected_cells"]
        == 128
    )
    assert registration["classification_admission_margin"] == {
        "min_total": 30,
        "min_outer_train": 20,
        "min_outer_test": 10,
    }
    with pytest.raises(ValueError, match="budget exceeds"):
        bind_preregistration({"manifests": [suite]}, suite["datasets"], config, 136801)


def test_final_release_exposure_snapshot_has_no_selected_matches():
    release = catalog.load_suite(SUITES / "final_reserved_confirmation_release_20260924.json")
    evidence = release["exposure_check"]
    rows = [[item[key] for key in ("family_id", "run_id", "role", "created_at")] for item in evidence["rows"]]
    assert len(rows) == evidence["row_count"] == 71
    assert (
        hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == evidence["logical_rows_sha256"]
    )
    selected = set(release["authorization"]["family_ids"])
    assert len(selected) == 12 and not selected.intersection(row[0] for row in rows)
    assert evidence["selected_family_matches"] == []


def test_released_confirmation_cannot_be_accessed_through_old_development_recipe(workspace):
    old = historical_manifest_snapshot(SUITES / "development_expansion_20260922.json")
    release = catalog.load_suite(SUITES / "final_reserved_confirmation_release_20260924.json")
    selected = set(release["authorization"]["family_ids"])
    for spec in old["datasets"]:
        if spec["family_id"] in selected:
            with pytest.raises(catalog.AdmissionError, match="prior_exposure mismatch|family role mismatch"):
                catalog.load_dataset(spec, workspace / "old-development-forbidden")
    assert not (workspace / "old-development-forbidden").exists()


@pytest.mark.usefixtures("pre_iid_release_registry")
def test_fresh_origin_reservation_has_exact_nonredistributed_observational_quotas(workspace):
    suite = catalog.load_suite(SUITES / "fresh_origin_reservation_20260924.json")
    assert len(suite["datasets"]) == suite["family_count"] == 10
    policy = suite["selection_policy"]
    assert policy["quotas"] == policy["actual_selected"] == {"regression": 4, "binary": 3, "multiclass": 3}
    assert policy["shortfalls"] == {"regression": 0, "binary": 0, "multiclass": 0}
    assert policy["quota_redistribution"] is False
    assert suite["model_execution_authorized"] is False and suite["payload_acquisition_authorized"] is False
    assert "preregistration" not in suite and "selection_parameters" not in suite
    for kind, count in policy["quotas"].items():
        names = [spec["family_id"] for spec in suite["datasets"] if spec["metadata_profile"]["stratum"] == kind]
        assert names == sorted(names) == policy["eligible_pool_order_by_stratum"][kind]
        assert len(names) == count
    for spec in suite["datasets"]:
        profile = spec["metadata_profile"]
        assert spec["role"] == "inventory" and spec["reservation_role"] == "confirmation_candidate"
        assert spec["origin_kind"] == "observational"
        assert spec["prior_exposure"] == spec["_family"]["prior_exposure"] == "unknown"
        assert spec["_family"]["reservation"]["status"] == "active"
        assert profile["rows"] <= 50000 and profile["predictors"] <= 200
        assert not profile["payload_checked"] and not profile["selected_fold_class_support_checked"]
        if spec["task"] == "classification":
            assert profile["minimum_class_count"] >= 30
        with pytest.raises(catalog.AdmissionError, match="must never download"):
            catalog.load_dataset(spec, workspace / "fresh-candidates-must-not-download")
    assert not (workspace / "fresh-candidates-must-not-download").exists()


@pytest.mark.usefixtures("pre_iid_release_registry")
def test_fresh_origins_do_not_reuse_old_registry_or_exposure_identities():
    suite = catalog.load_suite(SUITES / "fresh_origin_reservation_20260924.json")
    proof = suite["exposure_audit"]
    selected = {spec["family_id"] for spec in suite["datasets"]}
    assert not selected.intersection(proof["prior_registry_family_ids"])
    assert not selected.intersection(proof["prior_inventory_family_ids"])
    assert not selected.intersection(row["family_id"] for row in proof["ledger"]["rows"])
    rows = [[row[key] for key in ("family_id", "run_id", "role", "created_at")] for row in proof["ledger"]["rows"]]
    assert len(rows) == proof["ledger"]["row_count"] == 83
    assert (
        hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
        == proof["ledger"]["logical_rows_sha256"]
    )
    assert set(proof["previous_final_twelve_now_exposed"]) <= {row[0] for row in rows}
    assert not selected.intersection(proof["previous_final_twelve_now_exposed"])
    old_sources = set(proof["prior_inventory_source_ids"])
    for spec in suite["datasets"]:
        assert not old_sources.intersection(spec["_family"]["source_ids"])
    assert proof["ledger"]["selected_family_matches"] == []


@pytest.mark.usefixtures("pre_iid_release_registry")
def test_fresh_origin_primary_grants_and_target_exclusions_are_explicit():
    specs = {
        s["family_id"]: s for s in catalog.load_suite(SUITES / "fresh_origin_reservation_20260924.json")["datasets"]
    }
    assert Counter(s["rights"]["license"] for s in specs.values()) == {"CC-BY-4.0": 9, "CC0-1.0": 1}
    infrared = specs["infrared-oral-temperature"]
    assert infrared["target"] == "aveOralM"
    assert set(infrared["drop_columns"]) == {"SubjectID", "aveOralF"}
    assert infrared["source"]["target_columns"] == ["aveOralF", "aveOralM"]
    assert any("physionet.org/content/face-oral-temp-data/" in url for url in infrared["rights"]["evidence"])
    assert "not a homogeneous pixel" in infrared["evaluation_limitations"]
    hardware = specs["computer-hardware-performance"]
    assert hardware["source"]["dataset_id"] == 230 and hardware["target"] == "class"
    assert not {"ERP", "vendor", "model"}.intersection(hardware["source"]["columns"])
    assert "cpu-act" in hardware["evaluation_limitations"]
    crime = specs["communities-crime"]
    assert set(crime["drop_columns"]) == {"state", "county", "community", "communityname", "fold"}
    assert crime["metadata_profile"]["predictors"] == 122
    assert "communityname" not in crime["categorical"]
    australian = specs["australian-credit-approval"]
    assert {"uci:27", "uci:143", "openml:29", "openml:40981"} <= set(australian["_family"]["source_ids"])
    assert australian["source"]["dataset_id"] == australian["source"]["task_id"] == 29
    assert australian["metadata_profile"]["predictors"] == 15
    assert "preserved over" in australian["evaluation_limitations"]
    assert "undocumented filling" in australian["evaluation_limitations"]


@pytest.mark.usefixtures("pre_iid_release_registry")
def test_fresh_metadata_audit_preserves_every_old_registry_record_and_manifest():
    audit = catalog.load_suite(SUITES / "fresh_origin_reservation_audit_20260924.json")
    current = {f["family_id"]: f for f in json.loads(registry_before_cycle2_append())["families"]}
    append = audit["registry_append_only"]
    assert append["new_family_count"] == 10
    assert len(current) == append["old_family_count"] + 10
    for fid, expected in append["prior_registry_record_sha256"].items():
        assert (
            hashlib.sha256(json.dumps(current[fid], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == expected
        )
    registry = json.loads((ROOT / "benchmarks" / "families.json").read_bytes())
    assert {k: v for k, v in registry.items() if k != "families"} == append["prior_registry_top_level"]
    for filename, expected in audit["preserved_prior_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / filename).read_bytes()).hexdigest() == expected
    assert audit["runtime_changes"] is False
    assert audit["payload_downloads"] == audit["model_calls"] == audit["llm_calls"] == 0
    assert audit["reconnaissance"]["uci_records_inspected"] == 37
    excluded = {x["uci_id"]: x["reason"] for x in audit["reconnaissance"]["remaining_records"]}
    assert "SAME plant" in excluded[551]
    assert "Same original credit-approval" in excluded[27]
    assert "Existing wine-quality" in excluded[186]


@pytest.mark.usefixtures("pre_iid_release_registry")
def test_fresh_reservations_block_alternative_manifest_acquisition(workspace):
    suite = catalog.load_suite(SUITES / "fresh_origin_reservation_20260924.json")
    for candidate in suite["datasets"]:
        # Even a caller keeping inventory family metadata but relabeling the entry
        # cannot pass the existing canonical family-role/reservation gates.
        alternate = copy.deepcopy(candidate)
        alternate["role"] = "development"
        alternate["_family"]["reservation"]["status"] = "released"
        with pytest.raises(catalog.AdmissionError, match="family role mismatch|reserved as a confirmation candidate"):
            catalog.load_dataset(alternate, workspace / "fresh-alternate-blocked")
    assert not (workspace / "fresh-alternate-blocked").exists()


@pytest.mark.usefixtures("pre_iid_release_registry")
def test_frequency_confirmation_release_stops_on_official_class_support_blocker(workspace):
    audit = catalog.load_suite(SUITES / "frequency_fresh_confirmation_blocked_audit_20260924.json")
    reservation = catalog.load_suite(SUITES / "fresh_origin_reservation_20260924.json")
    assert audit["release_state"]["status"] == "blocked"
    assert audit["release_state"]["model_execution_authorized"] is False
    blocker = audit["blocker"]
    assert blocker["family_id"] == "vertebral-column" and blocker["task_id"] == 9939
    assert blocker["official_stratified_sampling"] is True
    assert blocker["reported_minority_class_count"] == 60
    assert blocker["official_fold_count"] == 10
    assert (
        blocker["reported_minority_class_count"] // blocker["official_fold_count"]
        == blocker["expected_minority_test_count"]
        == 6
    )
    assert blocker["required_minority_test_count"] == 10
    plan = audit["requested_protocol_not_executable"]
    assert plan["status"] == "blocked_not_released"
    assert plan["expected_total_splits"] == 14 and plan["expected_cells"] == 224
    assert plan["seeds"] == [2027] and plan["sample_seed"] == 2026
    assert plan["classification_admission_margin"] == {"min_total": 30, "min_outer_train": 20, "min_outer_test": 10}
    assert plan["total_active_seconds_ceiling"] == 86400
    assert "preregistration" not in audit
    assert audit["datasets"] == reservation["datasets"]
    for spec in audit["datasets"]:
        assert spec["role"] == "inventory" and spec["_family"]["reservation"]["status"] == "active"
        assert spec["_family"]["prior_exposure"] == "unknown"
        with pytest.raises(catalog.AdmissionError, match="must never download"):
            catalog.load_dataset(spec, workspace / "blocked-confirmation")
    assert not (workspace / "blocked-confirmation").exists()


@pytest.mark.usefixtures("pre_regression_stress_registry")
def test_three_fold_iid_release_preserves_every_reserved_dataset_schema():
    suite = catalog.load_suite(SUITES / "frequency_fresh_confirmation_iid_20260928.json", allow_confirmation=True)
    reservation = historical_manifest_snapshot(SUITES / "fresh_origin_reservation_20260924.json")
    assert suite["family_count"] == len(suite["datasets"]) == 10
    for current, original in zip(suite["datasets"], reservation["datasets"], strict=True):
        assert {key for key in original if current[key] != original[key]} == {"role", "prior_exposure"}
        assert current["role"] == current["_family"]["role"] == "confirmation"
        assert current["prior_exposure"] == current["_family"]["prior_exposure"] == "audited_unexposed"
        assert current["_family"]["reservation"]["status"] == "released"
        assert current["_family"]["exposure_audit"]["status"] == "complete"
        assert "class_support" not in current
        if current["provider"] == "openml_task":
            assert current["split_policy"] == "official"
            assert current["source"]["official_split_dimensions"] == {"repeats": 1, "folds": 10, "samples": 1}
        else:
            assert current["split_policy"] == "iid"
    with pytest.raises(catalog.AdmissionError, match="confirmation is locked"):
        catalog.load_suite(SUITES / "frequency_fresh_confirmation_iid_20260928.json")


@pytest.mark.usefixtures("pre_regression_stress_registry")
def test_three_fold_iid_release_has_exact_480_cells_and_no_implicit_resplitting():
    from benchmarks.execution import bind_preregistration

    suite = catalog.load_suite(SUITES / "frequency_fresh_confirmation_iid_20260928.json", allow_confirmation=True)
    protocol = suite["preregistration"]
    assert protocol["resplit_official_as_iid"] is True
    assert len(protocol["split_plan"]) == 10
    assert all(entry == {"policy": "iid", "folds": 3} for entry in protocol["split_plan"].values())
    assert protocol["expected_total_splits"] == 30 and protocol["expected_cells"] == 480
    assert protocol["seeds"] == [2027] and protocol["sample_seed"] == 2026
    assert protocol["max_rows"] == 0
    assert protocol["classification_admission_margin"] == {"min_total": 30, "min_outer_train": 20, "min_outer_test": 10}
    assert protocol["methods"] == [
        "baseline",
        "featcopilot-frequency-paired",
        "openfe-inductive-native",
        "autofeat-native",
    ]
    assert protocol["learners"] == ["linear", "lightgbm", "catboost", "xgboost"]
    config = {
        **protocol["selection_parameters"],
        **{
            key: protocol[key]
            for key in ("methods", "learners", "seeds", "folds", "max_rows", "threads", "memory_limit_mb")
        },
        "case_timeout": protocol["case_timeout_seconds"],
        "synthetic": False,
        "live_llm": False,
        "llm_ledger": None,
        "allow_confirmation": True,
    }
    assert bind_preregistration({"manifests": [suite]}, suite["datasets"], config, 86400) == protocol
    without_flag = copy.deepcopy(suite)
    del without_flag["preregistration"]["resplit_official_as_iid"]
    with pytest.raises(ValueError, match="policy differs"):
        bind_preregistration({"manifests": [without_flag]}, without_flag["datasets"], config, 86400)
    with pytest.raises(ValueError, match="max_rows|uncapped"):
        bind_preregistration({"manifests": [suite]}, suite["datasets"], {**config, "max_rows": 100}, 86400)
    with pytest.raises(ValueError, match="budget exceeds"):
        bind_preregistration({"manifests": [suite]}, suite["datasets"], config, 86401)


@pytest.mark.usefixtures("pre_regression_stress_registry")
def test_three_fold_iid_amendment_preserves_model_settings_margins_and_blocked_history():
    suite = catalog.load_suite(SUITES / "frequency_fresh_confirmation_iid_20260928.json", allow_confirmation=True)
    previous = historical_manifest_snapshot(SUITES / "frequency_fresh_confirmation_blocked_audit_20260924.json")
    old, new = previous["requested_protocol_not_executable"], suite["preregistration"]
    for field in (
        "family_ids",
        "methods",
        "learners",
        "learner_search",
        "selection_parameters",
        "seeds",
        "sample_seed",
        "max_rows",
        "threads",
        "memory_limit_mb",
        "case_timeout_seconds",
        "total_active_seconds_ceiling",
        "classification_admission_margin",
        "method_semantics_documentation_only",
        "primary_metrics",
    ):
        assert new[field] == old[field]
    assert new["evaluation_policy_amendment"]["scope"] == "entire_unchanged_ten_family_cohort"
    assert new["evaluation_policy_amendment"]["official_reproduction_claim"] is False
    assert new["total_active_seconds_ceiling"] == 86400
    assert new["expected_cells"] * new["case_timeout_seconds"] / 3600 == 120
    assert new["decision_gates"]["automatic_promotion"] is False
    assert new["decision_gates"]["automatic_sota"] is False
    assert "All 480" in new["decision_gates"]["completion"]


@pytest.mark.usefixtures("pre_regression_stress_registry")
def test_three_fold_iid_release_audit_is_metadata_only_and_changes_exactly_ten_records():
    release = catalog.load_suite(SUITES / "frequency_fresh_confirmation_iid_release_20260928.json")
    transition = release["registry_transition"]
    assert len(transition["changed_family_ids"]) == 10
    assert transition["nonselected_family_changes"] is False and transition["top_level_registry_changes"] is False
    assert hashlib.sha256(registry_before_cycle2_append()).hexdigest() == transition["after_sha256"]
    registry = catalog._families([])
    for fid, before in transition["before"].items():
        after = transition["after"][fid]
        changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
        assert changed == {"reservation", "role", "prior_exposure", "exposure_audit"}
        assert registry[fid] == after
        assert before["reservation"]["status"] == "active" and after["reservation"]["status"] == "released"
    for filename, expected in release["preserved_prior_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / filename).read_bytes()).hexdigest() == expected
    assert (
        release["no_payload_reads"] and release["no_models"] and release["no_llm_calls"] and release["no_ledger_writes"]
    )
    check = release["exposure_check"]
    rows = [[item[key] for key in ("family_id", "run_id", "role", "created_at")] for item in check["rows"]]
    assert len(rows) == check["row_count"] == 102
    assert hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == check["logical_rows_sha256"]
    assert check["selected_family_matches"] == []
    assert not set(transition["changed_family_ids"]).intersection(row[0] for row in rows)
    vertebral = next(item for item in release["per_family_metadata_checks"] if item["family_id"] == "vertebral-column")
    assert vertebral["reported_minimum_class"] == 60
    assert vertebral["expected_minimum_test_class"] == 20
    assert vertebral["expected_minimum_train_class"] == 40
    assert vertebral["payload_verified"] is False


@pytest.mark.usefixtures("pre_cycle2_robust_release_registry")
def test_cycle2_reserves_only_the_cleared_subset_without_transferring_quotas(workspace):
    suite = catalog.load_suite(SUITES / "cycle2_origin_reservation_20260929.json")
    policy = suite["selection_policy"]
    assert suite["family_count"] == len(suite["datasets"]) == 4
    assert policy["target_quotas"] == {"regression": 6, "binary": 3, "multiclass": 3}
    assert policy["actual_reserved"] == {"regression": 1, "binary": 1, "multiclass": 2}
    assert policy["shortfalls"] == {"regression": 5, "binary": 2, "multiclass": 1}
    assert policy["quota_redistribution"] is False
    assert not suite["model_execution_authorized"] and not suite["payload_acquisition_authorized"]
    assert "preregistration" not in suite
    for stratum in policy["target_quotas"]:
        selected = [s["family_id"] for s in suite["datasets"] if s["metadata_profile"]["stratum"] == stratum]
        assert selected == sorted(selected) == policy["eligible_order_by_stratum"][stratum]
    for spec in suite["datasets"]:
        profile = spec["metadata_profile"]
        assert 200 <= profile["rows"] <= 50000 and profile["predictors"] <= 200
        assert spec["role"] == spec["_family"]["role"] == "inventory"
        assert spec["prior_exposure"] == spec["_family"]["prior_exposure"] == "unknown"
        assert spec["_family"]["reservation"]["status"] == "active"
        assert spec["origin_kind"] == "observational"
        assert profile["payload_checked"] is False and profile["selected_fold_support_checked"] is False
        catalog._validate_rights(spec["rights"], spec["name"], runnable=True)
        catalog._validate_source(spec, runnable=True)
        if spec["task"] == "classification":
            assert profile["minimum_class_count"] >= 60
            assert profile["expected_three_fold_minimum_test_count"] >= 20
        with pytest.raises(catalog.AdmissionError, match="must never download"):
            catalog.load_dataset(spec, workspace / "cycle2-forbidden")
    assert not (workspace / "cycle2-forbidden").exists()


@pytest.mark.usefixtures("pre_cycle2_robust_release_registry")
def test_cycle2_schema_preserves_target_and_timing_exclusions():
    specs = {
        s["family_id"]: s for s in catalog.load_suite(SUITES / "cycle2_origin_reservation_20260929.json")["datasets"]
    }
    facebook = specs["facebook-cosmetics-posts"]
    assert facebook["target"] == "Total Interactions"
    assert facebook["metadata_profile"]["predictors"] == 7
    assert {"comment", "like", "share"} <= set(facebook["drop_columns"])
    assert all(
        name in facebook["drop_columns"] for name in facebook["source"]["columns"] if name.startswith("Lifetime")
    )
    assert "fixed page/year" in facebook["iid_suitability_review"]["observation_unit_basis"]
    thyroid = specs["differentiated-thyroid-recurrence"]
    assert set(thyroid["drop_columns"]) == {"Response", "Risk"}
    assert thyroid["metadata_profile"]["minimum_class_count"] == 108
    assert len(thyroid["categorical"]) == 13
    entrance = specs["assam-medical-entrance"]
    assert entrance["drop_columns"] == ["time"]
    assert len(entrance["categorical"]) == 10
    assert entrance["metadata_profile"]["minimum_class_count"] == 101
    drugs = specs["personality-drug-consumption"]
    assert drugs["target"] == "cannabis"
    assert set(drugs["source"]["target_columns"]) - {"cannabis"} <= set(drugs["drop_columns"])
    assert "id" in drugs["drop_columns"]
    assert drugs["metadata_profile"]["minimum_class_count"] == 140
    assert sum(drugs["metadata_profile"]["class_counts"].values()) == 1885
    for spec in specs.values():
        assert spec["rights"]["license"] == "CC-BY-4.0"
        if spec["provider"] == "openml_task":
            assert spec["split_policy"] == "official"
            assert spec["source"]["official_split_dimensions"] == {"repeats": 1, "folds": 10, "samples": 1}
            assert "NOMINAL and STRING" in spec["source"]["categorical_rule"]


@pytest.mark.usefixtures("pre_cycle2_robust_release_registry")
def test_cycle2_bounded_pool_records_concrete_holds_and_no_model_exposure():
    audit = catalog.load_suite(SUITES / "cycle2_origin_reservation_audit_20260929.json")
    assert len(audit["decisions"]) == audit["summary"]["candidate_origin_records"] == 30
    assert audit["summary"]["eligible_origins"] == 4
    assert audit["summary"]["pending_or_ineligible_origins"] == 26
    decisions = {entry["uci_id"]: entry for entry in audit["decisions"]}
    assert decisions[14]["category"] == "entity_boundary_pending"
    assert decisions[563]["category"] == "minimum_class_count_pending"
    assert decisions[607]["category"] == "experimental_sequence_pending"
    assert decisions[938]["category"] == "primary_noncommercial_grant"
    assert audit["specific_rights_hold"]["actual_license"] == "CC-BY-NC-4.0"
    assert decisions[544]["category"] == "synthetic_majority"
    assert decisions[401]["category"] == "predictor_cap"
    assert decisions[409]["category"] == "row_floor_and_temporal"
    proof = audit["exposure_audit"]
    selected = {s["family_id"] for s in audit["datasets"]}
    assert not selected.intersection(proof["prior_registry_family_ids"])
    assert not selected.intersection(proof["prior_inventory_family_ids"])
    rows = [[r[k] for k in ("family_id", "run_id", "role", "created_at")] for r in proof["ledger"]["rows"]]
    assert len(rows) == proof["ledger"]["row_count"] == 144
    assert (
        hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
        == proof["ledger"]["logical_rows_sha256"]
    )
    assert not selected.intersection(r[0] for r in rows)
    assert proof["ledger"]["selected_family_matches"] == []
    assert audit["runtime_changes"] is False
    assert (
        audit["summary"]["payload_acquisitions"]
        == audit["summary"]["model_calls"]
        == audit["summary"]["llm_calls"]
        == 0
    )


@pytest.mark.usefixtures("pre_cycle2_robust_release_registry")
def test_cycle2_append_preserves_all_previous_registry_records_and_suite_bytes():
    audit = catalog.load_suite(SUITES / "cycle2_origin_reservation_audit_20260929.json")
    registry = json.loads(registry_before_standard_iid_append())
    current = {f["family_id"]: f for f in registry["families"]}
    append = audit["registry_append_only"]
    assert len(current) == append["prior_family_count"] + append["appended_family_count"]
    assert append["appended_family_count"] == 4
    for fid, expected in append["prior_record_sha256"].items():
        assert (
            hashlib.sha256(json.dumps(current[fid], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == expected
        )
    assert {k: v for k, v in registry.items() if k != "families"} == append["prior_top_level"]
    for filename, expected in audit["preserved_prior_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / filename).read_bytes()).hexdigest() == expected
    assert (
        hashlib.sha256(registry_before_cycle2_append()).hexdigest() == audit["exposure_audit"]["registry_before_sha256"]
    )


def test_cycle2_robust_release_preserves_all_four_sources_and_requires_confirmation_allow():
    path = SUITES / "cycle2_robust_confirmation_20260930.json"
    with pytest.raises(catalog.AdmissionError, match="confirmation is locked"):
        catalog.load_suite(path)
    suite = catalog.load_suite(path, allow_confirmation=True)
    original = historical_manifest_snapshot(SUITES / "cycle2_origin_reservation_20260929.json")
    assert suite["family_count"] == len(suite["datasets"]) == 4
    for spec, before in zip(suite["datasets"], original["datasets"], strict=True):
        assert {key for key in before if spec[key] != before[key]} == {"role", "prior_exposure"}
        assert spec["role"] == spec["_family"]["role"] == "confirmation"
        assert spec["prior_exposure"] == spec["_family"]["prior_exposure"] == "audited_unexposed"
        assert spec["_family"]["reservation"]["status"] == "released"
        assert spec["_family"]["exposure_audit"]["status"] == "complete"
        assert "class_support" not in spec
        assert not any(spec.get(key) for key in ("group_column", "time_column", "label_end_column"))
        if spec["provider"] == "openml_task":
            assert spec["split_policy"] == "official"
            assert spec["source"]["official_split_dimensions"] == {"repeats": 1, "folds": 10, "samples": 1}


def test_cycle2_robust_release_binds_192_cells_with_matched_reference_clipping():
    from benchmarks import tabular
    from benchmarks.execution import bind_preregistration, validate_config

    suite = catalog.load_suite(SUITES / "cycle2_robust_confirmation_20260930.json", allow_confirmation=True)
    protocol = suite["preregistration"]
    expected_methods = [
        "baseline-robust",
        "featcopilot-frequency-aligned-robust",
        "openfe-inductive-native-robust",
        "autofeat-native-robust",
    ]
    assert protocol["methods"] == expected_methods
    assert set(expected_methods) <= set(tabular.METHODS) & tabular.ROBUST_METHODS
    assert tabular.ROBUST_CLIP_QUANTILES == (0.0, 1.0)
    assert protocol["learners"] == ["linear", "lightgbm", "catboost", "xgboost"]
    assert protocol["resplit_official_as_iid"] is True
    assert all(entry == {"policy": "iid", "folds": 3} for entry in protocol["split_plan"].values())
    assert len(protocol["split_plan"]) * 3 * len(expected_methods) * len(protocol["learners"]) == 192
    assert protocol["expected_cells"] == 192 and protocol["expected_total_splits"] == 12
    assert protocol["seeds"] == [2028] and protocol["sample_seed"] == 2026
    assert protocol["max_rows"] == 0 and protocol["threads"] == 8
    assert protocol["memory_limit_mb"] == 4096 and protocol["case_timeout_seconds"] == 900
    assert protocol["total_active_seconds_ceiling"] == 43200
    config = {
        **protocol["selection_parameters"],
        **{
            key: protocol[key]
            for key in ("methods", "learners", "seeds", "folds", "max_rows", "threads", "memory_limit_mb")
        },
        "case_timeout": protocol["case_timeout_seconds"],
        "synthetic": False,
        "live_llm": False,
        "llm_ledger": None,
        "allow_confirmation": True,
    }
    validate_config(config, 43200)
    assert bind_preregistration({"manifests": [suite]}, suite["datasets"], config, 43200) == protocol
    with pytest.raises(ValueError, match="budget exceeds"):
        bind_preregistration({"manifests": [suite]}, suite["datasets"], config, 43201)
    semantics = protocol["method_semantics_documentation_only"]
    candidate = semantics["featcopilot-frequency-aligned-robust"]
    assert candidate["importance_resamples"] == 1 and candidate["include_frequency"] is True
    assert candidate["screening"] == "importance" and candidate["selection_rule"] == "paired_gain"
    assert candidate["numeric_clip_quantiles"] == [0.0, 1.0]
    assert semantics["all_final_learners"]["numeric_clip_quantiles"] == [0.0, 1.0]
    for reference in expected_methods[2:]:
        assert semantics[reference]["clipping_scope"] == "final_matched_learner_only"
        assert semantics[reference]["internal_ranking_clipping"] is False
        assert semantics[reference]["max_added_features"] is None
    assert semantics["autofeat-native-robust"]["max_gb"] is None


def test_cycle2_robust_release_preserves_personal_data_exclusions_and_retrospective_scope():
    release = catalog.load_suite(SUITES / "cycle2_robust_confirmation_release_20260930.json")
    rows = {spec["family_id"]: spec for spec in release["datasets"]}
    assert {"comment", "like", "share"} <= set(rows["facebook-cosmetics-posts"]["drop_columns"])
    assert len(rows["facebook-cosmetics-posts"]["drop_columns"]) == 11
    assert set(rows["differentiated-thyroid-recurrence"]["drop_columns"]) == {"Response", "Risk"}
    assert rows["assam-medical-entrance"]["drop_columns"] == ["time"]
    drug = rows["personality-drug-consumption"]
    assert "id" in drug["drop_columns"]
    assert set(drug["source"]["target_columns"]) - {"cannabis"} <= set(drug["drop_columns"])
    exclusions = release["personal_data_and_timing_exclusions"]
    assert "not claimed" in exclusions["differentiated-thyroid-recurrence"]["meaning"]
    assert "Retrospective" in exclusions["personality-drug-consumption"]["meaning"]
    assert "no future admission" in exclusions["assam-medical-entrance"]["meaning"]
    for entry in release["metadata_reviews"]:
        assert set(entry["string_predictors_declared_categorical"]) <= set(entry["categorical_columns"])
        assert entry["payload_checked"] is False
        if entry["reported_minimum_class_count"] is not None:
            assert entry["reported_minimum_class_count"] >= 60
            assert entry["expected_three_fold_minimum_test_count"] >= 20
    claims = release["claim_gate"]
    assert len(claims["primary_family_ids"]) == 4
    assert len(claims["regression_family_ids"]) == 1
    assert claims["numeric_gate"]["required_complete_cells"] == 192
    assert claims["numeric_gate"]["maximum_raw_family_harm"] == 0.05
    assert claims["automatic_sota"] is False and claims["automatic_promotion"] is False
    assert "No between-family regression confidence interval" in claims["intervals"]["limits"]


def test_cycle2_robust_release_changes_only_four_registry_records_and_preserves_history():
    release = catalog.load_suite(SUITES / "cycle2_robust_confirmation_release_20260930.json")
    transition = release["registry_transition"]
    assert len(transition["changed_family_ids"]) == 4
    assert transition["nonselected_family_records_changed"] is False
    assert transition["registry_top_level_changed"] is False
    assert hashlib.sha256(registry_before_standard_iid_append()).hexdigest() == transition["after_sha256"]
    current = catalog._families([])
    for fid, before in transition["before"].items():
        after = transition["after"][fid]
        assert after == current[fid]
        assert {k for k in set(before) | set(after) if before.get(k) != after.get(k)} == {
            "role",
            "prior_exposure",
            "reservation",
            "exposure_audit",
        }
        assert before["reservation"]["status"] == "active" and after["reservation"]["status"] == "released"
    for filename, expected in release["preserved_prior_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / filename).read_bytes()).hexdigest() == expected
    exposure = release["exposure_check"]
    rows = [[item[k] for k in ("family_id", "run_id", "role", "created_at")] for item in exposure["rows"]]
    assert len(rows) == exposure["row_count"] == 176
    assert (
        hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == exposure["logical_rows_sha256"]
    )
    assert not set(transition["changed_family_ids"]).intersection(row[0] for row in rows)
    assert exposure["selected_family_matches"] == []
    assert (
        release["no_payload_reads"]
        and release["no_real_model_calls"]
        and release["no_llm_calls"]
        and release["no_ledger_writes"]
    )


@pytest.mark.usefixtures("pre_standard_iid_policy_release_registry")
def test_standard_iid_tier_is_explicit_inventory_only_and_not_clinical_generalization(workspace):
    suite = catalog.load_suite(SUITES / "standard_iid_entity_reservation_20261003.json")
    assert suite["family_count"] == suite["reserved_family_count"] == 2
    assert suite["evaluation_tier"] == "standard_iid_unverified_entity"
    assert suite["policy_amendment"]["user_choice"] == "separate-standard-iid"
    assert suite["model_execution_authorized"] is False and suite["payload_acquisition_authorized"] is False
    assert "preregistration" not in suite
    for spec in suite["datasets"]:
        assert spec["evaluation_tier"] == spec["_family"]["evaluation_tier"] == suite["evaluation_tier"]
        assert spec["role"] == spec["_family"]["role"] == "inventory"
        assert spec["prior_exposure"] == spec["_family"]["prior_exposure"] == "unknown"
        assert spec["_family"]["reservation"]["status"] == "active"
        boundary = spec["claim_boundary"]
        assert boundary == suite["claim_boundary"] == spec["_family"]["claim_boundary"]
        assert boundary["patient_entity_independence_verified"] is False
        assert boundary["patient_entity_disjoint_claims_allowed"] is False
        assert boundary["clinical_generalization_claims_allowed"] is False
        assert boundary["pool_with_verified_entity_cohorts"] is False
        assert boundary["known_group_time_constraints_must_be_preserved"] is True
        assert boundary["future_confirmation_requires_explicit_tier_binding"] is True
        catalog._validate_source(spec, runnable=True)
        catalog._validate_rights(spec["rights"], spec["name"], runnable=True)
        with pytest.raises(catalog.AdmissionError, match="must never download"):
            catalog.load_dataset(spec, workspace / "standard-iid-no-acquisition")
    assert not (workspace / "standard-iid-no-acquisition").exists()


@pytest.mark.usefixtures("pre_standard_iid_policy_release_registry")
def test_standard_iid_tier_preserves_native_targets_and_required_exclusions():
    specs = {
        s["family_id"]: s
        for s in catalog.load_suite(SUITES / "standard_iid_entity_reservation_20261003.json")["datasets"]
    }
    assert set(specs) == {"haberman-survival", "mammographic-mass"}
    mammographic = specs["mammographic-mass"]
    assert mammographic["source"]["uci_id"] == 161
    assert mammographic["target"] == "Severity"
    assert mammographic["drop_columns"] == ["BI-RADS"]
    assert mammographic["categorical"] == ["Shape", "Margin"]
    assert mammographic["metadata_profile"]["rows"] == 961
    assert mammographic["metadata_profile"]["predictors"] == 4
    assert mammographic["metadata_profile"]["class_counts"] == {"0": 516, "1": 445}
    haberman = specs["haberman-survival"]
    assert haberman["source"]["uci_id"] == 43
    assert haberman["source"]["columns"] == ["age", "operation_year", "positive_auxillary_nodes", "survival_status"]
    assert haberman["target"] == "survival_status"
    assert haberman["drop_columns"] == haberman["categorical"] == []
    assert haberman["metadata_profile"]["class_counts"] == {"1": 225, "2": 81}
    assert "five years" in haberman["prediction_time"]
    assert "not a preoperative" in haberman["prediction_time"]
    for spec in specs.values():
        assert spec["target_transform"] == "identity"
        assert spec["rights"]["license"] == "CC-BY-4.0"
        assert spec["metadata_profile"]["minimum_class_count"] >= 60
        assert spec["metadata_profile"]["entity_independence_verified"] is False
        assert spec["metadata_profile"]["payload_verified"] is False


@pytest.mark.usefixtures("pre_standard_iid_policy_release_registry")
def test_standard_iid_tier_does_not_retroactively_clear_strict_tier_or_waive_rights():
    audit = catalog.load_suite(SUITES / "standard_iid_entity_reservation_audit_20261003.json")
    assert audit["old_blocked_audit"]["preserved_unchanged"] is True
    assert audit["old_blocked_audit"]["old_decision_not_retracted"] is True
    for review in audit["selected_reviews"]:
        assert review["old_strict_tier_decision"]["decision"].startswith("pending_")
        assert review["unchanged_checks"]["entity_mapping_verified"] is False
        assert review["unchanged_checks"]["no_provided_group_ids_discarded"] is True
        assert review["unchanged_checks"]["no_known_mandatory_temporal_protocol_bypassed"] is True
    excluded = {r["uci_id"]: r for r in audit["excluded_origins_unchanged"]}
    assert set(excluded) == {176, 372}
    assert excluded[372]["status"] == "blocked_primary_license_conflict"
    assert "GPL3.0+" in excluded[372]["evidence"]
    assert excluded[176]["status"] == "excluded_already_exposed"
    assert excluded[176]["current_ledger_matches"]
    assert audit["catalog_runtime_support"]["metadata_reservation_supported"] is True
    assert audit["catalog_runtime_support"]["tier_aware_evaluation_and_reporting_enforced_now"] is False
    assert audit["runtime_changes"] is False


@pytest.mark.usefixtures("pre_standard_iid_policy_release_registry")
def test_standard_iid_tier_preserves_old_registry_records_suites_and_no_exposure_matches():
    audit = catalog.load_suite(SUITES / "standard_iid_entity_reservation_audit_20261003.json")
    registry = json.loads(registry_before_qualified_binary_append())
    by_id = {r["family_id"]: r for r in registry["families"]}
    append = audit["registry_append_only"]
    assert len(by_id) == append["prior_family_count"] + 2
    for fid, expected in append["prior_family_record_sha256"].items():
        assert (
            hashlib.sha256(json.dumps(by_id[fid], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == expected
        )
    assert {k: v for k, v in registry.items() if k != "families"} == append["prior_top_level"]
    for name, expected in audit["preserved_prior_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / name).read_bytes()).hexdigest() == expected
    exposure = audit["exposure_audit"]
    rows = [[row[k] for k in ("family_id", "run_id", "role", "created_at")] for row in exposure["ledger"]["rows"]]
    assert len(rows) == exposure["ledger"]["row_count"] == 198
    assert (
        hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
        == exposure["ledger"]["logical_rows_sha256"]
    )
    assert exposure["ledger"]["selected_family_matches"] == []
    assert not set(append["appended_family_ids"]).intersection(row[0] for row in rows)
    assert (
        audit["payloads_read"]
        == audit["labels_read"]
        == audit["model_calls"]
        == audit["llm_calls"]
        == audit["ledger_writes"]
        == 0
    )


def test_qualified_policy_release_preserves_exact_two_sources_and_separate_tier():
    path = SUITES / "standard_iid_policy_confirmation_20261003.json"
    with pytest.raises(catalog.AdmissionError, match="confirmation is locked"):
        catalog.load_suite(path)
    suite = catalog.load_suite(path, allow_confirmation=True)
    original = historical_manifest_snapshot(SUITES / "standard_iid_entity_reservation_20261003.json")
    assert suite["family_count"] == len(suite["datasets"]) == 2
    assert suite["evaluation_tier"] == "standard_iid_unverified_entity"
    for spec, before in zip(suite["datasets"], original["datasets"], strict=True):
        assert {key for key in before if spec[key] != before[key]} == {"role", "prior_exposure"}
        assert spec["role"] == spec["_family"]["role"] == "confirmation"
        assert spec["prior_exposure"] == spec["_family"]["prior_exposure"] == "audited_unexposed"
        assert spec["_family"]["reservation"]["status"] == "released"
        assert spec["evaluation_tier"] == suite["evaluation_tier"]
        assert spec["claim_boundary"] == suite["claim_boundary"] == suite["preregistration"]["claim_boundary"]
        assert spec["claim_boundary"]["pool_with_verified_entity_cohorts"] is False
        assert spec["claim_boundary"]["clinical_generalization_claims_allowed"] is False
        assert spec["claim_boundary"]["patient_entity_independence_verified"] is False


def test_qualified_policy_release_strictly_binds_sampling_pins_policy_and_job_counts():
    from benchmarks import calibration_evaluation as controller

    suite = catalog.load_suite(SUITES / "standard_iid_policy_confirmation_20261003.json", allow_confirmation=True)
    registration = suite["preregistration"]
    config = {
        **copy.deepcopy(controller.DEFAULTS),
        **copy.deepcopy(controller.CONFIRMATION_SETTINGS),
        "allow_confirmation": True,
        "suite": str(SUITES / "standard_iid_policy_confirmation_20261003.json"),
    }
    specs, bound = controller._bind_confirmation(suite, False, config, 36000)
    assert bound == registration and len(specs) == 2
    assert registration["expected_cells"] == 96
    assert registration["expected_physical_jobs"] == 72
    assert 2 * 3 * 2 * len(controller.CONFIRMATION_JOBS) == 72
    assert 2 * 3 * 2 * sum(len(methods) for methods in controller.CONFIRMATION_JOBS.values()) == 96
    assert registration["primary_method_by_learner"] == {
        "linear": "featcopilot-frequency-aligned-robust",
        "lightgbm": "featcopilot-ensemble-robust",
    }
    assert registration["calibrated_outcomes_diagnostic_only"] is True
    assert registration["selection_parameters"]["ensemble_sampling"] == "subsample85"
    assert registration["selection_parameters"]["numeric_clip_quantiles"] == [0.0, 1.0]
    assert registration["seeds"] == [2030] and registration["sample_seed"] == 2026
    assert registration["max_rows"] == 0 and registration["total_active_seconds_ceiling"] == 36000
    assert registration["classification_admission_margin"] == {
        "min_total": 30,
        "min_outer_train": 20,
        "min_outer_test": 10,
    }
    with pytest.raises(ValueError, match="ensemble_sampling"):
        controller._bind_confirmation(suite, False, {**config, "ensemble_sampling": "cv3"}, 36000)
    wrong = copy.deepcopy(suite)
    wrong["preregistration"]["expected_physical_jobs"] = 96
    with pytest.raises(ValueError, match="expected_physical_jobs"):
        controller._bind_confirmation(wrong, False, copy.deepcopy(config), 36000)
    wrong = copy.deepcopy(suite)
    wrong["datasets"][1]["drop_columns"] = []
    with pytest.raises(ValueError, match="source/schema"):
        controller._bind_confirmation(wrong, False, copy.deepcopy(config), 36000)
    wrong = copy.deepcopy(suite)
    wrong["claim_boundary"]["pool_with_verified_entity_cohorts"] = True
    with pytest.raises(ValueError, match="claim boundary"):
        controller._bind_confirmation(wrong, False, copy.deepcopy(config), 36000)


def test_qualified_policy_release_records_exact_claim_and_shared_fit_cost_limits():
    suite = catalog.load_suite(SUITES / "standard_iid_policy_confirmation_20261003.json", allow_confirmation=True)
    p = suite["preregistration"]
    assert p["sampling_semantics"]["members"] == 3
    assert p["sampling_semantics"]["training_fraction"] == 0.85
    assert p["sampling_semantics"]["calibration_fraction"] == 0.15
    report = p["reporting_and_acceptance"]
    assert report["primary_raw_comparators"] == ["baseline-robust", "baseline-ensemble-robust"]
    assert report["native_fe_comparators"] == ["openfe-inductive-native-robust", "autofeat-native-robust"]
    assert "not native FE ensembles" in report["counterfactual_limit"]
    assert "BOTH" in report["acceptance"]["positive_vs_each_raw_control"]
    assert "5%" in report["acceptance"]["family_harm"]
    assert "96" in report["acceptance"]["coverage"] and "72" in report["acceptance"]["coverage"]
    assert report["acceptance"]["automatic_sota"] is False
    assert report["acceptance"]["automatic_default_promotion"] is False
    assert "Do not pool" in report["tier_reporting"]
    assert p["budget"]["worst_case_physical_timeout_hours"] == 18
    assert p["budget"]["run_ceiling_hours"] == 10


def test_qualified_policy_release_changes_only_two_registry_records_and_preserves_history():
    release = catalog.load_suite(SUITES / "standard_iid_policy_confirmation_release_20261003.json")
    transition = release["registry_transition"]
    assert transition["changed_family_ids"] == ["haberman-survival", "mammographic-mass"]
    assert transition["tier_and_claim_boundary_unchanged"] is True
    assert transition["nonselected_records_changed"] is False
    assert hashlib.sha256(registry_before_qualified_binary_append()).hexdigest() == transition["after_sha256"]
    current = catalog._families([])
    for fid, before in transition["before"].items():
        after = transition["after"][fid]
        assert current[fid] == after
        assert {key for key in set(before) | set(after) if before.get(key) != after.get(key)} == {
            "role",
            "prior_exposure",
            "reservation",
            "exposure_audit",
        }
        assert before["reservation"]["status"] == "active" and after["reservation"]["status"] == "released"
        assert before["evaluation_tier"] == after["evaluation_tier"]
        assert before["claim_boundary"] == after["claim_boundary"]
    for filename, checksum in release["preserved_prior_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / filename).read_bytes()).hexdigest() == checksum
    e = release["exposure_check"]
    rows = [[r[k] for k in ("family_id", "run_id", "role", "created_at")] for r in e["rows"]]
    assert len(rows) == e["row_count"] == 207
    assert hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == e["logical_rows_sha256"]
    assert e["selected_family_matches"] == []
    assert not set(transition["changed_family_ids"]).intersection(r[0] for r in rows)
    assert release["strict_controller_binding_verified"] is True
    assert (
        release["no_payload_reads"] and release["no_models"] and release["no_llm_calls"] and release["no_ledger_writes"]
    )


@pytest.mark.usefixtures("pre_thoracic_release_registry")
def test_qualified_binary_followup_reserves_only_thoracic_without_model_authorization(workspace):
    suite = catalog.load_suite(SUITES / "qualified_binary_reservation_20261004.json")
    assert suite["requested_origins"] == 2 and suite["reserved_origins"] == suite["family_count"] == 1
    assert suite["evaluation_tier"] == "standard_iid_unverified_entity"
    assert not suite["model_execution_authorized"] and not suite["payload_acquisition_authorized"]
    assert "preregistration" not in suite
    spec = suite["datasets"][0]
    assert spec["family_id"] == "thoracic-surgery"
    assert spec["role"] == spec["_family"]["role"] == "inventory"
    assert spec["prior_exposure"] == spec["_family"]["prior_exposure"] == "unknown"
    assert spec["_family"]["reservation"]["status"] == "active"
    assert spec["claim_boundary"]["clinical_generalization_claims_allowed"] is False
    assert spec["claim_boundary"]["patient_entity_disjoint_claims_allowed"] is False
    assert spec["claim_boundary"]["pool_with_verified_entity_cohorts"] is False
    catalog._validate_source(spec, runnable=True)
    catalog._validate_rights(spec["rights"], spec["name"], runnable=True)
    with pytest.raises(catalog.AdmissionError, match="must never download"):
        catalog.load_dataset(spec, workspace / "thoracic-forbidden")
    assert not (workspace / "thoracic-forbidden").exists()


@pytest.mark.usefixtures("pre_thoracic_release_registry")
def test_qualified_thoracic_preserves_original_binary_outcome_and_reviewed_input_mapping():
    spec = catalog.load_suite(SUITES / "qualified_binary_reservation_20261004.json")["datasets"][0]
    source = spec["source"]
    assert source["uci_id"] == 277 and source["dataset_id"] == 1506
    assert source["dataset_version"] == 1 and source["task_id"] == 9969
    assert source["native_target_name"] == "Risk1Yr"
    assert spec["target"] == "Class" and spec["target_transform"] == "identity"
    assert source["label_meanings"] == {"1": "death within one year after surgery", "2": "survival"}
    assert spec["drop_columns"] == []
    assert len(spec["categorical"]) == 13
    assert source["column_semantics"]["V1"]["native_name"] == "DGN"
    assert source["column_semantics"]["V16"]["native_name"] == "AGE"
    assert source["column_semantics"]["Class"]["native_name"] == "Risk1Yr"
    assert spec["metadata_profile"]["rows"] == 470 and spec["metadata_profile"]["predictors"] == 16
    assert spec["metadata_profile"]["class_counts"] == {"1": 70, "2": 400}
    assert spec["metadata_profile"]["minimum_class_count"] == 70
    assert spec["metadata_profile"]["source_ten_fold_minority_test_count_expectation"] == 7
    assert spec["metadata_profile"]["expected_three_fold_minimum_test_count"] == 23
    assert spec["split_policy"] == "official"
    assert source["official_split_dimensions"] == {"repeats": 1, "folds": 10, "samples": 1}
    assert spec["metadata_profile"]["payload_verified"] is False
    assert "retrospective" in spec["timing_and_entity_review"]["timing_limit"]
    assert spec["rights"]["license"] == "CC-BY-4.0"


@pytest.mark.usefixtures("pre_thoracic_release_registry")
def test_qualified_binary_followup_does_not_erase_qsar_cross_source_hold():
    audit = catalog.load_suite(SUITES / "qualified_binary_reservation_audit_20261004.json")
    pending = audit["pending_review"]
    assert pending["uci_id"] == 254
    assert pending["previous_provisional_family_id"] == "tabarena-46952"
    assert pending["openml_source_ids"] == [1494, 46952]
    assert pending["decision"] == "pending_prior_cross_source_origin_independence_hold"
    assert pending["no_active_reservation_added"] is True
    assert pending["exposed_related_family_ledger_rows"]
    assert "tabarena-46952" not in {r["family_id"] for r in audit["datasets"]}
    assert "tabarena-46952" not in catalog._families([])
    assert audit["summary"] == {
        "reviewed_origins": 2,
        "reserved_origins": 1,
        "pending_origins": 1,
        "payloads": 0,
        "models": 0,
        "llm_calls": 0,
    }


@pytest.mark.usefixtures("pre_thoracic_release_registry")
def test_qualified_binary_append_preserves_old_records_and_current_exposure_snapshot():
    audit = catalog.load_suite(SUITES / "qualified_binary_reservation_audit_20261004.json")
    registry = json.loads(registry_before_regression_stress_reuse())
    current = {row["family_id"]: row for row in registry["families"]}
    append = audit["registry_append_only"]
    assert len(current) == append["prior_family_count"] + 1
    assert append["appended_family_ids"] == ["thoracic-surgery"]
    for fid, checksum in append["prior_record_sha256"].items():
        assert (
            hashlib.sha256(json.dumps(current[fid], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == checksum
        )
    assert {k: v for k, v in registry.items() if k != "families"} == append["prior_top_level"]
    for filename, checksum in audit["preserved_prior_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / filename).read_bytes()).hexdigest() == checksum
    ledger = audit["exposure_audit"]["ledger"]
    rows = [[row[k] for k in ("family_id", "run_id", "role", "created_at")] for row in ledger["rows"]]
    assert len(rows) == ledger["row_count"] == 245
    assert hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == ledger["logical_rows_sha256"]
    assert ledger["selected_family_matches"] == []
    assert "thoracic-surgery" not in {row[0] for row in rows}
    assert audit["runtime_changes"] is False and audit["ledger_writes"] == 0


def test_thoracic_fixed_release_preserves_source_tier_and_exact_controller_pin():
    from benchmarks import blending_evaluation as controller
    from benchmarks.run_state import fingerprint

    path = SUITES / "thoracic_fixed_average_confirmation_20261004.json"
    with pytest.raises(catalog.AdmissionError, match="confirmation is locked"):
        catalog.load_suite(path)
    suite = catalog.load_suite(path, allow_confirmation=True)
    reservation = historical_manifest_snapshot(SUITES / "qualified_binary_reservation_20261004.json")
    spec, old = suite["datasets"][0], reservation["datasets"][0]
    assert suite["family_count"] == 1 and spec["family_id"] == "thoracic-surgery"
    assert {key for key in old if old[key] != spec[key]} == {"role", "prior_exposure"}
    assert spec["role"] == spec["_family"]["role"] == "confirmation"
    assert spec["prior_exposure"] == spec["_family"]["prior_exposure"] == "audited_unexposed"
    assert spec["_family"]["reservation"]["status"] == "released"
    assert spec["evaluation_tier"] == "standard_iid_unverified_entity"
    assert spec["claim_boundary"] == suite["claim_boundary"] == suite["preregistration"]["claim_boundary"]
    assert spec["claim_boundary"]["pool_with_verified_entity_cohorts"] is False
    assert spec["claim_boundary"]["clinical_generalization_claims_allowed"] is False
    assert spec["split_policy"] == "official"
    assert spec["source"]["official_split_dimensions"] == {"repeats": 1, "folds": 10, "samples": 1}
    assert "class_support" not in spec
    pin = fingerprint({key: value for key, value in spec.items() if not key.startswith("_")})
    assert pin == controller.CONFIRMATION_SPEC_SHA256
    assert suite["preregistration"]["dataset_spec_sha256"] == {"thoracic-surgery": pin}


def test_thoracic_fixed_release_binds_constant_average_only_and_exact_workload():
    from benchmarks import blending_evaluation as controller

    suite = catalog.load_suite(SUITES / "thoracic_fixed_average_confirmation_20261004.json", allow_confirmation=True)
    p = suite["preregistration"]
    config = {
        **copy.deepcopy(controller.DEFAULTS),
        **copy.deepcopy(controller.CONFIRMATION_SETTINGS),
        "suite": str(SUITES / "thoracic_fixed_average_confirmation_20261004.json"),
    }
    for key in ("oof_folds", "oof_seed_offset"):
        config.pop(key, None)
    specs, bound = controller._bind_confirmation(suite, False, config, 14400)
    assert bound == p and len(specs) == 1
    assert p["methods"] == [
        "baseline-robust",
        "featcopilot-frequency-aligned-robust",
        "prediction-average-robust",
        "openfe-inductive-native-robust",
        "autofeat-native-robust",
    ]
    assert p["learners"] == ["lightgbm"]
    assert p["seeds"] == [2031] and p["sample_seed"] == 2026
    assert p["max_rows"] == 0 and p["threads"] == 8 and p["memory_limit_mb"] == 4096
    assert p["case_timeout_seconds"] == 900 and p["total_active_seconds_ceiling"] == 14400
    assert p["expected_cells"] == 15 and p["expected_physical_jobs"] == 9
    assert p["split_plan"] == {"thoracic-surgery": {"policy": "iid", "folds": 3}}
    assert p["resplit_official_as_iid"] is True
    assert p["classification_admission_margin"] == {"min_total": 30, "min_outer_train": 20, "min_outer_test": 10}
    assert p["blend_mode"] == "fixed_average" and p["fixed_weight"] == 0.5
    assert p["primary_method"] == "prediction-average-robust"
    assert (
        not p["prediction_rule"]["oof"]
        and not p["prediction_rule"]["weight_learning"]
        and not p["prediction_rule"]["calibration"]
    )
    assert p["reporting_and_acceptance"] == controller._confirmation_rules()
    invalid = copy.deepcopy(suite)
    invalid["preregistration"]["fixed_weight"] = 0.6
    with pytest.raises(ValueError):
        controller._bind_confirmation(invalid, False, copy.deepcopy(config), 14400)
    invalid = copy.deepcopy(suite)
    invalid["datasets"][0]["drop_columns"] = ["V1"]
    with pytest.raises(ValueError, match="source/schema"):
        controller._bind_confirmation(invalid, False, copy.deepcopy(config), 14400)


def test_thoracic_fixed_release_preserves_history_and_single_record_transition():
    release = catalog.load_suite(SUITES / "thoracic_fixed_average_confirmation_release_20261004.json")
    transition = release["registry_transition"]
    assert transition["changed_family_ids"] == ["thoracic-surgery"]
    assert transition["nonselected_records_changed"] is False
    assert transition["tier_claim_boundary_changed"] is False
    before, after = transition["before"]["thoracic-surgery"], transition["after"]["thoracic-surgery"]
    assert {key for key in set(before) | set(after) if before.get(key) != after.get(key)} == {
        "role",
        "prior_exposure",
        "reservation",
        "exposure_audit",
    }
    assert before["reservation"]["status"] == "active" and after["reservation"]["status"] == "released"
    assert hashlib.sha256(registry_before_regression_stress_reuse()).hexdigest() == transition["after_sha256"]
    for name, checksum in release["preserved_prior_manifest_sha256"].items():
        assert hashlib.sha256((SUITES / name).read_bytes()).hexdigest() == checksum
    assert release["qsar_pending_unchanged"]["decision"] == "pending_prior_cross_source_origin_independence_hold"
    assert release["qsar_pending_unchanged"]["no_active_reservation_added"] is True
    assert release["strict_controller_binding_verified"] is True
    assert release["runtime_changes_by_catalog_owner"] is False
    assert (
        release["no_payload_reads"] and release["no_models"] and release["no_llm_calls"] and release["no_ledger_writes"]
    )
    evidence = release["exposure_check"]
    rows = [[r[key] for key in ("family_id", "run_id", "role", "created_at")] for r in evidence["rows"]]
    assert len(rows) == evidence["row_count"] == 245
    assert (
        hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == evidence["logical_rows_sha256"]
    )
    assert not any(row[0] == "thoracic-surgery" for row in rows)


def test_regression_stress_restoration_reverses_only_the_audited_delta_byte_exactly():
    audit = json.loads((SUITES / "regression_exposed_stress_replay_audit_20261008.json").read_bytes())
    delta = audit["registry_delta"]
    current_bytes = (ROOT / "benchmarks" / "families.json").read_bytes()
    current = json.loads(current_bytes)
    restored_bytes = registry_before_regression_stress_reuse()
    restored = json.loads(restored_bytes)
    assert hashlib.sha256(current_bytes).hexdigest() == delta["after_file_sha256"]
    assert hashlib.sha256(restored_bytes).hexdigest() == delta["before_file_sha256"]
    assert len(current["families"]) == len(restored["families"]) == delta["family_count_unchanged"]
    assert current.keys() == restored.keys()
    assert {k: v for k, v in current.items() if k != "families"} == {
        k: v for k, v in restored.items() if k != "families"
    }
    selected = set(delta["changed_family_ids"])
    for after, before in zip(current["families"], restored["families"], strict=True):
        assert after["family_id"] == before["family_id"]
        if after["family_id"] not in selected:
            assert after == before
            continue
        assert {key for key in set(before) | set(after) if before.get(key) != after.get(key)} == {
            "role",
            "prior_exposure",
            "development_reuse_audit",
        }
        assert after["role"] == "development" and after["prior_exposure"] == "exposed"
        assert before["role"] == "confirmation" and before["prior_exposure"] == "audited_unexposed"
        for field in delta["preserved_fields"]:
            assert after[field] == before[field]
        assert after["development_reuse_audit"] == delta["added_audit_value"]
    assert (ROOT / "benchmarks" / "families.json").read_bytes() == current_bytes


def test_exposed_regression_stress_specs_preserve_all_original_public_fields():
    path = SUITES / "regression_exposed_stress_replay_20261008.json"
    suite = catalog.load_suite(path)
    audit = json.loads((SUITES / "regression_exposed_stress_replay_audit_20261008.json").read_bytes())
    original_path = SUITES / audit["source_manifest"]
    original = historical_manifest_snapshot(original_path)
    by_family = {spec["family_id"]: spec for spec in original["datasets"]}
    assert suite["family_count"] == len(suite["datasets"]) == 2
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == audit["source_manifest_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["replay_manifest_sha256"]
    for spec in suite["datasets"]:
        before = by_family[spec["family_id"]]
        public = {key: value for key, value in spec.items() if not key.startswith("_")}
        assert public.keys() == before.keys()
        assert {key for key in public if public[key] != before[key]} == {"role", "prior_exposure"}
        assert spec["role"] == spec["_family"]["role"] == "development"
        assert spec["prior_exposure"] == spec["_family"]["prior_exposure"] == "exposed"
        assert spec["_family"]["reservation"]["status"] == "released"
        expected = copy.deepcopy(before)
        expected.update(role="development", prior_exposure="exposed")
        normalized = catalog.validate_manifest(
            {"name": "expected-stress-reuse", "datasets": [expected]}, manifest_path=original_path
        )["datasets"][0]
        assert spec == normalized
        fingerprints = audit["dataset_delta"]["normalized_public_fingerprints"][spec["family_id"]]
        assert (
            hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == fingerprints["after"]
        )
        assert (
            hashlib.sha256(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == fingerprints["before"]
        )
        with pytest.raises(catalog.AdmissionError, match="already exposed"):
            catalog.validate_manifest(
                {"name": "forbidden-fresh-stress-reuse", "datasets": [before]}, allow_confirmation=True
            )


@pytest.mark.parametrize("max_rows", [1, 5000])
def test_exposed_hardware_rejects_every_nonzero_cap_before_provider_access(workspace, monkeypatch, max_rows):
    suite = catalog.load_suite(SUITES / "regression_exposed_stress_replay_20261008.json")
    hardware = next(spec for spec in suite["datasets"] if spec["family_id"] == "computer-hardware-performance")
    assert hardware["role"] == "development" and hardware["split_policy"] == "official"

    def forbidden_provider(*args, **kwargs):
        pytest.fail("Official row-cap rejection must precede any provider or payload access")

    monkeypatch.setattr(catalog, "_load_openml", forbidden_provider)
    cache = workspace / "hardware-cap-must-not-acquire"
    with pytest.raises(catalog.AdmissionError, match="Nonzero row caps are forbidden"):
        catalog.load_dataset(hardware, cache, max_rows=max_rows)
    assert not cache.exists()


def test_regression_stress_plan_retains_scope_separation_and_single_shared_budget():
    suite = catalog.load_suite(SUITES / "regression_exposed_stress_replay_20261008.json")
    plan = suite["study_plan"]
    assert "preregistration" not in suite
    assert len(plan["methods"]) == 3 and plan["learners"] == ["linear", "lightgbm"]
    assert plan["seeds"] == [42]
    assert plan["hardware_stress"]["folds"] == 10 and plan["hardware_stress"]["max_rows"] == 0
    assert plan["hardware_stress"]["split_policy"] == "official"
    assert plan["hardware_stress"]["all_source_official_splits"] is True
    assert plan["hardware_stress"]["official_split_dimensions"] == {"repeats": 1, "folds": 10, "samples": 1}
    assert plan["infrared_stress"]["folds"] == 3 and plan["infrared_stress"]["max_rows"] == 5000
    assert plan["infrared_stress"]["split_policy"] == "iid"
    assert plan["primary_seven"]["expected_cells"] == 126
    assert plan["infrared_stress"]["expected_cells"] == 18
    assert plan["hardware_stress"]["expected_cells"] == 60
    assert plan["expected_total_cells"] == 126 + 18 + 60
    assert plan["shared_total_active_seconds_ceiling"] == 14400
    assert plan["reporting_constraints"]["stress_in_primary_mean"] is False
    assert plan["reporting_constraints"]["combine_stress_scopes"] is False
    assert plan["reporting_constraints"]["fresh_confirmation_claim"] is False
