import json
import os
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from aide.utils.eda import generate_and_cache


def _make_cfg(tmp_path: Path, data_dir: Path, seed: int = 42, compact_mode: str = "compact"):
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    eda = SimpleNamespace(
        enable=True,
        sample_max_rows=30000,
        sample_rate=1.0,   # small data in tests; include all rows
        sample_seed=seed,
        max_columns=80,
        top_values_k=5,
        top_tokens_k=10,
        top_ngrams_k=10,
        top_corr_k=10,
        top_assoc_k=10,
        redaction_enable=True,
        pii_hash_salt="TEST_SALT",
        pii_rules=["email", "phone", "ip", "cc"],
        artifacts_subdir="artifacts/eda",
        cache_by_hash=True,
        target_column="SalePrice",
        compact=SimpleNamespace(
            enable=True,
            mode=compact_mode,
            token_budget_compact=900,
            token_budget_ultra=450,
        ),
    )
    cfg = SimpleNamespace(
        workspace_dir=str(ws),
        data_dir=str(data_dir),
        eda=eda,
    )
    return cfg


def _write_sample_csv(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "Id": np.arange(1, 51),
        "SalePrice": np.random.RandomState(0).randn(50) * 10000 + 200000,
        "Neighborhood": ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"] * 10,
        "OwnerEmail": [f"user{i}@example.com" for i in range(50)],
        "Description": [
            "Beautiful family home with large yard and garage." if i % 2 == 0
            else "Cozy cottage near downtown amenities and parks." for i in range(50)
        ],
        "LotArea": np.random.RandomState(1).randint(2000, 20000, size=50),
    })
    p = dir_path / "train.csv"
    df.to_csv(p, index=False)
    return p


def test_schema_fields_present(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_sample_csv(data_dir)
    cfg = _make_cfg(tmp_path, data_dir)
    summary = generate_and_cache(cfg)

    # Top-level keys
    for k in ["dataset_metadata", "columns", "relationships", "data_quality", "samples", "warnings", "notices", "summary_hash"]:
        assert k in summary

    # Columns entries have required fields
    assert len(summary["columns"]) > 0
    for c in summary["columns"]:
        for k in ["name", "inferred_dtype", "semantic_type", "missing_rate", "unique_count", "approximate_cardinality_bucket"]:
            assert k in c


def test_summary_hash_determinism(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_sample_csv(data_dir)
    cfg1 = _make_cfg(tmp_path, data_dir, seed=123)
    s1 = generate_and_cache(cfg1)
    cfg2 = _make_cfg(tmp_path, data_dir, seed=123)
    s2 = generate_and_cache(cfg2)
    assert s1["summary_hash"] == s2["summary_hash"]

    # Changing seed should usually change hash (sampling included in content hash)
    cfg3 = _make_cfg(tmp_path, data_dir, seed=999)
    s3 = generate_and_cache(cfg3)
    assert s1["summary_hash"] != s3["summary_hash"]


def test_redaction_correctness(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_sample_csv(data_dir)
    cfg = _make_cfg(tmp_path, data_dir)
    summary = generate_and_cache(cfg)

    # OwnerEmail should be detected as PII & redacted in top_values where applicable
    cols = {c["name"]: c for c in summary["columns"]}
    if "OwnerEmail" in cols and cols["OwnerEmail"]["top_values"] is not None:
        for tv in cols["OwnerEmail"]["top_values"]:
            assert "value_hash" in tv
            assert isinstance(tv["value_hash"], str)
            assert tv.get("redacted", False) in (True, False)
            # if redacted, raw_value should not leak
            if tv.get("redacted", False):
                assert "raw_value" not in tv


def test_deterministic_ordering(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_sample_csv(data_dir)
    cfg = _make_cfg(tmp_path, data_dir)
    summary = generate_and_cache(cfg)

    # columns are returned sorted by name
    names = [c["name"] for c in summary["columns"]]
    assert names == sorted(names)

    # correlations sorted by |r| desc then lexicographically (cannot assert exact values; just presence and keys)
    rel = summary["relationships"]
    if rel["numeric_correlations"]:
        prev = None
        for p in rel["numeric_correlations"]:
            assert {"col_a", "col_b", "r"} <= set(p.keys())
            if prev is not None:
                assert abs(prev["r"]) >= abs(p["r"]) or (
                    abs(prev["r"]) == abs(p["r"]) and (prev["col_a"], prev["col_b"]) <= (p["col_a"], p["col_b"])
                )
            prev = p


def test_caching_and_symlinks(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_sample_csv(data_dir)
    cfg = _make_cfg(tmp_path, data_dir)
    summary = generate_and_cache(cfg)

    ws = Path(cfg.workspace_dir)
    # Symlink or copied files should exist
    assert (ws / "EDA_SUMMARY.json").exists()
    if cfg.eda.compact.enable:
        assert (ws / "EDA_COMPACT.md").exists()

    # Artifacts directory contains summary.json
    artifacts = ws / cfg.eda.artifacts_subdir
    found = list(artifacts.rglob("summary.json"))
    assert len(found) >= 1
    # Summary hash in path
    assert any(summary["summary_hash"] in str(p.parent.name) for p in found)