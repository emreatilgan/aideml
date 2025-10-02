"""Deterministic EDA pipeline for AIDE.

Produces:
- A canonical, machine-readable JSON summary for caching/reuse.
- A compact Markdown summary optimized for LLM prompt injection (configurable modes).

Artifacts are stored under:
  workspace_dir/artifacts/eda/{dataset_name}__{summary_hash}/
  - summary.json
  - compact.md and/or ultra.md (if enabled)

Symlinks for convenience:
  workspace_dir/EDA_SUMMARY.json -> latest summary.json
  workspace_dir/EDA_COMPACT.md -> latest compact.md (if generated)
"""

from __future__ import annotations

import json
import math
import os
import re
import hashlib
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from collections import Counter

import numpy as np
import pandas as pd

# Mutual Information is optional; guard import
try:
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


# -----------------------
# Public entry points
# -----------------------

def generate_and_cache(cfg) -> dict:
    """
    Entry point. Reads primary dataset under cfg.workspace_dir/input, computes EDA summary,
    writes artifacts, and creates symlinks in workspace root.

    Returns:
        dict: The JSON summary object (already includes summary_hash).
    """
    base_input = Path(cfg.workspace_dir) / "input"
    dataset_path = _resolve_primary_dataset(base_input)
    # Prefer original data_dir name as dataset_name (more stable/semantic than workspace)
    dataset_name = Path(getattr(cfg, "data_dir", base_input)).name or dataset_path.stem

    # Read CSV with minimal memory surprises
    df = pd.read_csv(dataset_path, low_memory=False)

    # Sampling (deterministic)
    df_sampled = _apply_sampling(df, cfg.eda.sample_rate, cfg.eda.sample_max_rows, cfg.eda.sample_seed)

    # Column throttling (deterministic selection)
    target_col = cfg.eda.target_column
    time_col = _detect_time_column(df_sampled)
    df_throttled, kept_cols = _throttle_columns(df_sampled, max_columns=cfg.eda.max_columns,
                                               target_col=target_col, time_col=time_col)

    # Compute summary
    summary = compute_eda_summary(
        df=df_throttled,
        cfg=cfg.eda,
        data_path=str(dataset_path),
        dataset_name=dataset_name,
        kept_cols=kept_cols,
        original_rows=len(df),
        original_cols=df.shape[1],
        time_col=time_col,
        target_col=target_col
    )

    # Canonicalize and hash
    canonical_obj = _canonical_for_hash(summary)
    summary_hash = _stable_hash(canonical_obj, salt=cfg.eda.pii_hash_salt)
    summary["summary_hash"] = summary_hash

    # Persist artifacts
    artifacts_dir = Path(cfg.workspace_dir) / cfg.eda.artifacts_subdir
    run_dir = artifacts_dir / f"{dataset_name}__{summary_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)

    # Compact Markdown(s)
    compact_link_path = None
    if cfg.eda.compact.enable:
        mode = (cfg.eda.compact.mode or "compact").lower()
        if mode not in ("compact", "ultra"):
            mode = "compact"

        # Optionally produce both, but minimally produce the configured mode
        md_text = pack_compact_summary(summary, cfg.eda, mode=mode)
        md_name = "compact.md" if mode == "compact" else "ultra.md"
        md_path = run_dir / md_name
        md_path.write_text(md_text, encoding="utf-8")

        # Prepare symlink convenience name
        compact_link_path = Path(cfg.workspace_dir) / "EDA_COMPACT.md"
        _symlink_or_copy(md_path, compact_link_path)

    # Update top-level summary symlink
    top_summary = Path(cfg.workspace_dir) / "EDA_SUMMARY.json"
    _symlink_or_copy(summary_path, top_summary)

    return summary


def compute_eda_summary(
    df: pd.DataFrame,
    cfg,
    data_path: str,
    dataset_name: str,
    kept_cols: Optional[List[str]] = None,
    original_rows: Optional[int] = None,
    original_cols: Optional[int] = None,
    time_col: Optional[str] = None,
    target_col: Optional[str] = None,
) -> dict:
    """Compute the JSON EDA summary with deterministic ordering."""

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Basic metadata
    row_count = int(df.shape[0])
    col_count = int(df.shape[1])

    # Infer problem type
    inferred_target = target_col or _infer_target_column(df)
    problem_type = _infer_problem_type(df, inferred_target)

    # Content hash is a hash over a signature of the sampled data + config knobs that change summary content
    content_hash = _stable_hash({
        "schema": list(df.columns),
        "n_rows": row_count,
        "n_cols": col_count,
        "sample_seed": cfg.sample_seed,
        "sample_rate": cfg.sample_rate,
        "sample_max_rows": cfg.sample_max_rows,
        "max_columns": cfg.max_columns,
        "top_values_k": cfg.top_values_k,
        "top_tokens_k": cfg.top_tokens_k,
        "top_ngrams_k": cfg.top_ngrams_k,
        "top_corr_k": cfg.top_corr_k,
        "top_assoc_k": cfg.top_assoc_k,
        "target": inferred_target,
    }, salt=cfg.pii_hash_salt)

    # Per-column stats
    columns_info = _compute_column_stats(df, cfg, inferred_target)

    # Relationships, associations, leakage, drift
    relationships = _compute_relationships(df, cfg, inferred_target)

    # Data quality
    data_quality = _compute_data_quality(df)

    # Samples (sanitized)
    samples = _make_samples(df, cfg)

    # Warnings & notices collection
    warnings, notices = _collect_warnings_notices(columns_info, data_quality, relationships)

    # Time-aware heuristics
    time_info = _time_series_heuristics(df, time_col)

    summary = {
        "dataset_metadata": {
            "dataset_name": dataset_name,
            "data_path": data_path,
            "row_count": int(original_rows or row_count),
            "col_count": int(original_cols or col_count),
            "target_column": inferred_target,
            "problem_type": problem_type,
            "created_at": created_at,
            "content_hash": content_hash,
        },
        "columns": columns_info,
        "relationships": relationships,
        "data_quality": data_quality,
        "samples": samples,
        "time_series": time_info,
        "warnings": warnings,
        "notices": notices,
    }

    # Deterministic ordering of keys for nested dicts/lists is handled by:
    # - sorting columns by name ascending in _compute_column_stats
    # - sorting relationships arrays by tie-breakers
    # - json.dump(..., sort_keys=True) at write time
    return summary


def pack_compact_summary(summary: dict, eda_cfg, mode: str = "compact") -> str:
    """
    Convert summary JSON to token-efficient Markdown for LLM prompts.
    Token budget is approximated by characters/4. Sections have priority and hard truncation if needed.
    """
    # Budgets
    if mode == "ultra":
        token_budget = int(eda_cfg.compact.token_budget_ultra)
    else:
        token_budget = int(eda_cfg.compact.token_budget_compact)

    char_budget = token_budget * 4

    meta = summary.get("dataset_metadata", {})
    cols = summary.get("columns", [])
    rel = summary.get("relationships", {})
    dq = summary.get("data_quality", {})
    tinfo = summary.get("time_series", {})
    warnings = summary.get("warnings", [])
    notices = summary.get("notices", [])

    # Predictive salience ranking
    ranked_cols = _predictive_salience_ranking(summary, eda_cfg)

    # Section builders
    parts: List[str] = []

    # 1) Overview
    overview = _md_overview(meta, dq, tinfo)
    parts.append(overview)

    # 2) Top columns by salience
    n_cols = 12 if mode == "compact" else 8
    parts.append(_md_top_columns(cols, ranked_cols, n_cols))

    # 3) Data quality issues
    parts.append(_md_data_quality(dq, warnings))

    # 4) Correlations/associations + leakage
    parts.append(_md_relationships(rel))

    # 5) Feature engineering checklist
    parts.append(_md_fe_checklist(cols, tinfo, meta.get("problem_type")))

    # Join and truncate by budget
    text = "\n".join([p for p in parts if p and p.strip()])

    if len(text) > char_budget:
        # Priority order above, iterative truncation from the tail
        chunks = [p for p in parts if p and p.strip()]
        out = ""
        for i, ch in enumerate(chunks):
            if len(out) + len(ch) + 1 <= char_budget:
                out += ("\n" if out else "") + ch
            else:
                # Truncate the current section to fit
                remaining = char_budget - len(out) - 1
                if remaining > 0:
                    out += "\n" + ch[:remaining] + "\n... (truncated)"
                break
        text = out

    return text


# -----------------------
# Internal helpers
# -----------------------

def _resolve_primary_dataset(base_input: Path) -> Path:
    csvs = sorted(base_input.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {base_input}")

    # Prefer train.* if present; otherwise pick the largest by file size
    train_like = [p for p in csvs if "train" in p.name.lower()]
    if train_like:
        return train_like[0]

    try:
        return max(csvs, key=lambda p: p.stat().st_size)
    except Exception:
        return csvs[0]


def _apply_sampling(df: pd.DataFrame, sample_rate: float, sample_max_rows: int, seed: int) -> pd.DataFrame:
    n = len(df)
    max_by_rate = int(math.ceil(n * float(sample_rate)))
    k = min(n, max(sample_max_rows, 1), max_by_rate if max_by_rate > 0 else n)
    if k >= n:
        return df
    return df.sample(n=k, random_state=seed)


def _throttle_columns(df: pd.DataFrame, max_columns: int, target_col: Optional[str], time_col: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    cols = list(df.columns)
    if len(cols) <= max_columns:
        return df, cols

    # Always keep target/time if present
    keep = set()
    if target_col and target_col in cols:
        keep.add(target_col)
    if time_col and time_col in cols:
        keep.add(time_col)

    # Heuristic ranking: numeric high variance first, then categorical entropy
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if c not in keep and not pd.api.types.is_numeric_dtype(df[c])]

    var_pairs = []
    for c in num_cols:
        try:
            var = float(pd.Series(df[c]).var(skipna=True))
        except Exception:
            var = 0.0
        var_pairs.append((c, -var))  # negative for ascending sort to get largest first

    ent_pairs = []
    for c in cat_cols:
        try:
            s = df[c].astype("object")
            vc = s.value_counts(dropna=True)
            probs = (vc / max(vc.sum(), 1.0)).values
            entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        except Exception:
            entropy = 0.0
        ent_pairs.append((c, -entropy))

    ranked = sorted(var_pairs, key=lambda t: (t[1], t[0])) + sorted(ent_pairs, key=lambda t: (t[1], t[0]))

    for c in ranked:
        if len(keep) >= max_columns:
            break
        keep.add(c[0])

    kept_cols = sorted(list(keep))
    return df[kept_cols].copy(), kept_cols


def _infer_target_column(df: pd.DataFrame) -> Optional[str]:
    # Heuristics: look for typical target names
    candidates = ["target", "label", "y", "SalePrice", "Survived", "Class", "Response"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: last column if appears numeric and non-id-like
    last = df.columns[-1]
    if pd.api.types.is_numeric_dtype(df[last]) and "id" not in last.lower():
        return last
    return None


def _infer_problem_type(df: pd.DataFrame, target_col: Optional[str]) -> str:
    if target_col is None or target_col not in df.columns:
        return "unspecified"
    s = df[target_col]
    if pd.api.types.is_numeric_dtype(s):
        # Heuristic for classification: small unique set (<=10) and integer-like
        nunique = s.nunique(dropna=True)
        if (nunique <= 10) and (np.allclose(s.dropna() % 1, 0)):
            return "classification"
        return "regression"
    return "classification"


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
        # Try parse small sample
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            if parsed.notna().mean() > 0.9:
                return c
        except Exception:
            pass
    return None


def _dtype_category(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_string_dtype(series) or series.dtype == "object":
        # crude text vs categorical; short avg length -> categorical
        try:
            avg_len = series.dropna().astype(str).map(len).mean()
            if avg_len and avg_len > 40:
                return "text"
            return "category"
        except Exception:
            return "category"
    return "unknown"


_PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?\d{1,3})?[-. (]*\d{2,4}[-. )]*\d{3,4}[-. ]*\d{3,4}"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "cc": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def _is_pii_like(val: Any, rules: List[str]) -> bool:
    try:
        s = str(val)
    except Exception:
        return False
    for r in rules:
        pat = _PII_PATTERNS.get(r)
        if pat and pat.search(s):
            return True
    return False


def _hash_value(val: Any, salt: str) -> str:
    h = hashlib.sha256()
    h.update((salt + repr(val)).encode("utf-8", errors="ignore"))
    return "HASH:" + h.hexdigest()[:16]


def _redact_value(val: Any, cfg) -> Tuple[str, bool]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ("<NA>", False)
    if not cfg.redaction_enable:
        # safe-ish fallback: if string too long, still hash to contain prompt length
        if isinstance(val, str) and len(val) > 80:
            return (_hash_value(val, cfg.pii_hash_salt), True)
        return (str(val), False)
    if _is_pii_like(val, cfg.pii_rules) or (isinstance(val, str) and len(val) > 40):
        return (_hash_value(val, cfg.pii_hash_salt), True)
    return (str(val), False)


def _compute_column_stats(df: pd.DataFrame, cfg, target_col: Optional[str]) -> List[dict]:
    cols = sorted(list(df.columns))
    out: List[dict] = []
    for c in cols:
        s = df[c]
        dtype_cat = _dtype_category(s)
        missing_rate = float(pd.isna(s).mean())
        unique_count = int(s.nunique(dropna=True))

        # cardinality bucket
        if unique_count <= 10:
            card_bucket = "low"
        elif unique_count <= 100:
            card_bucket = "medium"
        elif unique_count <= 1000:
            card_bucket = "high"
        else:
            card_bucket = "huge"

        # top values
        top_values_payload = None
        if dtype_cat in ("category", "text", "boolean"):
            vc = s.astype("object").value_counts(dropna=True).head(int(cfg.top_values_k))
            tv = []
            for v, cnt in vc.items():
                rv, red = _redact_value(v, cfg)
                entry = {"value_hash": _hash_value(v, cfg.pii_hash_salt), "count": int(cnt), "redacted": red}
                # Only include raw_value when not redacted and short
                if not red and isinstance(v, str) and len(v) <= 40:
                    entry["raw_value"] = rv
                elif not red and not isinstance(v, str):
                    entry["raw_value"] = rv
                tv.append(entry)
            # deterministic sort: count desc, then value_hash asc
            tv = sorted(tv, key=lambda d: (-d["count"], d["value_hash"]))
            top_values_payload = tv

        numerical_stats = None
        categorical_stats = None
        text_stats = None

        if dtype_cat == "numeric":
            x = pd.to_numeric(s, errors="coerce")
            try:
                q = x.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            except Exception:
                q = pd.Series([np.nan]*7, index=[0.01,0.05,0.25,0.50,0.75,0.95,0.99])
            # Robust outliers via IQR
            try:
                q1, q3 = q.loc[0.25], q.loc[0.75]
                iqr = q3 - q1
                if pd.isna(iqr) or iqr == 0:
                    outlier_share = 0.0
                else:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_share = float(((x < lower) | (x > upper)).mean())
            except Exception:
                outlier_share = 0.0
            try:
                mean = float(x.mean())
                std = float(x.std())
                vmin = float(np.nanmin(x))
                vmax = float(np.nanmax(x))
            except Exception:
                mean = std = vmin = vmax = np.nan

            numerical_stats = {
                "min": vmin,
                "max": vmax,
                "mean": mean,
                "std": std,
                "quantiles": {
                    "p1": _safe_float(q.loc[0.01]),
                    "p5": _safe_float(q.loc[0.05]),
                    "p25": _safe_float(q.loc[0.25]),
                    "p50": _safe_float(q.loc[0.50]),
                    "p75": _safe_float(q.loc[0.75]),
                    "p95": _safe_float(q.loc[0.95]),
                    "p99": _safe_float(q.loc[0.99]),
                },
                "outlier_share": float(outlier_share),
            }

        elif dtype_cat in ("category", "boolean"):
            # Concentration = share of the most frequent value
            try:
                vc = s.astype("object").value_counts(dropna=True)
                total = max(int(vc.sum()), 1)
                concentration = float(vc.max() / total) if not vc.empty else 0.0
                probs = (vc / total).values if total > 0 else np.array([1.0])
                entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
                top_k = [
                    {
                        "value_hash": _hash_value(v, cfg.pii_hash_salt),
                        "count": int(cnt),
                    }
                    for v, cnt in vc.head(int(cfg.top_values_k)).items()
                ]
                top_k = sorted(top_k, key=lambda d: (-d["count"], d["value_hash"]))
            except Exception:
                concentration, entropy, top_k = 0.0, 0.0, []
            categorical_stats = {
                "top_k": top_k,
                "concentration": concentration,
                "entropy": entropy,
            }

        elif dtype_cat == "text":
            # Simple tokenization
            try:
                s_str = s.dropna().astype(str)
                lengths = s_str.map(len)
                lq = lengths.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
                # tokenization by alnum words
                toks = []
                for line in s_str.sample(min(len(s_str), 10000), random_state=cfg.sample_seed):
                    toks.extend(re.findall(r"[A-Za-z0-9_]+", line.lower()))
                token_counts = Counter(toks)
                top_tokens = [
                    {"token_hash": _hash_value(t, cfg.pii_hash_salt), "count": int(c)}
                    for t, c in token_counts.most_common(int(cfg.top_tokens_k))
                ]
                # bigrams
                bigrams = Counter(zip(toks, toks[1:])) if len(toks) > 1 else Counter()
                top_ngrams = [
                    {"ngram_hash": _hash_value(" ".join(bg), cfg.pii_hash_salt), "count": int(c)}
                    for bg, c in bigrams.most_common(int(cfg.top_ngrams_k))
                ]
                vocab_size = int(len(token_counts))
                # trivial OOV proxy is not applicable without reference vocab; set to null
                oov_rate = None
                text_stats = {
                    "length_quantiles": {
                        "p1": _safe_float(lq.loc[0.01]),
                        "p25": _safe_float(lq.loc[0.25]),
                        "p50": _safe_float(lq.loc[0.50]),
                        "p75": _safe_float(lq.loc[0.75]),
                        "p99": _safe_float(lq.loc[0.99]),
                    },
                    "top_tokens": top_tokens,
                    "top_ngrams": top_ngrams,
                    "vocab_size": vocab_size,
                    "oov_rate": oov_rate,
                }
            except Exception:
                text_stats = None

        out.append({
            "name": c,
            "inferred_dtype": str(s.dtype),
            "semantic_type": dtype_cat,
            "missing_rate": float(missing_rate),
            "unique_count": int(unique_count),
            "approximate_cardinality_bucket": card_bucket,
            "top_values": top_values_payload,
            "numerical_stats": numerical_stats,
            "categorical_stats": categorical_stats,
            "text_stats": text_stats,
        })
    return out


def _safe_float(x) -> Optional[float]:
    try:
        f = float(x)
        if np.isnan(f):
            return None
        return f
    except Exception:
        return None


def _compute_relationships(df: pd.DataFrame, cfg, target_col: Optional[str]) -> dict:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rel = {
        "numeric_correlations": [],
        "target_associations": [],
        "leakage_flags": [],
        "drift": None,
        "simple_interactions": [],
    }

    # Pairwise numeric correlations (top-K by |r|)
    try:
        corr = df[num_cols].corr(numeric_only=True) if num_cols else pd.DataFrame()
        pairs = []
        for i, a in enumerate(num_cols):
            for b in num_cols[i+1:]:
                r = corr.at[a, b] if a in corr.index and b in corr.columns else np.nan
                if pd.isna(r):
                    continue
                pairs.append({"col_a": a, "col_b": b, "r": float(r)})
        # sort deterministically by |r| desc then lexical
        pairs = sorted(pairs, key=lambda d: (-abs(d["r"]), d["col_a"], d["col_b"]))
        rel["numeric_correlations"] = pairs[: int(cfg.top_corr_k)]
    except Exception:
        pass

    # Target associations
    if target_col and target_col in df.columns:
        try:
            y = df[target_col]
            X = df.drop(columns=[target_col])
            assoc = []
            # Numeric correlation to target (pearson) for numeric features
            if pd.api.types.is_numeric_dtype(y):
                for c in X.columns:
                    if pd.api.types.is_numeric_dtype(X[c]):
                        try:
                            r = float(np.corrcoef(pd.to_numeric(X[c], errors="coerce"),
                                                  pd.to_numeric(y, errors="coerce"))[0, 1])
                        except Exception:
                            r = np.nan
                        if not pd.isna(r):
                            assoc.append({"column": c, "association": r, "metric": "pearson"})
                # Mutual info (optional)
                if HAVE_SKLEARN:
                    # Select numeric columns only for MI regression
                    X_num = X.select_dtypes(include=[np.number])
                    Xn = X_num.fillna(X_num.mean())
                    yn = pd.to_numeric(y, errors="coerce").fillna(y.mean())
                    try:
                        mi = mutual_info_regression(Xn, yn, random_state=cfg.sample_seed)
                        for c, v in zip(X_num.columns, mi):
                            assoc.append({"column": str(c), "association": float(v), "metric": "mi"})
                    except Exception:
                        pass
            else:
                # Classification-like target
                y_enc = pd.factorize(y, sort=True)[0]
                if HAVE_SKLEARN and len(np.unique(y_enc)) > 1:
                    X_num = X.select_dtypes(include=[np.number])
                    Xn = X_num.fillna(X_num.mean())
                    try:
                        mi = mutual_info_classif(Xn, y_enc, random_state=cfg.sample_seed)
                        for c, v in zip(X_num.columns, mi):
                            assoc.append({"column": str(c), "association": float(v), "metric": "mi"})
                    except Exception:
                        pass
            assoc = sorted(assoc, key=lambda d: (-abs(d["association"]), d["metric"], d["column"]))
            rel["target_associations"] = assoc[: int(cfg.top_assoc_k)]
        except Exception:
            pass

        # Leakage flags: id-like or suspiciously high target association
        try:
            flags = []
            for c in df.columns:
                if c == target_col:
                    continue
                if "id" in c.lower():
                    flags.append({"column": c, "reason": "id-like name"})
            # Also flag features with very high pearson (>|0.98|) to target
            for a in rel.get("target_associations", []):
                if a["metric"] == "pearson" and abs(a["association"]) > 0.98:
                    flags.append({"column": a["column"], "reason": "near-perfect correlation with target"})
            # Deterministic order
            flags = sorted(flags, key=lambda d: (d["column"], d["reason"]))
            rel["leakage_flags"] = flags
        except Exception:
            pass

    # simple_interactions placeholder (lightweight)
    rel["simple_interactions"] = []

    return rel


def _compute_data_quality(df: pd.DataFrame) -> dict:
    dq = {
        "duplicate_row_rate": 0.0,
        "constant_column_list": [],
        "high_missing_columns": [],
        "invalid_values_count": {},
        "schema_anomalies": [],
    }
    try:
        if len(df) > 0:
            dup_rate = float(df.duplicated().mean())
        else:
            dup_rate = 0.0
        dq["duplicate_row_rate"] = dup_rate
    except Exception:
        pass

    try:
        constants = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        dq["constant_column_list"] = sorted(constants)
    except Exception:
        pass

    try:
        miss = df.isna().mean()
        high_missing = miss[miss > 0.5].index.tolist()
        dq["high_missing_columns"] = sorted(high_missing)
    except Exception:
        pass

    try:
        invalid_counts = {}
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                s = pd.to_numeric(df[c], errors="coerce")
                invalid_counts[c] = int(np.isinf(s).sum())
        dq["invalid_values_count"] = invalid_counts
    except Exception:
        pass

    # Schema anomalies left as empty list placeholder for now
    return dq


def _make_samples(df: pd.DataFrame, cfg) -> dict:
    row_n = min(5, len(df))
    row_sample = []
    if row_n > 0:
        for _, row in df.sample(n=row_n, random_state=cfg.sample_seed).iterrows():
            row_payload = {}
            for c, v in row.items():
                rv, red = _redact_value(v, cfg)
                # store hashed value and (optionally) raw if not redacted
                row_payload[c] = rv if red else rv
            row_sample.append(row_payload)

    per_col_examples = {}
    for c in df.columns:
        try:
            ex_vals = df[c].dropna().unique().tolist()
            ex_vals = ex_vals[:3]
        except Exception:
            ex_vals = []
        ex_payload = []
        for v in ex_vals:
            rv, red = _redact_value(v, cfg)
            ex_payload.append(rv)
        per_col_examples[c] = ex_payload

    return {
        "row_sample": row_sample,
        "per_column_examples": per_col_examples,
    }


def _collect_warnings_notices(columns_info: List[dict], dq: dict, relationships: dict) -> Tuple[List[str], List[str]]:
    warnings = []
    notices = []

    # High missing columns
    for c in dq.get("high_missing_columns", []):
        warnings.append(f"High missing rate in {c} (>50%).")

    # Constant columns
    for c in dq.get("constant_column_list", []):
        notices.append(f"Constant column detected: {c}.")

    # Leakage flags
    for fl in relationships.get("leakage_flags", []):
        warnings.append(f"Potential leakage: {fl['column']} ({fl['reason']}).")

    return sorted(set(warnings)), sorted(set(notices))


def _time_series_heuristics(df: pd.DataFrame, time_col: Optional[str]) -> dict | None:
    if not time_col or time_col not in df.columns:
        return None
    out = {"time_col": time_col, "frequency": None, "trend_strength": None, "seasonality_strength": None, "stationary": None}
    try:
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        # infer frequency (rough heuristic)
        freq = pd.infer_freq(ts.sort_values().dropna())
        out["frequency"] = freq
    except Exception:
        pass

    # Trend/seasonality heuristics only with a numeric target might be meaningful;
    # leave None for now; can be extended later.
    return out


def _predictive_salience_ranking(summary: dict, cfg) -> List[str]:
    cols = summary.get("columns", [])
    target_assoc = {a["column"]: abs(a["association"]) for a in summary.get("relationships", {}).get("target_associations", [])}
    salience = []
    for c in cols:
        name = c["name"]
        miss = float(c.get("missing_rate", 0.0))
        card_bucket = c.get("approximate_cardinality_bucket", "low")
        # cardinality penalty
        card_pen = {"low": 1.0, "medium": 0.9, "high": 0.75, "huge": 0.6}.get(card_bucket, 0.8)
        # quality penalty
        q_pen = max(0.0, 1.0 - miss)
        assoc = float(target_assoc.get(name, 0.0))
        score = assoc * q_pen * card_pen
        salience.append((name, -score))  # negative for ascending
    ranked = sorted(salience, key=lambda t: (t[1], t[0]))
    return [n for n, _ in ranked]


def _md_overview(meta: dict, dq: dict, tinfo: Optional[dict]) -> str:
    name = meta.get("dataset_name") or "dataset"
    rows = meta.get("row_count")
    cols = meta.get("col_count")
    target = meta.get("target_column")
    ptype = meta.get("problem_type")
    overview = []
    overview.append(f"- Dataset: {name}, {rows} rows x {cols} cols, target={target}, type={ptype}")
    if tinfo and tinfo.get("time_col"):
        overview.append(f"- Time column: {tinfo['time_col']}, frequency={tinfo.get('frequency')}")
    dup = dq.get("duplicate_row_rate")
    overview.append(f"- Duplicate row rate: {dup:.3f}")
    return "\n".join(overview)


def _md_top_columns(cols: List[dict], ranked_cols: List[str], n: int) -> str:
    name2col = {c["name"]: c for c in cols}
    lines = ["- Top columns by salience:"]
    count = 0
    for name in ranked_cols:
        if name not in name2col:
            continue
        c = name2col[name]
        st = c.get("semantic_type")
        miss = c.get("missing_rate")
        card = c.get("approximate_cardinality_bucket")
        # association (if present)
        lines.append(f"  - {name} [{st}] miss={miss:.2f}, card={card}")
        count += 1
        if count >= n:
            break
    return "\n".join(lines)


def _md_data_quality(dq: dict, warnings: List[str]) -> str:
    lines = ["- Data quality:"]
    if dq.get("constant_column_list"):
        lines.append(f"  - Constant: {', '.join(dq['constant_column_list'][:8])}")
    if dq.get("high_missing_columns"):
        lines.append(f"  - High missing: {', '.join(dq['high_missing_columns'][:8])}")
    if warnings:
        sel = sorted(set(warnings))[:8]
        lines.append(f"  - Warnings: {', '.join(sel)}")
    return "\n".join(lines)


def _md_relationships(rel: dict) -> str:
    lines = ["- Associations & leakage:"]
    tassoc = rel.get("target_associations", [])
    if tassoc:
        top = tassoc[:8]
        short = [f"{a['column']}:{a['metric']}={a['association']:.3f}" for a in top]
        lines.append("  - Target assoc: " + ", ".join(short))
    leaks = rel.get("leakage_flags", [])
    if leaks:
        lines.append("  - Leakage: " + ", ".join(sorted({f['column'] for f in leaks})[:8]))
    nc = rel.get("numeric_correlations", [])
    if nc:
        top = nc[:6]
        short = [f"{p['col_a']}↔{p['col_b']} r={p['r']:.3f}" for p in top]
        lines.append("  - Corr: " + ", ".join(short))
    return "\n".join(lines)


def _md_fe_checklist(cols: List[dict], tinfo: Optional[dict], problem_type: Optional[str]) -> str:
    has_text = any(c.get("semantic_type") == "text" for c in cols)
    has_cats = any(c.get("semantic_type") == "category" for c in cols)
    needs_scale = any(c.get("semantic_type") == "numeric" for c in cols)
    lines = ["- Feature engineering checklist:"]
    if has_cats:
        lines.append("  - Encode categorical (one-hot or target-encoding based on cardinality).")
    if needs_scale and (problem_type in ("regression", "classification")):
        lines.append("  - Scale numeric where model benefits (e.g., linear models, NN).")
    if has_text:
        lines.append("  - Text: basic cleaning, tokenize, TF-IDF or embeddings.")
    if tinfo and tinfo.get("time_col"):
        lines.append("  - Time: derive lags, moving stats, calendar features.")
    lines.append("  - Handle missing values; consider 'Missing' category for cats with high missing.")
    return "\n".join(lines)


def _canonical_for_hash(obj: dict) -> dict:
    # Exclude summary_hash if present; ensure deterministic structure
    cpy = dict(obj)
    cpy.pop("summary_hash", None)
    return cpy


def _stable_hash(obj: Any, salt: str) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    h = hashlib.sha256()
    h.update((salt + data).encode("utf-8"))
    return "sha256:" + h.hexdigest()[:16]


def _symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except Exception:
                pass
        # Create relative symlink when possible
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)
    except Exception:
        # Fallback: copy content
        try:
            if src.suffix.lower() in (".json", ".md", ".txt"):
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                dst.write_bytes(src.read_bytes())
        except Exception:
            pass