"""Batched quality input parsing with verifiable, per-attachment caches.

Only raw features and descriptive text audits are cached. A1 normalization,
quality weights, domain aggregation and model fitting always run downstream.
Cache files contain NumPy arrays (never pickle). Each use hashes the compressed
source, so replacing data with the same size/mtime still invalidates the cache.
Changing this module, signal order or scalarization floors also invalidates it.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import re
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SIGNALS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness",
    "modernbert_readability", "modernbert_reasoning",
    "modernbert_professionalism", "dsir_books", "dsir_wiki", "dsir_math",
    "qurater", "ad_en", "rps_doc_word_count", "rps_doc_num_sentences",
    "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
AD_RE = re.compile(r"\b(?:buy now|subscribe|coupon|click here|affiliate|sponsored|free trial)\b", re.I)
URL_RE = re.compile(r"https?://", re.I)
SPACE_RE = re.compile(r"\s+")
CACHE_SCHEMA = 1


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _vector(values: list[object], width: int) -> np.ndarray:
    """Convert usual flat JSON lists in one operation, isolating malformed rows."""
    selected = [v if isinstance(v, list) and len(v) == width else [np.nan] * width
                for v in values]
    try:
        result = np.asarray(selected, dtype=np.float64)
        if result.shape == (len(values), width):
            return result
    except (TypeError, ValueError, OverflowError):
        pass
    result = np.full((len(values), width), np.nan, dtype=np.float64)
    for i, value in enumerate(values):
        try:
            candidate = np.asarray(value, dtype=np.float64)
            if candidate.shape == (width,):
                result[i] = candidate
        except (TypeError, ValueError, OverflowError):
            pass
    return result


def scalarize_rows(rows: list[dict], *, signals: list[str] = SIGNALS,
                   dsir_normalizer_floor: float = 20.0) -> np.ndarray:
    """Apply the existing 22 feature definitions to a bounded batch.

    Softmax and sigmoid are evaluated in float64 along the row axis. Their
    numerical agreement with the scalar implementation is at rounding error
    level (tested with rtol=0, atol=4e-16 on representative inputs).
    """
    result = np.empty((len(rows), len(signals)), dtype=np.float64)
    if not rows:
        return result
    words = np.fromiter((_float(row.get("rps_doc_word_count")) for row in rows),
                        dtype=np.float64, count=len(rows))
    # The original scalar parser uses the floor for unconvertible values but
    # retains a numeric NaN. Preserve that distinction for DSIR denominators.
    for i, row in enumerate(rows):
        try:
            float(row.get("rps_doc_word_count"))
        except (TypeError, ValueError, OverflowError):
            words[i] = dsir_normalizer_floor
    words = np.maximum(words, dsir_normalizer_floor)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for j, name in enumerate(signals):
            values = [row.get(name) for row in rows]
            if name.startswith("modernbert_"):
                arr = _vector(values, 6)
                finite = np.isfinite(arr).all(axis=1)
                z = np.exp(arr - arr.max(axis=1, keepdims=True))
                # np.dot has a different reduction order from sum. Match the
                # scalar formula's dot operation rather than a weighted sum.
                val = (z / z.sum(axis=1, keepdims=True)) @ (np.arange(6) / 5.0)
                val[~finite] = np.nan
            elif name in {"ad_en", "fluency_en"}:
                arr = _vector(values, 2)
                val = 1.0 / (1.0 + np.exp(-np.clip(arr[:, 1] - arr[:, 0], -50, 50)))
                val[~np.isfinite(arr).all(axis=1)] = np.nan
            elif name == "fineweb_edu":
                val = np.fromiter((_float(v[0]) if isinstance(v, list) and len(v) == 1
                                   else np.nan for v in values), dtype=np.float64)
            elif name == "qurater":
                arr = _vector(values, 4)
                val = arr.mean(axis=1)
                val[~np.isfinite(arr).all(axis=1)] = np.nan
                # Legacy len==4 permits nested lists for this one feature.
                for i, v in enumerate(values):
                    if isinstance(v, list) and len(v) == 4 and any(isinstance(x, list) for x in v):
                        try:
                            a = np.asarray(v, dtype=np.float64)
                            val[i] = a.mean() if np.isfinite(a).all() else np.nan
                        except (TypeError, ValueError, OverflowError):
                            val[i] = np.nan
            else:
                val = np.fromiter((_float(v) for v in values), dtype=np.float64)
                if name.startswith("dsir_"):
                    val /= words
                elif name in {"rps_doc_word_count", "rps_doc_num_sentences"}:
                    val = np.log1p(np.maximum(val, 0.0))
            val[~np.isfinite(val)] = np.nan
            result[:, j] = val
    return result


def _text_audit(content: str) -> tuple[int, int, float, str]:
    lines = [line.strip() for line in content.splitlines() if len(line.strip()) >= 20]
    repeat = 1.0 - len(set(lines)) / len(lines) if lines else 0.0
    return (int(bool(AD_RE.search(content))), int(len(URL_RE.findall(content)) >= 3),
            repeat, SPACE_RE.sub(" ", content)[:110])


def _hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _open_jsonl(path: Path):
    """Open a JSONL attachment, transparently accepting xz or plain bytes.

    The task statement and the extended-file naming use ``.jsonl.xz``, while some
    distributions ship the same records already expanded as ``.jsonl``. The
    container is detected from the xz magic bytes rather than the suffix, so both
    forms read identically and a mislabeled file cannot be silently mis-decoded.
    """
    with path.open("rb") as probe:
        magic = probe.read(6)
    if magic == b"\xfd7zXZ\x00":
        return lzma.open(path, "rb")
    return path.open("r", encoding="utf-8")


def resolve_attachment(path: Path) -> Path:
    """Return the shipped form of an attachment, allowing the suffix to differ.

    ``path`` is written as the task statement names it (``*.jsonl.xz``). A
    distribution that ships the same records already expanded is accepted through
    the ``.xz``-stripped name, so the reader is not tied to one packaging choice.
    """
    if path.is_file():
        return path
    if path.suffix == ".xz":
        expanded = path.with_suffix("")
        if expanded.is_file():
            return expanded
    raise FileNotFoundError(f"attachment not found: {path} (or its .xz-stripped form)")


def _schema_key(signals: list[str], dsir_floor: float, token_floor: float) -> str:
    config = {"cache_schema": CACHE_SCHEMA, "module_sha256": _hash_file(Path(__file__)),
              "signals": signals, "dsir_normalizer_floor": dsir_floor,
              "token_weight_floor": token_floor,
              "numpy": np.__version__}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _parse(path: Path, tag: str, domain: str | None, signals: list[str],
           dsir_floor: float, token_floor: float, batch_size: int) -> tuple[dict, int]:
    ids, domains, words, audits, chunks, rows = [], [], [], [], [], []
    invalid_json = 0
    with _open_jsonl(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
            ids.append(str(row.get("id", "")))
            domains.append(domain or str(row.get("_source_domain", "unknown")))
            words.append(max(_float(row.get("rps_doc_word_count")), token_floor))
            audits.append(_text_audit(row.get("content", "")) if tag == "A1" else (0, 0, 0.0, ""))
            rows.append(row)
            if len(rows) == batch_size:
                chunks.append(scalarize_rows(rows, signals=signals, dsir_normalizer_floor=dsir_floor))
                rows.clear()
    if rows:
        chunks.append(scalarize_rows(rows, signals=signals, dsir_normalizer_floor=dsir_floor))
    raw = np.concatenate(chunks) if chunks else np.empty((0, len(signals)), dtype=np.float64)
    arrays = {"id": np.asarray(ids, dtype=str), "domain": np.asarray(domains, dtype=str),
              "raw": raw, "word_count_raw": np.asarray(words, dtype=np.float64),
              "ad_marker": np.fromiter((v[0] for v in audits), dtype=np.int8),
              "many_urls": np.fromiter((v[1] for v in audits), dtype=np.int8),
              "repeat_line_rate": np.fromiter((v[2] for v in audits), dtype=np.float64),
              "excerpt": np.asarray([v[3] for v in audits], dtype=str)}
    return arrays, invalid_json


def _validate_arrays(arrays: dict, n: int, features: int) -> None:
    string_columns = {"id", "domain", "excerpt"}
    float_columns = {"word_count_raw", "repeat_line_rate"}
    expected = string_columns | float_columns | {"ad_marker", "many_urls", "raw"}
    if set(arrays) != expected:
        raise ValueError("cache columns do not match quality schema")
    for name, array in arrays.items():
        if array.shape != ((n, features) if name == "raw" else (n,)):
            raise ValueError(f"cache shape mismatch: {name}")
        if name in string_columns:
            if array.dtype.kind != "U":
                raise ValueError(f"cache string dtype mismatch: {name}")
        elif name in float_columns or name == "raw":
            if array.dtype != np.dtype("float64"):
                raise ValueError(f"cache float dtype mismatch: {name}")
        elif array.dtype != np.dtype("int8"):
            raise ValueError(f"cache flag dtype mismatch: {name}")


def _load_cache(cache: Path, key: str, source_hash: str, schema_key: str,
                features: int) -> tuple[dict, dict] | None:
    meta_path, array_path = cache / f"{key}.json", cache / f"{key}.npz"
    if not meta_path.is_file() or not array_path.is_file():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (metadata["source_sha256"] != source_hash or metadata["schema_key"] != schema_key
                or metadata["arrays_sha256"] != _hash_file(array_path)):
            return None
        with np.load(array_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        _validate_arrays(arrays, metadata["rows"], features)
        if not isinstance(metadata["invalid_json"], int) or metadata["invalid_json"] < 0:
            return None
        return arrays, metadata
    except (OSError, ValueError, TypeError, KeyError, EOFError):
        return None


def _write_cache(cache: Path, key: str, arrays: dict, metadata: dict) -> None:
    """Replace each file atomically; incomplete/racing pairs fail hash checks."""
    cache.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as output:
            array_temp = Path(output.name)
            temporary_paths.append(array_temp)
            np.savez(output, **arrays)
        metadata = dict(metadata, arrays_sha256=_hash_file(array_temp))
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".json", mode="w", encoding="utf-8",
                                         delete=False) as output:
            json_temp = Path(output.name)
            temporary_paths.append(json_temp)
            json.dump(metadata, output, ensure_ascii=False, indent=2)
        os.replace(array_temp, cache / f"{key}.npz")
        os.replace(json_temp, cache / f"{key}.json")
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def read_quality(data_dir: str | Path, *, signals: list[str] = SIGNALS,
                 dsir_normalizer_floor: float = 20.0, token_weight_floor: float = 1.0,
                 cache_dir: str | Path | None = None, cache_mode: str = "use",
                 batch_size: int = 4096) -> tuple[pd.DataFrame, dict]:
    """Read A1/A2/A3, preserving row order and the original diagnostics.

    ``cache_mode`` is ``use`` (read/write), ``refresh`` (rebuild), or ``off``.
    No cache is used unless ``cache_dir`` is supplied. ``diagnostics.io_cache``
    gives per-file source hashes, hit status and timings. Hashes identify raw
    input bytes; mtime is descriptive only and never controls invalidation.
    """
    if cache_mode not in {"use", "refresh", "off"}:
        raise ValueError("cache_mode must be 'use', 'refresh', or 'off'")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not signals or len(set(signals)) != len(signals) or set(signals) - set(SIGNALS):
        raise ValueError("signals must be unique supported quality features")
    if not (np.isfinite(dsir_normalizer_floor) and dsir_normalizer_floor > 0
            and np.isfinite(token_weight_floor) and token_weight_floor > 0):
        raise ValueError("normalizer floors must be positive and finite")
    data, cache = Path(data_dir), Path(cache_dir) if cache_dir is not None and cache_mode != "off" else None
    extended = data / "slimpajama_quality_extended"

    def require(pattern: str) -> Path:
        """Resolve one extended attachment, preferring the compressed variant.

        ``pattern`` is written with the ``.xz`` suffix used by the task statement.
        Distributions that ship the records already expanded are accepted through
        a suffix fallback; the reader itself detects the container from the bytes.
        """
        candidates = [pattern]
        if pattern.endswith(".xz"):
            candidates.append(pattern[:-3])
        for candidate in candidates:
            matches = sorted(extended.glob(candidate))
            if len(matches) > 1:
                raise ValueError(
                    f"Expected one attachment for {candidate!r}, found {len(matches)}; "
                    "define the shard list explicitly.")
            if matches:
                return matches[0]
        raise FileNotFoundError(
            f"no file matching {' or '.join(repr(c) for c in candidates)} under {extended}")

    files = [("A1", resolve_attachment(data / "slimpajama_quality_signal_sample.jsonl.xz"), None),
             ("A2", require("arxiv_*.jsonl.xz"), "arxiv"),
             ("A3", require("github_*.jsonl.xz"), "github")]
    schema = _schema_key(signals, dsir_normalizer_floor, token_weight_floor)
    parts, counts, bad_json, cache_diagnostics = [], {}, {}, []
    for tag, path, domain in files:
        started = time.perf_counter()
        source_hash = _hash_file(path)
        # Fixed domain and A1-only text audit affect cached values.
        key = hashlib.sha256(f"{tag}:{domain}:{source_hash}:{schema}".encode()).hexdigest()
        loaded = _load_cache(cache, key, source_hash, schema, len(signals)) if cache is not None and cache_mode == "use" else None
        if loaded is None:
            print(f"Reading {tag}: {path.name}", flush=True)
            arrays, invalid_json = _parse(path, tag, domain, signals, dsir_normalizer_floor,
                                          token_weight_floor, batch_size)
            # A file modified while it was being decoded must not enter a cache.
            if _hash_file(path) != source_hash:
                raise RuntimeError(f"source changed during parsing: {path}")
            metadata = {"source_sha256": source_hash, "schema_key": schema,
                        "rows": len(arrays["id"]), "invalid_json": invalid_json,
                        "source_path": str(path.resolve()), "source_bytes": path.stat().st_size}
            if cache is not None:
                try:
                    _write_cache(cache, key, arrays, metadata)
                except OSError as exc:
                    warnings.warn(f"quality cache could not be written: {exc}", RuntimeWarning)
        else:
            arrays, metadata = loaded
            invalid_json = metadata["invalid_json"]
            print(f"Cached {tag}: {path.name}", flush=True)
        part = pd.DataFrame(arrays["raw"], columns=signals)
        part.insert(0, "domain", arrays["domain"])
        part.insert(0, "source", tag)
        part.insert(0, "id", arrays["id"])
        for name in ("word_count_raw", "ad_marker", "many_urls", "repeat_line_rate", "excerpt"):
            part[name] = arrays[name]
        parts.append(part)
        counts[tag] = len(part)
        if invalid_json:
            bad_json[tag] = invalid_json
        cache_diagnostics.append({"source": tag, "source_sha256": source_hash,
                                  "cache_hit": loaded is not None, "cache_key": key,
                                  "rows": len(part), "seconds": time.perf_counter() - started})
    frame = pd.concat(parts, ignore_index=True)
    a1 = set(frame.loc[frame.source == "A1", "id"])
    diagnostics = {"counts": counts, "invalid_json": bad_json,
                   "a1_overlap": {tag: len(a1.intersection(frame.loc[frame.source == tag, "id"]))
                                  for tag in ("A2", "A3")},
                   "nonfinite_by_signal": {name: int(value) for name, value in
                                            (~np.isfinite(frame[signals])).sum().items()},
                   "io_cache": {"mode": cache_mode if cache is not None else "off",
                                "schema_key": schema, "files": cache_diagnostics}}
    return frame, diagnostics
