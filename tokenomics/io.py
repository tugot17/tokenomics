"""Shared I/O utilities for reading/writing benchmark results."""

import json
import os
import sys
import tempfile
from math import log10, floor
from pathlib import Path
from typing import Dict, Optional


def round_floats(obj, sig=4):
    """Round all floats in a nested structure to sig significant digits."""
    if isinstance(obj, float):
        if obj == 0:
            return 0.0
        return round(obj, -int(floor(log10(abs(obj)))) + (sig - 1))
    elif isinstance(obj, dict):
        return {k: round_floats(v, sig) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(v, sig) for v in obj]
    return obj


def atomic_write_json(path: str, obj, indent: int = 2) -> None:
    """Write JSON to *path* atomically (write-to-temp then rename).

    This prevents half-written files if the process is killed mid-write.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent)
        os.replace(tmp_path, path)  # atomic on POSIX
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _validate_metadata_consistency(entries: list, path: str) -> None:
    """Warn if key metadata fields differ across files in a results directory."""
    if len(entries) < 2:
        return

    check_fields = ("model", "scenario", "api_base")
    first_meta = entries[0].get("metadata", {})

    for entry in entries[1:]:
        meta = entry.get("metadata", {})
        for field in check_fields:
            v_first = first_meta.get(field)
            v_cur = meta.get(field)
            if v_first is not None and v_cur is not None and v_first != v_cur:
                print(f"WARNING: metadata mismatch in {path}: "
                      f"'{field}' is '{v_first}' in first file but '{v_cur}' in another",
                      file=sys.stderr)
                return  # one warning is enough


def load_results_dir(path: str, key_field: str = "sweep_value") -> Dict:
    """Load a results directory into the combined format.

    Reads all ``*.json`` files from *path*, validates metadata consistency,
    and reconstructs ``{"metadata": ..., "results": {key: result, ...}}``.

    Args:
        path: Directory containing per-experiment JSON files.
        key_field: JSON field used as the results dict key
                   (``"sweep_value"`` for completion, ``"config_key"`` for embedding).

    Returns:
        Combined dict with ``metadata`` and ``results``.
    """
    p = Path(path)
    files = sorted(p.glob("*.json"))
    if not files:
        print(f"Error: no .json files found in {path}", file=sys.stderr)
        sys.exit(1)

    entries = []
    for f in files:
        with open(f, "r") as fh:
            entries.append(json.load(fh))

    _validate_metadata_consistency(entries, path)

    combined_metadata = entries[0].get("metadata", {})
    combined_results = {}
    for entry, f in zip(entries, files):
        key = str(entry.get(key_field, f.stem))
        combined_results[key] = entry.get("result", {})

    return {"metadata": combined_metadata, "results": combined_results}
