"""
Normalising what ``ziDAQServer.poll()`` hands back.

``poll()`` does not return one shape. Depending on the LabOne version and how
the subscription was made, a single impedance sample can arrive as:

* flat keys per field -- ``{"/dev3519/imps/0/sample/param0": array([...])}``
* dot-joined keys -- ``{"/dev3519/imps/0/sample.param0": array([...])}``
* one nested dict -- ``{"/dev3519/imps/0/sample": {"param0": array([...])}}``
* a nested tree -- ``{"dev3519": {"imps": {"0": {"sample": {...}}}}}``
* a structured numpy array with named fields

and each field may be a scalar, a list, or an array of several readings, of
which we want the most recent. The device id in the reply also does not always
match the one discovery reported, so lookups fall back to matching on the path
suffix.

Ported from ``cv_cf_recipe_ui.py:651-745`` and kept dependency-free -- numpy
arrays are handled by duck typing, so this module (and its tests) run without
``numpy`` or ``zhinst`` installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def normalize_node_path(path: str) -> str:
    """Lowercase, single-slash-separated, leading slash."""
    parts = [p for p in str(path).strip().split("/") if p]
    if not parts:
        return "/"
    return "/" + "/".join(parts).lower()


def node_path_suffix(path: str) -> str:
    """Drop a leading ``devNNNN`` component so paths compare across devices."""
    norm = normalize_node_path(path)
    parts = [p for p in norm.split("/") if p]
    if len(parts) >= 2 and parts[0].startswith("dev"):
        return "/" + "/".join(parts[1:])
    return norm


def last_value(value: Any) -> Any:
    """Reduce a field to its most recent reading.

    Duck-typed rather than isinstance-checked so numpy arrays work without
    importing numpy.
    """
    if hasattr(value, "reshape") and hasattr(value, "size"):
        try:
            flat = value.reshape(-1)
            return flat[-1] if value.size > 0 else None
        except Exception:
            return value
    if isinstance(value, (list, tuple)):
        return value[-1] if value else None
    return value


def sample_payload_to_last_values(sample: Any) -> Dict[str, Any]:
    """Turn any sample shape into ``{field: latest value}``."""
    names = getattr(getattr(sample, "dtype", None), "names", None)
    if names:
        out: Dict[str, Any] = {}
        for name in names:
            try:
                out[name] = last_value(sample[name])
            except Exception:
                continue
        return out

    if not isinstance(sample, dict):
        return {}
    return {key: last_value(value) for key, value in sample.items()}


def sample_payload_has_data(sample: Any) -> bool:
    if sample is None:
        return False
    if getattr(getattr(sample, "dtype", None), "names", None):
        return getattr(sample, "size", 1) > 0
    try:
        return len(sample) > 0
    except TypeError:
        return True


def flatten_poll_sample_nodes(data: Any, prefix: str = "") -> Dict[str, Dict[str, Any]]:
    """Collect every ``.../sample`` node in a poll reply into one flat mapping."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, dict):
        return out

    for raw_key, raw_value in data.items():
        key = str(raw_key)
        path = normalize_node_path(key if key.startswith("/") else f"{prefix}/{key}")

        for separator in ("/sample/", "/sample."):
            if separator in path:
                sample_path, field = path.split(separator, 1)
                bucket = out.setdefault(f"{sample_path}/sample", {})
                bucket[field] = raw_value
                break
        else:
            if path.endswith("/sample") and sample_payload_has_data(raw_value):
                out[path] = raw_value
            elif isinstance(raw_value, dict):
                out.update(flatten_poll_sample_nodes(raw_value, path))

    return out


def extract_sample_payload(
    data: Any, requested_path: str
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Find the requested sample.

    Returns ``(fields, seen_paths)``. ``seen_paths`` is for diagnostics -- when
    the requested node produced nothing, reporting what *did* arrive turns a bare
    timeout into an actionable message.
    """
    sample_nodes = flatten_poll_sample_nodes(data)
    if not sample_nodes:
        return None, []

    requested = normalize_node_path(requested_path)
    candidates = [sample_nodes.get(requested)]
    suffix = node_path_suffix(requested)
    candidates += [
        payload for node_path, payload in sample_nodes.items()
        if node_path_suffix(node_path) == suffix
    ]

    for payload in candidates:
        if sample_payload_has_data(payload):
            values = sample_payload_to_last_values(payload)
            if values:
                return values, list(sample_nodes.keys())

    return None, list(sample_nodes.keys())
