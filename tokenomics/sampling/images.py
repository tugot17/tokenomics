"""
Synthetic image generation for multimodal (VL) benchmarking.

Produces images as OpenAI ``image_url`` base64 ``data:`` URIs, so any completion
request can carry them. Kept alongside the text samplers since it's just another
form of synthetic request content.
"""

import base64
import io
import os
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image


def _pil_to_uri(img: "Image.Image") -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def image_value_to_uris(value: Any) -> List[str]:
    """Coerce a dataset image-column value into a list of ``data:`` URIs.

    Handles the shapes HuggingFace vision datasets produce: a decoded PIL image,
    a ``{"bytes"/"path"}`` dict, a filesystem path, or a list of any of these
    (multi-image rows). Unrecognized/empty values yield an empty list.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for v in value:
            out.extend(image_value_to_uris(v))
        return out
    # Decoded PIL image (the common HF Image-feature case)
    if hasattr(value, "save"):
        return [_pil_to_uri(value)]
    # HF Image dict: {"bytes": ..., "path": ...}
    if isinstance(value, dict):
        if value.get("bytes"):
            return [_pil_to_uri(Image.open(io.BytesIO(value["bytes"])))]
        if value.get("path") and os.path.exists(value["path"]):
            return [_pil_to_uri(Image.open(value["path"]))]
        return []
    # Filesystem path
    if isinstance(value, str) and os.path.exists(value):
        return [_pil_to_uri(Image.open(value))]
    return []


def parse_image_size(spec: str) -> Tuple[int, int]:
    """Parse an image size spec: '512' -> (512, 512), '640x480' -> (640, 480)."""
    parts = str(spec).lower().split("x")
    if len(parts) == 1:
        w = h = int(parts[0])
    elif len(parts) == 2:
        w, h = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Invalid image size '{spec}'; use N or WxH")
    return w, h


def build_synthetic_image_uris(size: str, count: int, start_id: int = 0) -> List[str]:
    """Return `count` random-noise PNG images (as base64 ``data:`` URIs).

    Each image is filled with random pixels seeded by a unique id (``start_id +
    k``), so it is deterministic (reproducible across runs) yet unique across the
    whole benchmark — which keeps the server's prefix and multimodal caches from
    deduplicating requests and undercounting vision-encoder compute. Pixel
    *content* doesn't change VL compute (patch count is fixed by size), but note
    noise is nearly incompressible, so payloads are much larger than a flat
    image (~MBs at 1024x1024) — keep image size/count sane for the transport.
    """
    if count <= 0:
        return []
    w, h = parse_image_size(size)
    uris = []
    for k in range(count):
        rng = np.random.default_rng(start_id + k)  # deterministic, unique per image
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uris.append(f"data:image/png;base64,{b64}")
    return uris
