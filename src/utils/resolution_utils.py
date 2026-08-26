from __future__ import annotations

import re
from typing import Optional, Tuple

def bp_to_label(bp: int) -> str:
    if bp <= 0:
        raise ValueError(f"resolution must be a positive integer (bp); got {bp}")
    if bp % 1_000_000 == 0:
        return f"{bp // 1_000_000}mb"
    if bp % 1000 == 0:
        return f"{bp // 1000}kb"
    return f"{bp}bp"

def parse_resolution(value: str) -> Tuple[str, int]:
    s = str(value).strip().lower().replace(" ", "")
    if not s:
        raise ValueError("resolution value cannot be empty")

    match = re.fullmatch(r"(\d+)(kb|mb|bp)?", s)
    if not match:
        raise ValueError(
            f"Invalid resolution {value!r}. Use e.g. 5kb, 10kb, 1mb, or 5000."
        )

    n = int(match.group(1))
    unit = match.group(2)
    if unit == "kb":
        bp = n * 1000
    elif unit == "mb":
        bp = n * 1_000_000
    else:
        bp = n

    if bp <= 0:
        raise ValueError(f"resolution must be positive; got {bp} bp from {value!r}")

    return bp_to_label(bp), bp

def resolve_resolution(
    res: Optional[str] = None,
    res_val: Optional[int] = None,
    default: str = "5kb",
) -> Tuple[str, int]:
    if res is None and res_val is None:
        return parse_resolution(default)

    if res is not None:
        label, bp = parse_resolution(res)
        if res_val is not None and int(res_val) != bp:
            raise ValueError(
                f"--res {res!r} resolves to {bp} bp, but --res-val={res_val} was also given. "
                f"Pass only one, or make them match."
            )
        return label, bp

    bp = int(res_val)
    return bp_to_label(bp), bp
