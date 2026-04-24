"""Named geographic presets (bounding boxes) aligned with EPA AQS bbox queries."""

from __future__ import annotations

from aqsparsebench.types import BoundingBox

# Approximate boxes for common labels; tighten or add presets as needed.
REGION_PRESETS: dict[str, BoundingBox] = {
    "northeast": BoundingBox(
        min_lat=38.0,
        max_lat=47.5,
        min_lon=-82.0,
        max_lon=-66.5,
    ),
    "southeast": BoundingBox(
        min_lat=24.0,
        max_lat=38.5,
        min_lon=-95.0,
        max_lon=-75.0,
    ),
    "midwest": BoundingBox(
        min_lat=36.0,
        max_lat=49.5,
        min_lon=-104.5,
        max_lon=-80.5,
    ),
    "southwest": BoundingBox(
        min_lat=25.5,
        max_lat=37.5,
        min_lon=-124.5,
        max_lon=-102.0,
    ),
    "west": BoundingBox(
        min_lat=32.0,
        max_lat=49.5,
        min_lon=-125.0,
        max_lon=-102.0,
    ),
    "pacific_northwest": BoundingBox(
        min_lat=41.5,
        max_lat=49.5,
        min_lon=-125.0,
        max_lon=-116.0,
    ),
    "california": BoundingBox(
        min_lat=32.2,
        max_lat=42.2,
        min_lon=-124.6,
        max_lon=-114.0,
    ),
    "new_york": BoundingBox(
        min_lat=40.3,
        max_lat=45.1,
        min_lon=-79.9,
        max_lon=-71.8,
    ),
    "texas": BoundingBox(
        min_lat=25.5,
        max_lat=36.6,
        min_lon=-106.8,
        max_lon=-93.4,
    ),
    "contiguous_us": BoundingBox(
        min_lat=24.0,
        max_lat=49.5,
        min_lon=-125.0,
        max_lon=-66.0,
    ),
}
