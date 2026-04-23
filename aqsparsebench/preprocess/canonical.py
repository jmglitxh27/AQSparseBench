"""Canonical column names for the preprocessing pipeline (provider-agnostic)."""

from __future__ import annotations

# Air quality — long daily panel
COL_STATION_ID = "station_id"
COL_DATE = "date"
COL_CONCENTRATION = "concentration"  # µg/m³ for PM2.5 after normalization

# Monitor catalog
COL_LATITUDE = "latitude"
COL_LONGITUDE = "longitude"

# Alignment helpers
COL_MISSING_FLAG = "missing_flag"

REQUIRED_DAILY_COLUMNS = frozenset({COL_STATION_ID, COL_DATE, COL_CONCENTRATION})
REQUIRED_MONITOR_COLUMNS = frozenset({COL_STATION_ID, COL_LATITUDE, COL_LONGITUDE})
