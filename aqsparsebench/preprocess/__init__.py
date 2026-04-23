"""Alignment, imputation, QC, scaling, geo helpers, and source-aware loaders."""

from aqsparsebench.preprocess.align import align_daily, merge_exogenous
from aqsparsebench.preprocess.canonical import (
    COL_CONCENTRATION,
    COL_DATE,
    COL_LATITUDE,
    COL_LONGITUDE,
    COL_MISSING_FLAG,
    COL_STATION_ID,
)
from aqsparsebench.preprocess.epa_normalize import (
    coerce_custom_daily_to_canonical,
    coerce_custom_monitors_to_canonical,
    looks_like_aqs_daily,
    looks_like_aqs_monitors,
    normalize_aqs_daily_df,
    normalize_aqs_monitors_df,
)
from aqsparsebench.preprocess.from_sources import (
    build_merged_daily_panel,
    load_air_quality_for_preprocess,
    load_weather_for_monitors,
    resolve_air_quality_normalization,
)
from aqsparsebench.preprocess.geo import haversine_km, pairwise_distance_matrix
from aqsparsebench.preprocess.impute import interpolate_time, seasonal_mean_fill
from aqsparsebench.preprocess.qc import (
    drop_duplicate_station_ids,
    drop_invalid_coordinates,
    drop_short_station_series,
)
from aqsparsebench.preprocess.scaling import minmax_scale_columns

__all__ = [
    "COL_CONCENTRATION",
    "COL_DATE",
    "COL_LATITUDE",
    "COL_LONGITUDE",
    "COL_MISSING_FLAG",
    "COL_STATION_ID",
    "align_daily",
    "build_merged_daily_panel",
    "coerce_custom_daily_to_canonical",
    "coerce_custom_monitors_to_canonical",
    "drop_duplicate_station_ids",
    "drop_invalid_coordinates",
    "drop_short_station_series",
    "haversine_km",
    "interpolate_time",
    "load_air_quality_for_preprocess",
    "load_weather_for_monitors",
    "looks_like_aqs_daily",
    "looks_like_aqs_monitors",
    "merge_exogenous",
    "minmax_scale_columns",
    "normalize_aqs_daily_df",
    "normalize_aqs_monitors_df",
    "pairwise_distance_matrix",
    "resolve_air_quality_normalization",
    "seasonal_mean_fill",
]
