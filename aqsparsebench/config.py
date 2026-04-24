"""Scoring weights, API settings, and run configuration with validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from aqsparsebench.types import RegionSpec


class UtilityWeights(BaseModel):
    alpha: float = Field(0.30, description="Weight for concentration component")
    beta: float = Field(0.25, description="Weight for population component")
    gamma: float = Field(0.20, description="Weight for wind/transport component")
    delta: float = Field(0.15, description="Weight for variability component")
    eta: float = Field(0.10, description="Weight for background component")

    @model_validator(mode="after")
    def utility_sum(self) -> UtilityWeights:
        total = self.alpha + self.beta + self.gamma + self.delta + self.eta
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Utility weights must sum to 1.0, got {total:.6f}")
        return self


class ConcentrationWeights(BaseModel):
    lambda_mean: float = 0.4
    lambda_q95: float = 0.4
    lambda_max: float = 0.2

    @model_validator(mode="after")
    def concentration_sum(self) -> ConcentrationWeights:
        total = self.lambda_mean + self.lambda_q95 + self.lambda_max
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Concentration weights must sum to 1.0, got {total:.6f}")
        return self


class VariabilityWeights(BaseModel):
    rho_std: float = 0.5
    rho_iqr: float = 0.3
    rho_seasonal: float = 0.2

    @model_validator(mode="after")
    def variability_sum(self) -> VariabilityWeights:
        total = self.rho_std + self.rho_iqr + self.rho_seasonal
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Variability weights must sum to 1.0, got {total:.6f}")
        return self


class WindWeights(BaseModel):
    omega_speed: float = 0.5
    omega_directional_variability: float = 0.5

    @model_validator(mode="after")
    def wind_sum(self) -> WindWeights:
        total = self.omega_speed + self.omega_directional_variability
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"Wind weights must sum to 1.0, got {total:.6f}")
        return self


class ScoringConfig(BaseModel):
    utility: UtilityWeights = Field(default_factory=UtilityWeights)
    concentration: ConcentrationWeights = Field(default_factory=ConcentrationWeights)
    variability: VariabilityWeights = Field(default_factory=VariabilityWeights)
    wind: WindWeights = Field(default_factory=WindWeights)
    normalize_range: tuple[float, float] = (0.0, 1.0)

    @field_validator("normalize_range")
    @classmethod
    def validate_norm_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if lo >= hi:
            raise ValueError("normalize_range must be strictly increasing")
        return v

    @classmethod
    def from_partial(
        cls,
        utility: dict[str, Any] | None = None,
        concentration: dict[str, Any] | None = None,
        variability: dict[str, Any] | None = None,
        wind: dict[str, Any] | None = None,
        normalize_range: tuple[float, float] | None = None,
    ) -> ScoringConfig:
        """Merge user dicts into defaults (supports builder.set_scoring_config style)."""
        base: dict[str, Any] = {}
        if utility:
            base["utility"] = UtilityWeights(**{**UtilityWeights().model_dump(), **utility})
        if concentration:
            base["concentration"] = ConcentrationWeights(
                **{**ConcentrationWeights().model_dump(), **concentration}
            )
        if variability:
            base["variability"] = VariabilityWeights(
                **{**VariabilityWeights().model_dump(), **variability}
            )
        if wind:
            base["wind"] = WindWeights(**{**WindWeights().model_dump(), **wind})
        if normalize_range is not None:
            base["normalize_range"] = normalize_range
        return cls(**base) if base else cls()


class ApiConfig(BaseModel):
    """Credentials and endpoints for data sources."""

    aqs_email: str | None = Field(
        default=None,
        description="EPA AQS API registered email (required for production AQS calls)",
    )
    aqs_key: str | None = Field(
        default=None,
        description="EPA AQS API key paired with aqs_email",
    )
    census_api_key: str | None = Field(
        default=None,
        description="U.S. Census API key (optional under 500 queries/IP/day without key)",
    )
    cache_dir: str | None = Field(
        default=None,
        description="Root directory for LocalCache; None disables persistent cache",
    )
    aqs_request_sleep_seconds: float = Field(
        default=6.5,
        ge=0.0,
        description="Pause between AQS requests to stay near official courtesy limits (~10/min)",
    )
    aqs_read_timeout_seconds: float = Field(
        default=300.0,
        gt=0.0,
        description="HTTP read timeout (seconds) for each AQS GET (large byBox/byState payloads often need several minutes)",
    )
    open_meteo_base_url: str | None = Field(
        default=None,
        description="Historical weather API base URL; None uses Open-Meteo public archive endpoint",
    )
    open_meteo_api_key: str | None = Field(
        default=None,
        description="Optional apikey query parameter for Open-Meteo customer API hosts",
    )
    open_meteo_request_sleep_seconds: float = Field(
        default=0.2,
        ge=0.0,
        description="Pause between Open-Meteo HTTP calls when batching many coordinates",
    )
    open_meteo_read_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        description="HTTP read timeout (seconds) for each Open-Meteo GET",
    )


class SparseGenerationSettings(BaseModel):
    """Defaults for :func:`~aqsparsebench.benchmark.generate.generate_sparse_candidates`."""

    strategy: str = Field(default="default_weighted", description="Registered strategy id")
    min_pairwise_km: float | None = Field(
        default=None,
        description="Reject candidates whose closest pair of retained sites is closer than this",
    )
    score_power: float = Field(default=1.0, ge=0.1, le=10.0)
    spatial_diversity_scale_km: float = Field(default=25.0, gt=0.0)
    max_resample_attempts: int = Field(default=500, ge=1)


class TargetSelectionWeights(BaseModel):
    """Weights for conditional target suitability ``T_s`` (independent from utility weights)."""

    lambda_J: float = Field(0.35, description="Base station utility")
    lambda_G: float = Field(0.25, description="Gap-coverage benefit")
    lambda_U: float = Field(0.25, description="Underrepresentation benefit")
    lambda_E: float = Field(0.15, description="Expansion / novelty benefit")

    @model_validator(mode="after")
    def lambdas_sum(self) -> TargetSelectionWeights:
        t = self.lambda_J + self.lambda_G + self.lambda_U + self.lambda_E
        if abs(t - 1.0) > 1e-5:
            raise ValueError(f"Target selection weights must sum to 1.0, got {t:.6f}")
        return self


class TargetSelectionConfig(BaseModel):
    """Eligibility thresholds and weights for marginal target siting."""

    weights: TargetSelectionWeights = Field(default_factory=TargetSelectionWeights)
    min_days: int = Field(default=365, ge=1)
    min_completeness: float = Field(default=0.8, ge=0.0, le=1.0)
    min_feature_distance: float | None = Field(
        default=None,
        description="Minimum feature-space distance to nearest retained site (z-space L2)",
    )
    min_geo_distance_km: float | None = Field(
        default=None,
        description="Minimum geographic distance (km) to nearest retained site",
    )


class BenchmarkRunConfig(BaseModel):
    """Top-level knobs for a benchmark build (extends builder kwargs)."""

    pollutant: str = Field(default="pm25", description="Logical pollutant id (mapped to AQS param)")
    region: RegionSpec
    years: list[int] = Field(min_length=1)
    retention_levels: list[float] = Field(
        default_factory=lambda: [1.0, 0.75, 0.50, 0.25, 0.10],
    )
    random_state: int = Field(default=42)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

    @field_validator("retention_levels")
    @classmethod
    def validate_retention(cls, v: list[float]) -> list[float]:
        for x in v:
            if not (0.0 < x <= 1.0):
                raise ValueError(f"retention level must be in (0, 1], got {x}")
        return sorted(set(v), reverse=True)

    @field_validator("years")
    @classmethod
    def validate_years(cls, v: list[int]) -> list[int]:
        if len(v) != len(set(v)):
            raise ValueError("years must be unique")
        return sorted(v)
