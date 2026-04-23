# Architecture overview

## Data flow

1. **Ingest** — `io` modules (`AQSClient`, `DataFrameAirQualitySource`, `OpenMeteoClient`, caching) load raw air quality and optional weather.
2. **Preprocess** — `preprocess` normalizes EPA-style frames, aligns daily series, merges exogenous fields, geodesics, imputation, and QC helpers.
3. **Features** — `features` builds station-level component scores (concentration, utility, variability, wind, etc.) and the station component table used for scoring sparse sets.
4. **Targets** — `target` scores and filters candidate *target* stations for the transfer-learning setting.
5. **Benchmark generation** — `benchmark.generate` produces many `SparseNetworkCandidate` instances using pluggable retention strategies (`SparseGenerationStrategy`, registry).
6. **Representation & clustering** — `benchmark.represent` attaches numeric vectors; `benchmark.cluster` runs k-medoids (`KMedoids`); `benchmark.select` picks medoid networks as representatives.
7. **Export** — `export.write_benchmark_bundle` writes Parquet manifests, per-network folders, and training job tables; `split_export` can emit source/target Parquet splits.

## Key types

Defined in `aqsparsebench.types` and re-exported from the package root: `StationRecord`, `SparseNetworkCandidate`, `RepresentativeSparseNetwork`, `BenchmarkManifest`, `RegionSpec`, `BoundingBox`, region and scoring configuration models from `config`.

## Configuration

`BenchmarkRunConfig`, `SparseGenerationSettings`, `ScoringConfig`, `TargetSelectionConfig`, and related Pydantic models in `aqsparsebench.config` control runs without hard-coding constants in notebooks.
