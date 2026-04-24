# Colab cookbook

Install from Git when needed: `https://github.com/jmglitxh27/AQSparseBench` (see {doc}`installation` for exact `pip` / `git clone` lines).

Minimal pattern after `pip install`:

```python
import aqsparsebench as aqb
from aqsparsebench import (
    BenchmarkRunConfig,
    attach_candidate_vectors,
    cluster_candidates,
    generate_sparse_candidates,
    select_representative_networks,
    write_benchmark_bundle,
)

# Wire your own loaded frames / sources using the preprocess + feature APIs,
# then generate candidates, represent, cluster, select, and export:

# candidates = generate_sparse_candidates(...)
# attach_candidate_vectors(candidates, station_features_df)
# cluster_candidates(candidates, ...)
# reps = select_representative_networks(candidates)
# write_benchmark_bundle(..., output_dir="/content/aqb_out")
```

## Tips

- **Caching** — set `ApiConfig.cache_dir` to a folder on disk. Both `AQSClient` and `OpenMeteoClient` (from `DataSources.from_us_epa_defaults`) read and write JSON under that root (`aqs/…` and `open_meteo/…`). If `cache_dir` is `None`, nothing is persisted between runs.

### Cache directory on Colab

**Session-only** (fast, lost when the runtime disconnects):

```python
from pathlib import Path

from aqsparsebench.config import ApiConfig
from aqsparsebench.io import DataSources

cache_root = Path("/content/aqsparsebench_cache")
cache_root.mkdir(parents=True, exist_ok=True)

api = ApiConfig(
    aqs_email="your@email",
    aqs_key="your_key",
    cache_dir=str(cache_root),
)
sources = DataSources.from_us_epa_defaults(api)
```

**Across sessions** — mount Drive and point `cache_dir` at a folder under `MyDrive` (same `ApiConfig` pattern, different path):

```python
from google.colab import drive

drive.mount("/content/drive")

api = ApiConfig(
    aqs_email="your@email",
    aqs_key="your_key",
    cache_dir="/content/drive/MyDrive/aqsparsebench_cache",
)
sources = DataSources.from_us_epa_defaults(api)
```

Create the directory once if needed (`mkdir -p` style with `Path(...).mkdir(parents=True, exist_ok=True)`).
- **Disk** — Parquet exports can grow; use Google Drive (`drive.mount`) for `output_dir` if outputs are large.
- **Determinism** — pass explicit random seeds where the API accepts them (generation strategies, clustering) for reproducible notebooks.

For full signatures and defaults, use the {doc}`api` page or `help(function)` in a cell after import.
