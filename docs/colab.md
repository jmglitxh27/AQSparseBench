# Colab cookbook

Minimal pattern after `pip install` (see {doc}`installation`):

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

- **Caching** — set cache directories on `LocalCache` / clients so repeated Colab runs do not refetch the same AQS or Open-Meteo slices.
- **Disk** — Parquet exports can grow; use Google Drive (`drive.mount`) for `output_dir` if outputs are large.
- **Determinism** — pass explicit random seeds where the API accepts them (generation strategies, clustering) for reproducible notebooks.

For full signatures and defaults, use the {doc}`api` page or `help(function)` in a cell after import.
