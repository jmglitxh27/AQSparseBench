# Installation

The canonical source repository is **https://github.com/jmglitxh27/AQSparseBench** (use that URL for `git clone` and `pip` VCS installs unless you are working from a fork).

## Requirements

- Python 3.10 or newer
- Dependencies are declared in `pyproject.toml` (pandas, numpy, scipy, requests, pyarrow, scikit-learn, scikit-learn-extra, pydantic, joblib).

## Local install (editable)

From the repository root:

```bash
pip install -e ".[dev]"
```

## PyPI (when published)

```bash
pip install aqsparsebench
```

Until the project is published, use a [Git URL](https://pip.pypa.io/en/stable/topics/vcs-support/) or a local path as shown below.

## Google Colab

Colab is a normal Python environment: install the package, then import it like any library.

**Option A — clone + editable install** (good while developing):

```text
!git clone https://github.com/jmglitxh27/AQSparseBench.git
%cd AQSparseBench
!pip install -e .
```

If you use a fork, substitute your fork’s clone URL.

**Option B — install from a Git URL** (no clone in the notebook):

```text
!pip install "aqsparsebench @ git+https://github.com/jmglitxh27/AQSparseBench.git@main"
```

Use a branch or tag instead of `main` if you pin versions.

**Option C — upload the project as a Colab file archive** or mount Drive, then:

```text
!pip install -e /content/path/to/AQSparseBench
```

After installation:

```python
import aqsparsebench as aqb
from aqsparsebench import write_benchmark_bundle, cluster_candidates
```

There is no separate HTTP service: the **Python package is the API** (functions, types, and configs). For long-running jobs in Colab, keep API keys (if any) in Colab secrets and write outputs to Drive if needed.

## Building this manual (HTML or PDF)

Install documentation extras, then build:

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
python -m sphinx -b rinoh docs docs/_build/rinoh
```

The rinoh builder writes **PDF** output under `docs/_build/rinoh/` (for example `aqsparsebench.pdf`). It does **not** require a LaTeX distribution.
