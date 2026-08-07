# pde-icl

Train a **TabICL** (SODA-INRIA, fully open-source) in-context learner on the
`pde_priors` geometry-conditioned PDE prior.

`pde_priors` emits, on demand and deterministically, one in-context-learning
dataset per boundary-value problem (`pde_priors.icl.ICLStreamDataset` +
`collate_tabicl` → `X [B, T, H]`, `y [B, T]`). This repo bridges that stream into
the actual TabICL API and provides zero-shot / fine-tuning / pre-training paths.

TabICL is preferred over the `tabpfn` inference package because it is fully open:
the v2 checkpoint downloads from Hugging Face with no auth token, and its
pre-training code is open source in this repo (`tabicl.prior`, `tabicl.train`).

## Status

- **Data bridge — done.** `pde_icl.icldata` slots `TabICLBatch` rows into the
  sklearn-style `(X, y)` split (`split_table`, `tables_from_batch`) that
  `TabICLRegressor.fit/predict` consumes.
- **Zero-shot — verified.** `pde_icl.eval_zero_shot` fits a pre-trained
  `TabICLRegressor` on each table's support rows and predicts its query rows
  (no per-table gradient updates). Verified end-to-end on real `pde_priors`
  tables (e.g. query RMSE ≈ 0.6 in standardized units on a 15-row support /
  15-row query table).
- **In-context fine-tuning** — the same `fit` mechanism adapts a table; the
  `tabicl[finetune]` extra also exposes single-dataset weight fine-tuning.
- **Full pre-training — next.** TabICL's open pre-training pipeline lives in
  `tabicl.prior` (deep-prior SCM samplers) + `tabicl.train`. Plugging the
  geometry-conditioned `pde_priors` prior in means adapting our BVP tables into
  TabICL's `DeepInstruction`/nested-tensor training format and pointing their
  trainer at it. This adapter is the active build target (see TODO below).

## Try zero-shot

```bash
uv sync --all-groups
uv run python - <<'PY'
import numpy as np
from pde_priors import GeometryPrior, GeometryAwareGenerator
from pde_priors.config import GenerationRequest, BoundaryRequest
from pde_priors.typing import EquationName, BoundaryKind
from pde_priors.random import BatchKey
from pde_priors.icl import adapt, collate_tabicl
from pde_icl.icldata import tables_from_batch
from tabicl import TabICLRegressor

pri = GeometryPrior(); gen = GeometryAwareGenerator(prior=pri)
req = GenerationRequest(equation=EquationName.POISSON, spatial_dim=2, batch_size=1,
    interior_points=24,
    boundary=BoundaryRequest(kind=BoundaryKind.DIRICHLET, points_per_face=6))
key = BatchKey(3, 0, 0)
sp = tables_from_batch(collate_tabicl([adapt(gen.generate(req, key), pri.sample(key))]))[0]
m = TabICLRegressor(); m.fit(sp.X_support, sp.y_support)
print("RMSE:", float(np.sqrt(np.mean((np.asarray(m.predict(sp.X_query)) - sp.y_query)**2))))
PY
```

## TODO

- `tabicl.prior` adapter: convert `pde_priors.icl` tables to TabICL's
  deep-prior training batches and wire the pretraining trainer (`tabicl.train`).
- Cross-geometry eval harness (train on circle/star prior, eval on ellipse /
  unseen stars).
- Raw-target normalization for physics-meaningful RMSE reporting.
