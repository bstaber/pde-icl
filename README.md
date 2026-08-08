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
    boundary=BoundaryRequest(kind=BoundaryKind.DIRICHLET, boundary_points=6),
)
key = BatchKey(3, 0, 0)
sp = tables_from_batch(collate_tabicl([adapt(gen.generate(req, key), pri.sample(key))]))[0]
m = TabICLRegressor(); m.fit(sp.X_support, sp.y_support)
print("RMSE:", float(np.sqrt(np.mean((np.asarray(m.predict(sp.X_query)) - sp.y_query)**2))))
PY
```

## Online training (no disk round-trip)

`python -m pde_icl.train_online` trains TabICL directly on the live
`pde_priors` generator — no pre-generated batches on disk. It subclasses TabICL's
`Trainer` (`PdeTrainer`, `pde_icl.train_online`) and overrides `configure_prior`
to feed a `PdePriorIterable` (one fresh BVP per table, generated every step;
`pde_icl.online_prior`), reusing all the rest of the trainer unchanged.

```bash
uv run python -m pde_icl.train_online \
    --regression_method quantile --num_quantiles 999 \
    --max_steps 60000 --batch_size 16 --micro_batch_size 4 \
    --embed_dim 96 --icl_num_blocks 6 --row_num_blocks 2 --col_num_blocks 2 --col_num_inds 64 --ff_factor 3 \
    --min_seq_len 144 --max_seq_len 160 --min_features 9 --max_features 9 \
    --device cuda --amp True --wandb_mode disabled \
    --pde-request-interior 128 --pde-request-boundary 16 --n_jobs 4
```

- `--pde-request-interior`/`--pde-request-boundary` size the single-BVP request
  (uniform tables within a batch), `--pde-prior-root-seed` seeds the stream,
  `--n_jobs>0` prefetches generation in worker processes.
- Benefits over the disk path: no storage, no file-count step cap, and a fresh
  (unbounded) distribution of geometries/tables every step.

## Pre-training TabICL on the geometry prior (from disk)

`pde_icl.pretrain_data` exports the geometry prior as TabICL's on-disk
pre-training batches (the exact sparse `{X, y, d, seq_lens, train_sizes,
batch_size}` format `python -m tabicl.train` reads via
`tabicl.prior._genload.LoadPriorDataset`). One BVP = one table; every table in a
batch shares one fixed request, so row counts are uniform within a batch as the
loader expects.

```bash
# 1. Generate prior batches on disk
uv run python -m pde_icl.pretrain_data \
    --save-dir prior_data \
    --n-batches 1000 --batch-size 32 \
    --interior-points 64 --boundary-points 16 \
    --support-fraction 0.5

# 2. Pre-train TabICL on the geometry prior (needs a GPU; long run)
uv run python -m tabicl.train \
    --prior_dir prior_data \
    --regression_method quantile --num_quantiles 999 \
    --max_steps 60000 --device cuda \
    --wandb_mode disabled
```

Round-trip compatibility with `LoadPriorDataset` is covered by
`tests/test_pretrain_data.py` (export → load → reconstructed dense `X [B,T,H]`,
`y`, `d`, `seq_lens`).

## Cross-geometry eval

`pde_icl.eval_zero_shot.cross_geometry_summary(request, n_tables=...)` evaluates
zero-shot TabICL RMSE per geometry family (`circle`, `ellipse`, `fourier_star`)
and on the full prior. Results are reported per family in both standardized
(`rmse_std`) and de-normalized raw (`rmse_raw`) units (using each table's
support-target std). This is the harness for probing a pre-trained model on
held-out geometry families.

De-normalization: `TabICLSample` now carries the support-target stats
(`y_mean`, `y_std`), so a standardized prediction is converted back to raw
physics units via `pred * y_std + y_mean` (and `rmse_raw = rmse_std * y_std`).

## Evaluating a pre-trained checkpoint

`pde_icl.eval_checkpoint` rebuilds a `tabicl.train` checkpoint of our *custom*
architecture (the stock `TabICLRegressor` only loads the fixed HF checkpoint)
and evaluates it per geometry family:

- `load_trained_model(ckpt, train_args, device)` / `evaluate_batch(...)` — in-context
  forward `model(X, y_train, d)` giving median-quantile predictions.
- `trained_cross_geometry(ckpt, train_args, request, n_tables=...)` — per-family
  `circle`/`ellipse`/`fourier_star`/`all` zero-shot RMSE (std + de-normalized raw).

## Results

**Experiment.** We train a TabICL v2 (fully open `soda-inria/tabicl`, quantile
regression) as an in-context PDE solver on the `pde_priors` geometry-conditioned
Poisson prior. One table = one BVP; the context is all Dirichlet boundary rows
plus a subset of observed interior rows, and the task is to predict the withheld
interior solution. We compare three models by **zero-shot** RMSE (fit context,
predict query, no per-table gradient updates) on fresh tables per geometry
family. RMSE is de-normalized to raw target units (`rmse_raw = rmse_std · y_std`).

**Training setup.** `python -m pde_icl.train_online` (see above) pre-trains on a
*fresh BVP every step* — no fixed dataset on disk. One run: 8,000 steps,
`batch_size 16` / `micro_batch_size 4`, embed 96 / 6 ICL blocks / 2 row / 2 col
blocks, `seq_len 144` (128 interior + 16 boundary points, 9 features), 64
quantiles, AMP fp16, ~11 it/s on an RTX 4080 SUPER (~12 min). The disk-trained
comparison used the same arch but the pre-generated archive instead.

**Result (raw RMSE, lower is better; n=24 tables/family, fresh seed):**

| geometry family | stock TabICL | disk-trained | **online-trained** |
|---|---|---|---|
| circle | 0.098 | 0.045 | **0.0039** |
| ellipse | 0.078 | 0.066 | **0.0058** |
| fourier_star | 0.171 | 0.040 | **0.0021** |
| all | 0.197 | 0.052 | **0.0037** |

Pre-training on the geometry prior markedly beats stock TabICL, and **online
training dominates the disk path (~10–30×)**: streaming a fresh, unbounded
geometry distribution each step yields a model that reproduces the withheld
interior solution nearly exactly from the boundary + observed interior
observations.

**Caveats.** These are *in-family* results — training and eval drew from the same
geometry-prior distribution (all three families). True held-out cross-geometry
generalization (train on a restricted prior, eval on an unseen family) and the
numerical `SolverPrior` baseline are the next scientific steps; the tooling for
both is in place.

## TODO

**Scientific**
- Validate a truly held-out cross-geometry split: pre-train on a *restricted*
  prior (e.g. `{circle, fourier_star}`) and evaluate the trained model on the
  held-out family (`ellipse`) — the tooling supports this directly.
- Numerical `SolverPrior` (V2) baseline: establish whether the in-context PDE
  solver genuinely *generalizes* across geometry, not just memorizes the prior.

**Training / scale**
- Scale up the online training recipe (steps, model size, interior/boundary
  points) now that online is the canonical path and clearly better than disk.

**Packaging / reproducibility**
- Publish the best trained checkpoint (`online_ckpt_real/step-8000.ckpt`) as a
  GitHub release (it's gitignored, not in the repo), with a reproduction note.
