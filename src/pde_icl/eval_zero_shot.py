"""Zero-shot in-context evaluation of a pre-trained TabICL on pde_priors tables.

For each generated table (one BVP), a pre-trained ``TabICLRegressor`` is
``fit`` on the support rows and ``predict``s the query rows -- no per-table
gradient updates, i.e. pure in-context learning.  Returns per-table RMSE (in
standardized target units) and the mean.

Cross-geometry: pass a ``request``/``gen`` built from a different geometry prior
to probe generalization to unseen geometry families.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from pde_priors.config import GenerationRequest
from pde_priors.generators.geometry_aware import GeometryAwareGenerator
from pde_priors.geometries import Domain
from pde_priors.icl import collate_tabicl
from tabicl import TabICLRegressor

from pde_icl.icldata import TableSplit, tables_from_batch


def zero_shot_rmse_on_tables(
    splits: list[TableSplit],
    *,
    device: str = "auto",
    verbose: bool = False,
) -> list[float]:
    """Fit+predict a shared tabicl model on each table; return per-table RMSE."""
    model = TabICLRegressor(device=device, verbose=verbose)
    rmses: list[float] = []
    for split in splits:
        model.fit(split.X_support, split.y_support)
        pred = model.predict(split.X_query)
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        rmse = float(np.sqrt(float(np.mean((pred - split.y_query) ** 2))))
        rmses.append(rmse)
    return rmses


def evaluate_stream(
    gen: GeometryAwareGenerator,
    request: GenerationRequest,
    *,
    root_seed: int,
    n_tables: int,
    device: str = "cpu",
) -> float:
    """Generate ``n_tables`` BVPs and return mean zero-shot query RMSE."""
    from pde_priors.icl import adapt
    from pde_priors.random import BatchKey

    samples = []
    for index in range(n_tables):
        k = BatchKey(root_seed=root_seed, epoch=0, global_batch_index=index)
        batch = gen.generate(request, k)
        samples.append(adapt(batch, cast(Domain, gen.prior.sample(k))))
    return float(
        np.mean(zero_shot_rmse_on_tables(tables_from_batch(collate_tabicl(samples)), device=device))
    )
