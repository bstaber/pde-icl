"""Zero-shot in-context evaluation of a pre-trained TabICL on pde_priors tables,
plus de-normalized (raw) RMSE and a cross-geometry harness.

For each generated table (one BVP), a pre-trained ``TabICLRegressor`` is
``fit`` on the support rows and ``predict``s the query rows -- no per-table
gradient updates (pure in-context learning).

``TableEval.rmse_std`` is in standardized target units; ``rmse_raw`` is
de-normalized back to raw physics units using the table's support-set target std
(``rmse_raw = rmse_std * y_std``).

The cross-geometry harness evaluates per geometry family (circle / ellipse /
fourier_star) and on the full default prior, so a model pre-trained on a
restricted prior can be probed on held-out geometry families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pde_priors.config import GenerationRequest
from pde_priors.generators.geometry_aware import GeometryAwareGenerator
from pde_priors.geometries import GeometryPrior
from pde_priors.icl import adapt, collate_tabicl
from pde_priors.random import BatchKey
from tabicl import TabICLRegressor

from pde_icl.icldata import TableSplit, tables_from_batch


@dataclass(frozen=True, slots=True)
class TableEval:
    """Per-table zero-shot result."""

    rmse_std: float  # RMSE in standardized target units
    rmse_raw: float  # RMSE de-normalized to raw target units


def raw_rmse(rmse_std: float, y_std: float) -> float:
    """De-normalize a standardized RMSE back to raw target units.

    RMSE scales linearly with the target scale under an affine standardization,
    so ``rmse_raw == rmse_std * y_std`` (the support-set target std).
    """
    return rmse_std * y_std


def zero_shot_rmse_on_tables(
    splits: list[TableSplit],
    *,
    device: str = "cpu",
    verbose: bool = False,
) -> list[TableEval]:
    """Fit+predict a shared tabicl model on each table; return per-table results."""
    model = TabICLRegressor(device=device, verbose=verbose)
    results: list[TableEval] = []
    for split in splits:
        model.fit(split.X_support, split.y_support)
        pred = model.predict(split.X_query)
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        rmse_std = float(np.sqrt(float(np.mean((pred - split.y_query) ** 2))))
        results.append(TableEval(rmse_std=rmse_std, rmse_raw=raw_rmse(rmse_std, split.y_std)))
    return results


def _evaluate_prior(
    gen: GeometryAwareGenerator,
    request: GenerationRequest,
    *,
    n_tables: int,
    root_seed: int,
    device: str,
    verbose: bool,
) -> TableEval:
    samples = []
    for index in range(n_tables):
        k = BatchKey(root_seed=root_seed, epoch=0, global_batch_index=index)
        batch, domain = gen.generate_with_domain(request, k)
        samples.append(adapt(batch, domain, key=k))
    results = zero_shot_rmse_on_tables(
        tables_from_batch(collate_tabicl(samples)), device=device, verbose=verbose
    )
    return TableEval(
        rmse_std=float(np.mean([r.rmse_std for r in results])),
        rmse_raw=float(np.mean([r.rmse_raw for r in results])),
    )


def evaluate_stream(
    gen: GeometryAwareGenerator,
    request: GenerationRequest,
    *,
    root_seed: int,
    n_tables: int,
    device: str = "cpu",
    verbose: bool = False,
) -> TableEval:
    """Generate ``n_tables`` BVPs from ``gen`` and return mean zero-shot RMSE."""
    return _evaluate_prior(
        gen, request, n_tables=n_tables, root_seed=root_seed, device=device, verbose=verbose
    )


def evaluate_geometry_family(
    families: dict[str, float] | None,
    request: GenerationRequest,
    *,
    n_tables: int,
    root_seed: int,
    device: str = "cpu",
    verbose: bool = False,
) -> TableEval:
    """Zero-shot RMSE on tables drawn only from the given geometry families.

    ``families=None`` uses the default full prior.  Keys are one of
    ``circle`` / ``ellipse`` / ``fourier_star``.
    """
    prior = GeometryPrior(families=families) if families is not None else GeometryPrior()
    gen = GeometryAwareGenerator(prior=prior)
    return _evaluate_prior(
        gen, request, n_tables=n_tables, root_seed=root_seed, device=device, verbose=verbose
    )


def cross_geometry_summary(
    request: GenerationRequest,
    *,
    n_tables: int,
    root_seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> dict[str, TableEval]:
    """Zero-shot RMSE per geometry family and on the full prior.

    Lets a model pre-trained on a restricted prior be probed on held-out
    geometry families (the key scientific generalization test).
    """
    configs: dict[str, dict[str, float] | None] = {
        "circle": {"circle": 1.0},
        "ellipse": {"ellipse": 1.0},
        "fourier_star": {"fourier_star": 1.0},
        "all": None,
    }
    return {
        label: evaluate_geometry_family(
            fam,
            request,
            n_tables=n_tables,
            root_seed=root_seed,
            device=device,
            verbose=verbose,
        )
        for label, fam in configs.items()
    }
