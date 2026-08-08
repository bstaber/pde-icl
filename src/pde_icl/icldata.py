"""Bridge `pde_priors.icl` tables into the tabpfn `fit`/`predict` format.

Each `TabICLSample` is one boundary-value problem -> one table.  `collate_tabicl`
packs a batch into `X [B, T, H]` / `y [B, T]` with `train_size[B]` context rows.
This module splits each table into a (support, query) pair as flat `(n_rows,
n_features)` arrays that `TabPFNRegressor.fit/predict` consume.

Note on standardization: `pde_priors.icl.adapt` z-standardizes features and the
target from the support set, so the target `y` here is in standardized units. For
physics-meaningful RMSE you can re-express predictions using the support
mean/std of the raw target (returned by this module's raw normalizer when
available).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pde_priors.icl import TabICLBatch


@dataclass(frozen=True, slots=True)
class TableSplit:
    """One table split into tabpfn-style support/core tensors.

    ``y_mean``/``y_std`` are the table's support-set target statistics that were
    used to standardize ``y``; predictions in standardized units are converted
    back to raw physics units via ``y_pred * y_std + y_mean``.
    """

    X_support: np.ndarray  # [k, H]
    y_support: np.ndarray  # [k]
    X_query: np.ndarray  # [T-k, H]
    y_query: np.ndarray  # [T-k]
    y_mean: float = 0.0  # raw target mean (standardization offset)
    y_std: float = 1.0  # raw target std (standardization scale)


def split_table(
    X: np.ndarray,
    y: np.ndarray,
    train_size: int,
    *,
    y_mean: float = 0.0,
    y_std: float = 1.0,
) -> TableSplit:
    """Split one standardized table into support and query arrays."""
    support = slice(0, train_size)
    query = slice(train_size, None)
    return TableSplit(
        X_support=X[support],
        y_support=y[support],
        X_query=X[query],
        y_query=y[query],
        y_mean=y_mean,
        y_std=y_std,
    )


def tables_from_batch(batch: TabICLBatch) -> list[TableSplit]:
    """Materialize every table in a collated batch into tabpfn-form splits."""
    sizes = [batch.X.shape[0], batch.y.shape[0], batch.train_size.shape[0]]
    if len(set(sizes)) != 1:
        raise ValueError("X, y, train_size must agree on the batch dimension")
    if batch.X.shape[0] == 0:
        raise ValueError("batch contains no tables")
    tables: list[TableSplit] = []
    for index in range(batch.X.shape[0]):
        # trim padding: `collate_tabicl` pads every table to the batch's longest
        # seq_len, so only the first `seq_len[index]` rows are real.  Without the
        # trim, padded zero rows would leak into the support/query split.
        seq_len = int(batch.seq_len[index].item())
        X = np.asarray(batch.X[index, :seq_len].detach().cpu().numpy(), dtype=np.float32)
        y = np.asarray(batch.y[index, :seq_len].detach().cpu().numpy(), dtype=np.float32)
        train_size = int(batch.train_size[index].item())
        y_mean = float(batch.y_mean[index].item())
        y_std = float(batch.y_std[index].item())
        tables.append(split_table(X, y, train_size, y_mean=y_mean, y_std=y_std))
    return tables
