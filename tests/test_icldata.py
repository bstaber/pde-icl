"""Tests for the pde_priors -> tabicl table bridge (no network / no checkpoint)."""

import numpy as np
import pytest
import torch
from pde_priors.icl import TabICLBatch

from pde_icl.icldata import split_table, tables_from_batch


def _batch(n_tables: int = 2, t: int = 8, h: int = 9, k: int = 5) -> TabICLBatch:
    X = torch.arange(n_tables * t * h, dtype=torch.float32).reshape(n_tables, t, h)
    y = torch.arange(n_tables * t, dtype=torch.float32).reshape(n_tables, t)
    return TabICLBatch(
        X=X,
        y=y,
        d=torch.tensor(h, dtype=torch.int64),
        seq_len=torch.tensor([t] * n_tables, dtype=torch.int64),
        train_size=torch.tensor([k, k], dtype=torch.int64),
        y_mean=torch.zeros(n_tables, dtype=torch.float32),
        y_std=torch.ones(n_tables, dtype=torch.float32),
    )


def test_split_table_returns_support_and_query() -> None:
    X = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
    y = np.arange(10, dtype=np.float32)
    split = split_table(X, y, train_size=4)
    assert split.X_support.shape == (4, 3)
    assert split.y_support.shape == (4,)
    assert split.X_query.shape == (6, 3)
    assert split.y_query.shape == (6,)
    assert np.allclose(split.X_support, X[:4])
    assert np.allclose(split.X_query, X[4:])


def test_tables_from_batch_materializes_each_table() -> None:
    tables = tables_from_batch(_batch(n_tables=2, t=8, h=9, k=5))
    assert len(tables) == 2
    assert tables[0].X_support.shape == (5, 9)
    assert tables[0].X_query.shape == (3, 9)


def test_tables_from_batch_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no tables"):
        batch = TabICLBatch(
            X=torch.zeros(0, 4, 3),
            y=torch.zeros(0, 4),
            d=torch.tensor(3, dtype=torch.int64),
            seq_len=torch.tensor([], dtype=torch.int64),
            train_size=torch.tensor([], dtype=torch.int64),
            y_mean=torch.tensor([], dtype=torch.float32),
            y_std=torch.tensor([], dtype=torch.float32),
        )
        tables_from_batch(batch)


def test_tables_from_batch_trims_padded_rows_by_seq_len() -> None:
    h = 9
    max_t = 20
    # table 0 has 10 real rows, table 1 has 20; both padded in a [2, max_t, h] batch
    X = torch.zeros(2, max_t, h, dtype=torch.float32)
    y = torch.zeros(2, max_t, dtype=torch.float32)
    X[0, :10] = torch.arange(10 * h, dtype=torch.float32).reshape(10, h)
    y[0, :10] = torch.arange(10, dtype=torch.float32)
    X[1, :20] = torch.arange(20 * h, dtype=torch.float32).reshape(20, h)
    y[1, :20] = torch.arange(20, dtype=torch.float32)
    batch = TabICLBatch(
        X=X,
        y=y,
        d=torch.tensor(h, dtype=torch.int64),
        seq_len=torch.tensor([10, 20], dtype=torch.int64),
        train_size=torch.tensor([4, 8], dtype=torch.int64),
        y_mean=torch.zeros(2, dtype=torch.float32),
        y_std=torch.tensor([1.0, 2.0], dtype=torch.float32),
    )
    tables = tables_from_batch(batch)
    # table 0 trimmed to 10 rows -> support 4, query 6 (NO padded zero rows leaked)
    assert tables[0].X_support.shape == (4, h)
    assert tables[0].X_query.shape == (6, h)
    assert tables[0].y_query.shape == (6,)
    # table 1 full length -> support 8, query 12
    assert tables[1].X_support.shape == (8, h)
    assert tables[1].X_query.shape == (12, h)
    # per-table target stats are carried for de-normalization
    assert tables[0].y_std == 1.0
    assert tables[1].y_std == 2.0


def test_split_table_carries_target_stats() -> None:
    X = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
    y = np.arange(10, dtype=np.float32)
    split = split_table(X, y, train_size=4, y_mean=0.25, y_std=1.5)
    assert split.y_mean == 0.25
    assert split.y_std == 1.5
