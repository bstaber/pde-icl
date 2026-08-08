"""Tests for the online pde_priors batch generator (CPU only, no checkpoint)."""

import pytest
import torch
from pde_priors.generators.geometry_aware import GeometryAwareGenerator
from pde_priors.geometries import GeometryPrior
from pde_priors.icl.schemas import N_COLUMNS

from pde_icl.online_prior import PdePriorIterable, make_online_request


def _dataset(batch_size: int = 4, root_seed: int = 7):
    gen = GeometryAwareGenerator(prior=GeometryPrior())
    request = make_online_request(interior_points=24, boundary_points=6)  # T=30
    return PdePriorIterable(gen, request, batch_size=batch_size, root_seed=root_seed)


def test_batch_contract_shapes_and_uniformity() -> None:
    X, y, d, seq_len, train_size = next(_dataset(batch_size=4))
    B, T, H = X.shape
    assert B == 4
    assert T == 30  # interior 24 + boundary 6
    assert H == N_COLUMNS == 9
    assert y.shape == (B, T)
    # d / seq_len / train_size are [B] tensors matching run_batch's dim-0 split
    assert d.shape == (B,) and (d == N_COLUMNS).all() and d.dtype == torch.int64
    assert seq_len.shape == (B,) and (seq_len == T).all()
    assert train_size.shape == (B,)
    # uniform train_size within the batch, and a non-empty query (train < seq)
    ts = int(train_size[0].item())
    assert (train_size == ts).all()
    assert 0 < ts < T


def test_batch_is_fresh_each_step() -> None:
    it = _dataset(batch_size=2)
    X1, *_ = next(it)
    X2, *_ = next(it)
    assert not torch.equal(X1, X2)


def test_deterministic_given_same_seed() -> None:
    Xa, *_ = next(_dataset(batch_size=2, root_seed=5))
    Xb, *_ = next(_dataset(batch_size=2, root_seed=5))
    Xc, *_ = next(_dataset(batch_size=2, root_seed=6))
    assert torch.equal(Xa, Xb)  # same seed -> same tables
    assert not torch.equal(Xa, Xc)  # different seed -> different tables


def test_support_fraction_is_respected() -> None:
    gen = GeometryAwareGenerator(prior=GeometryPrior())
    request = make_online_request(interior_points=24, boundary_points=0)
    it = PdePriorIterable(gen, request, batch_size=1, root_seed=1, support_fraction=0.9)
    X, y, d, seq_len, train_size = next(it)
    T = X.shape[1]
    assert int(seq_len[0]) == T == 24
    # context = round(0.9*24) = 22 interior rows (boundary_points=0)
    assert int(train_size[0]) == 22
    assert 0 < train_size[0] < seq_len[0]


def test_make_online_request_rejects_degenerate_interior() -> None:
    with pytest.raises(ValueError):
        make_online_request(interior_points=1, boundary_points=0)
