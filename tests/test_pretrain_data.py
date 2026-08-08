"""Tests for exporting pde_priors tables in TabICL's pre-training format."""

import torch
from pde_priors.generators.geometry_aware import GeometryAwareGenerator
from pde_priors.geometries import GeometryPrior

from pde_icl.pretrain_data import build_geometry_request, generate_and_save


def test_saved_batches_round_trip_through_tabicl_loader(tmp_path) -> None:
    from tabicl.prior._genload import LoadPriorDataset

    gen = GeometryAwareGenerator(prior=GeometryPrior())
    request = build_geometry_request(interior_points=24, boundary_points=6)
    save_dir = generate_and_save(
        gen,
        request,
        save_dir=tmp_path,
        n_batches=2,
        batch_size=4,
        root_seed=7,
    )
    assert (save_dir / "metadata.json").exists()

    loaded = LoadPriorDataset(save_dir, batch_size=4, max_batches=2, device="cpu")
    it = iter(loaded)
    for _ in range(2):
        X, y, d, seq_lens, train_sizes = next(it)
        # reconstructed dense table batch: [B, T, H], one BVP per table
        assert X.ndim == 3
        B, T, H = X.shape
        assert B == 4
        assert H == 9
        assert T == 30  # interior 24 + boundary 6
        assert y.shape == (B, T)
        assert (d == 9).all()
        assert (seq_lens == T).all()
        # context (train_sizes) < seq_len for every table; query is non-empty
        assert (train_sizes > 3).all() and (train_sizes < T).all()
        assert torch.isfinite(X).all()
        assert torch.isfinite(y).all()


def test_build_geometry_request_single_bvp_per_table() -> None:
    request = build_geometry_request(interior_points=32, boundary_points=8)
    assert request.batch_size == 1
    assert request.boundary is not None
    assert request.boundary.boundary_points == 8
