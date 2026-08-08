"""Tests for de-normalization and the cross-geometry eval harness (no network)."""

import pde_icl.eval_zero_shot as ev
from pde_icl.eval_zero_shot import TableEval, raw_rmse
from pde_icl.pretrain_data import build_geometry_request


def test_raw_rmse_der_normalizes_by_target_std() -> None:
    assert raw_rmse(1.0, 1.0) == 1.0
    assert raw_rmse(0.5, 2.0) == 1.0  # standardized 0.5 @ std 2 -> raw 1.0
    assert raw_rmse(0.3, 0.0) == 0.0


def test_cross_geometry_summary_structure(monkeypatch) -> None:
    fake = TableEval(rmse_std=0.5, rmse_raw=1.25)
    monkeypatch.setattr(
        ev,
        "evaluate_geometry_family",
        lambda fam, request, **kwargs: fake,
    )
    request = build_geometry_request(interior_points=16, boundary_points=4)
    out = ev.cross_geometry_summary(request, n_tables=1)
    assert set(out) == {"circle", "ellipse", "fourier_star", "all"}
    assert all(r == fake for r in out.values())


def test_evaluate_geometry_family_builds_family_prior(monkeypatch) -> None:
    captured = {}

    def fake_eval(gen, request, *, n_tables, root_seed, device, verbose):
        captured["families"] = dict(gen.prior.family_weights)
        return TableEval(rmse_std=1.0, rmse_raw=1.0)

    monkeypatch.setattr(ev, "_evaluate_prior", fake_eval)
    request = build_geometry_request(interior_points=16, boundary_points=4)
    out = ev.evaluate_geometry_family({"ellipse": 1.0}, request, n_tables=1, root_seed=0)
    assert captured["families"] == {"ellipse": 1.0}
    assert out == TableEval(rmse_std=1.0, rmse_raw=1.0)
