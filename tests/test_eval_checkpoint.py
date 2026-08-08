"""Tests for the trained-checkpoint evaluator (no GPU / no checkpoint file)."""

import pytest

from pde_icl.eval_checkpoint import build_model_config


def test_build_model_config_quantile_regression() -> None:
    cfg = build_model_config(
        [
            "--regression_method",
            "quantile",
            "--num_quantiles",
            "64",
            "--embed_dim",
            "96",
            "--icl_num_blocks",
            "6",
            "--row_num_blocks",
            "2",
            "--col_num_inds",
            "64",
            "--ff_factor",
            "3",
        ]
    )
    assert cfg["max_classes"] == 0  # quantile regression -> no classification head
    assert cfg["num_quantiles"] == 64
    assert cfg["embed_dim"] == 96
    assert cfg["icl_num_blocks"] == 6
    assert cfg["ff_factor"] == 3
    assert cfg["bias_free_ln"] is False  # default norm_type


def test_build_model_config_rejects_non_quantile() -> None:
    # default regression_method is None (classification) -> unsupported here
    with pytest.raises(ValueError):
        build_model_config(["--min_seq_len", "144", "--max_seq_len", "160"])
