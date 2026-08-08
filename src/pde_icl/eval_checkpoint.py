"""Evaluate a `tabicl.train` checkpoint (our custom architecture) on pde_priors
tables, and measure cross-geometry (per-family) zero-shot RMSE.

The stock ``TabICLRegressor`` only loads the fixed-architecture HF checkpoint;
our pre-trained model has a custom architecture, so this module rebuilds it via
``TabICL(**model_config)`` (mirroring ``Trainer.build_model``), loads the
saved ``state_dict``, and runs the same in-context forward used at training time::

    pred = model(X, y_train, d)   # -> (B, test_size, num_quantiles)

de-normalizing the median quantile prediction with the carried support-target std
(``rmse_raw = rmse_std * y_std``).
"""

from __future__ import annotations

from typing import cast

import torch
from pde_priors.config import GenerationRequest
from pde_priors.generators.geometry_aware import GeometryAwareGenerator
from pde_priors.geometries import GeometryPrior
from pde_priors.icl import TabICLBatch, adapt, collate_tabicl
from pde_priors.icl.schemas import N_COLUMNS
from pde_priors.random import BatchKey
from tabicl._model.tabicl import TabICL
from tabicl.train._train_config import build_parser

from pde_icl.eval_zero_shot import TableEval


def build_model_config(train_args: list[str]) -> dict[str, object]:
    """Rebuild the model config exactly as `Trainer.build_model` did for the run.

    ``train_args`` are the same CLI args used to launch `tabicl.train`.
    """
    cfg = build_parser().parse_args(train_args)
    if cfg.regression_method != "quantile":
        raise ValueError("eval_checkpoint supports only regression_method='quantile'")
    bias_free_ln = cfg.norm_type == "layernorm_nobias"
    return {
        "max_classes": 0,  # quantile regression -> no classification head
        "num_quantiles": cfg.num_quantiles,
        "embed_dim": cfg.embed_dim,
        "col_num_blocks": cfg.col_num_blocks,
        "col_nhead": cfg.col_nhead,
        "col_num_inds": cfg.col_num_inds,
        "col_affine": cfg.col_affine,
        "col_feature_group": cfg.col_feature_group,
        "col_feature_group_size": cfg.col_feature_group_size,
        "col_target_aware": cfg.col_target_aware,
        "col_ssmax": cfg.ssmax_type if cfg.col_ssmax else False,
        "row_num_blocks": cfg.row_num_blocks,
        "row_nhead": cfg.row_nhead,
        "row_num_cls": cfg.row_num_cls,
        "row_rope_base": cfg.row_rope_base,
        "row_rope_interleaved": cfg.row_rope_interleaved,
        "icl_num_blocks": cfg.icl_num_blocks,
        "icl_nhead": cfg.icl_nhead,
        "icl_ssmax": cfg.ssmax_type if cfg.icl_ssmax else False,
        "ff_factor": cfg.ff_factor,
        "dropout": cfg.dropout,
        "activation": cfg.activation,
        "norm_first": cfg.norm_first,
        "bias_free_ln": bias_free_ln,
        "zero_init": cfg.zero_init,
        "recompute": cfg.recompute,
    }


def load_trained_model(ckpt_path: str, train_args: list[str], device: str = "cuda") -> TabICL:
    """Build our architecture and load the trained ``state_dict`` from a checkpoint."""
    model = TabICL(**build_model_config(train_args))
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = checkpoint["state_dict"]
    # strip any DDP 'module.' prefix (not present for single-GPU, defensive)
    state_dict = {k: v for k, v in state_dict.items()}
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def evaluate_batch(
    model: TabICL,
    batch: TabICLBatch,
    num_quantiles: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-context forward on one collated batch; return per-table (std, raw) RMSE.

    Mirror the training-time forward: ``pred = model(X, y_train, d)`` produces
    quantile predictions on the query rows; the median quantile is the point
    prediction.
    """
    B, T, H = batch.X.shape
    train_size = int(batch.train_size[0].item())
    X = batch.X.to(device)
    y = batch.y.to(device)
    y_train = y[:, :train_size]
    d = torch.full((B,), N_COLUMNS, dtype=torch.int64, device=device)
    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(X, y_train, d)
        else:
            pred = model(X, y_train, d)
    median = pred[:, :, num_quantiles // 2].float()  # (B, test_size)
    y_test = y[:, train_size:]  # (B, test_size)
    rmse_std = (median - y_test).pow(2).mean(dim=1).sqrt()  # (B,)
    rmse_raw = rmse_std * batch.y_std.to(device)
    return rmse_std, rmse_raw


def evaluate_trained_on_family(
    ckpt_path: str,
    train_args: list[str],
    families: dict[str, float] | None,
    request: GenerationRequest,
    *,
    n_tables: int,
    root_seed: int,
    device: str,
) -> TableEval:
    """Zero-shot RMSE of the trained model on tables from one geometry prior."""
    model = load_trained_model(ckpt_path, train_args, device)
    prior = GeometryPrior(families=families) if families is not None else GeometryPrior()
    gen = GeometryAwareGenerator(prior=prior)
    samples = [
        adapt(
            *gen.generate_with_domain(
                request, BatchKey(root_seed=root_seed, epoch=0, global_batch_index=i)
            )
        )
        for i in range(n_tables)
    ]
    batch = collate_tabicl(samples)
    num_quantiles = cast(int, build_model_config(train_args)["num_quantiles"])
    rmse_std, rmse_raw = evaluate_batch(model, batch, num_quantiles, device)
    return TableEval(
        rmse_std=float(rmse_std.mean().item()),
        rmse_raw=float(rmse_raw.mean().item()),
    )


def trained_cross_geometry(
    ckpt_path: str,
    train_args: list[str],
    request: GenerationRequest,
    *,
    n_tables: int,
    root_seed: int,
    device: str,
) -> dict[str, TableEval]:
    """Per-family cross-geometry zero-shot RMSE of the trained model."""
    configs: dict[str, dict[str, float] | None] = {
        "circle": {"circle": 1.0},
        "ellipse": {"ellipse": 1.0},
        "fourier_star": {"fourier_star": 1.0},
        "all": None,
    }
    return {
        label: evaluate_trained_on_family(
            ckpt_path,
            train_args,
            fam,
            request,
            n_tables=n_tables,
            root_seed=root_seed,
            device=device,
        )
        for label, fam in configs.items()
    }
