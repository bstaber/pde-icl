"""Export pde_priors in-context tables in TabICL's pre-training file format.

TabICL's open pre-training pipeline (`tabicl.train`) consumes prior datasets from
disk via ``tabicl.prior._genload.LoadPriorDataset``, which reads per-batch files
saved as::

    {"X": sparse, "y": [B,T], "d": [B], "seq_lens": [B],
     "train_sizes": [B], "batch_size": B}

``X`` is stored in tabicl's dense2sparse format (each of the B*T rows truncated to
``d`` features) and reconstructed by ``LoadPriorDataset`` using ``seq_lens[0]``
(so every table in one batch must share the same row count).

This module turns a ``pde_priors`` stream -- one BVP per table, with the episode
semantics (all boundary rows + observed interior in the context, query = withheld
interior) -- into exactly those batch files, so the geometry prior can be fed to
``python -m tabicl.train`` unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pde_priors.config import BoundaryRequest, GenerationRequest
from pde_priors.generators.geometry_aware import GeometryAwareGenerator
from pde_priors.geometries import GeometryPrior
from pde_priors.icl import ICLStreamDataset, collate_tabicl
from pde_priors.icl.schemas import N_COLUMNS
from pde_priors.typing import BoundaryKind, EquationName


def _dense2sparse(
    dense: torch.Tensor, row_lengths: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Concatenate each row truncated to ``row_lengths`` valid entries (tabicl)."""
    num_rows, num_cols = dense.shape
    indices = torch.arange(num_cols, device=dense.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    return dense[mask].to(dtype)


def build_geometry_request(
    *,
    interior_points: int,
    boundary_points: int,
    device: torch.device | None = None,
) -> GenerationRequest:
    """A single-BVP Poisson request whose tables all share the same row count."""
    boundary = (
        None
        if boundary_points == 0
        else BoundaryRequest(kind=BoundaryKind.DIRICHLET, boundary_points=boundary_points)
    )
    return GenerationRequest(
        equation=EquationName.POISSON,
        spatial_dim=2,
        batch_size=1,  # one BVP per batch; the generator is single-BVP
        interior_points=interior_points,
        boundary=boundary,
        device=device if device is not None else torch.device("cpu"),
    )


def generate_and_save(
    gen: GeometryAwareGenerator,
    request: GenerationRequest,
    *,
    save_dir: Path,
    n_batches: int,
    batch_size: int,
    root_seed: int,
    support_fraction: float = 0.5,
) -> Path:
    """Generate ``n_batches`` batches of ``batch_size`` tables and save them.

    Each batch file is written atomically (tmp + rename) in tabicl's format so it
    can be loaded by ``tabicl.train`` via ``LoadPriorDataset``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    seq_len = request.interior_points + (
        request.boundary.points_per_face if request.boundary else 0
    )
    metadata = {
        "regression": True,
        "prior_type": "pde_priors_geometry_poisson",
        "batch_size": batch_size,
        "min_seq_len": seq_len,
        "max_seq_len": seq_len,
        "min_features": N_COLUMNS,
        "max_features": N_COLUMNS,
        "min_train_size": 0,
        "max_train_size": 0,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for idx in range(n_batches):
        stream = ICLStreamDataset(
            gen,
            request,
            root_seed=root_seed,
            batches_per_epoch=batch_size,
            support_fraction=support_fraction,
        )
        collated = collate_tabicl(list(stream))  # X [B,T,H], y [B,T], ...
        B, T, H = collated.X.shape
        d_batch = torch.full((B,), int(collated.d.item()), dtype=torch.int64)
        X_sparse = _dense2sparse(
            collated.X.reshape(-1, H),
            d_batch.repeat_interleave(T),
        )
        batch_file = save_dir / f"batch_{idx:06d}.pt"
        tmp_file = batch_file.with_suffix(".pt.tmp")
        torch.save(
            {
                "X": X_sparse,
                "y": collated.y,
                "d": d_batch,
                "seq_lens": collated.seq_len,
                "train_sizes": collated.train_size,
                "batch_size": B,
            },
            tmp_file,
        )
        tmp_file.replace(batch_file)
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pde-icl-pretrain-data",
        description="Generate TabICL pre-training batches from the pde_priors geometry prior.",
    )
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--n-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--root-seed", type=int, default=0)
    parser.add_argument("--interior-points", type=int, default=64)
    parser.add_argument("--boundary-points", type=int, default=16)
    parser.add_argument("--support-fraction", type=float, default=0.5)
    parser.add_argument("--coefficient-mode", default="global")
    parser.add_argument("--shape-solution", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    request = build_geometry_request(
        interior_points=args.interior_points,
        boundary_points=args.boundary_points,
    )
    gen = GeometryAwareGenerator(
        prior=GeometryPrior(),
        coefficient_mode=args.coefficient_mode,
        shape_solution=args.shape_solution,
    )
    path = generate_and_save(
        gen,
        request,
        save_dir=args.save_dir,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        root_seed=args.root_seed,
        support_fraction=args.support_fraction,
    )
    print(f"wrote {args.n_batches} batches to {path.resolve()}")


if __name__ == "__main__":
    main()
