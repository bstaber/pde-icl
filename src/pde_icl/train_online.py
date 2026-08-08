"""Train TabICL online on the pde_priors geometry prior (no disk round-trip).

Replaces TabICL's on-the-fly SCM ``PriorDataset`` / on-disk ``LoadPriorDataset``
with a live ``PdePriorIterable`` by subclassing the Trainer and overriding
``configure_prior()``.  Everything else (model build, AMP, optimizer/scheduler,
DDP, checkpointing, the training loop) is inherited unchanged.

Run::

    uv run python -m pde_icl.train_online [tabicl.train args...] \\
        [--pde-request-interior N] [--pde-request-boundary N] [--n_jobs K]

The extra ``--pde-*`` args size the single-BVP geometry request and the online
generator's sampling; ``--n_jobs`` (default 0) enables multiprocess prefetching.
"""

from __future__ import annotations

import argparse

from pde_priors.generators.geometry_aware import GeometryAwareGenerator
from pde_priors.geometries import GeometryPrior
from tabicl.prior._genload import seed_worker
from tabicl.train._run import Trainer
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from pde_icl.online_prior import PdePriorIterable, make_online_request


def _build_parser() -> argparse.ArgumentParser:
    """TabICL's trainer CLI plus the online pde_priors options."""
    from tabicl.train._train_config import build_parser

    parser: argparse.ArgumentParser = build_parser()
    parser.add_argument("--pde-request-interior", type=int, default=128)
    parser.add_argument("--pde-request-boundary", type=int, default=16)
    parser.add_argument("--pde-prior-root-seed", type=int, default=0)
    parser.add_argument("--pde-prior-support-fraction", type=float, default=0.5)
    return parser


class PdeTrainer(Trainer):  # type: ignore[misc]  # Trainer base is untyped (tabicl)
    """Trainer fed by the live pde_priors geometry generator."""

    def configure_prior(self) -> None:
        dataset = PdePriorIterable(
            self.config.pde_prior_generator,
            self.config.pde_prior_request,
            batch_size=self.config.batch_size,
            root_seed=self.config.pde_prior_root_seed,
            support_fraction=self.config.pde_prior_support_fraction,
        )
        num_workers = max(0, self.config.n_jobs)
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        if self.master_process:
            print(
                f"Online pde_priors dataloader: n_jobs={num_workers}, "
                f"batch_size={self.config.batch_size}"
            )


def main() -> None:
    config = _build_parser().parse_args()

    # Attach the online pde_priors configuration to the argparse Namespace so
    # PdeTrainer.configure_prior() can reach it (avoids overriding __init__).
    config.pde_prior_generator = GeometryAwareGenerator(prior=GeometryPrior())
    config.pde_prior_request = make_online_request(
        interior_points=config.pde_request_interior,
        boundary_points=config.pde_request_boundary,
    )
    config.pde_prior_support_fraction = config.pde_prior_support_fraction
    config.prior_dir = None  # always online; never read a pre-generated dir

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    trainer = PdeTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
