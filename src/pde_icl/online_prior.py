"""Online pde_priors batch generation for TabICL pre-training.

TabICL's own ``PriorDataset`` is hardwired to its internal SCM priors; to train
on our geometry-conditioned PDE prior without pre-generating to disk we inject a
replacement ``IterableDataset`` that produces a fresh full batch every iteration.

Each ``__next__`` returns a 5-tuple matching ``Trainer.run_batch``'s contract::

    (X [B,T,H], y [B,T], d [B], seq_len [B], train_size [B])

where ``B == config.batch_size``, generated live from ``GeometryAwareGenerator``
+ ``adapt`` + ``collate_tabicl``.  ``validate_micro_batch`` requires identical
``seq_len``/``train_size`` within a micro-batch -- satisfied because a single
fixed ``GenerationRequest`` yields uniform tables.

Multi-process (`DataLoader(..., num_workers>0)`): each worker derives a disjoint
table stream from ``root_seed + worker_id * stride`` so workers never regenerate
the same tables.
"""

from __future__ import annotations

import torch
from pde_priors.config import GenerationRequest
from pde_priors.generators.geometry_aware import GeometryAwareGenerator
from pde_priors.icl import adapt, collate_tabicl
from pde_priors.icl.schemas import N_COLUMNS
from pde_priors.random import BatchKey
from torch.utils.data import IterableDataset, get_worker_info

from pde_icl.pretrain_data import build_geometry_request

_WORKER_STRIDE = 10_000_003  # disjoint key space per dataloader worker


class PdePriorIterable(IterableDataset[tuple[torch.Tensor, ...]]):
    """Infinite stream of pde_priors training batches (one live BVP per table)."""

    def __init__(
        self,
        generator: GeometryAwareGenerator,
        request: GenerationRequest,
        *,
        batch_size: int,
        root_seed: int,
        support_fraction: float = 0.5,
    ) -> None:
        self.generator = generator
        self.request = request
        self.batch_size = batch_size
        self.root_seed = root_seed
        self.support_fraction = support_fraction
        self._offset = 0

    def _stream_seed(self) -> int:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        return self.root_seed + worker_id * _WORKER_STRIDE

    def __iter__(self) -> PdePriorIterable:
        return self

    def __next__(self) -> tuple[torch.Tensor, ...]:
        seed = self._stream_seed()
        samples = []
        for index in range(self.batch_size):
            key = BatchKey(root_seed=seed, epoch=self._offset, global_batch_index=index)
            bvp, domain = self.generator.generate_with_domain(self.request, key)
            samples.append(adapt(bvp, domain, support_fraction=self.support_fraction, key=key))
        collated = collate_tabicl(samples)  # X [B,T,H], y [B,T]
        B, _, _ = collated.X.shape
        d = torch.full((B,), N_COLUMNS, dtype=torch.int64)
        self._offset += 1
        return collated.X, collated.y, d, collated.seq_len, collated.train_size


def make_online_request(*, interior_points: int, boundary_points: int) -> GenerationRequest:
    """A single-BVP request with fixed rows, so tables are uniform within a batch."""
    if interior_points < 2:
        raise ValueError(
            "interior_points must be >= 2 (need at least one observed and one "
            "withheld interior row)"
        )
    return build_geometry_request(interior_points=interior_points, boundary_points=boundary_points)
