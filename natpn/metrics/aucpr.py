from typing import Any, cast, List
import torch
import torchmetrics
import torchmetrics.functional as M
from torchmetrics.utilities.compute import auc


class AUCPR(torchmetrics.Metric):
    """
    Computes the area under the precision recall curve.
    """

    values: List[torch.Tensor]
    targets: List[torch.Tensor]

    def __init__(self, dist_sync_fn: Any = None):
        super().__init__(dist_sync_fn=dist_sync_fn)

        self.add_state("values", [], dist_reduce_fx="cat")
        self.add_state("targets", [], dist_reduce_fx="cat")

    def update(self, values: torch.Tensor, targets: torch.Tensor) -> None:
        self.values.append(values)
        self.targets.append(targets)

    def compute(self) -> torch.Tensor:
        precision, recall, _ = M.precision_recall_curve(
            torch.cat(self.values), torch.cat(self.targets), task="binary"
        )
        return auc(cast(torch.Tensor, recall), cast(torch.Tensor, precision), reorder=True)
