from typing import Any, Optional

from torch import Tensor

from torchmetrics.utilities.enums import AverageMethod, DataType
from torchmetrics.classification.stat_scores import StatScores
from torchmetrics.functional.classification.f_beta import _fbeta_compute
from torchmetrics.functional.classification.precision_recall import _precision_compute, _recall_compute

class ClassificationStatScores(StatScores):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
        if average not in allowed_average:
            raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

        _reduce_options = (AverageMethod.WEIGHTED, AverageMethod.NONE, None)
        if "reduce" not in kwargs:
            kwargs["reduce"] = AverageMethod.MACRO if average in _reduce_options else average
        if "mdmc_reduce" not in kwargs:
            kwargs["mdmc_reduce"] = mdmc_average

        super().__init__(
            threshold=threshold,
            top_k=top_k,
            num_classes=num_classes,
            multiclass=multiclass,
            ignore_index=ignore_index,
            **kwargs,
        )

        self.average = average
        self.beta = 1.0

    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._get_final_stats()
        p = _precision_compute(tp, fp, fn, self.average, self.mdmc_reduce)
        r = _recall_compute(tp, fp, fn, self.average, self.mdmc_reduce)
        f1 = _fbeta_compute(tp, fp, tn, fn, 1.0, self.ignore_index, self.average, self.mdmc_reduce)
        f2 = _fbeta_compute(tp, fp, tn, fn, 2.0, self.ignore_index, self.average, self.mdmc_reduce)
        f0_5 = _fbeta_compute(tp, fp, tn, fn, 0.5, self.ignore_index, self.average, self.mdmc_reduce)
        
        return {'Precision': p, 'Recall': r, 'F1-Score': f1, 'F2-Score': f2, 'F0.5-Score': f0_5}
    