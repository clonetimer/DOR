from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .opinion_fusion import DirichletOpinion

@dataclass(frozen=True)
class ClassEvidenceBatch:
    evidence: np.ndarray
    class_frame: tuple[str,...]
    source: str
    provenance: str

    def __post_init__(self):
        e=np.asarray(self.evidence,dtype=float)
        if e.ndim != 2 or e.shape[1] != len(self.class_frame) or e.shape[1] < 2:
            raise ValueError('evidence must be n_samples x n_classes and match class_frame')
        if np.any(~np.isfinite(e)) or np.any(e<0):
            raise ValueError('class evidence must be finite and non-negative')
        if len(set(self.class_frame)) != len(self.class_frame):
            raise ValueError('class_frame labels must be unique')
        if not self.source or not self.provenance:
            raise ValueError('source and provenance are required')
        object.__setattr__(self,'evidence',e.copy())

    def opinions(self) -> list[DirichletOpinion]:
        return [DirichletOpinion(row,source=self.source) for row in self.evidence]


def validate_shared_frame(batches: Sequence[ClassEvidenceBatch]) -> tuple[str,...]:
    if not batches:
        raise ValueError('at least one class-evidence batch is required')
    frame=batches[0].class_frame; n=len(batches[0].evidence)
    for b in batches[1:]:
        if b.class_frame != frame:
            raise ValueError('all class-evidence sources must share exactly the same ordered class frame')
        if len(b.evidence) != n:
            raise ValueError('all class-evidence sources must contain the same number of samples')
    return frame
