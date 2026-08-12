from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence
import numpy as np

from fec.core.fec_core import (
    ContactSpec,
    RootCalibrationReference,
    make_root_reference,
    calibrated_p,
)


class CalibrationGroupUnavailable(KeyError):
    """Raised when no sufficiently populated calibration stratum exists."""


@dataclass(frozen=True)
class CalibrationGroupInfo:
    key: Hashable
    n_reference: int
    scope: str
    provenance: str


class MondrianRootRegistry:
    """Group-conditional root calibration registry.

    FEC adaptation of Mondrian/group-conditional conformal calibration.
    A group key is external to the Core and may encode, for example:
      - operating context id;
      - modality/contact presence mask;
      - (context id, presence mask).

    Each group still enforces the exact FEC topology/A/G signature.

    Important: this helper produces empirical group-conditional p-values for
    FEC root scores. It does NOT claim the prediction-set theorem of MCCF.
    Statistical validity still requires exchangeability inside the declared
    calibration stratum.
    """

    def __init__(self, *, min_group_size: int = 20):
        if min_group_size < 2:
            raise ValueError('min_group_size must be >= 2')
        self.min_group_size = int(min_group_size)
        self._refs: dict[Hashable, RootCalibrationReference] = {}

    def fit_group(
        self,
        key: Hashable,
        scores: np.ndarray,
        specs: Sequence[ContactSpec],
        *,
        A: np.ndarray | None = None,
        G: np.ndarray | None = None,
        scope: str = 'root',
        dependency_set: Iterable[str] = (),
        provenance: str = 'trusted-normal',
    ) -> RootCalibrationReference:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError('scores must be 1D')
        if len(scores) < self.min_group_size:
            raise ValueError(
                f'calibration group {key!r} has {len(scores)} samples; '
                f'minimum is {self.min_group_size}'
            )
        ref = make_root_reference(
            scores, specs, A=A, G=G, scope=scope,
            dependency_set=dependency_set,
            provenance=f'{provenance};mondrian_group={key!r}',
        )
        self._refs[key] = ref
        return ref

    def fit_groups(
        self,
        grouped_scores: Mapping[Hashable, np.ndarray],
        specs: Sequence[ContactSpec],
        *,
        A: np.ndarray | None = None,
        G: np.ndarray | None = None,
        scope: str = 'root',
        dependency_set: Iterable[str] = (),
        provenance: str = 'trusted-normal',
    ) -> None:
        for key, scores in grouped_scores.items():
            self.fit_group(
                key, scores, specs, A=A, G=G, scope=scope,
                dependency_set=dependency_set, provenance=provenance,
            )

    def has_group(self, key: Hashable) -> bool:
        return key in self._refs

    def info(self, key: Hashable) -> CalibrationGroupInfo:
        if key not in self._refs:
            raise CalibrationGroupUnavailable(key)
        ref = self._refs[key]
        return CalibrationGroupInfo(key, len(ref.scores), ref.scope, ref.provenance)

    def p_value(
        self,
        key: Hashable,
        scores: np.ndarray,
        specs: Sequence[ContactSpec],
        *,
        A: np.ndarray | None = None,
        G: np.ndarray | None = None,
    ) -> np.ndarray:
        if key not in self._refs:
            raise CalibrationGroupUnavailable(
                f'no calibrated reference for group {key!r}; '
                'FEC should report reference-unsupported rather than select '
                'the most permissive reference.'
            )
        return calibrated_p(self._refs[key], scores, specs, A=A, G=G)

    @staticmethod
    def presence_mask(G: np.ndarray) -> tuple[int, ...]:
        return tuple(int(x > 0) for x in np.asarray(G, dtype=float))

    @staticmethod
    def context_presence_key(context: Hashable, G: np.ndarray) -> tuple[Hashable, tuple[int, ...]]:
        return context, MondrianRootRegistry.presence_mask(G)
