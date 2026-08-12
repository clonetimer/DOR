from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable, Mapping


@dataclass(frozen=True)
class GroupRouteDecision:
    requested_key: Hashable
    selected_key: Hashable | None
    mode: str
    supported: bool
    power_limited: bool
    reason: str


class SparseGroupPolicy:
    """Safe routing policy for sparse or empirically conservative strata.

    Exact groups are preferred. A declared parent may be used only when the
    Adapter explicitly supplies a parent mapping *and* the parent's trusted
    sentinel audit is acceptable. Conservative-but-adequate groups are kept
    as-is and marked power-limited; they are not silently pooled away merely to
    make the empirical FPR look closer to alpha.
    """
    def __init__(self, *, min_group_size: int = 20):
        self.min_group_size = int(min_group_size)

    def decide(
        self,
        key: Hashable,
        *,
        n_reference: Mapping[Hashable, int],
        health_status: Mapping[Hashable, str] | None = None,
        declared_parent: Mapping[Hashable, Hashable] | None = None,
    ) -> GroupRouteDecision:
        health_status = health_status or {}
        declared_parent = declared_parent or {}
        n = int(n_reference.get(key, 0))
        health = str(health_status.get(key, 'unknown')).lower()
        if n >= self.min_group_size:
            if health == 'anti_conservative':
                return GroupRouteDecision(key, key, 'exact_group', False, False, 'exact group is anti-conservative on trusted sentinel')
            if health == 'conservative':
                return GroupRouteDecision(key, key, 'exact_group', True, True, 'exact group is conservative; keep semantics and report reduced power')
            return GroupRouteDecision(key, key, 'exact_group', True, False, 'exact group is adequately populated')
        parent = declared_parent.get(key)
        if parent is None:
            return GroupRouteDecision(key, None, 'abstain', False, False, 'sparse group and no declared compatible parent')
        pn = int(n_reference.get(parent, 0)); ph = str(health_status.get(parent, 'unknown')).lower()
        if pn < self.min_group_size:
            return GroupRouteDecision(key, None, 'abstain', False, False, 'declared parent is also sparse')
        if ph in {'anti_conservative'}:
            return GroupRouteDecision(key, None, 'abstain', False, False, 'declared parent is anti-conservative')
        return GroupRouteDecision(key, parent, 'declared_parent', True, ph == 'conservative', 'using explicitly declared compatible parent')
