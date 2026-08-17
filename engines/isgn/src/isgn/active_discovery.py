"""ISGN Active Mechanism Discovery backend.

This module extends the existing closed-loop information-gap contract with a device-neutral
Bayesian experiment-design layer.  It deliberately separates:

* candidate generation (untrusted hypotheses / parameter priors),
* evidence acquisition (truth-quarantined experiment observations),
* simulation-based likelihood / SMC updating,
* value-of-information experiment ranking, and
* predictive-adequacy / M-open expansion.

The implementation is intentionally solver-agnostic.  A device adapter supplies forward
simulation samples; this module never reads hidden test truth or fault labels.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Iterable, Mapping, Sequence
import hashlib
import math
import numpy as np


def entropy_bits(values: Mapping[str, float] | Sequence[float]) -> float:
    x = np.asarray(list(values.values()) if isinstance(values, Mapping) else list(values), dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(-(x * np.log2(x)).sum()) if x.size else 0.0


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("posterior weights must be a non-empty finite non-negative vector")
    s = float(x.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("posterior mass collapsed")
    return x / s


@dataclass(frozen=True)
class ParameterParticle:
    particle_id: str
    structure_id: str
    parameters: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.particle_id).strip() or not str(self.structure_id).strip():
            raise ValueError("particle_id and structure_id must be non-empty")
        for name, value in self.parameters.items():
            if not str(name).strip() or not np.isfinite(float(value)):
                raise ValueError("particle parameters must have non-empty names and finite numeric values")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentCandidate:
    experiment_id: str
    payload: Any = None
    repeats: int = 1
    unit_cost: float = 1.0
    safety_eligible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    execution_mode: str = "ADVISORY_ONLY"
    safety_evidence_id: str | None = None

    def __post_init__(self) -> None:
        if not str(self.experiment_id).strip():
            raise ValueError("experiment_id must be non-empty")
        if isinstance(self.repeats, bool) or int(self.repeats) != self.repeats or int(self.repeats) < 1:
            raise ValueError("repeats must be a positive integer")
        if not np.isfinite(float(self.unit_cost)) or float(self.unit_cost) <= 0:
            raise ValueError("unit_cost must be finite and positive")
        mode = str(self.execution_mode or "ADVISORY_ONLY").upper()
        if mode not in {"ADVISORY_ONLY", "OFFLINE_REPLAY", "SIMULATION", "LIVE"}:
            raise ValueError(f"unsupported execution_mode {mode!r}")
        if self.safety_evidence_id is not None and not str(self.safety_evidence_id).strip():
            raise ValueError("safety_evidence_id must be non-empty when supplied")

    @property
    def cost(self) -> float:
        return float(self.repeats) * float(self.unit_cost)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost"] = self.cost
        return d


@dataclass(frozen=True)
class ExperimentExecutionPolicy:
    """Fail-closed execution policy for experiment recommendations.

    Active Discovery is an advisory planner by default.  Offline replay, simulation and advisory-only
    candidates may be ranked when they satisfy the declared budget/safety gate.  A ``LIVE`` experiment
    is excluded unless live actuation is explicitly enabled *and* the candidate carries a non-empty
    ``safety_evidence_id``.  This module never executes an actuator.
    """

    allow_live_actuation: bool = False
    require_live_safety_evidence: bool = True
    maximum_candidate_cost: float | None = None
    approved_live_experiment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.maximum_candidate_cost is not None:
            value = float(self.maximum_candidate_cost)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("maximum_candidate_cost must be finite and positive")
        ids = tuple(str(x) for x in self.approved_live_experiment_ids)
        if any(not x.strip() for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("approved_live_experiment_ids must be unique non-empty strings")


@dataclass(frozen=True)
class ExperimentAdmission:
    experiment_id: str
    status: str
    eligible: bool
    cost: float
    execution_mode: str
    reason: str
    truth_channels_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def admit_experiment_candidate(
    candidate: ExperimentCandidate,
    *,
    remaining_budget: float,
    policy: ExperimentExecutionPolicy | None = None,
) -> ExperimentAdmission:
    policy = policy or ExperimentExecutionPolicy()
    if not np.isfinite(float(remaining_budget)) or float(remaining_budget) < 0:
        raise ValueError("remaining_budget must be finite and non-negative")
    mode = str(candidate.execution_mode or "ADVISORY_ONLY").upper()
    if not candidate.safety_eligible:
        return ExperimentAdmission(candidate.experiment_id, "BLOCKED_UNSAFE", False, candidate.cost, mode,
                                   "candidate is not safety eligible")
    if candidate.cost > float(remaining_budget) + 1e-12:
        return ExperimentAdmission(candidate.experiment_id, "BLOCKED_BUDGET", False, candidate.cost, mode,
                                   "candidate cost exceeds remaining budget")
    if policy.maximum_candidate_cost is not None and candidate.cost > float(policy.maximum_candidate_cost) + 1e-12:
        return ExperimentAdmission(candidate.experiment_id, "BLOCKED_POLICY_COST", False, candidate.cost, mode,
                                   "candidate cost exceeds execution policy maximum")
    if mode == "LIVE":
        if not policy.allow_live_actuation:
            return ExperimentAdmission(candidate.experiment_id, "BLOCKED_LIVE_NOT_AUTHORIZED", False, candidate.cost, mode,
                                       "live actuation is disabled by default")
        if policy.require_live_safety_evidence and not candidate.safety_evidence_id:
            return ExperimentAdmission(candidate.experiment_id, "BLOCKED_MISSING_SAFETY_EVIDENCE", False, candidate.cost, mode,
                                       "live actuation requires explicit safety evidence")
        approved = set(policy.approved_live_experiment_ids)
        if approved and candidate.experiment_id not in approved:
            return ExperimentAdmission(candidate.experiment_id, "BLOCKED_LIVE_NOT_APPROVED", False, candidate.cost, mode,
                                       "live experiment is not in the approved execution set")
    return ExperimentAdmission(candidate.experiment_id, "ELIGIBLE", True, candidate.cost, mode,
                               "candidate is eligible for ranking; execution remains external to ISGN")


@dataclass(frozen=True)
class ParticleSimulationSummary:
    particle_id: str
    experiment_id: str
    repeats: int
    n_simulations: int
    mean: np.ndarray
    covariance: np.ndarray


class ParticleSimulationBank:
    """Simulation-only likelihood bank.

    ``samples[(particle_id, experiment_id, repeats)]`` must be a matrix ``(n_sim, n_features)``.
    The bank estimates a shrinkage covariance per particle/experiment and is therefore compatible
    with likelihood-free / SBI workflows where candidate simulations are available but a tractable
    analytical likelihood is not.
    """

    def __init__(
        self,
        samples: Mapping[tuple[str, str, int], np.ndarray],
        *,
        shrinkage: float = 0.35,
        ridge: float = 1e-3,
        covariance_floor: Sequence[float] | None = None,
        student_df: float = 5.0,
    ) -> None:
        if not np.isfinite(float(shrinkage)) or not 0.0 <= float(shrinkage) <= 1.0:
            raise ValueError("shrinkage must be finite and in [0, 1]")
        if not np.isfinite(float(ridge)) or float(ridge) < 0:
            raise ValueError("ridge must be finite and non-negative")
        if not np.isfinite(float(student_df)) or float(student_df) <= 0:
            raise ValueError("student_df must be finite and positive")
        self.student_df = float(student_df)
        self._summaries: dict[tuple[str, str, int], ParticleSimulationSummary] = {}
        self._precision: dict[tuple[str, str, int], np.ndarray] = {}
        self._logdet: dict[tuple[str, str, int], float] = {}
        if not samples:
            raise ValueError("simulation bank must contain at least one particle/experiment sample")
        raw_floor = None if covariance_floor is None else np.asarray(covariance_floor, dtype=float)
        if raw_floor is not None and (raw_floor.ndim != 1 or not np.all(np.isfinite(raw_floor)) or np.any(raw_floor < 0)):
            raise ValueError("covariance_floor must be a finite non-negative vector")
        floor = None if raw_floor is None else raw_floor ** 2
        dimensions: dict[tuple[str, int], int] = {}
        for key, raw in samples.items():
            p, e, r = key
            if not str(p).strip() or not str(e).strip() or isinstance(r, bool) or int(r) != r or int(r) < 1:
                raise ValueError("simulation-bank keys require non-empty particle/experiment ids and positive integer repeats")
            x = np.asarray(raw, dtype=float)
            if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1 or not np.all(np.isfinite(x)):
                raise ValueError(f"simulation bank {key!r} must have shape (n,d), n>=3")
            dim_key = (str(e), int(r))
            if dim_key in dimensions and dimensions[dim_key] != x.shape[1]:
                raise ValueError(f"simulation-bank feature dimension mismatch for experiment/repeats {dim_key!r}")
            dimensions[dim_key] = int(x.shape[1])
            mu = x.mean(axis=0)
            cov = np.cov(x, rowvar=False)
            if np.ndim(cov) == 0:
                cov = np.array([[float(cov)]], dtype=float)
            diag = np.diag(np.diag(cov))
            cov = (1.0 - shrinkage) * cov + shrinkage * diag
            if floor is not None:
                if floor.shape != (x.shape[1],):
                    raise ValueError("covariance_floor dimension mismatch")
                cov = cov + np.diag(floor)
            scale = np.maximum(np.diag(cov), 1e-8)
            cov = cov + np.diag(ridge * scale + 1e-8)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                cov = cov + np.eye(cov.shape[0]) * 1e-6
                sign, logdet = np.linalg.slogdet(cov)
            k = (str(p), str(e), int(r))
            self._summaries[k] = ParticleSimulationSummary(str(p), str(e), int(r), len(x), mu, cov)
            self._precision[k] = np.linalg.inv(cov)
            self._logdet[k] = float(logdet)

    def summary(self, particle_id: str, experiment_id: str, repeats: int = 1) -> ParticleSimulationSummary:
        key = (str(particle_id), str(experiment_id), int(repeats))
        if key in self._summaries:
            return self._summaries[key]
        # Fail closed rather than assuming a 1/r covariance law.  Repeat observation semantics are
        # device specific (NeuronBench, for example, averages spike counts but exposes one voltage trace).
        raise KeyError(f"no repeat-aware simulation bank for {key!r}")

    def log_likelihood(self, particle_id: str, experiment_id: str, observation: np.ndarray, repeats: int = 1) -> float:
        k = (str(particle_id), str(experiment_id), int(repeats))
        s = self.summary(*k)
        y = np.asarray(observation, dtype=float)
        delta = y - s.mean
        d = y.size
        nu = self.student_df
        quad = float(delta @ self._precision[k] @ delta)
        return float(
            math.lgamma((nu + d) / 2.0)
            - math.lgamma(nu / 2.0)
            - 0.5 * (d * math.log(nu * math.pi) + self._logdet[k])
            - 0.5 * (nu + d) * math.log1p(quad / nu)
        )


@dataclass
class ParticleUpdateRecord:
    experiment_id: str
    repeats: int
    prior_structure_entropy_bits: float
    posterior_structure_entropy_bits: float
    ess_before_resample: float
    resampled: bool
    structure_posterior: dict[str, float]
    parameter_summary: dict[str, dict[str, float]]
    backend: str = "SIMULATION_BASED_PARAMETER_PARTICLE_SMC"
    truth_channels_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParameterParticleSMC:
    """Joint structure + continuous-parameter particle posterior.

    Resampling is structure-support preserving: every declared structure keeps at least one sentinel
    particle whenever the particle budget allows it.  This prevents an approximate synthetic likelihood
    from irreversibly killing a structure before a later discriminating intervention is observed.
    """

    def __init__(
        self,
        particles: Sequence[ParameterParticle],
        *,
        weights: Sequence[float] | None = None,
        seed: int = 0,
    ) -> None:
        if not particles:
            raise ValueError("at least one parameter particle is required")
        self.particles = list(particles)
        self.rng = np.random.default_rng(seed)
        self.weights = _normalise(np.ones(len(self.particles)) if weights is None else np.asarray(weights, dtype=float))
        self._index = {p.particle_id: i for i, p in enumerate(self.particles)}
        if len(self._index) != len(self.particles):
            raise ValueError("particle_id must be unique")

    def copy(self) -> "ParameterParticleSMC":
        return ParameterParticleSMC(self.particles, weights=self.weights.copy(), seed=0)

    def ess(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))

    def structure_posterior(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p, w in zip(self.particles, self.weights):
            out[p.structure_id] = out.get(p.structure_id, 0.0) + float(w)
        return out

    def structure_entropy(self) -> float:
        return entropy_bits(self.structure_posterior())

    def parameter_summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for structure in sorted({p.structure_id for p in self.particles}):
            idx = np.array([i for i, p in enumerate(self.particles) if p.structure_id == structure], dtype=int)
            mass = float(self.weights[idx].sum())
            if mass <= 0:
                continue
            w = self.weights[idx] / mass
            names = sorted({k for i in idx for k in self.particles[i].parameters})
            row: dict[str, float] = {"posterior_mass": mass}
            for name in names:
                vals = np.array([float(self.particles[i].parameters.get(name, np.nan)) for i in idx])
                good = np.isfinite(vals)
                if not good.any():
                    continue
                wg = w[good]
                wg = wg / wg.sum()
                vg = vals[good]
                mean = float(np.sum(wg * vg))
                var = float(np.sum(wg * (vg - mean) ** 2))
                row[f"{name}_mean"] = mean
                row[f"{name}_sd"] = math.sqrt(max(var, 0.0))
            out[structure] = row
        return out

    def update(
        self,
        bank: ParticleSimulationBank,
        experiment_id: str,
        observation: np.ndarray,
        *,
        repeats: int = 1,
        resample_threshold: float = 0.35,
    ) -> ParticleUpdateRecord:
        if not np.isfinite(float(resample_threshold)) or not 0.0 <= float(resample_threshold) <= 1.0:
            raise ValueError("resample_threshold must be finite and in [0, 1]")
        prior_entropy = self.structure_entropy()
        logw = np.log(np.maximum(self.weights, 1e-300))
        for i, p in enumerate(self.particles):
            logw[i] += bank.log_likelihood(p.particle_id, experiment_id, observation, repeats)
        m = float(np.max(logw))
        w = np.exp(logw - m)
        self.weights = _normalise(w)
        ess = self.ess()
        resampled = False
        if ess < float(resample_threshold) * len(self.particles):
            self._support_preserving_resample()
            resampled = True
        return ParticleUpdateRecord(
            str(experiment_id), int(repeats), prior_entropy, self.structure_entropy(), ess, resampled,
            self.structure_posterior(), self.parameter_summary(),
        )

    def _support_preserving_resample(self) -> None:
        """Support-preserving particle rejuvenation without changing particle identity.

        A simulation bank is keyed by immutable particle IDs, so ordinary multinomial resampling
        would require regenerating likelihood banks for duplicated/jittered particles.  The release
        candidate instead performs a conservative weight rejuvenation: it preserves the categorical
        structure posterior exactly, while mixing each structure's conditional parameter weights with
        a small uniform component.  This prevents zero-weight parameter particles from becoming
        irrecoverable and keeps the external SBI bank valid.
        """
        structure_mass = self.structure_posterior()
        new_w = self.weights.copy()
        rejuvenation = 0.08
        for structure, mass in structure_mass.items():
            idx = np.array([i for i, p in enumerate(self.particles) if p.structure_id == structure], dtype=int)
            if idx.size == 0 or mass <= 0:
                continue
            conditional = self.weights[idx] / max(float(self.weights[idx].sum()), 1e-300)
            conditional = (1.0 - rejuvenation) * conditional + rejuvenation / idx.size
            new_w[idx] = mass * conditional
        self.weights = _normalise(new_w)



@dataclass(frozen=True)
class DesignScore:
    experiment_id: str
    repeats: int
    cost: float
    expected_structure_information_gain_bits: float
    utility_per_cost: float
    current_structure_entropy_bits: float
    status: str = "RANKED"
    truth_channels_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _posterior_from_loglikes(prior_w: np.ndarray, loglikes: np.ndarray) -> np.ndarray:
    z = np.log(np.maximum(prior_w, 1e-300)) + loglikes
    z = z - np.max(z)
    return _normalise(np.exp(z))


def _structure_entropy_for_particle_weights(particles: Sequence[ParameterParticle], weights: np.ndarray) -> float:
    out: dict[str, float] = {}
    for p, w in zip(particles, weights):
        out[p.structure_id] = out.get(p.structure_id, 0.0) + float(w)
    return entropy_bits(out)


def expected_structure_information_gain(
    posterior: ParameterParticleSMC,
    bank: ParticleSimulationBank,
    candidate: ExperimentCandidate,
    *,
    max_source_particles: int = 32,
    predictive_draws_per_particle: int = 2,
) -> float:
    """Monte-Carlo expected reduction in *structure* entropy.

    Parameter uncertainty is retained inside each structure, but the design objective is the reduction
    in mechanism-structure ambiguity.  This avoids mistaking narrow nuisance-parameter learning for
    successful mechanism discrimination.
    """
    if int(max_source_particles) < 1 or int(predictive_draws_per_particle) < 1:
        raise ValueError("max_source_particles and predictive_draws_per_particle must be positive")
    if not candidate.safety_eligible:
        return 0.0
    H0 = posterior.structure_entropy()
    n = len(posterior.particles)
    if n <= max_source_particles:
        src = np.arange(n)
        src_w = posterior.weights.copy()
    else:
        # deterministic weighted quantile-like sampling for reproducible design scores
        seed = int(hashlib.sha256(candidate.experiment_id.encode()).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        src = rng.choice(np.arange(n), size=max_source_particles, replace=True, p=posterior.weights)
        src_w = posterior.weights[src]
        src_w = _normalise(src_w)
    expected_H = 0.0
    for j, i in enumerate(src):
        p = posterior.particles[int(i)]
        s = bank.summary(p.particle_id, candidate.experiment_id, candidate.repeats)
        seed = int(hashlib.sha256(f"{candidate.experiment_id}|{p.particle_id}".encode()).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        ys = rng.multivariate_normal(s.mean, s.covariance, size=int(predictive_draws_per_particle))
        for y in ys:
            ll = np.array([
                bank.log_likelihood(q.particle_id, candidate.experiment_id, y, candidate.repeats)
                for q in posterior.particles
            ])
            post_w = _posterior_from_loglikes(posterior.weights, ll)
            expected_H += float(src_w[j]) / predictive_draws_per_particle * _structure_entropy_for_particle_weights(posterior.particles, post_w)
    return max(float(H0 - expected_H), 0.0)


def rank_experiments(
    posterior: ParameterParticleSMC,
    bank: ParticleSimulationBank,
    candidates: Sequence[ExperimentCandidate],
    *,
    remaining_budget: float,
    max_source_particles: int = 32,
    predictive_draws_per_particle: int = 2,
    execution_policy: ExperimentExecutionPolicy | None = None,
) -> list[DesignScore]:
    if not np.isfinite(float(remaining_budget)) or float(remaining_budget) < 0:
        raise ValueError("remaining_budget must be finite and non-negative")
    rows: list[DesignScore] = []
    H = posterior.structure_entropy()
    for c in candidates:
        admission = admit_experiment_candidate(c, remaining_budget=remaining_budget, policy=execution_policy)
        if not admission.eligible:
            continue
        eig = expected_structure_information_gain(
            posterior, bank, c,
            max_source_particles=max_source_particles,
            predictive_draws_per_particle=predictive_draws_per_particle,
        )
        rows.append(DesignScore(c.experiment_id, c.repeats, c.cost, eig, eig / max(c.cost, 1e-12), H))
    return sorted(rows, key=lambda r: (-r.utility_per_cost, -r.expected_structure_information_gain_bits, r.cost, r.experiment_id))


def predictive_adequacy(
    posterior: ParameterParticleSMC,
    bank: ParticleSimulationBank,
    experiment_id: str,
    observation: np.ndarray,
    *,
    repeats: int = 1,
    mahalanobis_threshold: float = 5.0,
) -> dict[str, Any]:
    if not np.isfinite(float(mahalanobis_threshold)) or float(mahalanobis_threshold) <= 0:
        raise ValueError("mahalanobis_threshold must be finite and positive")
    rows: list[tuple[float, str, str]] = []
    y = np.asarray(observation, dtype=float)
    for p in posterior.particles:
        s = bank.summary(p.particle_id, experiment_id, repeats)
        key = (p.particle_id, experiment_id, int(repeats))
        # use the bank's likelihood geometry but report an intuitive Mahalanobis distance
        delta = y - s.mean
        md = float(np.sqrt(max(delta @ bank._precision[key] @ delta, 0.0)))
        rows.append((md, p.structure_id, p.particle_id))
    md, structure, particle = min(rows)
    return {
        "status": "ADEQUATE" if md <= mahalanobis_threshold else "HYPOTHESIS_CLASS_INADEQUATE",
        "min_mahalanobis": md,
        "threshold": float(mahalanobis_threshold),
        "nearest_structure": structure,
        "nearest_particle": particle,
        "truth_channels_allowed": False,
    }



@dataclass(frozen=True)
class EvidenceStabilityDecision:
    status: str
    current_top_structure: str
    robust_top_structure: str
    n_observations: int
    loo_same_top_fraction: float
    loo_top_structures: tuple[str, ...]
    reason: str
    truth_channels_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["loo_top_structures"] = list(self.loo_top_structures)
        return d


def _posterior_from_history_without_resampling(
    prior: ParameterParticleSMC,
    bank: ParticleSimulationBank,
    observations: Sequence[tuple[str, np.ndarray, int]],
) -> tuple[np.ndarray, dict[str, float]]:
    """Recompute one static-likelihood posterior for an evidence subset.

    This intentionally bypasses sequential resampling/rejuvenation so it can act as an independent
    stability check on the online SMC path.  It does not replace the online posterior.
    """
    logw = np.log(np.maximum(np.asarray(prior.weights, dtype=float), 1e-300))
    for experiment_id, observation, repeats in observations:
        y = np.asarray(observation, dtype=float)
        ll = np.array([
            bank.log_likelihood(p.particle_id, str(experiment_id), y, int(repeats))
            for p in prior.particles
        ], dtype=float)
        logw += ll
    m = float(np.max(logw))
    w = _normalise(np.exp(logw - m))
    sp: dict[str, float] = {}
    for p, wi in zip(prior.particles, w):
        sp[p.structure_id] = sp.get(p.structure_id, 0.0) + float(wi)
    return w, sp


def assess_evidence_stability(
    posterior: ParameterParticleSMC,
    prior: ParameterParticleSMC,
    bank: ParticleSimulationBank,
    observations: Sequence[tuple[str, np.ndarray, int]],
    *,
    minimum_loo_same_top_fraction: float = 0.80,
) -> EvidenceStabilityDecision:
    """Jackknife-style stability guard for a unique mechanism claim.

    The final online posterior is compared with an independently recomputed, no-resampling posterior
    over the same evidence.  Then each observed intervention is removed once.  A unique advisory
    identification is considered stable only when the robust full-history top agrees with the online
    top and that top survives a declared fraction of leave-one-intervention-out replays.  This guard
    catches confidently-wrong posteriors produced by approximate likelihoods or over-flexible nuisance
    parameters without consulting hidden mechanism truth.
    """
    if not 0.0 <= float(minimum_loo_same_top_fraction) <= 1.0:
        raise ValueError("minimum_loo_same_top_fraction must be in [0, 1]")
    if len(prior.particles) != len(posterior.particles) or {p.particle_id for p in prior.particles} != {p.particle_id for p in posterior.particles}:
        return EvidenceStabilityDecision(
            "UNSTABLE", max(posterior.structure_posterior(), key=posterior.structure_posterior().get), "",
            len(observations), 0.0, tuple(),
            "prior and posterior particle supports differ; independent evidence-stability replay is not valid",
        )
    cur_sp = posterior.structure_posterior(); current_top = max(cur_sp, key=cur_sp.get)
    if len(observations) < 2:
        return EvidenceStabilityDecision(
            "INSUFFICIENT", current_top, current_top, len(observations), 0.0, tuple(),
            "at least two independent intervention observations are required for leave-one-intervention-out stability",
        )
    _, robust_sp = _posterior_from_history_without_resampling(prior, bank, observations)
    robust_top = max(robust_sp, key=robust_sp.get)
    loo_tops: list[str] = []
    for i in range(len(observations)):
        subset = list(observations[:i]) + list(observations[i + 1:])
        _, sp = _posterior_from_history_without_resampling(prior, bank, subset)
        loo_tops.append(max(sp, key=sp.get))
    frac = float(sum(t == robust_top for t in loo_tops) / len(loo_tops))
    stable = robust_top == current_top and frac + 1e-12 >= float(minimum_loo_same_top_fraction)
    if stable:
        reason = "online and no-resampling full-history posteriors agree, and the top structure is stable under leave-one-intervention-out replay"
        status = "STABLE"
    elif robust_top != current_top:
        reason = "online posterior top disagrees with the independent no-resampling evidence replay"
        status = "UNSTABLE"
    else:
        reason = "top structure is too sensitive to removal of one intervention for a unique mechanism claim"
        status = "UNSTABLE"
    return EvidenceStabilityDecision(status, current_top, robust_top, len(observations), frac, tuple(loo_tops), reason)


@dataclass(frozen=True)
class StructureIdentifiabilityDecision:
    status: str
    top_structure: str
    top_posterior_mass: float
    alternative_structures: tuple[str, ...]
    credible_structures: tuple[str, ...]
    remaining_discriminating_experiments: tuple[str, ...]
    max_pairwise_separation: Mapping[str, float]
    reason: str
    certification_allowed: bool = False
    truth_channels_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["alternative_structures"] = list(self.alternative_structures)
        d["credible_structures"] = list(self.credible_structures)
        d["remaining_discriminating_experiments"] = list(self.remaining_discriminating_experiments)
        return d


def _structure_predictive_moments(
    posterior: ParameterParticleSMC,
    bank: ParticleSimulationBank,
    structure_id: str,
    experiment_id: str,
    repeats: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.array([i for i, p in enumerate(posterior.particles) if p.structure_id == structure_id], dtype=int)
    if idx.size == 0:
        raise KeyError(structure_id)
    w = posterior.weights[idx].astype(float)
    if float(w.sum()) <= 0:
        w = np.ones(idx.size, dtype=float) / idx.size
    else:
        w = w / w.sum()
    means = []
    covs = []
    for i in idx:
        p = posterior.particles[int(i)]
        sm = bank.summary(p.particle_id, experiment_id, repeats)
        means.append(np.asarray(sm.mean, dtype=float))
        covs.append(np.asarray(sm.covariance, dtype=float))
    means = np.asarray(means, dtype=float)
    mu = np.sum(w[:, None] * means, axis=0)
    cov = np.zeros_like(covs[0], dtype=float)
    for wi, mi, ci in zip(w, means, covs):
        d = mi - mu
        cov += float(wi) * (ci + np.outer(d, d))
    return mu, cov


def structure_pair_separation(
    posterior: ParameterParticleSMC,
    bank: ParticleSimulationBank,
    structure_a: str,
    structure_b: str,
    candidate: ExperimentCandidate,
) -> float:
    """Symmetric predictive separation in pooled Mahalanobis units.

    The score includes simulation noise and within-structure parameter uncertainty.  Small values mean
    the two structures are observationally equivalent under this experiment, even if posterior mass has
    accidentally concentrated on one of them.
    """
    ma, ca = _structure_predictive_moments(posterior, bank, structure_a, candidate.experiment_id, candidate.repeats)
    mb, cb = _structure_predictive_moments(posterior, bank, structure_b, candidate.experiment_id, candidate.repeats)
    pooled = 0.5 * (ca + cb)
    diag = np.maximum(np.diag(pooled), 1e-8)
    pooled = pooled + np.diag(1e-6 * diag + 1e-10)
    delta = ma - mb
    try:
        prec = np.linalg.inv(pooled)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(pooled)
    return float(np.sqrt(max(delta @ prec @ delta, 0.0)))


def assess_structure_identifiability(
    posterior: ParameterParticleSMC,
    bank: ParticleSimulationBank,
    candidates: Sequence[ExperimentCandidate],
    *,
    remaining_budget: float,
    executed_experiment_ids: Iterable[str] = (),
    execution_policy: ExperimentExecutionPolicy | None = None,
    adequacy_status: str = "ADEQUATE",
    evidence_stability_status: str | None = None,
    hypothesis_space_coverage_complete: bool = True,
    unresolved_candidate_structures: Sequence[str] = (),
    credible_mass: float = 0.90,
    minimum_structure_mass: float = 0.02,
    unique_mass_threshold: float = 0.80,
    equivalence_threshold: float = 1.25,
    discriminating_threshold: float = 2.0,
) -> StructureIdentifiabilityDecision:
    """Map an Active Discovery posterior to calibrated identifiability semantics.

    ``ALTERNATIVES`` means more than one live structure remains but at least one admissible experiment
    can still discriminate the set.  ``UNIDENTIFIABLE`` means the live alternatives are observationally
    equivalent under every currently admissible experiment (or no admissible experiment remains).
    ``UNIQUE_SUPPORTED`` is only an advisory identifiability state: it never bypasses the normal ISGN
    certification pipeline, hence ``certification_allowed`` is always false here.
    """
    numeric_thresholds = {
        "credible_mass": credible_mass,
        "minimum_structure_mass": minimum_structure_mass,
        "unique_mass_threshold": unique_mass_threshold,
    }
    for name, value in numeric_thresholds.items():
        if not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    for name, value in {
        "equivalence_threshold": equivalence_threshold,
        "discriminating_threshold": discriminating_threshold,
    }.items():
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(discriminating_threshold) < float(equivalence_threshold):
        raise ValueError("discriminating_threshold must be >= equivalence_threshold")
    if not np.isfinite(float(remaining_budget)) or float(remaining_budget) < 0:
        raise ValueError("remaining_budget must be finite and non-negative")
    sp = posterior.structure_posterior()
    ranked = sorted(sp.items(), key=lambda kv: (-kv[1], kv[0]))
    top, top_mass = ranked[0]
    unresolved = tuple(sorted({str(s) for s in unresolved_candidate_structures if str(s) and str(s) not in sp}))
    if str(adequacy_status).upper() != "ADEQUATE":
        return StructureIdentifiabilityDecision(
            "HYPOTHESIS_CLASS_INADEQUATE", top, float(top_mass), tuple(), tuple(s for s, _ in ranked), tuple(), {},
            "predictive adequacy failed; expand the hypothesis class before mechanism identification",
        )
    if evidence_stability_status is not None and str(evidence_stability_status).upper() != "STABLE":
        return StructureIdentifiabilityDecision(
            "HYPOTHESIS_CLASS_INADEQUATE", top, float(top_mass), tuple(), tuple(s for s, _ in ranked), tuple(), {},
            "evidence-stability guard did not pass; unique mechanism identification is vetoed",
        )

    # M-open safeguard: a concentrated posterior over the *currently admitted* structures is not evidence
    # that the hypothesis universe is complete.  Dormant/unreplayed proposals may only veto uniqueness;
    # they are not promoted to evidence-backed alternatives.  When coverage is explicitly incomplete,
    # Active Discovery must remain advisory and request hypothesis expansion / replay before mechanism
    # identification.  This closes the confidently-wrong failure mode observed under stochastic M-open.
    if not bool(hypothesis_space_coverage_complete):
        reason = "hypothesis-space coverage is incomplete; unresolved candidate structures must be admitted and replay-scored before unique mechanism identification"
        if unresolved:
            reason += ": " + ", ".join(unresolved)
        return StructureIdentifiabilityDecision(
            "HYPOTHESIS_CLASS_INADEQUATE", top, float(top_mass), tuple(), tuple(s for s, _ in ranked), tuple(), {}, reason,
        )

    credible: list[str] = []
    mass = 0.0
    for sid, prob in ranked:
        if prob >= minimum_structure_mass or not credible:
            credible.append(sid); mass += float(prob)
        if mass >= credible_mass:
            break
    # Do not let posterior concentration erase a structure that is predictively indistinguishable from
    # the top hypothesis across the declared experiment pool.  Such a structure is a genuine alternative.
    all_structures = [s for s, _ in ranked]
    executed = set(str(x) for x in executed_experiment_ids)
    eligible_remaining = []
    eligible_all = []
    for c in candidates:
        # Intrinsic observational equivalence is evaluated over the whole declared safe experiment
        # contract, including experiments already executed.  Otherwise, consuming the one discriminating
        # experiment would paradoxically make two already-separated structures appear unidentifiable.
        adm_all = admit_experiment_candidate(c, remaining_budget=max(float(remaining_budget), float(c.cost)), policy=execution_policy)
        if adm_all.eligible:
            eligible_all.append(c)
        if c.experiment_id in executed:
            continue
        adm = admit_experiment_candidate(c, remaining_budget=remaining_budget, policy=execution_policy)
        if adm.eligible:
            eligible_remaining.append(c)

    pair_max: dict[str, float] = {}
    separating_experiments: set[str] = set()
    alternatives = set(credible)
    for sid in all_structures:
        if sid == top:
            continue
        scores_all = []
        for c in eligible_all:
            try:
                sep = structure_pair_separation(posterior, bank, top, sid, c)
            except KeyError:
                continue
            scores_all.append((sep, c.experiment_id))
        for c in eligible_remaining:
            try:
                sep = structure_pair_separation(posterior, bank, top, sid, c)
            except KeyError:
                continue
            if sid in credible and sep >= discriminating_threshold:
                separating_experiments.add(c.experiment_id)
        mx = max((x[0] for x in scores_all), default=0.0)
        pair_max[f"{top}|{sid}"] = float(mx)
        if mx < equivalence_threshold:
            alternatives.add(sid)

    alternatives = {s for s in alternatives if s == top or sp.get(s, 0.0) >= minimum_structure_mass or pair_max.get(f"{top}|{s}", 0.0) < equivalence_threshold}
    alt_sorted = tuple(sorted(alternatives, key=lambda s: (-sp.get(s, 0.0), s)))

    # If predictive-equivalent alternatives exist, no amount of posterior concentration may certify
    # uniqueness under the current intervention contract.
    equiv_alts = [s for s in alt_sorted if s != top and pair_max.get(f"{top}|{s}", float("inf")) < equivalence_threshold]
    if equiv_alts:
        return StructureIdentifiabilityDecision(
            "UNIDENTIFIABLE", top, float(top_mass), alt_sorted, tuple(credible), tuple(sorted(separating_experiments)), pair_max,
            "one or more alternative structures are observationally equivalent to the top structure under all admissible remaining experiments",
        )

    if len(credible) > 1:
        if separating_experiments:
            return StructureIdentifiabilityDecision(
                "ALTERNATIVES", top, float(top_mass), tuple(credible), tuple(credible), tuple(sorted(separating_experiments)), pair_max,
                "multiple mechanism structures remain credible and at least one admissible experiment can still discriminate them",
            )
        return StructureIdentifiabilityDecision(
            "UNIDENTIFIABLE", top, float(top_mass), tuple(credible), tuple(credible), tuple(), pair_max,
            "multiple mechanism structures remain credible but no admissible remaining experiment can distinguish them",
        )

    if float(top_mass) >= unique_mass_threshold:
        return StructureIdentifiabilityDecision(
            "UNIQUE_SUPPORTED", top, float(top_mass), (top,), tuple(credible), tuple(), pair_max,
            "one structure is strongly supported and no predictive-equivalent alternative survives the declared intervention pool",
        )
    return StructureIdentifiabilityDecision(
        "ALTERNATIVES", top, float(top_mass), tuple(s for s, p in ranked if p >= minimum_structure_mass), tuple(credible),
        tuple(sorted(separating_experiments)), pair_max,
        "posterior support is not concentrated enough for a unique advisory identification",
    )



def admit_proposal_particles(
    posterior: ParameterParticleSMC,
    proposals: Sequence[ParameterParticle],
    *,
    proposal_prior_mass: float = 0.12,
    max_new_particles: int | None = None,
) -> ParameterParticleSMC:
    """Admit untrusted candidate particles with bounded prior mass.

    Admission is not certification: the returned particles merely become eligible to compete for
    posterior evidence.  Candidate text/source metadata is never converted into posterior evidence.
    """
    if not np.isfinite(float(proposal_prior_mass)) or not 0.0 < float(proposal_prior_mass) <= 0.45:
        raise ValueError("proposal_prior_mass must be finite and in (0, 0.45]")
    if max_new_particles is not None and (
        isinstance(max_new_particles, bool) or int(max_new_particles) != max_new_particles or int(max_new_particles) < 1
    ):
        raise ValueError("max_new_particles must be a positive integer when supplied")
    existing = {p.particle_id for p in posterior.particles}
    new = [p for p in proposals if p.particle_id not in existing]
    if max_new_particles is not None:
        new = new[: int(max_new_particles)]
    if not new:
        return posterior.copy()
    q = float(proposal_prior_mass)
    particles = list(posterior.particles) + list(new)
    weights = np.concatenate([posterior.weights * (1.0 - q), np.full(len(new), q / len(new))])
    return ParameterParticleSMC(particles, weights=weights, seed=0)


__all__ = [
    "ParameterParticle", "ExperimentCandidate", "ExperimentExecutionPolicy", "ExperimentAdmission",
    "admit_experiment_candidate", "ParticleSimulationSummary", "ParticleSimulationBank",
    "ParticleUpdateRecord", "ParameterParticleSMC", "DesignScore", "EvidenceStabilityDecision", "StructureIdentifiabilityDecision", "entropy_bits",
    "expected_structure_information_gain", "rank_experiments", "predictive_adequacy", "assess_evidence_stability",
    "structure_pair_separation", "assess_structure_identifiability", "admit_proposal_particles",
]
