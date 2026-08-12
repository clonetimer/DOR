from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def weighted_upper_p(reference_scores, reference_weights, scores, *, test_weight=1.0):
    """Importance-weighted upper-tail conformal-style p-value.

    FEC adaptation of weighted conformal under covariate shift. Larger scores
    mean more anomalous. `reference_weights` should approximate the target/source
    covariate density ratio at calibration points. `test_weight` is the same
    ratio at the test point (scalar or one per score).

    The method only has the intended covariate-shift interpretation when the
    relevant conditional mechanism is stable and the likelihood ratio is valid.
    """
    r=np.asarray(reference_scores,dtype=float)
    w=np.asarray(reference_weights,dtype=float)
    s=np.atleast_1d(np.asarray(scores,dtype=float))
    tw=np.asarray(test_weight,dtype=float)
    if r.ndim!=1 or w.ndim!=1 or len(r)!=len(w) or len(r)<2:
        raise ValueError('reference_scores/reference_weights must be equal-length 1D arrays with >=2 values')
    if np.any(~np.isfinite(r)) or np.any(~np.isfinite(w)) or np.any(w<0):
        raise ValueError('reference scores/weights must be finite and weights non-negative')
    if np.all(w<=0):
        raise ValueError('at least one reference weight must be positive')
    if tw.ndim==0: tw=np.full(len(s),float(tw))
    if tw.shape!=(len(s),) or np.any(~np.isfinite(tw)) or np.any(tw<0):
        raise ValueError('test_weight must be non-negative scalar or match scores')
    out=np.empty(len(s),dtype=float)
    for i,x in enumerate(s):
        numerator=tw[i]+np.sum(w[r>=x])
        denominator=tw[i]+np.sum(w)
        out[i]=numerator/denominator
    return out


@dataclass
class LogisticDensityRatio:
    """Classifier-based target/source density-ratio estimator.

    This is a practical helper, not a guarantee that the ratio model is correct.
    It trains a logistic classifier to discriminate source vs target covariates;
    odds are adjusted by the source/target sample-count ratio.
    """
    model: object
    n_source: int
    n_target: int
    clip: tuple[float,float]=(1e-3,1e3)

    @classmethod
    def fit(cls, source_X, target_X, *, C=1.0, random_state=0, clip=(1e-3,1e3)):
        from sklearn.linear_model import LogisticRegression
        Xs=np.asarray(source_X,dtype=float);Xt=np.asarray(target_X,dtype=float)
        if Xs.ndim==1: Xs=Xs[:,None]
        if Xt.ndim==1: Xt=Xt[:,None]
        if Xs.ndim!=2 or Xt.ndim!=2 or Xs.shape[1]!=Xt.shape[1]:
            raise ValueError('source_X and target_X must be 2D with matching features')
        X=np.vstack([Xs,Xt]);y=np.r_[np.zeros(len(Xs),int),np.ones(len(Xt),int)]
        m=LogisticRegression(C=C,max_iter=2000,random_state=random_state).fit(X,y)
        return cls(m,len(Xs),len(Xt),tuple(map(float,clip)))

    def ratio(self,X):
        X=np.asarray(X,dtype=float)
        if X.ndim==1:X=X[:,None]
        p=self.model.predict_proba(X)[:,1]
        eps=1e-9
        odds=np.clip(p,eps,1-eps)/np.clip(1-p,eps,1-eps)
        ratio=odds*(self.n_target and self.n_source/self.n_target)
        return np.clip(ratio,self.clip[0],self.clip[1])
