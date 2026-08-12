#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observation-side numerical frontend for system identification.

This module is deliberately *not* part of the Mechanism Graph vocabulary.
Its purpose is to prevent measurement noise from being amplified by finite-difference
state-increment targets.  The current implementation uses a local polynomial
(Savitzky-Golay) smoother; the semantic grammar remains unchanged.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from scipy.signal import savgol_filter
from .contracts import PublicDataset

def filtered_state_observations(pub: PublicDataset, window: int = 0, polyorder: int = 3) -> PublicDataset:
    """Return a dataset with only X smoothed along time; U/schema/traj ids are unchanged.

    window<=0 means identity.  An odd window is enforced and clipped to trajectory length.
    This function never touches private truth and never filters public commands/contexts.
    """
    if not window or window < 5:
        return pub
    w=int(window)
    if w%2==0:w+=1
    maxw=pub.X.shape[1] if pub.X.shape[1]%2==1 else pub.X.shape[1]-1
    w=min(w,maxw)
    if w < 5:return pub
    p=min(int(polyorder),w-2)
    X=savgol_filter(np.asarray(pub.X,dtype=np.float64),window_length=w,polyorder=p,axis=1,mode='interp')
    return replace(pub,X=X.astype(np.float32))
