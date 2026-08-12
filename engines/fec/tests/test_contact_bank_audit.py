import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fec.core.fec_core import ContactSpec
from fec.extensions.contact_bank_audit import audit_contact_bank

def test_contact_bank_audit_flags_capacity_and_redundancy_without_auto_equivalence():
    x=np.arange(30,dtype=float)
    X=np.c_[x, x*2, np.sin(x), np.cos(x)]
    s=[ContactSpec('a',('big',)),ContactSpec('a_copylike',('big',)),ContactSpec('b',('big',)),ContactSpec('c',('small',))]
    r=audit_contact_bank(X,s,corr_threshold=.999)
    assert r.branch_counts['big']==3 and r.branch_counts['small']==1
    assert any({p.key_i,p.key_j}=={'a','a_copylike'} for p in r.redundancy_pairs)
    assert all(spec.equivalence_id is None for spec in s)
