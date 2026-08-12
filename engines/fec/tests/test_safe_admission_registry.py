import numpy as np
from fec.core.fec_core import ContactSpec, make_root_reference
from fec.extensions.safe_admission import FrozenReserveRegistry


def test_registry_requires_external_key_and_dependency_claims():
    specs=[ContactSpec('x',('a',),'x')]
    ref=make_root_reference(np.linspace(0,1,100),specs,dependency_set={'ctx0','surface'})
    reg=FrozenReserveRegistry({'ctx0':np.linspace(0,1,100)},alpha=.05)
    miss=reg.check('ctx1',.2,trusted_claims={'ctx0','surface'},target_reference=ref)
    assert not miss.allowed
    dep=reg.check('ctx0',.2,trusted_claims={'ctx0'},target_reference=ref)
    assert not dep.allowed and not dep.dependency_ok
    ok=reg.check('ctx0',.2,trusted_claims={'ctx0','surface'},target_reference=ref)
    assert ok.allowed and ok.dependency_ok
