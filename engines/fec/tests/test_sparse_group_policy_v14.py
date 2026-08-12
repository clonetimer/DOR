from fec.extensions.sparse_group_policy import SparseGroupPolicy

def test_sparse_group_abstains_without_declared_parent():
    d=SparseGroupPolicy(min_group_size=20).decide('child',n_reference={'child':5})
    assert not d.supported and d.mode=='abstain'

def test_sparse_group_uses_only_declared_healthy_parent():
    p=SparseGroupPolicy(min_group_size=20)
    d=p.decide('child',n_reference={'child':5,'parent':50},health_status={'parent':'healthy'},declared_parent={'child':'parent'})
    assert d.supported and d.selected_key=='parent'
    bad=p.decide('child',n_reference={'child':5,'parent':50},health_status={'parent':'anti_conservative'},declared_parent={'child':'parent'})
    assert not bad.supported

def test_conservative_group_is_kept_and_power_limited():
    d=SparseGroupPolicy().decide(2,n_reference={2:200},health_status={2:'conservative'})
    assert d.supported and d.power_limited and d.selected_key==2
