import json
from pathlib import Path
from isgn.atlas import MechanismAtlas, MechanismScopeRecord, save_atlas
from isgn.cli import main


def test_route_evidence_cli(tmp_path, monkeypatch):
    a=MechanismAtlas('demo')
    a.add_scope(MechanismScopeRecord('s',['x'],completion_status='COMPLETE',phm_coverage={'evidence_kinds':['mechanism_residual']}))
    ap=tmp_path/'atlas.json'; save_atlas(a,ap)
    ep=tmp_path/'events.json'; ep.write_text(json.dumps([{'event_id':'e','evidence_kind':'mechanism_residual','variables':['x']}]))
    out=tmp_path/'route.json'
    monkeypatch.setattr('sys.argv',['isgn','route-evidence','--atlas',str(ap),'--events',str(ep),'--out',str(out)])
    main()
    d=json.loads(out.read_text())
    assert d['routes'][0]['decision']=='ROUTED_CERTIFIED_SCOPE'
