import json
from isgn.cli import main


def test_synthesize_and_compile_cli(tmp_path, monkeypatch):
    evidence={
      'system_id':'cli-demo',
      'variables':[{'name':'x','role':'driver','observation_status':'READY'},{'name':'y','role':'state','observation_status':'READY'}],
      'relations':[{'relation_id':'r','target':'y','sources':['x'],'kind':'dynamic','origin':'known','confidence':1.0}],
    }
    inp=tmp_path/'evidence.json'; inp.write_text(json.dumps(evidence))
    atlas=tmp_path/'atlas.json'; report=tmp_path/'report.json'
    monkeypatch.setattr('sys.argv',['isgn','synthesize-atlas','--evidence',str(inp),'--out',str(atlas),'--report',str(report)])
    main()
    a=json.loads(atlas.read_text()); sid=next(iter(a['scopes'])); a['scopes'][sid]['completion_status']='COMPLETE'; atlas.write_text(json.dumps(a))
    out=tmp_path/'arch.json'
    monkeypatch.setattr('sys.argv',['isgn','compile-architecture','--atlas',str(atlas),'--out',str(out)])
    main()
    d=json.loads(out.read_text()); assert d['output_variables']==['y']; assert d['scopes'][0]['targets'][0]['sources']==['x']
