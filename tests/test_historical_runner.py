import json
import sys

from geo2wf.historical import run


def test_runner_collects_and_verifies_without_queuing_training(tmp_path, monkeypatch):
    root=tmp_path/'data';root.mkdir()
    output=tmp_path/'run'
    calls=[]
    monkeypatch.setattr(sys,'argv',['run','--root',str(root),'--output',str(output)])
    monkeypatch.setattr(run.subprocess,'run',lambda args,**kwargs:calls.append(args))
    run.main()
    assert [args[3] for args in calls]==['collect','collect','verify']
    assert all(args[2]=='geo2wf.historical.dataset' for args in calls)
    assert '--retry-gaps' in calls[1]
    state=json.loads((output/'status.json').read_text())
    assert state['stage']=='dataset_ready'
    assert state['automatic_training'] is False
