"""Bound successful audit payloads without changing returned data or errors."""
import json
import pytest
from gamax.gama_client import GamaClient


@pytest.mark.parametrize('record',(False,True))
def test_successful_logging_does_not_change_rpc_result(monkeypatch,record):
    class Socket:
        def __init__(self):self.request=None
        def send(self,text):self.request=json.loads(text)
        def recv(self):
            if self.request is None:return json.dumps({'type':'ConnectionSuccessful'})
            return json.dumps({'type':'CommandExecutedSuccessfully','command':self.request,'content':{'values':[0.125]*10000}})
        def close(self):pass
    monkeypatch.setattr('gamax.gama_client.websocket.create_connection',lambda *args,**kwargs:Socket())
    client=GamaClient(record_response_payloads=record)
    assert client.command('expression',exp_id='arbitrary',expr='snapshot')['values']==[0.125]*10000
    response=client.commands[0]['response']
    if record:assert 'content' in response
    else:
        assert 'content' not in response
        assert response['payload_bytes']>50000
        assert len(response['payload_sha256'])==64
        assert len(json.dumps(client.commands))<1000


def test_failed_requests_retain_full_diagnostics(monkeypatch):
    class Socket:
        def __init__(self):self.request=None
        def send(self,text):self.request=json.loads(text)
        def recv(self):
            if self.request is None:return json.dumps({'type':'ConnectionSuccessful'})
            return json.dumps({'type':'UnableToExecuteRequest','command':self.request,'content':{'error':'full diagnostic'}})
        def close(self):pass
    monkeypatch.setattr('gamax.gama_client.websocket.create_connection',lambda *args,**kwargs:Socket())
    client=GamaClient(record_response_payloads=False)
    with pytest.raises(RuntimeError,match='full diagnostic'):client.command('step',exp_id='arbitrary',nb_step=1)
    assert client.commands[0]['response']['content']=={'error':'full diagnostic'}
