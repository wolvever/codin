import pytest
import asyncio
from codin.replay.base import ReplayService

class DummyStep:
    def __init__(self, step_id='step1', step_type='typeA'):
        self.step_id = step_id
        self.step_type = step_type
    def __str__(self):
        return f"DummyStep({self.step_id}, {self.step_type})"

class DummyResult:
    def __init__(self, value='result1'):
        self.value = value
    def __str__(self):
        return f"DummyResult({self.value})"

@pytest.mark.asyncio
async def test_replay_service_record_and_get():
    service = ReplayService()
    session_id = 'test_session'
    step = DummyStep()
    result = DummyResult()

    await service.record_step(session_id, step, result)
    log = await service.get_replay_log(session_id)

    assert isinstance(log, list)
    assert len(log) == 1
    entry = log[0]
    assert entry['step_id'] == 'step1'
    assert entry['step_type'] == 'typeA'
    assert entry['step_data']['type'] == 'DummyStep'
    assert 'DummyStep' in entry['step_data']['data']
    assert entry['result']['type'] == 'DummyResult'
    assert 'DummyResult' in entry['result']['data'] 