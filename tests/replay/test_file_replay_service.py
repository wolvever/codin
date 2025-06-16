from pathlib import Path
import asyncio
import json


import pytest

from codin.replay.file import FileReplayService


class DummyStep:
    def __init__(self, step_id="step1", step_type="typeA"):
        self.step_id = step_id
        self.step_type = step_type

    def __str__(self) -> str:
        return f"DummyStep({self.step_id}, {self.step_type})"


class DummyResult:
    def __init__(self, value="result1"):
        self.value = value

    def __str__(self) -> str:
        return f"DummyResult({self.value})"


def test_file_replay_service_writes(tmp_path: Path):
    session_id = "test_session"

    async def _run():
        service = FileReplayService(base_dir=tmp_path)

        await service.record_step(session_id, DummyStep(), DummyResult())
        await service.cleanup()

    asyncio.run(_run())

    files = list((tmp_path / "sessions").glob(f"replay-*{session_id}.jsonl"))
    assert len(files) == 1

    lines = files[0].read_text().splitlines()
    assert len(lines) == 2

    meta = json.loads(lines[0])
    assert meta["id"] == session_id

    entry = json.loads(lines[1])
    assert entry["step_id"] == "step1"
    assert entry["step_type"] == "typeA"
