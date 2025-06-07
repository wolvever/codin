"""File-based replay service for persisting execution logs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import typing as _t

from .base import ReplayService


@dataclass
class _Writer:
    queue: asyncio.Queue[str]
    task: asyncio.Task
    file: _t.TextIO


class FileReplayService(ReplayService):
    """Replay service that persists logs to JSONL files on disk."""

    def __init__(self, base_dir: Path | None = None) -> None:
        super().__init__()
        self._base_dir = Path(base_dir or Path.home() / ".codin") / "sessions"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, _Writer] = {}

    async def record_step(self, session_id: str, step: _t.Any, result: _t.Any) -> None:
        await super().record_step(session_id, step, result)
        entry = self._replay_logs[session_id][-1]
        writer = await self._ensure_writer(session_id)
        await writer.queue.put(json.dumps(entry))

    async def cleanup(self) -> None:
        """Flush and close all session writers."""
        for writer in self._writers.values():
            await writer.queue.put(None)  # type: ignore[arg-type]
            await writer.task
        self._writers.clear()

    async def _ensure_writer(self, session_id: str) -> _Writer:
        if session_id in self._writers:
            return self._writers[session_id]

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"replay-{timestamp}-{session_id}.jsonl"
        path = self._base_dir / filename
        file = open(path, "a", encoding="utf-8")
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=256)

        async def writer_loop() -> None:
            while True:
                line = await queue.get()
                if line is None:
                    break
                await asyncio.to_thread(file.write, line + "\n")
                await asyncio.to_thread(file.flush)
            file.close()

        task = asyncio.create_task(writer_loop())
        writer = _Writer(queue=queue, task=task, file=file)
        self._writers[session_id] = writer

        meta = {
            "id": session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        await queue.put(json.dumps(meta))
        return writer


__all__ = ["FileReplayService"]
