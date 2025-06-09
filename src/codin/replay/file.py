"""File-based replay service for persisting execution logs."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .base import BaseReplay


@dataclass
class _Writer:
    queue: asyncio.Queue[str | None]
    task: asyncio.Task
    file: Any  # Actually TextIO, but Any for simplicity with open modes


class FileReplay(BaseReplay):
    """Replay service that persists logs to JSONL files on disk for a single session."""

    def __init__(self, session_id: str, base_dir: Path | None = None) -> None:
        self.session_id = session_id
        self._base_dir = Path(base_dir or Path.home() / ".codin") / "sessions"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._writer: _Writer | None = None

    async def record_message_exchange(
        self, client_message: Any, agent_message: Any, **kwargs: Any
    ) -> None:
        """Records a client message and the corresponding agent message."""
        writer = await self._ensure_writer()
        exchange_type = kwargs.pop('exchange_type', 'message_exchange')
        entry: dict[str, Any] = {
            "type": exchange_type,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "client_message": self._serialize_message(client_message),
            "agent_message": self._serialize_message(agent_message),
        }
        if kwargs: # Add any remaining kwargs to the entry
            entry.update(kwargs)
        await writer.queue.put(json.dumps(entry))

    async def cleanup(self) -> None:
        """Flush and close the session writer."""
        if self._writer:
            await self._writer.queue.put(None)
            await self._writer.task
            self._writer = None

    async def _ensure_writer(self) -> _Writer:
        if self._writer is not None:
            return self._writer

        # Corrected filename to include session_id and current timestamp for uniqueness
        filename = f"replay-{self.session_id}-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.jsonl"
        path = self._base_dir / filename

        # Using Any for file type hint due to open returning _io.TextIOWrapper
        file_handle: Any = open(path, "a", encoding="utf-8")
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)

        async def writer_loop() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done() # Signal that None has been processed
                    break
                await asyncio.to_thread(file_handle.write, item + "\n")
                await asyncio.to_thread(file_handle.flush)
                queue.task_done()
            file_handle.close()

        task = asyncio.create_task(writer_loop())
        self._writer = _Writer(queue=queue, task=task, file=file_handle)

        meta = {
            "type": "session_start",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
        }
        await self._writer.queue.put(json.dumps(meta))
        return self._writer


__all__ = ["FileReplay"]
