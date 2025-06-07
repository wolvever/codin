from __future__ import annotations

import asyncio
import json
import typing as _t

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..actor import LocalActorManager, LocalDispatcher, Dispatcher


class SubmitRequest(BaseModel):
    a2a_request: dict[str, _t.Any]


class SubmitResponse(BaseModel):
    runner_id: str


class SignalRequest(BaseModel):
    ctrl: str


async def _sse_stream(queue: asyncio.Queue) -> _t.AsyncIterator[str]:
    while True:
        item = await queue.get()
        if item is None:
            break
        data = json.dumps(item)
        yield f"data: {data}\n\n"


def create_app(dispatcher: Dispatcher | None = None) -> FastAPI:
    dispatcher = dispatcher or LocalDispatcher(LocalActorManager())
    app = FastAPI()

    @app.post("/v1/submit", response_model=SubmitResponse)
    async def submit(req: SubmitRequest) -> SubmitResponse:
        runner_id = await dispatcher.submit(req.a2a_request)
        return SubmitResponse(runner_id=runner_id)

    @app.post("/v1/signal/{agent_id}")
    async def signal(agent_id: str, req: SignalRequest) -> dict[str, str]:
        await dispatcher.signal(agent_id, req.ctrl)
        return {"status": "ok"}

    @app.get("/v1/status/{runner_id}")
    async def status(runner_id: str):
        result = await dispatcher.get_status(runner_id)
        if result is None:
            raise HTTPException(status_code=404, detail="runner not found")
        return result

    @app.get("/v1/stream/{runner_id}")
    async def stream(runner_id: str):
        queue = dispatcher.get_stream_queue(runner_id)
        if queue is None:
            raise HTTPException(status_code=404, detail="runner not found")
        return StreamingResponse(_sse_stream(queue), media_type="text/event-stream")

    return app


# Convenience application for `uvicorn codin.api.app:app`
dispatcher = LocalDispatcher(LocalActorManager())
app = create_app(dispatcher)
