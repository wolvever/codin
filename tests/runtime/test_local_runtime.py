from __future__ import annotations

import pytest

from src.codin.runtime.base import Workload, WorkloadType
from src.codin.runtime.local import LocalRuntime


@pytest.mark.asyncio
async def test_run_cli_streaming(tmp_path):
    script = tmp_path / "script.sh"
    script.write_text("printf 'one\\ntwo\\nthree\\n'")

    runtime = LocalRuntime()
    workload = Workload(kind=WorkloadType.CLI, command=f"bash {script}")

    result = await runtime.run(workload, stream=True)
    assert result.stream is not None

    lines = []
    async for chunk in result.stream:
        lines.append(chunk.strip())

    assert lines == ["one", "two", "three"]
    assert result.output == "one\ntwo\nthree\n"
    assert result.error == ""
    assert result.success
