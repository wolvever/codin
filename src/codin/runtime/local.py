"""Local runtime implementation for codin agents.

This module provides a runtime implementation that executes workloads
locally using subprocess for development and testing purposes.
"""

from __future__ import annotations

import asyncio
import shlex
import types
import typing as _t

from codin.runtime.base import Runtime, RuntimeResult, Workload, WorkloadType


__all__ = [
    'LocalRuntime',
]


class LocalRuntime(Runtime):
    """Execute workloads directly on the host Python interpreter / shell.

    * Security note*: This runtime should only be used in trusted environments.
    """

    name = 'local'

    async def _run(
        self,
        workload: Workload,
        /,
        *,
        stream: bool = False,
    ) -> RuntimeResult:
        if workload.kind in (WorkloadType.FUNCTION, WorkloadType.CLASS):
            return await self._run_callable(workload)
        if workload.kind is WorkloadType.CLI:
            return await self._run_cli(workload, stream=stream)
        raise NotImplementedError(f'LocalRuntime does not support kind={workload.kind}')

    async def _run_callable(self, workload: Workload) -> RuntimeResult:
        if workload.callable is None:
            raise ValueError('callable must be provided for FUNCTION/CLASS workload')

        func = workload.callable
        if isinstance(func, types.FunctionType):
            result = func(*(workload.args or ()), **(workload.kwargs or {}))
        else:
            # Assume *callable* is class.
            # Instantiate then call ``__call__`` if defined.
            instance = func(*(workload.args or ()), **(workload.kwargs or {}))
            if not callable(instance):
                return RuntimeResult(success=True, output=instance)
            result = instance()
        return RuntimeResult(success=True, output=result)

    async def _run_cli(
        self,
        workload: Workload,
        /,
        *,
        stream: bool = False,
    ) -> RuntimeResult:
        if workload.command is None:
            raise ValueError('command must be provided for CLI workload')

        cmd_list = shlex.split(workload.command)
        proc = await asyncio.create_subprocess_exec(
            *cmd_list,
            cwd=workload.working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        if stream:
            assert proc.stdout is not None

            result = RuntimeResult(success=False, output='', error=None)

            async def _iter() -> _t.AsyncIterator[str]:
                output_parts = []
                async for line in proc.stdout:
                    text = line.decode()
                    output_parts.append(text)
                    yield text
                err = b''
                if proc.stderr is not None:
                    err = await proc.stderr.read()
                await proc.wait()
                result.output = ''.join(output_parts)
                result.error = err.decode()
                result.success = proc.returncode == 0

            result.stream = _iter()
            return result
        stdout, stderr = await proc.communicate()
        return RuntimeResult(
            success=proc.returncode == 0,
            output=stdout.decode(),
            error=stderr.decode(),
        )
