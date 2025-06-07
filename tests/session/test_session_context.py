import importlib
import sys
import types

import pytest


@pytest.mark.asyncio
async def test_session_context_manager():
    sys.modules['src.codin.agent.types'] = types.SimpleNamespace(State=object)
    SessionManager = importlib.import_module('src.codin.session.base').SessionManager
    mgr = SessionManager()
    async with mgr.session('s1') as session:
        assert session.session_id == 's1'
        assert mgr.get_session('s1') is not None
    # After context exit the session should be removed
    assert mgr.get_session('s1') is None

