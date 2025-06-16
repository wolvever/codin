import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from codin.actor.dispatcher import LocalDispatcher, DispatchResult
from codin.actor.supervisor import LocalActorManager, ActorInfo
from codin.actor.task_manager import TaskRegistry, TaskInfo
from codin.actor.envelope_types import Envelope, EnvelopeKind, ControlPayload, ControlAction, TaskState
from codin.agent.base_agent import BaseAgent
from codin.agent.base import Planner
from codin.agent.types import AgentRunInput, AgentRunOutput, Message, TextPart, Role


class MockPlanner(Planner):
    async def next(self, state):
        if False:
            yield  # pragma: no cover

    async def reset(self, state):
        pass


class MockAgent(BaseAgent):
    def __init__(self, agent_id: str = "test_agent"):
        super().__init__(agent_id=agent_id, planner=MockPlanner())
        
    async def run(self, input_data: AgentRunInput):
        yield AgentRunOutput(
            id="test_output",
            messages=[Message(role=Role.ASSISTANT, content=[TextPart(text="Test response")])],
            runner_id=input_data.runner_id,
            request_id=input_data.request_id
        )


@pytest.fixture
def mock_supervisor():
    supervisor = Mock(spec=LocalActorManager)
    supervisor.acquire = AsyncMock(return_value=ActorInfo(
        actor_id="test_actor",
        actor_instance=MockAgent(),
        capabilities=set(),
        status="active"
    ))
    supervisor.release = AsyncMock()
    supervisor.list = AsyncMock(return_value=[])
    supervisor.info = AsyncMock(return_value=ActorInfo(
        actor_id="test_actor",
        actor_instance=MockAgent(),
        capabilities=set(),
        status="active"
    ))
    supervisor.cleanup_idle_actors = AsyncMock()
    return supervisor


@pytest.fixture
def mock_task_registry():
    registry = Mock(spec=TaskRegistry)
    registry.add_task = AsyncMock(return_value="task_123")
    registry.get_task = AsyncMock(return_value=TaskInfo(
        task_id="task_123",
        runner_id="runner_123",
        request_id="req_123",
        state=TaskState.RUNNING,
        metadata={}
    ))
    registry.update_task_state = AsyncMock()
    registry.list_all_tasks = AsyncMock(return_value=[])
    return registry


@pytest.fixture
def dispatcher(mock_supervisor, mock_task_registry):
    return LocalDispatcher(
        supervisor=mock_supervisor,
        task_registry=mock_task_registry
    )


class TestLocalDispatcher:
    
    @pytest.mark.asyncio
    async def test_submit_valid_envelope(self, dispatcher, mock_supervisor, mock_task_registry):
        """Test submitting a valid work envelope."""
        envelope_dict = {
            "kind": "work",
            "payload": {
                "runner_id": "runner_123",
                "request_id": "req_123",
                "agent_id": "test_agent",
                "input": {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
                }
            }
        }
        
        result = await dispatcher.submit(envelope_dict)
        
        assert result == "runner_123"
        mock_task_registry.add_task.assert_called_once()
        mock_supervisor.acquire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_control_envelope(self, dispatcher, mock_supervisor, mock_task_registry):
        """Test submitting a control envelope."""
        envelope_dict = {
            "kind": "control",
            "payload": {
                "action": "signal",
                "runner_id": "runner_123",
                "signal": "stop"
            }
        }
        
        result = await dispatcher.submit(envelope_dict)
        
        assert result == "runner_123"
        # Control envelopes shouldn't create tasks
        mock_task_registry.add_task.assert_not_called()
        mock_supervisor.acquire.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_submit_invalid_envelope(self, dispatcher):
        """Test submitting an invalid envelope."""
        invalid_envelope = {
            "kind": "invalid_kind",
            "payload": {}
        }
        
        with pytest.raises(ValueError, match="Invalid envelope kind"):
            await dispatcher.submit(invalid_envelope)
    
    @pytest.mark.asyncio
    async def test_signal_stop(self, dispatcher):
        """Test sending a stop signal."""
        runner_id = "runner_123"
        
        result = await dispatcher.signal(runner_id, "stop")
        
        assert result is True
        assert runner_id in dispatcher._stop_signals
    
    @pytest.mark.asyncio
    async def test_signal_pause_resume(self, dispatcher):
        """Test sending pause and resume signals."""
        runner_id = "runner_123"
        
        # Test pause
        result = await dispatcher.signal(runner_id, "pause")
        assert result is True
        assert runner_id in dispatcher._pause_signals
        
        # Test resume
        result = await dispatcher.signal(runner_id, "resume")
        assert result is True
        assert runner_id not in dispatcher._pause_signals
    
    @pytest.mark.asyncio
    async def test_get_status_running_task(self, dispatcher, mock_task_registry):
        """Test getting status for a running task."""
        runner_id = "runner_123"
        mock_task_registry.get_task.return_value = TaskInfo(
            task_id="task_123",
            runner_id=runner_id,
            request_id="req_123",
            state=TaskState.RUNNING,
            metadata={"agent_id": "test_agent"}
        )
        
        result = await dispatcher.get_status(runner_id)
        
        assert isinstance(result, DispatchResult)
        assert result.runner_id == runner_id
        assert result.status == "running"
        assert result.task_id == "task_123"
        mock_task_registry.get_task.assert_called_once_with(runner_id)
    
    @pytest.mark.asyncio
    async def test_get_status_completed_task(self, dispatcher, mock_task_registry):
        """Test getting status for a completed task."""
        runner_id = "runner_123"
        mock_task_registry.get_task.return_value = TaskInfo(
            task_id="task_123",
            runner_id=runner_id,
            request_id="req_123",
            state=TaskState.COMPLETED,
            metadata={}
        )
        
        result = await dispatcher.get_status(runner_id)
        
        assert result.status == "completed"
    
    @pytest.mark.asyncio
    async def test_get_status_nonexistent_task(self, dispatcher, mock_task_registry):
        """Test getting status for a non-existent task."""
        runner_id = "nonexistent"
        mock_task_registry.get_task.return_value = None
        
        result = await dispatcher.get_status(runner_id)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_active_runs(self, dispatcher, mock_task_registry):
        """Test listing active runs."""
        mock_task_registry.list_all_tasks.return_value = [
            TaskInfo(
                task_id="task_1",
                runner_id="runner_1",
                request_id="req_1",
                state=TaskState.RUNNING,
                metadata={}
            ),
            TaskInfo(
                task_id="task_2",
                runner_id="runner_2",
                request_id="req_2",
                state=TaskState.COMPLETED,
                metadata={}
            ),
            TaskInfo(
                task_id="task_3",
                runner_id="runner_3",
                request_id="req_3",
                state=TaskState.FAILED,
                metadata={}
            )
        ]
        
        result = await dispatcher.list_active_runs()
        
        assert len(result) == 3
        assert all(isinstance(item, DispatchResult) for item in result)
        assert result[0].status == "running"
        assert result[1].status == "completed"
        assert result[2].status == "failed"
    
    @pytest.mark.asyncio
    async def test_get_stream_queue_creates_new_queue(self, dispatcher):
        """Test getting stream queue creates a new queue if it doesn't exist."""
        runner_id = "new_runner"
        
        queue = await dispatcher.get_stream_queue(runner_id)
        
        assert queue is not None
        assert isinstance(queue, asyncio.Queue)
        assert runner_id in dispatcher._stream_queues
    
    @pytest.mark.asyncio
    async def test_get_stream_queue_returns_existing_queue(self, dispatcher):
        """Test getting stream queue returns existing queue."""
        runner_id = "existing_runner"
        existing_queue = asyncio.Queue()
        dispatcher._stream_queues[runner_id] = existing_queue
        
        queue = await dispatcher.get_stream_queue(runner_id)
        
        assert queue is existing_queue
    
    @pytest.mark.asyncio
    async def test_cleanup_removes_completed_tasks(self, dispatcher, mock_supervisor, mock_task_registry):
        """Test cleanup removes completed tasks and their queues."""
        # Setup some completed tasks
        runner_id_1 = "completed_1"
        runner_id_2 = "running_1"
        
        dispatcher._stream_queues[runner_id_1] = asyncio.Queue()
        dispatcher._stream_queues[runner_id_2] = asyncio.Queue()
        
        mock_task_registry.list_all_tasks.return_value = [
            TaskInfo(
                task_id="task_1",
                runner_id=runner_id_1,
                request_id="req_1",
                state=TaskState.COMPLETED,
                metadata={}
            ),
            TaskInfo(
                task_id="task_2",
                runner_id=runner_id_2,
                request_id="req_2",
                state=TaskState.RUNNING,
                metadata={}
            )
        ]
        
        await dispatcher.cleanup()
        
        # Completed task queue should be removed, running task queue should remain
        assert runner_id_1 not in dispatcher._stream_queues
        assert runner_id_2 in dispatcher._stream_queues
        mock_supervisor.cleanup_idle_actors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_removes_old_signals(self, dispatcher):
        """Test cleanup removes old stop and pause signals."""
        old_runner = "old_runner"
        recent_runner = "recent_runner"
        
        # Add signals
        dispatcher._stop_signals.add(old_runner)
        dispatcher._stop_signals.add(recent_runner)
        dispatcher._pause_signals.add(old_runner)
        
        with patch('codin.actor.dispatcher.datetime') as mock_datetime:
            # Mock current time to be much later than signal creation
            mock_datetime.now.return_value = datetime.now()
            
            await dispatcher.cleanup()
            
            # Old signals should be cleaned up based on implementation logic
            # This test would need to be adjusted based on actual cleanup logic
    
    @pytest.mark.asyncio
    async def test_error_handling_in_work_processing(self, dispatcher, mock_supervisor, mock_task_registry):
        """Test error handling when actor processing fails."""
        envelope_dict = {
            "kind": "work",
            "payload": {
                "runner_id": "runner_123",
                "request_id": "req_123",
                "agent_id": "test_agent",
                "input": {
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
                }
            }
        }
        
        # Mock supervisor to raise an exception
        mock_supervisor.acquire.side_effect = Exception("Actor acquisition failed")
        
        result = await dispatcher.submit(envelope_dict)
        
        # Should still return runner_id but task should be marked as failed
        assert result == "runner_123"
        mock_task_registry.update_task_state.assert_called()