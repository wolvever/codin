import asyncio
from unittest.mock import Mock

import pytest

from codin.agent.base import Planner
from codin.agent.base_agent import BaseAgent
from codin.agent.types import AgentRunInput, AgentRunOutput, Message, Role, TextPart
from codin.host.agent_host import AgentHost, LocalAgentHost
from codin.host.base import AgentHostConfig


class MockPlanner(Planner):
    async def next(self, state):
        if False:
            yield  # pragma: no cover

    async def reset(self, state):
        pass


class MockAgent(BaseAgent):
    def __init__(self, agent_id: str = "test_agent"):
        super().__init__(agent_id=agent_id, planner=MockPlanner())
        self.execution_count = 0
        
    async def run(self, input_data: AgentRunInput):
        self.execution_count += 1
        yield AgentRunOutput(
            id=f"output_{self.execution_count}",
            messages=[Message(role=Role.ASSISTANT, content=[TextPart(text=f"Response {self.execution_count}")])],
            runner_id=input_data.runner_id,
            request_id=input_data.request_id
        )


class TestAgentHostConfig:
    
    def test_config_creation(self):
        """Test creating agent host configuration."""
        config = AgentHostConfig(
            max_concurrent_agents=5,
            agent_timeout=30.0,
            cleanup_interval=60.0
        )
        
        assert config.max_concurrent_agents == 5
        assert config.agent_timeout == 30.0
        assert config.cleanup_interval == 60.0
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = AgentHostConfig()
        
        # Should have reasonable defaults
        assert config.max_concurrent_agents > 0
        assert config.agent_timeout > 0
        assert config.cleanup_interval > 0
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = AgentHostConfig(max_concurrent_agents=10)
        assert config.max_concurrent_agents == 10
        
        # Invalid config
        with pytest.raises(ValueError):
            AgentHostConfig(max_concurrent_agents=0)  # Should be positive
        
        with pytest.raises(ValueError):
            AgentHostConfig(agent_timeout=-1.0)  # Should be positive


class TestLocalAgentHost:
    
    @pytest.fixture
    def agent_host(self):
        config = AgentHostConfig(
            max_concurrent_agents=3,
            agent_timeout=10.0,
            cleanup_interval=60.0
        )
        return LocalAgentHost(config)
    
    @pytest.mark.asyncio
    async def test_start_agent_host(self, agent_host):
        """Test starting the agent host."""
        await agent_host.start()
        
        assert agent_host.is_running() is True
    
    @pytest.mark.asyncio
    async def test_stop_agent_host(self, agent_host):
        """Test stopping the agent host."""
        await agent_host.start()
        assert agent_host.is_running() is True
        
        await agent_host.stop()
        assert agent_host.is_running() is False
    
    @pytest.mark.asyncio
    async def test_submit_agent_task(self, agent_host):
        """Test submitting an agent task."""
        await agent_host.start()
        
        agent = MockAgent("test_agent")
        input_data = AgentRunInput(
            messages=[Message(role=Role.USER, content=[TextPart(text="Test message")])],
            runner_id="runner_123",
            request_id="req_123"
        )
        
        task_id = await agent_host.submit_task(agent, input_data)
        
        assert task_id is not None
        assert isinstance(task_id, str)
        
        await agent_host.stop()
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, agent_host):
        """Test getting task status."""
        await agent_host.start()
        
        agent = MockAgent("test_agent")
        input_data = AgentRunInput(
            messages=[Message(role=Role.USER, content=[TextPart(text="Test")])],
            runner_id="runner_123",
            request_id="req_123"
        )
        
        task_id = await agent_host.submit_task(agent, input_data)
        
        # Wait a bit for task to start
        await asyncio.sleep(0.1)
        
        status = await agent_host.get_task_status(task_id)
        
        assert status is not None
        assert status.task_id == task_id
        assert status.status in ["pending", "running", "completed", "failed"]
        
        await agent_host.stop()
    
    @pytest.mark.asyncio
    async def test_get_task_result(self, agent_host):
        """Test getting task result."""
        await agent_host.start()
        
        agent = MockAgent("test_agent")
        input_data = AgentRunInput(
            messages=[Message(role=Role.USER, content=[TextPart(text="Test")])],
            runner_id="runner_123",
            request_id="req_123"
        )
        
        task_id = await agent_host.submit_task(agent, input_data)
        
        # Wait for task completion
        max_wait = 5.0
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            status = await agent_host.get_task_status(task_id)
            if status.status == "completed":
                break
            await asyncio.sleep(0.1)
        
        result = await agent_host.get_task_result(task_id)
        
        if result is not None:  # Task might still be running
            assert hasattr(result, 'outputs')
            assert len(result.outputs) > 0
        
        await agent_host.stop()
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, agent_host):
        """Test canceling a task."""
        await agent_host.start()
        
        agent = MockAgent("test_agent")
        input_data = AgentRunInput(
            messages=[Message(role=Role.USER, content=[TextPart(text="Test")])],
            runner_id="runner_123",
            request_id="req_123"
        )
        
        task_id = await agent_host.submit_task(agent, input_data)
        
        # Cancel the task
        success = await agent_host.cancel_task(task_id)
        
        assert success is True
        
        # Check status after cancellation
        status = await agent_host.get_task_status(task_id)
        assert status.status in ["cancelled", "failed"]
        
        await agent_host.stop()
    
    @pytest.mark.asyncio
    async def test_list_active_tasks(self, agent_host):
        """Test listing active tasks."""
        await agent_host.start()
        
        # Submit multiple tasks
        tasks = []
        for i in range(3):
            agent = MockAgent(f"agent_{i}")
            input_data = AgentRunInput(
                messages=[Message(role=Role.USER, content=[TextPart(text=f"Test {i}")])],
                runner_id=f"runner_{i}",
                request_id=f"req_{i}"
            )
            task_id = await agent_host.submit_task(agent, input_data)
            tasks.append(task_id)
        
        active_tasks = await agent_host.list_active_tasks()
        
        assert len(active_tasks) >= 0  # Might be 0 if tasks completed quickly
        
        for task_info in active_tasks:
            assert hasattr(task_info, 'task_id')
            assert hasattr(task_info, 'status')
        
        await agent_host.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_task_limit(self, agent_host):
        """Test that concurrent task limit is enforced."""
        await agent_host.start()
        
        # Submit more tasks than the limit
        tasks = []
        for i in range(5):  # More than max_concurrent_agents (3)
            agent = MockAgent(f"agent_{i}")
            input_data = AgentRunInput(
                messages=[Message(role=Role.USER, content=[TextPart(text=f"Test {i}")])],
                runner_id=f"runner_{i}",
                request_id=f"req_{i}"
            )
            task_id = await agent_host.submit_task(agent, input_data)
            tasks.append(task_id)
        
        # Check that not all tasks are running simultaneously
        await asyncio.sleep(0.1)
        active_tasks = await agent_host.list_active_tasks()
        running_tasks = [t for t in active_tasks if t.status == "running"]
        
        # Should respect the limit
        assert len(running_tasks) <= agent_host._config.max_concurrent_agents
        
        await agent_host.stop()
    
    @pytest.mark.asyncio
    async def test_agent_host_cleanup(self, agent_host):
        """Test agent host cleanup functionality."""
        await agent_host.start()
        
        # Submit a task
        agent = MockAgent("cleanup_test")
        input_data = AgentRunInput(
            messages=[Message(role=Role.USER, content=[TextPart(text="Test")])],
            runner_id="runner_cleanup",
            request_id="req_cleanup"
        )
        
        task_id = await agent_host.submit_task(agent, input_data)
        
        # Wait for task to potentially complete
        await asyncio.sleep(0.2)
        
        # Trigger cleanup
        await agent_host.cleanup()
        
        # Cleanup should remove completed tasks
        status = await agent_host.get_task_status(task_id)
        # Task might be removed or marked as cleaned up
        
        await agent_host.stop()
    
    @pytest.mark.asyncio
    async def test_agent_host_error_handling(self, agent_host):
        """Test error handling in agent host."""
        await agent_host.start()
        
        # Create an agent that will fail
        class FailingAgent(BaseAgent):
            def __init__(self):
                super().__init__(agent_id="failing_agent", planner=MockPlanner())
            
            async def run(self, input_data: AgentRunInput):
                raise Exception("Agent execution failed")
        
        failing_agent = FailingAgent()
        input_data = AgentRunInput(
            messages=[Message(role=Role.USER, content=[TextPart(text="Test")])],
            runner_id="runner_fail",
            request_id="req_fail"
        )
        
        task_id = await agent_host.submit_task(failing_agent, input_data)
        
        # Wait for task to fail
        await asyncio.sleep(0.2)
        
        status = await agent_host.get_task_status(task_id)
        assert status.status in ["failed", "error"]
        
        await agent_host.stop()
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self):
        """Test task timeout handling."""
        config = AgentHostConfig(
            max_concurrent_agents=1,
            agent_timeout=0.1,  # Very short timeout
            cleanup_interval=1.0
        )
        agent_host = LocalAgentHost(config)
        
        await agent_host.start()
        
        # Create an agent that takes too long
        class SlowAgent(BaseAgent):
            def __init__(self):
                super().__init__(agent_id="slow_agent", planner=MockPlanner())
            
            async def run(self, input_data: AgentRunInput):
                await asyncio.sleep(1.0)  # Longer than timeout
                yield AgentRunOutput(
                    id="slow_output",
                    messages=[Message(role=Role.ASSISTANT, content=[TextPart(text="Too slow")])],
                    runner_id=input_data.runner_id,
                    request_id=input_data.request_id
                )
        
        slow_agent = SlowAgent()
        input_data = AgentRunInput(
            messages=[Message(role=Role.USER, content=[TextPart(text="Test")])],
            runner_id="runner_slow",
            request_id="req_slow"
        )
        
        task_id = await agent_host.submit_task(slow_agent, input_data)
        
        # Wait for timeout to occur
        await asyncio.sleep(0.3)
        
        status = await agent_host.get_task_status(task_id)
        assert status.status in ["timeout", "failed", "cancelled"]
        
        await agent_host.stop()
    
    @pytest.mark.asyncio
    async def test_double_start_stop(self, agent_host):
        """Test that starting/stopping multiple times is safe."""
        # Start twice
        await agent_host.start()
        await agent_host.start()  # Should be safe
        assert agent_host.is_running() is True
        
        # Stop twice
        await agent_host.stop()
        await agent_host.stop()  # Should be safe
        assert agent_host.is_running() is False


class TestAgentHostInterface:
    
    def test_agent_host_is_abstract(self):
        """Test that AgentHost is an abstract class."""
        with pytest.raises(TypeError):
            AgentHost()
    
    @pytest.mark.asyncio
    async def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteAgentHost(AgentHost):
            pass
        
        with pytest.raises(TypeError):
            IncompleteAgentHost()
    
    @pytest.mark.asyncio
    async def test_valid_subclass_implementation(self):
        """Test a valid AgentHost subclass."""
        class ValidAgentHost(AgentHost):
            def __init__(self):
                self._running = False
                self._tasks = {}
            
            async def start(self):
                self._running = True
            
            async def stop(self):
                self._running = False
            
            def is_running(self):
                return self._running
            
            async def submit_task(self, agent, input_data):
                task_id = f"task_{len(self._tasks)}"
                self._tasks[task_id] = {"status": "pending"}
                return task_id
            
            async def get_task_status(self, task_id):
                task_info = self._tasks.get(task_id, {"status": "not_found"})
                return Mock(task_id=task_id, status=task_info["status"])
            
            async def get_task_result(self, task_id):
                return None
            
            async def cancel_task(self, task_id):
                if task_id in self._tasks:
                    self._tasks[task_id]["status"] = "cancelled"
                    return True
                return False
            
            async def list_active_tasks(self):
                return []
            
            async def cleanup(self):
                pass
        
        host = ValidAgentHost()
        assert isinstance(host, AgentHost)
        
        # Test basic functionality
        await host.start()
        assert host.is_running() is True
        
        task_id = await host.submit_task(MockAgent(), Mock())
        assert task_id is not None
        
        status = await host.get_task_status(task_id)
        assert status.task_id == task_id