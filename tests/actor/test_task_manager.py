import asyncio

import pytest

from codin.actor.envelope_types import TaskState
from codin.actor.task_manager import TaskInfo, TaskRegistry


class TestTaskInfo:
    
    def test_task_info_creation(self):
        """Test creating TaskInfo instance."""
        info = TaskInfo(
            task_id="task_123",
            runner_id="runner_123",
            request_id="req_123",
            state=TaskState.RUNNING,
            metadata={"agent_id": "test_agent", "priority": "high"}
        )
        
        assert info.task_id == "task_123"
        assert info.runner_id == "runner_123"
        assert info.request_id == "req_123"
        assert info.state == TaskState.RUNNING
        assert info.metadata["agent_id"] == "test_agent"
        assert info.metadata["priority"] == "high"
    
    def test_task_info_default_metadata(self):
        """Test TaskInfo with default metadata."""
        info = TaskInfo(
            task_id="task_123",
            runner_id="runner_123", 
            request_id="req_123",
            state=TaskState.PENDING,
            metadata={}
        )
        
        assert len(info.metadata) == 0
        assert info.state == TaskState.PENDING


class TestTaskRegistry:
    
    @pytest.fixture
    def registry(self):
        return TaskRegistry()
    
    @pytest.mark.asyncio
    async def test_add_task(self, registry):
        """Test adding a new task."""
        runner_id = "runner_123"
        request_id = "req_123"
        metadata = {"agent_id": "test_agent"}
        
        task_id = await registry.add_task(runner_id, request_id, metadata)
        
        assert task_id is not None
        assert isinstance(task_id, str)
        
        # Verify task was added
        task_info = await registry.get_task(runner_id)
        assert task_info.task_id == task_id
        assert task_info.runner_id == runner_id
        assert task_info.request_id == request_id
        assert task_info.state == TaskState.PENDING
        assert task_info.metadata == metadata
    
    @pytest.mark.asyncio
    async def test_add_task_generates_unique_ids(self, registry):
        """Test that adding tasks generates unique IDs."""
        task_id1 = await registry.add_task("runner1", "req1", {})
        task_id2 = await registry.add_task("runner2", "req2", {})
        
        assert task_id1 != task_id2
    
    @pytest.mark.asyncio
    async def test_get_task_existing(self, registry):
        """Test getting an existing task."""
        runner_id = "runner_123"
        metadata = {"test": "data"}
        
        task_id = await registry.add_task(runner_id, "req_123", metadata)
        
        task_info = await registry.get_task(runner_id)
        
        assert task_info.task_id == task_id
        assert task_info.runner_id == runner_id
        assert task_info.metadata == metadata
    
    @pytest.mark.asyncio
    async def test_get_task_nonexistent(self, registry):
        """Test getting a non-existent task."""
        task_info = await registry.get_task("nonexistent_runner")
        
        assert task_info is None
    
    @pytest.mark.asyncio
    async def test_update_task_state(self, registry):
        """Test updating task state."""
        runner_id = "runner_123"
        await registry.add_task(runner_id, "req_123", {})
        
        # Update to running
        await registry.update_task_state(runner_id, TaskState.RUNNING)
        
        task_info = await registry.get_task(runner_id)
        assert task_info.state == TaskState.RUNNING
        
        # Update to completed
        await registry.update_task_state(runner_id, TaskState.COMPLETED)
        
        task_info = await registry.get_task(runner_id)
        assert task_info.state == TaskState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_update_task_state_with_metadata(self, registry):
        """Test updating task state with additional metadata."""
        runner_id = "runner_123"
        initial_metadata = {"agent_id": "test_agent"}
        
        await registry.add_task(runner_id, "req_123", initial_metadata)
        
        additional_metadata = {"completion_time": "2023-01-01T12:00:00Z"}
        await registry.update_task_state(
            runner_id, 
            TaskState.COMPLETED, 
            additional_metadata
        )
        
        task_info = await registry.get_task(runner_id)
        assert task_info.state == TaskState.COMPLETED
        assert task_info.metadata["agent_id"] == "test_agent"  # Original metadata preserved
        assert task_info.metadata["completion_time"] == "2023-01-01T12:00:00Z"  # New metadata added
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_task_state(self, registry):
        """Test updating state of non-existent task."""
        # Should not raise an error
        await registry.update_task_state("nonexistent", TaskState.COMPLETED)
    
    @pytest.mark.asyncio
    async def test_remove_task(self, registry):
        """Test removing a task."""
        runner_id = "runner_123"
        await registry.add_task(runner_id, "req_123", {})
        
        # Verify task exists
        assert await registry.get_task(runner_id) is not None
        
        # Remove task
        await registry.remove_task(runner_id)
        
        # Verify task is removed
        assert await registry.get_task(runner_id) is None
    
    @pytest.mark.asyncio
    async def test_remove_nonexistent_task(self, registry):
        """Test removing a non-existent task."""
        # Should not raise an error
        await registry.remove_task("nonexistent")
    
    @pytest.mark.asyncio
    async def test_list_all_tasks_empty(self, registry):
        """Test listing tasks when registry is empty."""
        tasks = await registry.list_all_tasks()
        
        assert len(tasks) == 0
        assert isinstance(tasks, list)
    
    @pytest.mark.asyncio
    async def test_list_all_tasks_with_data(self, registry):
        """Test listing all tasks with data."""
        # Add multiple tasks
        await registry.add_task("runner1", "req1", {"agent": "agent1"})
        await registry.add_task("runner2", "req2", {"agent": "agent2"})
        await registry.add_task("runner3", "req3", {"agent": "agent3"})
        
        # Update some states
        await registry.update_task_state("runner1", TaskState.RUNNING)
        await registry.update_task_state("runner2", TaskState.COMPLETED)
        
        tasks = await registry.list_all_tasks()
        
        assert len(tasks) == 3
        
        # Verify all tasks are returned
        runner_ids = {task.runner_id for task in tasks}
        assert runner_ids == {"runner1", "runner2", "runner3"}
        
        # Verify states
        task_states = {task.runner_id: task.state for task in tasks}
        assert task_states["runner1"] == TaskState.RUNNING
        assert task_states["runner2"] == TaskState.COMPLETED
        assert task_states["runner3"] == TaskState.PENDING
    
    @pytest.mark.asyncio
    async def test_list_tasks_by_state(self, registry):
        """Test listing tasks filtered by state."""
        # Add tasks with different states
        await registry.add_task("runner1", "req1", {})
        await registry.add_task("runner2", "req2", {})
        await registry.add_task("runner3", "req3", {})
        
        await registry.update_task_state("runner1", TaskState.RUNNING)
        await registry.update_task_state("runner2", TaskState.COMPLETED)
        # runner3 stays PENDING
        
        # Test filtering (if supported by implementation)
        all_tasks = await registry.list_all_tasks()
        running_tasks = [t for t in all_tasks if t.state == TaskState.RUNNING]
        completed_tasks = [t for t in all_tasks if t.state == TaskState.COMPLETED]
        pending_tasks = [t for t in all_tasks if t.state == TaskState.PENDING]
        
        assert len(running_tasks) == 1
        assert len(completed_tasks) == 1
        assert len(pending_tasks) == 1
        assert running_tasks[0].runner_id == "runner1"
        assert completed_tasks[0].runner_id == "runner2"
        assert pending_tasks[0].runner_id == "runner3"
    
    @pytest.mark.asyncio
    async def test_task_lifecycle(self, registry):
        """Test complete task lifecycle."""
        runner_id = "lifecycle_runner"
        request_id = "lifecycle_req"
        metadata = {"agent_id": "lifecycle_agent"}
        
        # Add task
        await registry.add_task(runner_id, request_id, metadata)
        
        # Verify initial state
        task_info = await registry.get_task(runner_id)
        assert task_info.state == TaskState.PENDING
        assert task_info.metadata["agent_id"] == "lifecycle_agent"
        
        # Update to running
        await registry.update_task_state(runner_id, TaskState.RUNNING, {"start_time": "now"})
        task_info = await registry.get_task(runner_id)
        assert task_info.state == TaskState.RUNNING
        assert "start_time" in task_info.metadata
        
        # Update to completed
        await registry.update_task_state(runner_id, TaskState.COMPLETED, {"end_time": "later"})
        task_info = await registry.get_task(runner_id)
        assert task_info.state == TaskState.COMPLETED
        assert "end_time" in task_info.metadata
        assert task_info.metadata["agent_id"] == "lifecycle_agent"  # Original metadata preserved
        
        # Verify in list
        all_tasks = await registry.list_all_tasks()
        lifecycle_task = next(t for t in all_tasks if t.runner_id == runner_id)
        assert lifecycle_task.state == TaskState.COMPLETED
        
        # Remove task
        await registry.remove_task(runner_id)
        assert await registry.get_task(runner_id) is None
    
    @pytest.mark.asyncio
    async def test_concurrent_task_operations(self, registry):
        """Test concurrent task operations."""
        runner_ids = [f"runner_{i}" for i in range(10)]
        
        # Concurrently add tasks
        add_tasks = [
            registry.add_task(runner_id, f"req_{i}", {"index": i})
            for i, runner_id in enumerate(runner_ids)
        ]
        task_ids = await asyncio.gather(*add_tasks)
        
        assert len(task_ids) == 10
        assert len(set(task_ids)) == 10  # All unique
        
        # Concurrently update states
        update_tasks = [
            registry.update_task_state(runner_id, TaskState.RUNNING)
            for runner_id in runner_ids[:5]
        ]
        await asyncio.gather(*update_tasks)
        
        # Verify updates
        all_tasks = await registry.list_all_tasks()
        running_count = sum(1 for t in all_tasks if t.state == TaskState.RUNNING)
        pending_count = sum(1 for t in all_tasks if t.state == TaskState.PENDING)
        
        assert running_count == 5
        assert pending_count == 5
    
    @pytest.mark.asyncio
    async def test_task_metadata_edge_cases(self, registry):
        """Test task metadata edge cases."""
        runner_id = "metadata_test"
        
        # Test with empty metadata
        await registry.add_task(runner_id, "req", {})
        task_info = await registry.get_task(runner_id)
        assert task_info.metadata == {}
        
        # Test with None values in metadata
        await registry.update_task_state(runner_id, TaskState.RUNNING, {"none_value": None})
        task_info = await registry.get_task(runner_id)
        assert task_info.metadata["none_value"] is None
        
        # Test with complex metadata
        complex_metadata = {
            "nested": {"deep": {"value": 42}},
            "list": [1, 2, 3],
            "boolean": True,
            "string": "test"
        }
        await registry.update_task_state(runner_id, TaskState.COMPLETED, complex_metadata)
        task_info = await registry.get_task(runner_id)
        assert task_info.metadata["nested"]["deep"]["value"] == 42
        assert task_info.metadata["list"] == [1, 2, 3]
        assert task_info.metadata["boolean"] is True
    
    @pytest.mark.asyncio
    async def test_task_state_transitions(self, registry):
        """Test valid and invalid task state transitions."""
        runner_id = "state_test"
        await registry.add_task(runner_id, "req", {})
        
        # Valid transitions
        await registry.update_task_state(runner_id, TaskState.RUNNING)
        await registry.update_task_state(runner_id, TaskState.COMPLETED)
        
        # Transition from completed (might be valid for retry scenarios)
        await registry.update_task_state(runner_id, TaskState.PENDING)
        task_info = await registry.get_task(runner_id)
        assert task_info.state == TaskState.PENDING
        
        # Test failed state
        await registry.update_task_state(runner_id, TaskState.FAILED)
        task_info = await registry.get_task(runner_id)
        assert task_info.state == TaskState.FAILED