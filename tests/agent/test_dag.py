"""Tests for DAG types.

This module contains tests for the data types used in the DAG-based planning and execution system.
"""

import json
import uuid
from datetime import datetime, timedelta

import pytest

from codin.agent.dag_types import Plan, PlanResult, Task, TaskStatus
from codin.id import new_id


class TestTask:
    """Test the Task class."""
    
    def test_init(self):
        """Test Task initialization."""
        task = Task(
            id="task-1",
            name="Test Task",
            description="A test task"
        )
        
        assert task.id == "task-1"
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.status == TaskStatus.PENDING
        assert task.requires == []
        assert task.result is None
        assert task.error_message is None
        assert task.created_at is not None
        assert task.started_at is None
        assert task.completed_at is None
    
    def test_auto_id(self):
        """Test automatic ID generation."""
        task = Task(
            id="",
            name="Auto ID Task",
            description="A task with auto-generated ID"
        )
        
        assert task.id != ""
        assert task.id.startswith("task-")  # Should use new_id with prefix
    
    def test_legacy_dependencies_handling(self):
        """Test handling of legacy dependencies field."""
        # Test with dependencies in the input dictionary
        task_dict = {
            "id": "task-1",
            "name": "Task 1",
            "description": "Task with dependencies",
            "dependencies": ["dep1", "dep2"]
        }
        
        task = Task.from_dict(task_dict)
        assert task.requires == ["dep1", "dep2"]
        
        # Test with both dependencies and requires
        task_dict = {
            "id": "task-2",
            "name": "Task 2",
            "description": "Task with both fields",
            "dependencies": ["dep1", "dep2"],
            "requires": ["dep3"]
        }
        
        task = Task.from_dict(task_dict)
        assert "dep1" in task.requires
        assert "dep2" in task.requires
        assert "dep3" in task.requires
    
    def test_error_sync(self):
        """Test error and error_message sync."""
        # Test with error
        task1 = Task(
            id="task-1",
            name="Task 1",
            description="Task with error",
            error="Test error"
        )
        assert task1.error_message == "Test error"
        
        # Test with error_message
        task2 = Task(
            id="task-2",
            name="Task 2",
            description="Task with error_message",
            error_message="Another error"
        )
        assert task2.error == "Another error"
    
    def test_hash(self):
        """Test Task hash implementation."""
        task1 = Task(
            id="task-1",
            name="Task 1",
            description="First task"
        )
        
        task2 = Task(
            id="task-1",  # Same ID
            name="Different Name",  # Different name
            description="Different description"  # Different description
        )
        
        task3 = Task(
            id="task-3",  # Different ID
            name="Task 1",  # Same name
            description="First task"  # Same description
        )
        
        # Same ID should give same hash
        assert hash(task1) == hash(task2)
        
        # Different ID should give different hash
        assert hash(task1) != hash(task3)
    
    def test_equality(self):
        """Test Task equality implementation."""
        task1 = Task(
            id="task-1",
            name="Task 1",
            description="First task"
        )
        
        task2 = Task(
            id="task-1",  # Same ID
            name="Different Name",  # Different name
            description="Different description"  # Different description
        )
        
        task3 = Task(
            id="task-3",  # Different ID
            name="Task 1",  # Same name
            description="First task"  # Same description
        )
        
        # Same ID should be equal
        assert task1 == task2
        
        # Different ID should not be equal
        assert task1 != task3
        
        # Different type should not be equal
        assert task1 != "task-1"
    
    def test_update_status(self):
        """Test updating task status."""
        task = Task(
            id="task-1",
            name="Test Task",
            description="A test task"
        )
        
        # Initial state
        assert task.status == TaskStatus.PENDING
        assert task.started_at is None
        assert task.completed_at is None
        
        # Update to in-progress
        task.update_status(TaskStatus.IN_PROGRESS)
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None
        assert task.completed_at is None
        
        # Update to completed with result
        result = {"key": "value"}
        task.update_status(TaskStatus.COMPLETED, result=result)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.completed_at is not None
        
        # Update with error
        task = Task(
            id="task-2",
            name="Error Task",
            description="A task with error"
        )
        
        task.update_status(TaskStatus.FAILED, error="Test error")
        assert task.status == TaskStatus.FAILED
        assert task.error == "Test error"
        assert task.error_message == "Test error"
    
    def test_add_dependency(self):
        """Test adding dependencies."""
        task = Task(
            id="task-1",
            name="Test Task",
            description="A test task"
        )
        
        task.add_dependency("dep1")
        assert "dep1" in task.requires
        
        # Adding same dependency again should have no effect
        task.add_dependency("dep1")
        assert len(task.requires) == 1
        
        # Add another dependency
        task.add_dependency("dep2")
        assert "dep2" in task.requires
        assert len(task.requires) == 2
    
    def test_ready(self):
        """Test ready() method."""
        # Create a test plan
        plan = Plan(
            id="test-plan",
            name="Test Plan",
            description="A test plan"
        )
        
        # Create tasks
        task1 = Task(
            id="task-1",
            name="Task 1",
            description="First task"
        )
        
        task2 = Task(
            id="task-2",
            name="Task 2",
            description="Second task",
            requires=["task-1"]
        )
        
        task3 = Task(
            id="task-3",
            name="Task 3",
            description="Third task",
            requires=["task-1", "task-2"]
        )
        
        # Add tasks to plan
        plan.add_task(task1)
        plan.add_task(task2)
        plan.add_task(task3)
        
        # Initial state
        assert task1.ready() is True  # No dependencies
        assert task2.ready() is False  # task-1 not completed
        assert task3.ready() is False  # Dependencies not completed
        
        # Complete task1
        task1.status = TaskStatus.COMPLETED
        
        # Check ready status again
        assert task2.ready() is True  # task-1 completed
        assert task3.ready() is False  # task-2 not completed
        
        # Complete task2
        task2.status = TaskStatus.COMPLETED
        
        # Check ready status again
        assert task3.ready() is True  # All dependencies completed
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        created_at = datetime.utcnow()
        started_at = created_at + timedelta(minutes=1)
        completed_at = started_at + timedelta(minutes=5)
        
        task = Task(
            id="task-1",
            name="Test Task",
            description="A test task",
            tool="test_tool",
            parameters={"param1": "value1"},
            requires=["dep1", "dep2"],
            status=TaskStatus.COMPLETED,
            result={"test": "result"},
            error_message="No error",
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            input_data={"input": "data"},
            output_data={"output": "data"},
            metadata={"meta": "data"}
        )
        
        task_dict = task.to_dict()
        
        # Check fields
        assert task_dict["id"] == "task-1"
        assert task_dict["name"] == "Test Task"
        assert task_dict["description"] == "A test task"
        assert task_dict["tool"] == "test_tool"
        assert task_dict["parameters"] == {"param1": "value1"}
        assert task_dict["requires"] == ["dep1", "dep2"]
        assert task_dict["status"] == "completed"
        assert task_dict["result"] == {"test": "result"}
        assert task_dict["error_message"] == "No error"
        assert task_dict["created_at"] == created_at.isoformat()
        assert task_dict["started_at"] == started_at.isoformat()
        assert task_dict["completed_at"] == completed_at.isoformat()
        assert task_dict["input_data"] == {"input": "data"}
        assert task_dict["output_data"] == {"output": "data"}
        assert task_dict["metadata"] == {"meta": "data"}
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        created_at = datetime.utcnow()
        started_at = created_at + timedelta(minutes=1)
        completed_at = started_at + timedelta(minutes=5)
        
        task_dict = {
            "id": "task-1",
            "name": "Test Task",
            "description": "A test task",
            "tool": "test_tool",
            "parameters": {"param1": "value1"},
            "requires": ["dep1", "dep2"],
            "status": "completed",
            "result": {"test": "result"},
            "error_message": "No error",
            "created_at": created_at.isoformat(),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "input_data": {"input": "data"},
            "output_data": {"output": "data"},
            "metadata": {"meta": "data"}
        }
        
        task = Task.from_dict(task_dict)
        
        # Check fields
        assert task.id == "task-1"
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.tool == "test_tool"
        assert task.parameters == {"param1": "value1"}
        assert task.requires == ["dep1", "dep2"]
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"test": "result"}
        assert task.error_message == "No error"
        assert task.created_at.isoformat() == created_at.isoformat()
        assert task.started_at.isoformat() == started_at.isoformat()
        assert task.completed_at.isoformat() == completed_at.isoformat()
        assert task.input_data == {"input": "data"}
        assert task.output_data == {"output": "data"}
        assert task.metadata == {"meta": "data"}


class TestPlan:
    """Test the Plan class."""
    
    def test_init(self):
        """Test Plan initialization."""
        plan = Plan(
            id="plan-1",
            name="Test Plan",
            description="A test plan"
        )
        
        assert plan.id == "plan-1"
        assert plan.name == "Test Plan"
        assert plan.description == "A test plan"
        assert plan.status == TaskStatus.PENDING
        assert plan.tasks == {}
        assert plan.created_at is not None
        assert plan.started_at is None
        assert plan.completed_at is None
        assert plan.artifacts == []
        assert plan.metadata == {}
    
    def test_auto_id(self):
        """Test automatic ID generation."""
        plan = Plan(
            name="Auto ID Plan",
            description="A plan with auto-generated ID"
        )
        
        assert plan.id != ""
        assert plan.id.startswith("plan-")  # Should use new_id with prefix
    
    def test_add_task(self):
        """Test adding tasks to a plan."""
        plan = Plan(
            id="plan-1",
            name="Test Plan",
            description="A test plan"
        )
        
        task = Task(
            id="task-1",
            name="Test Task",
            description="A test task"
        )
        
        plan.add_task(task)
        
        assert "task-1" in plan.tasks
        assert plan.tasks["task-1"] == task
        assert task._plan == plan
    
    def test_get_task(self):
        """Test getting a task by ID."""
        plan = Plan(
            id="plan-1",
            name="Test Plan",
            description="A test plan"
        )
        
        task1 = Task(
            id="task-1",
            name="Task 1",
            description="First task"
        )
        
        task2 = Task(
            id="task-2",
            name="Task 2",
            description="Second task"
        )
        
        plan.add_task(task1)
        plan.add_task(task2)
        
        assert plan.get_task("task-1") == task1
        assert plan.get_task("task-2") == task2
        assert plan.get_task("non-existent") is None
    
    def test_task_helpers(self):
        """Test task helper methods."""
        plan = Plan(
            id="plan-1",
            name="Test Plan",
            description="A test plan"
        )
        
        # Create tasks with different statuses
        task_pending = Task(
            id="pending",
            name="Pending Task",
            description="A pending task",
            status=TaskStatus.PENDING
        )
        
        task_running = Task(
            id="running",
            name="Running Task",
            description="A running task",
            status=TaskStatus.IN_PROGRESS
        )
        
        task_blocked = Task(
            id="blocked",
            name="Blocked Task",
            description="A blocked task",
            status=TaskStatus.BLOCKED
        )
        
        task_completed = Task(
            id="completed",
            name="Completed Task",
            description="A completed task",
            status=TaskStatus.COMPLETED
        )
        
        task_failed = Task(
            id="failed",
            name="Failed Task",
            description="A failed task",
            status=TaskStatus.FAILED
        )
        
        task_ready = Task(
            id="ready",
            name="Ready Task",
            description="A ready task",
            status=TaskStatus.PENDING
        )
        
        # Add tasks to plan
        plan.add_task(task_pending)
        plan.add_task(task_running)
        plan.add_task(task_blocked)
        plan.add_task(task_completed)
        plan.add_task(task_failed)
        plan.add_task(task_ready)
        
        # Test helper methods
        assert len(plan.get_ready_tasks()) >= 1  # Includes task_pending and task_ready
        assert len(plan.get_running_tasks()) == 1
        assert plan.get_running_tasks()[0] == task_running
        
        assert len(plan.get_failed_tasks()) == 1
        assert plan.get_failed_tasks()[0] == task_failed
        
        assert len(plan.get_successful_tasks()) == 1
        assert plan.get_successful_tasks()[0] == task_completed
        
        # Since both task_pending and task_ready are in PENDING status,
        # we expect to have 4 incomplete tasks: pending, ready, running, and blocked
        assert len(plan.get_incomplete_tasks()) == 4
    
    def test_is_done(self):
        """Test is_done method."""
        plan = Plan(
            id="plan-1",
            name="Test Plan",
            description="A test plan"
        )
        
        # Add tasks with non-terminal statuses
        plan.add_task(Task(
            id="pending",
            name="Pending Task",
            description="A pending task",
            status=TaskStatus.PENDING
        ))
        
        # Plan should not be done
        assert plan.is_done() is False
        
        # Clear and add only terminal status tasks
        plan.tasks = {}
        
        plan.add_task(Task(
            id="completed",
            name="Completed Task",
            description="A completed task",
            status=TaskStatus.COMPLETED
        ))
        
        plan.add_task(Task(
            id="failed",
            name="Failed Task",
            description="A failed task",
            status=TaskStatus.FAILED
        ))
        
        plan.add_task(Task(
            id="skipped",
            name="Skipped Task",
            description="A skipped task",
            status=TaskStatus.SKIPPED
        ))
        
        # Now plan should be done
        assert plan.is_done() is True
    
    def test_is_successful(self):
        """Test is_successful method."""
        plan = Plan(
            id="plan-1",
            name="Test Plan",
            description="A test plan"
        )
        
        # Add completed tasks
        plan.add_task(Task(
            id="completed1",
            name="Completed Task 1",
            description="First completed task",
            status=TaskStatus.COMPLETED
        ))
        
        plan.add_task(Task(
            id="completed2",
            name="Completed Task 2",
            description="Second completed task",
            status=TaskStatus.COMPLETED
        ))
        
        # Plan should be successful
        assert plan.is_successful() is True
        
        # Add a failed task
        plan.add_task(Task(
            id="failed",
            name="Failed Task",
            description="A failed task",
            status=TaskStatus.FAILED
        ))
        
        # Now plan should not be successful
        assert plan.is_successful() is False
    
    def test_reset(self):
        """Test reset method."""
        plan = Plan(
            id="plan-1",
            name="Test Plan",
            description="A test plan"
        )
        
        # Add tasks with various statuses
        task1 = Task(
            id="task1", 
            name="Task 1", 
            description="First task",
            status=TaskStatus.COMPLETED,
            result="Result 1",
            error=None,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            output_data={"output": "data1"}
        )
        
        task2 = Task(
            id="task2", 
            name="Task 2", 
            description="Second task",
            status=TaskStatus.FAILED,
            result=None,
            error="Error 2",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            output_data={"output": "data2"}
        )
        
        plan.add_task(task1)
        plan.add_task(task2)
        
        # Reset the plan
        plan.reset()
        
        # Check that tasks were reset
        for task in plan.tasks.values():
            assert task.status == TaskStatus.PENDING
            assert task.result is None
            assert task.error is None
            assert task.error_message is None
            assert task.started_at is None
            assert task.completed_at is None
            assert task.output_data == {}
    
    def test_update_status(self):
        """Test update_status method."""
        plan = Plan(
            id="plan-1",
            name="Test Plan",
            description="A test plan"
        )
        
        # Test with pending tasks
        plan.add_task(Task(
            id="pending1",
            name="Pending Task 1",
            description="First pending task",
            status=TaskStatus.PENDING
        ))
        
        plan.add_task(Task(
            id="pending2",
            name="Pending Task 2",
            description="Second pending task",
            status=TaskStatus.PENDING
        ))
        
        plan.update_status()
        assert plan.status == TaskStatus.PENDING
        
        # Test with one in-progress task
        plan.tasks["pending1"].status = TaskStatus.IN_PROGRESS
        plan.update_status()
        assert plan.status == TaskStatus.IN_PROGRESS
        assert plan.started_at is not None
        
        # Test with one completed and one pending
        plan.tasks["pending1"].status = TaskStatus.COMPLETED
        plan.update_status()
        assert plan.status == TaskStatus.IN_PROGRESS
        
        # Test with all completed
        plan.tasks["pending2"].status = TaskStatus.COMPLETED
        plan.update_status()
        assert plan.status == TaskStatus.COMPLETED
        assert plan.completed_at is not None
        
        # Test with a failed task
        plan = Plan(
            id="plan-2",
            name="Failed Plan",
            description="A plan with a failed task"
        )
        
        plan.add_task(Task(
            id="completed",
            name="Completed Task",
            description="A completed task",
            status=TaskStatus.COMPLETED
        ))
        
        plan.add_task(Task(
            id="failed",
            name="Failed Task",
            description="A failed task",
            status=TaskStatus.FAILED
        ))
        
        plan.update_status()
        assert plan.status == TaskStatus.FAILED
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        created_at = datetime.utcnow()
        started_at = created_at + timedelta(minutes=1)
        completed_at = started_at + timedelta(minutes=5)
        
        plan = Plan(
            id="plan-1",
            name="Test Plan",
            description="A test plan",
            status=TaskStatus.COMPLETED,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            metadata={"meta": "data"}
        )
        
        # Add a task
        task = Task(
            id="task-1",
            name="Test Task",
            description="A test task",
            status=TaskStatus.COMPLETED
        )
        
        plan.add_task(task)
        
        plan_dict = plan.to_dict()
        
        # Check fields
        assert plan_dict["id"] == "plan-1"
        assert plan_dict["name"] == "Test Plan"
        assert plan_dict["description"] == "A test plan"
        assert plan_dict["status"] == "completed"
        assert plan_dict["created_at"] == created_at.isoformat()
        assert plan_dict["started_at"] == started_at.isoformat()
        assert plan_dict["completed_at"] == completed_at.isoformat()
        assert plan_dict["metadata"] == {"meta": "data"}
        
        # Check task
        assert "task-1" in plan_dict["tasks"]
        assert plan_dict["tasks"]["task-1"]["name"] == "Test Task"
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        created_at = datetime.utcnow()
        started_at = created_at + timedelta(minutes=1)
        completed_at = started_at + timedelta(minutes=5)
        
        plan_dict = {
            "id": "plan-1",
            "name": "Test Plan",
            "description": "A test plan",
            "status": "completed",
            "created_at": created_at.isoformat(),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "metadata": {"meta": "data"},
            "tasks": {
                "task-1": {
                    "id": "task-1",
                    "name": "Test Task",
                    "description": "A test task",
                    "status": "completed",
                    "requires": [],
                    "created_at": created_at.isoformat()
                }
            }
        }
        
        plan = Plan.from_dict(plan_dict)
        
        # Check fields
        assert plan.id == "plan-1"
        assert plan.name == "Test Plan"
        assert plan.description == "A test plan"
        assert plan.status == TaskStatus.COMPLETED
        assert plan.created_at.isoformat() == created_at.isoformat()
        assert plan.started_at.isoformat() == started_at.isoformat()
        assert plan.completed_at.isoformat() == completed_at.isoformat()
        assert plan.metadata == {"meta": "data"}
        
        # Check tasks
        assert "task-1" in plan.tasks
        assert plan.tasks["task-1"].name == "Test Task"
        assert plan.tasks["task-1"].status == TaskStatus.COMPLETED
        
        # Test with legacy dependencies field
        plan_dict = {
            "id": "plan-2",
            "name": "Legacy Plan",
            "description": "A plan with legacy task format",
            "status": "pending",
            "tasks": {
                "task-1": {
                    "id": "task-1",
                    "name": "Legacy Task",
                    "description": "A task with dependencies field",
                    "status": "pending",
                    "dependencies": ["dep1", "dep2"],
                    "created_at": created_at.isoformat()
                }
            }
        }
        
        plan = Plan.from_dict(plan_dict)
        assert "task-1" in plan.tasks
        assert plan.tasks["task-1"].requires == ["dep1", "dep2"]


class TestPlanResult:
    """Test PlanResult class."""
    
    def test_init(self):
        """Test PlanResult initialization."""
        # Create a plan
        plan = Plan(
            id="plan-1",
            name="Test Plan",
            description="A test plan",
            status=TaskStatus.COMPLETED
        )
        
        # Create a successful result
        result1 = PlanResult(
            plan=plan,
            success=True,
            execution_time=10.5
        )
        
        assert result1.plan == plan
        assert result1.success is True
        assert result1.error_message is None
        assert result1.execution_time == 10.5
        
        # Create a failed result
        result2 = PlanResult(
            plan=plan,
            success=False,
            error_message="Test error",
            execution_time=5.2
        )
        
        assert result2.plan == plan
        assert result2.success is False
        assert result2.error_message == "Test error"
        assert result2.execution_time == 5.2 