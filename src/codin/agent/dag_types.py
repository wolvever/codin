"""DAG-based task planning types."""

import typing as _t
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

from ..id import new_id

__all__ = [
    "TaskStatus",
    "Task", 
    "Plan",
    "PlanResult",
]


class TaskStatus(str, Enum):
    """Status enum for DAG tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    """
    A task in a DAG plan.
    
    Each task has:
    - An ID (unique within a plan)
    - A name and description
    - Optional tool to use for execution
    - Optional parameters for the tool
    - Dependencies on other tasks (stored in requires)
    - Status tracking
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str
    name: str
    description: str
    tool: str = ""
    parameters: _t.Dict[str, _t.Any] = Field(default_factory=dict)
    requires: _t.List[str] = Field(default_factory=list)  # Task dependencies
    status: TaskStatus = TaskStatus.PENDING
    result: _t.Any = None
    error_message: str | None = None
    error: _t.Optional[str] = None  # For backwards compatibility
    plan_ref: _t.Optional['Plan'] = Field(default=None, repr=False, exclude=True)
    depends_on_results: bool = False  # Whether this task needs results from dependencies
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_data: dict[str, _t.Any] = Field(default_factory=dict)
    output_data: dict[str, _t.Any] = Field(default_factory=dict)
    artifacts: list[dict] = Field(default_factory=list)  # TODO: Use Artifact type
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        """Initialize task with proper defaults."""
        if not data.get('id'):
            data['id'] = new_id(prefix="task")
        
        super().__init__(**data)
        
        # Sync error fields
        if self.error and not self.error_message:
            self.error_message = self.error
        elif self.error_message and not self.error:
            self.error = self.error_message
            
    def __hash__(self) -> int:
        """
        Make Task hashable by using its ID.
        
        Returns:
            Hash of the task ID
        """
        return hash(self.id)
        
    def __eq__(self, other: object) -> bool:
        """
        Compare tasks by ID.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if the objects are equal
        """
        if not isinstance(other, Task):
            return False
        return self.id == other.id

    def update_status(self, status: TaskStatus, result: _t.Any = None, error: _t.Optional[str] = None) -> None:
        """
        Update the status of this task.
        
        Args:
            status: New status
            result: Optional result to set
            error: Optional error message
        """
        self.status = status
        
        if result is not None:
            self.result = result
            
        if error is not None:
            self.error = error
            self.error_message = error
        
        # Update timestamps based on status
        now = datetime.utcnow()
        if status == TaskStatus.IN_PROGRESS and not self.started_at:
            self.started_at = now
        elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED) and not self.completed_at:
            self.completed_at = now
            
    def add_dependency(self, task_id: str) -> None:
        """
        Add a dependency to this task.
        
        Args:
            task_id: ID of the task this task depends on
        """
        if task_id not in self.requires:
            self.requires.append(task_id)
    
    def ready(self) -> bool:
        """
        Check if this task is ready to run.
        
        A task is ready if:
        1. It's in PENDING status
        2. All its dependencies have completed successfully
        
        Returns:
            True if the task is ready to run
        """
        if self.status != TaskStatus.PENDING:
            return False
            
        # Check dependencies
        if not self.plan_ref:
            return len(self.requires) == 0
            
        for dep_id in self.requires:
            dep_task = self.plan_ref.get_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
                
        return True
    
    def to_dict(self) -> _t.Dict[str, _t.Any]:
        """
        Convert the task to a dictionary for serialization.
        
        Returns:
            Task as a dictionary
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tool": self.tool,
            "parameters": self.parameters,
            "requires": self.requires,
            "status": self.status.value,
            "result": self.result if not hasattr(self.result, 'to_dict') else self.result.to_dict(),
            "error_message": self.error_message,
            "error": self.error,
            "depends_on_results": self.depends_on_results,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, _t.Any]) -> "Task":
        """Create from dictionary after deserialization."""
        task_data = data.copy()
        
        # Handle status
        if "status" in task_data:
            task_data["status"] = TaskStatus(task_data["status"])
            
        # Handle timestamps
        for ts_field in ["created_at", "started_at", "completed_at"]:
            if task_data.get(ts_field):
                task_data[ts_field] = datetime.fromisoformat(task_data[ts_field])
        
        # Handle legacy 'dependencies' field
        if "dependencies" in task_data:
            # Merge dependencies into requires
            dependencies = task_data.pop("dependencies", [])
            if "requires" not in task_data:
                task_data["requires"] = []
            for dep in dependencies:
                if dep not in task_data["requires"]:
                    task_data["requires"].append(dep)
        
        # Convert artifacts if they exist
        if "artifacts" in task_data and isinstance(task_data["artifacts"], list):
            # This would need artifact conversion logic if needed
            pass
            
        return cls(**task_data)


class Plan(BaseModel):
    """
    A DAG plan consisting of multiple tasks with dependencies.
    
    The plan stores tasks and manages their relationships and execution state.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: str = Field(default_factory=lambda: new_id(prefix="plan"))
    name: str = "DAG Plan"
    description: str = "A directed acyclic graph task plan"
    tasks: dict[str, Task] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    artifacts: list[dict] = Field(default_factory=list)
    metadata: dict[str, _t.Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        """Initialize plan with proper defaults and back-references."""
        super().__init__(**data)
        # Set back reference to plan in each task
        for _, task in self.tasks.items():
            task.plan_ref = self
    
    def add_task(self, task: Task) -> None:
        """
        Add a task to the plan.
        
        Args:
            task: The task to add
        """
        task.plan_ref = self
        self.tasks[task.id] = task
    
    def get_task(self, task_id: str) -> _t.Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task to get
            
        Returns:
            The task if found, None otherwise
        """
        return self.tasks.get(task_id)
    
    def get_ready_tasks(self) -> list[Task]:
        """
        Get all tasks that are ready to run.
        
        Returns:
            List of ready tasks
        """
        return [task for task in self.tasks.values() if task.ready()]
    
    def get_running_tasks(self) -> list[Task]:
        """
        Get all currently running tasks.
        
        Returns:
            List of running tasks
        """
        return [task for task in self.tasks.values() if task.status == TaskStatus.IN_PROGRESS]
    
    def get_failed_tasks(self) -> list[Task]:
        """
        Get all failed tasks.
        
        Returns:
            List of failed tasks
        """
        return [task for task in self.tasks.values() if task.status == TaskStatus.FAILED]
    
    def get_successful_tasks(self) -> list[Task]:
        """
        Get all successfully completed tasks.
        
        Returns:
            List of successful tasks
        """
        return [task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED]
    
    def get_incomplete_tasks(self) -> list[Task]:
        """
        Get all incomplete tasks (not success, failure, or canceled).
        
        Returns:
            List of incomplete tasks
        """
        incomplete_statuses = {TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED}
        return [task for task in self.tasks.values() if task.status in incomplete_statuses]
    
    def is_done(self) -> bool:
        """
        Check if the plan is complete (all tasks finished).
        
        Returns:
            True if all tasks are in a terminal state
        """
        terminal_statuses = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
        return all(task.status in terminal_statuses for task in self.tasks.values())
    
    def is_successful(self) -> bool:
        """
        Check if the plan completed successfully (all tasks succeeded).
        
        Returns:
            True if all tasks succeeded
        """
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks.values())
    
    def reset(self) -> None:
        """Reset all tasks to PENDING status."""
        for task in self.tasks.values():
            task.status = TaskStatus.PENDING
            task.result = None
            task.error = None
            task.error_message = None
            task.started_at = None
            task.completed_at = None
            task.output_data = {}
    
    def update_status(self) -> None:
        """Update the overall plan status based on task statuses."""
        # If any task failed, the plan failed
        if any(task.status == TaskStatus.FAILED for task in self.tasks.values()):
            self.status = TaskStatus.FAILED
            if not self.completed_at:
                self.completed_at = datetime.utcnow()
            return
            
        # If all tasks completed or skipped, the plan is complete
        if all(task.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) 
               for task in self.tasks.values()):
            self.status = TaskStatus.COMPLETED
            if not self.completed_at:
                self.completed_at = datetime.utcnow()
            return
            
        # If any task is in progress, the plan is in progress
        if any(task.status == TaskStatus.IN_PROGRESS for task in self.tasks.values()):
            self.status = TaskStatus.IN_PROGRESS
            if not self.started_at:
                self.started_at = datetime.utcnow()
            return
        
        # If any task is neither pending nor completed, the plan is in progress
        if any(task.status not in (TaskStatus.PENDING, TaskStatus.COMPLETED, TaskStatus.SKIPPED) 
               for task in self.tasks.values()):
            self.status = TaskStatus.IN_PROGRESS
            if not self.started_at:
                self.started_at = datetime.utcnow()
            return
            
        # If we have some completed tasks but not all, the plan is in progress
        if any(task.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for task in self.tasks.values()) and \
           any(task.status == TaskStatus.PENDING for task in self.tasks.values()):
            self.status = TaskStatus.IN_PROGRESS
            if not self.started_at:
                self.started_at = datetime.utcnow()
            return
            
        # Otherwise, the plan is still pending
        self.status = TaskStatus.PENDING
    
    def to_dict(self) -> _t.Dict[str, _t.Any]:
        """
        Convert the plan to a dictionary for serialization.
        
        Returns:
            Plan as a dictionary
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, _t.Any]) -> "Plan":
        """Create from dictionary after deserialization."""
        plan_data = data.copy()
        
        # Handle status
        if "status" in plan_data:
            plan_data["status"] = TaskStatus(plan_data["status"])
            
        # Handle timestamps
        for ts_field in ["created_at", "started_at", "completed_at"]:
            if plan_data.get(ts_field):
                plan_data[ts_field] = datetime.fromisoformat(plan_data[ts_field])
            
        # Process tasks
        tasks_dict = {}
        raw_tasks = plan_data.pop("tasks", {})
        for task_id, task_data in raw_tasks.items():
            tasks_dict[task_id] = Task.from_dict(task_data)
            
        plan_data["tasks"] = tasks_dict
        return cls(**plan_data)


class PlanResult(BaseModel):
    """Result of plan execution."""
    
    plan: Plan
    success: bool
    error_message: str | None = None
    execution_time: float | None = None 