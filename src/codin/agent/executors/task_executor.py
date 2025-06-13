from ..tool.base import Toolset


class TaskExecutor:
    """Unified executor for different plan types."""

    def __init__(self, tools: Toolset):
        self.tools = tools

    async def execute_dag_plan(self, plan: 'dag_types.Plan') -> 'PlanResult':
        from ..dag_planner import DAGExecutor
        dag_executor = DAGExecutor(tools=self.tools)
        return await dag_executor.execute_plan(plan)

    async def execute_workflow_plan(self, plan: 'WorkflowPlan') -> 'PlanResult':
        """Execute workflow-based plan (placeholder)."""
        return None
