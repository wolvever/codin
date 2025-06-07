"""DAG-based planning and execution system.

This module provides a directed acyclic graph (DAG) planner that breaks down complex coding
tasks into smaller, interdependent subtasks, and an executor that runs these tasks in the
correct order.
"""

# Annotations are enabled by default in Python 3.13

import asyncio
import json
import logging
import time
from datetime import datetime

from a2a.types import (
    Artifact,
    DataPart,
    Message,
    TaskState,
    TextPart,
)
from a2a.types import (
    Task as A2ATask,
)
from a2a.types import (
    TaskStatus as A2ATaskStatus,
)

from ..id import new_id
from ..model.base import BaseLLM
from ..prompt import prompt_run
from ..tool.base import Tool
from .base import Agent, AgentRunInput, AgentRunOutput
from .dag_types import Plan, PlanResult, Task, TaskStatus

logger = logging.getLogger(__name__)


class DAGPlanner(Agent):
    """Agent that creates directed acyclic graph (DAG) plans for coding tasks.

    The DAG planner breaks down complex tasks into simpler, interdependent subtasks
    organized in a DAG that can be executed by a DAGExecutor.
    """

    def __init__(
        self,
        *,
        llm: BaseLLM,
        id: str | None = None,
        name: str = 'DAG Planner',
        description: str = 'Creates directed acyclic graph plans for coding tasks',
        version: str = '1.0.0',
        tools: list[Tool] | None = None,
        max_retries: int = 3,
    ) -> None:
        """Initialize the DAG planner.

        Args:
            llm: Language model to use for planning
            id: Unique identifier for the agent
            name: Name of the agent
            description: Description of the agent
            version: Version string
            tools: List of tools the agent can use
            max_retries: Maximum number of retries for LLM calls
        """
        super().__init__(
            id=id,
            name=name,
            description=description,
            version=version,
            tools=tools,
        )
        self.llm = llm
        self.max_retries = max_retries

    async def run(self, inp: AgentRunInput) -> AgentRunOutput:
        """Execute planning logic and return structured output."""
        try:
            # Extract the user query
            query = inp.message.parts[0].text

            # Generate a plan based on the query
            plan_id = new_id(prefix='plan')
            plan = await self._create_plan(plan_id, query)

            # Return the plan as output
            response_text = f'Created plan: {plan.name}\n\nDescription: {plan.description}\n\nTasks:\n'

            for i, task in enumerate(plan.tasks.values(), 1):
                dependencies = ', '.join(task.requires) if task.requires else 'None'
                response_text += f'{i}. {task.name}: {task.description} (Dependencies: {dependencies})\n'

            # Create response message
            response_message = Message(role='assistant', parts=[TextPart(text=response_text)])

            # Create A2A task
            a2a_task = A2ATask(
                id=inp.id or new_id(prefix='task'),
                status=A2ATaskStatus(state=TaskState.COMPLETED),
                message=response_message,
            )

            # Serialize the plan to JSON and add as an artifact
            artifact = Artifact(
                id=new_id(prefix='artifact'),
                name='plan.json',
                description='DAG plan for the coding task',
                parts=[DataPart(type='data', data={'plan': plan.to_dict()})],
            )

            return AgentRunOutput(id=inp.id, result=a2a_task, artifacts=[artifact], metadata={'plan_id': plan.id})

        except Exception as e:
            logger.exception('Error in DAG planner')

            # Create error response
            error_message = Message(role='assistant', parts=[TextPart(text=f'Error creating plan: {e!s}')])

            # Create A2A task with error
            a2a_task = A2ATask(
                id=inp.id or new_id(prefix='task'),
                status=A2ATaskStatus(state=TaskState.FAILED),
                message=error_message,
            )

            return AgentRunOutput(id=inp.id, result=a2a_task, artifacts=None, metadata={'error': str(e)})

    async def _create_plan(self, plan_id: str, query: str) -> Plan:
        """Create a plan by calling the LLM with the query.

        Args:
            plan_id: Unique ID for the plan
            query: User query or task description

        Returns:
            A Plan object with tasks organized in a DAG
        """
        # Call the LLM to generate a plan using template
        for attempt in range(self.max_retries):
            try:
                response = await prompt_run('dag_planner', variables={'query': query})

                # Extract content from response
                content = ''
                if hasattr(response, 'message') and response.message:
                    # Extract text from A2A message parts
                    for part in response.message.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            content += part.root.text
                        elif hasattr(part, 'text'):
                            content += part.text
                elif hasattr(response, 'content'):
                    content = str(response.content)
                else:
                    content = str(response)

                # Parse the response as JSON
                try:
                    # Extract JSON from the response (handling potential markdown code blocks)
                    json_str = content
                    if '```json' in json_str:
                        json_str = json_str.split('```json')[1].split('```')[0].strip()
                    elif '```' in json_str:
                        json_str = json_str.split('```')[1].split('```')[0].strip()

                    plan_data = json.loads(json_str)

                    # Create the Plan object
                    plan = Plan(
                        id=plan_id,
                        name=plan_data['name'],
                        description=plan_data['description'],
                        status=TaskStatus.PENDING,
                        created_at=datetime.utcnow(),
                    )

                    # Create Task objects
                    for task_data in plan_data['tasks']:
                        # Handle both requires and dependencies fields for backward compatibility
                        requires = task_data.get('requires', [])
                        if 'dependencies' in task_data and not requires:
                            requires = task_data['dependencies']

                        task = Task(
                            id=task_data['id'],
                            name=task_data['name'],
                            description=task_data['description'],
                            requires=requires,
                            depends_on_results=True,  # Default to requiring results
                            status=TaskStatus.PENDING,
                            created_at=datetime.utcnow(),
                        )
                        plan.add_task(task)

                    return plan

                except json.JSONDecodeError as e:
                    logger.error(f'Failed to parse JSON response: {e}, attempt {attempt + 1}/{self.max_retries}')
                    if attempt == self.max_retries - 1:
                        raise ValueError(f'Failed to parse planner output as JSON: {e}') from e

            except Exception as e:
                logger.error(f'Error during plan creation: {e}, attempt {attempt + 1}/{self.max_retries}')
                if attempt == self.max_retries - 1:
                    raise

        # This should never be reached due to the exception in the loop
        raise RuntimeError('Failed to create plan after maximum retries')


class DAGExecutor(Agent):
    """Agent that executes DAG plans created by the DAGPlanner.

    The DAG executor runs tasks in the correct order, based on their dependencies,
    and tracks the overall progress of the plan.
    """

    def __init__(
        self,
        *,
        llm: BaseLLM,
        id: str | None = None,
        name: str = 'DAG Executor',
        description: str = 'Executes directed acyclic graph plans for coding tasks',
        version: str = '1.0.0',
        tools: list[Tool] | None = None,
        max_retries: int = 3,
        max_concurrent_tasks: int = 5,
    ) -> None:
        """Initialize the DAG executor.

        Args:
            llm: Language model to use for task execution
            id: Unique identifier for the agent
            name: Name of the agent
            description: Description of the agent
            version: Version string
            tools: List of tools the agent can use
            max_retries: Maximum number of retries for LLM calls
            max_concurrent_tasks: Maximum number of tasks to run concurrently
        """
        super().__init__(
            id=id,
            name=name,
            description=description,
            version=version,
            tools=tools,
        )
        self.llm = llm
        self.max_retries = max_retries
        self.max_concurrent_tasks = max_concurrent_tasks

    async def run(self, inp: AgentRunInput) -> AgentRunOutput:
        """Execute the DAG plan and return structured output."""
        try:
            # Extract the plan data from input
            try:
                message_parts = inp.message.parts
                plan_data = None

                # First look for DataPart with plan data
                for part in message_parts:
                    if (
                        part.type == 'data'
                        and hasattr(part, 'data')
                        and isinstance(part.data, dict)
                        and 'plan' in part.data
                    ):
                        plan_data = part.data['plan']
                        break

                # If not found, check if the message text is JSON
                if not plan_data:
                    text_part = next((p for p in message_parts if p.type == 'text'), None)
                    if text_part and hasattr(text_part, 'text'):
                        try:
                            json_data = json.loads(text_part.text)
                            if 'plan' in json_data:
                                plan_data = json_data['plan']
                            else:
                                plan_data = json_data
                        except json.JSONDecodeError as err:
                            # Not JSON, check if there's plan_id in metadata
                            if inp.metadata and 'plan_id' in inp.metadata:
                                raise ValueError(
                                    f"Plan with ID {inp.metadata['plan_id']} not found in input"
                                ) from err
                            raise ValueError('No plan data found in input') from err

                if not plan_data:
                    raise ValueError('No plan data found in input')

                # Create Plan object
                plan = Plan.from_dict(plan_data)

            except (KeyError, json.JSONDecodeError, ValueError) as e:
                logger.error(f'Failed to parse plan data: {e}')
                raise ValueError(f'Invalid plan data: {e!s}') from e

            # Execute the plan
            result = await self._execute_plan(plan)

            # Create response message based on execution result
            if result.success:
                response_text = f'Successfully executed plan: {result.plan.name}\n\nAll tasks completed successfully.'
            else:
                response_text = (
                    f'Failed to execute plan: {result.plan.name}\n\n'
                    f'Error: {result.error_message}\n\n'
                    f'Completed tasks: {len(result.plan.get_successful_tasks())}/{len(result.plan.tasks)}'
                )

            # Create response message
            response_message = Message(role='assistant', parts=[TextPart(text=response_text)])

            # Create A2A task
            a2a_task = A2ATask(
                id=inp.id or new_id(prefix='task'),
                status=A2ATaskStatus(state=TaskState.COMPLETED if result.success else TaskState.FAILED),
                message=response_message,
            )

            # Serialize the plan to JSON and add as an artifact
            artifact = Artifact(
                id=new_id(prefix='artifact'),
                name='executed_plan.json',
                description='Executed DAG plan for the coding task',
                parts=[DataPart(type='data', data={'plan': result.plan.to_dict()})],
            )

            # Combine our artifact with any artifacts created during plan execution
            artifacts = [artifact]
            if result.plan.artifacts:
                artifacts.extend(result.plan.artifacts)

            return AgentRunOutput(
                id=inp.id,
                result=a2a_task,
                artifacts=artifacts,
                metadata={
                    'plan_id': result.plan.id,
                    'success': result.success,
                    'execution_time': result.execution_time,
                },
            )

        except Exception as e:
            logger.exception('Error in DAG executor')

            # Create error response
            error_message = Message(role='assistant', parts=[TextPart(text=f'Error executing plan: {e!s}')])

            # Create A2A task with error
            a2a_task = A2ATask(
                id=inp.id or new_id(prefix='task'),
                status=A2ATaskStatus(state=TaskState.FAILED),
                message=error_message,
            )

            return AgentRunOutput(id=inp.id, result=a2a_task, artifacts=None, metadata={'error': str(e)})

    async def _execute_plan(self, plan: Plan) -> PlanResult:
        """Execute a DAG plan.

        Args:
            plan: The Plan object to execute

        Returns:
            A PlanResult object with execution results
        """
        start_time = time.time()

        try:
            # Set plan status to in progress
            plan.status = TaskStatus.IN_PROGRESS
            plan.started_at = datetime.utcnow()

            # Create a semaphore to limit concurrent tasks
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

            # Keep track of running tasks
            running_tasks: dict[str, asyncio.Task] = {}

            # Continue until all tasks are in a terminal state
            while not plan.is_done():
                # Update plan status
                plan.update_status()

                # Get ready tasks
                ready_tasks = plan.get_ready_tasks()

                # Start new tasks
                for task in ready_tasks:
                    if task.id not in running_tasks:
                        task_coroutine = self._execute_task(semaphore, plan, task)
                        running_tasks[task.id] = asyncio.create_task(task_coroutine)

                # Clean up completed tasks
                done_task_ids = []
                for task_id, task_obj in running_tasks.items():
                    if task_obj.done():
                        try:
                            await task_obj  # Retrieve any exceptions
                        except Exception as e:
                            logger.error(f'Task {task_id} failed: {e}')
                            # The task should have updated its own status

                        done_task_ids.append(task_id)

                for task_id in done_task_ids:
                    running_tasks.pop(task_id)

                # If no ready tasks and no running tasks, but plan is not done,
                # we might have a cycle or blocked tasks
                if not ready_tasks and not running_tasks and not plan.is_done():
                    blocked_tasks = [t for t in plan.tasks.values() if t.status == TaskStatus.BLOCKED]
                    if blocked_tasks:
                        blocked_ids = [t.id for t in blocked_tasks]
                        error_message = (
                            f'Plan execution blocked. The following tasks are blocked: {", ".join(blocked_ids)}'
                        )
                        logger.error(error_message)

                        # Mark blocked tasks as failed
                        for task in blocked_tasks:
                            task.status = TaskStatus.FAILED
                            task.error_message = 'Task blocked due to dependency cycle or failed dependencies'

                        # Update plan status
                        plan.update_status()
                        break

                # Short sleep to avoid busy waiting
                await asyncio.sleep(0.1)

            # Check if all tasks succeeded
            success = plan.is_successful()
            error_message = None
            if not success:
                failed_tasks = plan.get_failed_tasks()
                failed_ids = [t.id for t in failed_tasks]
                error_message = f'Plan execution failed. The following tasks failed: {", ".join(failed_ids)}'
                logger.error(error_message)

            # Mark plan as completed
            plan.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            plan.completed_at = datetime.utcnow()

            # Return result
            execution_time = time.time() - start_time
            return PlanResult(
                plan=plan,
                success=success,
                error_message=error_message,
                execution_time=execution_time,
            )

        except Exception as e:
            # If an exception occurs during execution, mark the plan as failed
            execution_time = time.time() - start_time
            logger.exception('Error executing plan')

            plan.status = TaskStatus.FAILED
            plan.completed_at = datetime.utcnow()

            return PlanResult(
                plan=plan,
                success=False,
                error_message=str(e),
                execution_time=execution_time,
            )

    async def _execute_task(self, semaphore: asyncio.Semaphore, plan: Plan, task: Task) -> None:
        """Execute a single task.

        Args:
            semaphore: Semaphore to limit concurrent tasks
            plan: The Plan object containing the task
            task: The Task object to execute
        """
        async with semaphore:
            try:
                # Mark task as in progress
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.utcnow()

                # For each task, call the LLM with:
                # 1. The task description
                # 2. The context (plan description, dependencies, etc.)
                # 3. The available tools

                # Build context from dependencies
                dependency_context = ''
                for dep_id in task.requires:
                    dep_task = plan.get_task(dep_id)
                    if dep_task:
                        dependency_context += f'Dependency: {dep_task.name}\n'
                        dependency_context += f'Description: {dep_task.description}\n'
                        if dep_task.output_data:
                            dep_output = json.dumps(dep_task.output_data, indent=2)
                            dependency_context += f'Output: {dep_output}\n'
                        dependency_context += '\n'

                # Build the system prompt
                system_prompt = f"""
                You are executing a task as part of a DAG-based coding plan.
                
                Plan: {plan.name}
                Plan Description: {plan.description}
                
                Current Task: {task.name}
                Task Description: {task.description}
                
                {dependency_context}
                
                Execute this task and provide the following:
                1. A summary of what you did
                2. The result or output of the task
                3. Any observations or issues encountered
                
                Your response should be in JSON format:
                {{
                  "summary": "Summary of task execution",
                  "result": "Result or output of the task",
                  "observations": "Any observations or issues",
                  "status": "completed"  // Or "failed" if unsuccessful
                }}
                """

                # Execute the task using the LLM
                for attempt in range(self.max_retries):
                    try:
                        # If we have tools, use the generate_with_tools method
                        if self.tools:
                            tool_definitions = [
                                {
                                    'name': tool.name,
                                    'description': tool.description,
                                    'parameters': tool.to_openai_schema().get('function', {}).get('parameters', {}),
                                }
                                for tool in self.tools
                            ]

                            response = await self.llm.generate_with_tools(
                                [
                                    {'role': 'system', 'content': system_prompt},
                                    {'role': 'user', 'content': f'Execute task: {task.name}'},
                                ],
                                tools=tool_definitions,
                                temperature=0.2,
                            )

                            # Process the response (tool calls and/or content)
                            if isinstance(response, dict):
                                # Process tool calls if any
                                if response.get('tool_calls'):
                                    # Implement tool call handling
                                    tool_call_results = []
                                    for tool_call in response['tool_calls']:
                                        tool_name = tool_call.get('name')
                                        tool_args = tool_call.get('arguments', {})

                                        # Find the tool
                                        tool = next((t for t in self.tools if t.name == tool_name), None)
                                        if tool:
                                            try:
                                                # Call the tool
                                                result = await tool.run(tool_args, None)
                                                tool_call_results.append(
                                                    {
                                                        'tool': tool_name,
                                                        'args': tool_args,
                                                        'result': result,
                                                        'success': True,
                                                    }
                                                )
                                            except Exception as e:
                                                tool_call_results.append(
                                                    {
                                                        'tool': tool_name,
                                                        'args': tool_args,
                                                        'error': str(e),
                                                        'success': False,
                                                    }
                                                )

                                    # Add tool call results to task output
                                    task.output_data['tool_calls'] = tool_call_results

                                # Process content
                                if 'content' in response:
                                    content = response['content']
                                    # Try to parse as JSON
                                    try:
                                        if '```json' in content:
                                            content = content.split('```json')[1].split('```')[0].strip()
                                        elif '```' in content:
                                            content = content.split('```')[1].split('```')[0].strip()

                                        result_data = json.loads(content)
                                        task.output_data.update(result_data)

                                        # Update task status based on result
                                        if result_data.get('status') == 'failed':
                                            task.status = TaskStatus.FAILED
                                            task.error_message = result_data.get('observations', 'Task failed')
                                        else:
                                            task.status = TaskStatus.COMPLETED

                                    except (json.JSONDecodeError, ValueError):
                                        # Not JSON, just use the content as is
                                        task.output_data['content'] = content
                                        task.status = TaskStatus.COMPLETED

                            else:
                                # Response is a string, try to parse as JSON
                                try:
                                    if '```json' in response:
                                        response = response.split('```json')[1].split('```')[0].strip()
                                    elif '```' in response:
                                        response = response.split('```')[1].split('```')[0].strip()

                                    result_data = json.loads(response)
                                    task.output_data.update(result_data)

                                    # Update task status based on result
                                    if result_data.get('status') == 'failed':
                                        task.status = TaskStatus.FAILED
                                        task.error_message = result_data.get('observations', 'Task failed')
                                    else:
                                        task.status = TaskStatus.COMPLETED

                                except (json.JSONDecodeError, ValueError):
                                    # Not JSON, just use the response as is
                                    task.output_data['content'] = response
                                    task.status = TaskStatus.COMPLETED

                        else:
                            # No tools, just use the generate method
                            response = await self.llm.generate(
                                [
                                    {'role': 'system', 'content': system_prompt},
                                    {'role': 'user', 'content': f'Execute task: {task.name}'},
                                ],
                                temperature=0.2,
                            )

                            # Try to parse the response as JSON
                            try:
                                if '```json' in response:
                                    response = response.split('```json')[1].split('```')[0].strip()
                                elif '```' in response:
                                    response = response.split('```')[1].split('```')[0].strip()

                                result_data = json.loads(response)
                                task.output_data.update(result_data)

                                # Update task status based on result
                                if result_data.get('status') == 'failed':
                                    task.status = TaskStatus.FAILED
                                    task.error_message = result_data.get('observations', 'Task failed')
                                else:
                                    task.status = TaskStatus.COMPLETED

                            except (json.JSONDecodeError, ValueError):
                                # Not JSON, just use the response as is
                                task.output_data['content'] = response
                                task.status = TaskStatus.COMPLETED

                        # Task completed successfully
                        break

                    except Exception as e:
                        logger.error(f'Error during task execution: {e}, attempt {attempt + 1}/{self.max_retries}')
                        if attempt == self.max_retries - 1:
                            # Max retries reached, mark task as failed
                            task.status = TaskStatus.FAILED
                            task.error_message = f'Failed after {self.max_retries} attempts: {e!s}'

                # Record task completion time
                task.completed_at = datetime.utcnow()

            except Exception as e:
                # If an exception occurs, mark the task as failed
                logger.exception(f'Error executing task {task.id}: {e}')

                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                task.completed_at = datetime.utcnow()

            finally:
                # Update the plan status
                plan.update_status()
