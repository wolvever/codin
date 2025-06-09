import pytest
import asyncio
import uuid
from typing import AsyncGenerator, Any

from src.codin.agent.base import Planner
from src.codin.agent.base_agent import BaseAgent
from src.codin.agent.types import (
    AgentRunInput,
    Message,
    TextPart,
    Role,
    State,
    Task,
    FinishStep,
    Step,
    StepType,
    AgentRunOutput,
    TaskState,
    TaskStatus,
)
from src.codin.memory.base import MemMemoryService


# Mock Planner
class MockPlanner(Planner):
    def __init__(self, steps_to_yield: list[Step] | None = None):
        self.steps_to_yield = steps_to_yield if steps_to_yield is not None else []
        super().__init__()

    async def next(self, state: State) -> AsyncGenerator[Step, None]:
        for step in self.steps_to_yield:
            yield step
        # Default to finish if no steps provided, to prevent infinite loops in some tests
        if not self.steps_to_yield:
            yield FinishStep(step_id=str(uuid.uuid4()), reason="Mock planner finished")

    async def reset(self, state: State) -> None:
        pass


class MockFinishPlanner(Planner):
    async def next(self, state: State) -> AsyncGenerator[Step, None]:
        yield FinishStep(step_id=str(uuid.uuid4()), reason="Finished immediately by MockFinishPlanner")

    async def reset(self, state: State) -> None:
        pass


@pytest.mark.asyncio
async def test_base_agent_state_initialization_and_task():
    agent = BaseAgent(planner=MockPlanner(), memory=MemMemoryService())
    session_id = f"test_session_{uuid.uuid4().hex}"
    message_id = f"test_msg_{uuid.uuid4().hex}"
    input_text = "Hello, agent!"

    input_data = AgentRunInput(
        session_id=session_id,
        message=Message(
            messageId=message_id,
            role=Role.user,
            parts=[TextPart(text=input_text)],
            contextId=session_id # Important: contextId of message should match session_id for memory
        )
    )

    # Directly call _build_state for focused testing
    # Note: In normal operation, run() calls _build_state and handles session_id propagation.
    # Here, we pass session_id explicitly as _build_state expects it.
    state = await agent._build_state(session_id, input_data)

    assert state.session_id == session_id
    assert state.agent_id == agent.id
    assert state.task is not None
    assert isinstance(state.task, Task)
    assert state.task.id == message_id
    assert state.task.message is not None
    assert len(state.task.message.parts) == 1
    assert isinstance(state.task.message.parts[0], TextPart)
    assert state.task.message.parts[0].text == input_text

    # Check history via memory as _build_state interacts with self.memory
    history = await agent.memory.get_history()
    assert len(history) == 1
    assert history[0].messageId == message_id
    assert history[0].contextId == session_id


@pytest.mark.asyncio
async def test_base_agent_memory_usage_in_build_state():
    memory_service = MemMemoryService()
    agent = BaseAgent(planner=MockPlanner(), memory=memory_service)
    session_id = f"test_session_mem_{uuid.uuid4().hex}"
    message_id = f"test_msg_mem_{uuid.uuid4().hex}"
    input_text = "Test memory initialization."

    input_data = AgentRunInput(
        session_id=session_id,
        message=Message(
            messageId=message_id,
            role=Role.user,
            parts=[TextPart(text=input_text)],
            contextId=session_id # Ensure message contextId is the session_id
        )
    )

    await agent._build_state(session_id, input_data)

    # For MemMemoryService, session_id is set internally via set_session_id
    # We can't directly assert memory_service.session_id as it's not a public attribute
    # Instead, we verify that the message added to memory has the correct session_id (contextId)

    history = await memory_service.get_history(session_id=session_id) # Pass session_id to get_history
    assert len(history) == 1
    assert history[0].messageId == message_id
    assert history[0].contextId == session_id


@pytest.mark.asyncio
async def test_base_agent_run_with_immediate_finish():
    agent = BaseAgent(planner=MockFinishPlanner(), memory=MemMemoryService())
    session_id = f"test_session_finish_{uuid.uuid4().hex}"
    message_id = f"test_msg_finish_{uuid.uuid4().hex}"
    input_text = "Run and finish."

    input_data = AgentRunInput(
        session_id=session_id,
        message=Message(
            messageId=message_id,
            role=Role.user,
            parts=[TextPart(text=input_text)],
            contextId=session_id
        )
    )

    outputs = []
    async for output in agent.run(input_data):
        outputs.append(output)

    assert len(outputs) > 0
    final_output = outputs[-1]
    assert isinstance(final_output, AgentRunOutput)

    # The result of a FinishStep is usually a Message
    assert isinstance(final_output.result, Message)
    final_message: Message = final_output.result

    assert final_message.role == Role.agent
    assert "Finished immediately by MockFinishPlanner" in final_message.get_text_content()

    # Verify that an event for task completion was emitted (indirectly, via mailbox)
    # This test focuses on the run output, direct event testing might require mailbox mocking
    # For now, check if metadata indicates success or finish
    assert final_output.metadata is not None
    assert final_output.metadata.get("step_type") == "finish"
    assert final_output.metadata.get("reason") == "Finished immediately by MockFinishPlanner"

    # Check memory for the initial user message and the final agent message
    history = await agent.memory.get_history(session_id=session_id)
    assert len(history) == 2 # User message + Agent's final message
    assert history[0].messageId == message_id
    assert history[1].messageId == final_message.messageId
