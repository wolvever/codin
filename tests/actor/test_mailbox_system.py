"""Tests for the actor mailbox system implementation."""

import asyncio
import pytest
from datetime import datetime

from a2a.types import Message, Role, TextPart

from codin.actor import (
    Mailbox, LocalAsyncMailbox, MailboxMessage,
    ActorManager, LocalActorManager, ActorInfo,
    Dispatcher, LocalDispatcher
)
from codin.agent.types import ControlSignal, RunnerControl, RunnerInput


@pytest.mark.asyncio
async def test_local_async_mailbox():
    """Test the LocalAsyncMailbox implementation."""
    mailbox = LocalAsyncMailbox("test_agent", maxsize=10)
    
    # Test basic inbox/outbox operations
    msg1 = Message(
        messageId="msg1",
        role=Role.user,
        parts=[TextPart(text="Hello")],
        contextId="test",
        kind="message"
    )
    
    # Put message in inbox
    await mailbox.put_inbox(msg1)
    
    # Get message from inbox
    received = await mailbox.get_inbox(timeout=1.0)
    assert received.messageId == "msg1"
    assert received.parts[0].root.text == "Hello"
    
    # Test outbox
    msg2 = Message(
        messageId="msg2",
        role=Role.agent,
        parts=[TextPart(text="World")],
        contextId="test",
        kind="message"
    )
    
    await mailbox.put_outbox(msg2)
    received_out = await mailbox.get_outbox(timeout=1.0)
    assert received_out.messageId == "msg2"


@pytest.mark.asyncio
async def test_local_actor_manager():
    """Test the LocalActorManager implementation."""
    manager = LocalActorManager()
    
    # Test get_or_create
    agent1 = await manager.get_or_create("test_agent", "key1")
    assert agent1 is not None
    assert hasattr(agent1, 'agent_id')
    
    # Test getting same agent
    agent2 = await manager.get_or_create("test_agent", "key1")
    assert agent1 is agent2  # Should be same instance
    
    # Test list actors
    actors = await manager.list_actors()
    assert len(actors) == 1
    assert actors[0].actor_type == "test_agent"
    assert actors[0].key == "key1"
    
    # Test deactivate
    agent_id = actors[0].actor_id
    await manager.deactivate(agent_id)
    
    # Should be empty now
    actors_after = await manager.list_actors()
    assert len(actors_after) == 0


@pytest.mark.asyncio
async def test_local_dispatcher():
    """Test the LocalDispatcher implementation."""
    manager = LocalActorManager()
    dispatcher = LocalDispatcher(manager)
    
    # Test submit request
    a2a_request = {
        "contextId": "test_context",
        "message": {
            "messageId": "test_msg",
            "role": "user",
            "parts": [{"kind": "text", "text": "Test message"}],
            "kind": "message"
        }
    }
    
    runner_id = await dispatcher.submit(a2a_request)
    assert runner_id.startswith("run_")
    
    # Test get status
    status = await dispatcher.get_status(runner_id)
    assert status is not None
    assert status.runner_id == runner_id
    assert status.status in ("started", "completed", "failed")
    
    # Wait a bit for processing
    await asyncio.sleep(0.1)
    
    # Test list active runs
    active_runs = await dispatcher.list_active_runs()
    assert len(active_runs) >= 1
    
    # Clean up
    await dispatcher.cleanup()


@pytest.mark.asyncio
async def test_control_signals():
    """Test control signal types and runner input."""
    # Test control signal creation
    control = RunnerControl(signal=ControlSignal.PAUSE)
    assert control.signal == ControlSignal.PAUSE
    assert isinstance(control.timestamp, datetime)
    
    # Test runner input from control
    runner_input = RunnerInput.from_control(ControlSignal.RESUME, {"test": "metadata"})
    assert runner_input.control is not None
    assert runner_input.control.signal == ControlSignal.RESUME
    assert runner_input.control.metadata["test"] == "metadata"
    
    # Test runner input from message
    message = Message(
        messageId="test",
        role=Role.user,
        parts=[TextPart(text="Test")],
        contextId="test",
        kind="message"
    )
    runner_input2 = RunnerInput.from_message(message)
    assert runner_input2.message is not None
    assert runner_input2.message.messageId == "test"


@pytest.mark.asyncio
async def test_mailbox_timeout():
    """Test mailbox timeout behavior."""
    mailbox = LocalAsyncMailbox("test_agent")
    
    # Test timeout on empty inbox
    with pytest.raises(asyncio.TimeoutError):
        await mailbox.get_inbox(timeout=0.1)
    
    # Test timeout on empty outbox
    with pytest.raises(asyncio.TimeoutError):
        await mailbox.get_outbox(timeout=0.1)


@pytest.mark.asyncio
async def test_mailbox_subscription():
    """Test mailbox subscription patterns."""
    mailbox = LocalAsyncMailbox("test_agent")
    
    # Put a message in inbox
    msg = Message(
        messageId="sub_test",
        role=Role.user,
        parts=[TextPart(text="Subscription test")],
        contextId="test",
        kind="message"
    )
    await mailbox.put_inbox(msg)
    
    # Test subscription
    async def check_subscription():
        async for message in mailbox.subscribe_inbox():
            assert message.messageId == "sub_test"
            break  # Exit after first message
    
    # Run subscription check with timeout
    await asyncio.wait_for(check_subscription(), timeout=1.0) 