import pytest
from codin.agent.types import Message, Role, TextPart

from codin.actor.mailbox import RayMailbox


@pytest.mark.asyncio
async def test_ray_mailbox_basic():
    ray = pytest.importorskip("ray")
    ray.init(local_mode=True, ignore_reinit_error=True)

    mailbox = RayMailbox("agent1", maxsize=10)
    msg = Message(
        messageId="m1",
        role=Role.user,
        parts=[TextPart(text="hi")],
        contextId="ctx",
        kind="message",
    )
    await mailbox.put_inbox(msg)
    received = (await mailbox.get_inbox(timeout=1.0))[0]
    assert received.messageId == "m1"

    ray.shutdown()
