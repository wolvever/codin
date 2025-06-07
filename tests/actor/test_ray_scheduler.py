import pytest

from codin.actor.ray_scheduler import RayActorManager


@pytest.mark.asyncio
async def test_ray_actor_manager_basic():
    ray = pytest.importorskip("ray")
    ray.init(local_mode=True, ignore_reinit_error=True)

    manager = RayActorManager()
    agent = await manager.acquire("test_agent", "1")
    assert agent is not None

    actors = await manager.list()
    assert len(actors) == 1
    info = await manager.info("test_agent:1")
    assert info is not None

    await manager.release("test_agent:1")
    actors_after = await manager.list()
    assert len(actors_after) == 0
    ray.shutdown()
