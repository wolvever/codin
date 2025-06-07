import pytest

from codin.actor.ray_scheduler import RayActorManager


@pytest.mark.asyncio
async def test_ray_actor_manager_basic():
    ray = pytest.importorskip("ray")
    ray.init(local_mode=True, ignore_reinit_error=True)

    manager = RayActorManager()
    agent = await manager.get_or_create("test_agent", "1")
    assert agent is not None

    actors = await manager.list_actors()
    assert len(actors) == 1
    info = await manager.get_actor_info("test_agent:1")
    assert info is not None

    await manager.deactivate("test_agent:1")
    actors_after = await manager.list_actors()
    assert len(actors_after) == 0
    ray.shutdown()
