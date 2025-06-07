import pytest

from codin.host import LocalHost, RayHost
from codin.lifecycle import LifecycleState


@pytest.mark.asyncio
async def test_local_host_lifecycle():
    host = LocalHost()
    assert host.state == LifecycleState.DOWN
    await host.up()
    assert host.state == LifecycleState.UP
    assert host.dispatcher is not None
    assert host.actor_manager is not None
    await host.down()
    assert host.state == LifecycleState.DOWN


@pytest.mark.asyncio
async def test_ray_host_lifecycle():
    ray = pytest.importorskip("ray")
    host = RayHost()
    await host.up()
    assert host.dispatcher is not None
    assert host.actor_manager is not None
    await host.down()
    assert not ray.is_initialized()
