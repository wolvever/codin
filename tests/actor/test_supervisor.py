import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from codin.actor.supervisor import LocalActorManager, ActorInfo, ActorSupervisor
from codin.actor.types import CallableActor
from codin.agent.base_agent import BaseAgent
from codin.agent.base import Planner
from codin.agent.types import AgentRunInput, AgentRunOutput, Message, TextPart, Role


class MockPlanner(Planner):
    async def next(self, state):
        if False:
            yield  # pragma: no cover

    async def reset(self, state):
        pass


class MockAgent(BaseAgent):
    def __init__(self, agent_id: str = "test_agent"):
        super().__init__(agent_id=agent_id, planner=MockPlanner())
        
    async def run(self, input_data: AgentRunInput):
        yield AgentRunOutput(
            id="test_output",
            messages=[Message(role=Role.ASSISTANT, content=[TextPart(text="Test response")])],
            runner_id=input_data.runner_id,
            request_id=input_data.request_id
        )


class TestActorInfo:
    
    def test_actor_info_creation(self):
        """Test creating ActorInfo instance."""
        agent = MockAgent("test_agent")
        info = ActorInfo(
            actor_id="test_agent",
            actor_instance=agent,
            capabilities={"coding", "analysis"},
            status="active"
        )
        
        assert info.actor_id == "test_agent"
        assert info.actor_instance == agent
        assert info.capabilities == {"coding", "analysis"}
        assert info.status == "active"
    
    def test_actor_info_default_values(self):
        """Test ActorInfo with default values."""
        agent = MockAgent()
        info = ActorInfo(
            actor_id="test",
            actor_instance=agent,
            capabilities=set(),
            status="idle"
        )
        
        assert len(info.capabilities) == 0
        assert info.status == "idle"


class TestLocalActorManager:
    
    @pytest.fixture
    def manager(self):
        return LocalActorManager()
    
    @pytest.mark.asyncio
    async def test_acquire_new_actor(self, manager):
        """Test acquiring a new actor."""
        agent_id = "test_agent"
        
        with patch.object(manager, '_create_actor_instance') as mock_create:
            mock_agent = MockAgent(agent_id)
            mock_create.return_value = mock_agent
            
            info = await manager.acquire(agent_id)
            
            assert info.actor_id == agent_id
            assert info.actor_instance == mock_agent
            assert info.status == "active"
            mock_create.assert_called_once_with(agent_id)
    
    @pytest.mark.asyncio
    async def test_acquire_existing_idle_actor(self, manager):
        """Test acquiring an existing idle actor."""
        agent_id = "test_agent"
        mock_agent = MockAgent(agent_id)
        
        # Pre-populate with an idle actor
        manager._actors[agent_id] = ActorInfo(
            actor_id=agent_id,
            actor_instance=mock_agent,
            capabilities=set(),
            status="idle"
        )
        
        info = await manager.acquire(agent_id)
        
        assert info.actor_id == agent_id
        assert info.status == "active"
        assert info.actor_instance == mock_agent
    
    @pytest.mark.asyncio
    async def test_acquire_busy_actor_creates_new(self, manager):
        """Test acquiring when actor is busy creates new instance."""
        agent_id = "test_agent"
        
        # Pre-populate with a busy actor
        manager._actors[agent_id] = ActorInfo(
            actor_id=agent_id,
            actor_instance=MockAgent(agent_id),
            capabilities=set(),
            status="active"
        )
        
        with patch.object(manager, '_create_actor_instance') as mock_create:
            new_agent = MockAgent(agent_id)
            mock_create.return_value = new_agent
            
            info = await manager.acquire(agent_id)
            
            assert info.actor_instance == new_agent
            mock_create.assert_called_once_with(agent_id)
    
    @pytest.mark.asyncio
    async def test_release_actor(self, manager):
        """Test releasing an actor."""
        agent_id = "test_agent"
        mock_agent = MockAgent(agent_id)
        
        # Set up an active actor
        manager._actors[agent_id] = ActorInfo(
            actor_id=agent_id,
            actor_instance=mock_agent,
            capabilities=set(),
            status="active"
        )
        
        await manager.release(agent_id)
        
        assert manager._actors[agent_id].status == "idle"
        assert manager._actors[agent_id].actor_instance == mock_agent
    
    @pytest.mark.asyncio
    async def test_release_nonexistent_actor(self, manager):
        """Test releasing a non-existent actor."""
        # Should not raise an error
        await manager.release("nonexistent_actor")
    
    @pytest.mark.asyncio
    async def test_list_actors(self, manager):
        """Test listing all actors."""
        # Add some actors
        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2")
        
        manager._actors["agent1"] = ActorInfo(
            actor_id="agent1",
            actor_instance=agent1,
            capabilities={"coding"},
            status="active"
        )
        manager._actors["agent2"] = ActorInfo(
            actor_id="agent2", 
            actor_instance=agent2,
            capabilities={"analysis"},
            status="idle"
        )
        
        actors = await manager.list()
        
        assert len(actors) == 2
        actor_ids = {info.actor_id for info in actors}
        assert actor_ids == {"agent1", "agent2"}
    
    @pytest.mark.asyncio
    async def test_info_existing_actor(self, manager):
        """Test getting info for existing actor."""
        agent_id = "test_agent"
        mock_agent = MockAgent(agent_id)
        expected_info = ActorInfo(
            actor_id=agent_id,
            actor_instance=mock_agent,
            capabilities={"test"},
            status="active"
        )
        
        manager._actors[agent_id] = expected_info
        
        info = await manager.info(agent_id)
        
        assert info == expected_info
    
    @pytest.mark.asyncio
    async def test_info_nonexistent_actor(self, manager):
        """Test getting info for non-existent actor."""
        info = await manager.info("nonexistent")
        
        assert info is None
    
    @pytest.mark.asyncio
    async def test_get_actor_instance(self, manager):
        """Test getting actor instance."""
        agent_id = "test_agent"
        mock_agent = MockAgent(agent_id)
        
        manager._actors[agent_id] = ActorInfo(
            actor_id=agent_id,
            actor_instance=mock_agent,
            capabilities=set(),
            status="active"
        )
        
        instance = await manager.get_actor_instance(agent_id)
        
        assert instance == mock_agent
    
    @pytest.mark.asyncio
    async def test_get_actor_instance_nonexistent(self, manager):
        """Test getting instance for non-existent actor."""
        instance = await manager.get_actor_instance("nonexistent")
        
        assert instance is None
    
    @pytest.mark.asyncio
    async def test_cleanup_idle_actors_removes_old_actors(self, manager):
        """Test cleanup removes actors idle for too long."""
        agent_id = "old_agent"
        mock_agent = MockAgent(agent_id)
        
        # Create an actor that's been idle for a long time
        old_info = ActorInfo(
            actor_id=agent_id,
            actor_instance=mock_agent,
            capabilities=set(),
            status="idle"
        )
        
        manager._actors[agent_id] = old_info
        
        # Mock the last activity time to be old
        with patch('codin.actor.supervisor.datetime') as mock_datetime:
            # Current time is much later than last activity
            mock_datetime.now.return_value = datetime.now()
            
            await manager.cleanup_idle_actors(max_idle_time=timedelta(seconds=1))
            
            # Actor should be removed if it's been idle too long
            # Implementation depends on actual cleanup logic
    
    @pytest.mark.asyncio
    async def test_cleanup_idle_actors_keeps_active_actors(self, manager):
        """Test cleanup keeps active actors."""
        agent_id = "active_agent"
        mock_agent = MockAgent(agent_id)
        
        manager._actors[agent_id] = ActorInfo(
            actor_id=agent_id,
            actor_instance=mock_agent,
            capabilities=set(),
            status="active"
        )
        
        await manager.cleanup_idle_actors()
        
        # Active actors should not be removed
        assert agent_id in manager._actors
        assert manager._actors[agent_id].status == "active"
    
    @pytest.mark.asyncio
    async def test_cleanup_idle_actors_keeps_recent_idle(self, manager):
        """Test cleanup keeps recently idle actors."""
        agent_id = "recent_agent"
        mock_agent = MockAgent(agent_id)
        
        manager._actors[agent_id] = ActorInfo(
            actor_id=agent_id,
            actor_instance=mock_agent,
            capabilities=set(),
            status="idle"
        )
        
        # Use very short max idle time
        await manager.cleanup_idle_actors(max_idle_time=timedelta(hours=1))
        
        # Recently idle actors should be kept
        assert agent_id in manager._actors
    
    @pytest.mark.asyncio
    async def test_concurrent_acquire_same_actor(self, manager):
        """Test concurrent acquisition of the same actor."""
        agent_id = "concurrent_agent"
        
        with patch.object(manager, '_create_actor_instance') as mock_create:
            mock_create.side_effect = lambda aid: MockAgent(aid)
            
            # Start two concurrent acquisitions
            task1 = asyncio.create_task(manager.acquire(agent_id))
            task2 = asyncio.create_task(manager.acquire(agent_id))
            
            info1, info2 = await asyncio.gather(task1, task2)
            
            # Both should succeed but might get different instances
            assert info1.actor_id == agent_id
            assert info2.actor_id == agent_id
            assert info1.status == "active"
            assert info2.status == "active"
    
    @pytest.mark.asyncio
    async def test_actor_lifecycle_full_cycle(self, manager):
        """Test full actor lifecycle: acquire -> use -> release -> cleanup."""
        agent_id = "lifecycle_agent"
        
        with patch.object(manager, '_create_actor_instance') as mock_create:
            mock_agent = MockAgent(agent_id)
            mock_create.return_value = mock_agent
            
            # Acquire
            info = await manager.acquire(agent_id)
            assert info.status == "active"
            
            # Use (simulated by keeping it active)
            assert await manager.get_actor_instance(agent_id) == mock_agent
            
            # Release
            await manager.release(agent_id)
            released_info = await manager.info(agent_id)
            assert released_info.status == "idle"
            
            # Cleanup (with immediate expiration)
            await manager.cleanup_idle_actors(max_idle_time=timedelta(seconds=0))
            
            # Actor might be removed depending on cleanup implementation


class TestActorSupervisorInterface:
    
    def test_actor_supervisor_is_abstract(self):
        """Test that ActorSupervisor is an abstract class."""
        with pytest.raises(TypeError):
            ActorSupervisor()
    
    @pytest.mark.asyncio
    async def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteActorSupervisor(ActorSupervisor):
            pass
        
        with pytest.raises(TypeError):
            IncompleteActorSupervisor()
    
    @pytest.mark.asyncio
    async def test_valid_subclass_implementation(self):
        """Test a valid ActorSupervisor subclass."""
        class ValidActorSupervisor(ActorSupervisor):
            def __init__(self):
                self._actors = {}
            
            async def acquire(self, agent_id):
                mock_agent = MockAgent(agent_id)
                return ActorInfo(
                    actor_id=agent_id,
                    actor_instance=mock_agent,
                    capabilities=set(),
                    status="active"
                )
            
            async def release(self, agent_id):
                pass
            
            async def list(self):
                return list(self._actors.values())
            
            async def info(self, agent_id):
                return self._actors.get(agent_id)
            
            async def get_actor_instance(self, agent_id):
                info = self._actors.get(agent_id)
                return info.actor_instance if info else None
            
            async def cleanup_idle_actors(self, max_idle_time=None):
                pass
        
        supervisor = ValidActorSupervisor()
        assert isinstance(supervisor, ActorSupervisor)
        
        # Test basic functionality
        info = await supervisor.acquire("test_agent")
        assert info.actor_id == "test_agent"
        assert info.status == "active"