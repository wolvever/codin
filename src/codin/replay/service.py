"""Unified replay service using endpoint configuration."""

import json
import logging
from typing import Any

from codin.endpoint import EndpointConfig, ReplayConfig
from codin.endpoint.resolver import EndpointManager

from .base import BaseReplay

logger = logging.getLogger(__name__)


class Replay(BaseReplay):
    """Unified replay service using endpoint configuration system."""

    def __init__(self, session_id: str, config: ReplayConfig | EndpointConfig | None = None):
        self.session_id = session_id
        if isinstance(config, ReplayConfig):
            self.config = config.endpoint
        elif isinstance(config, EndpointConfig):
            self.config = config
        else:
            # Auto-detect from environment
            replay_config = ReplayConfig.from_env()
            self.config = replay_config.endpoint
        self.manager = EndpointManager(self.config)
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure the replay service is initialized."""
        if not self._initialized:
            await self._initialize_session()
            self._initialized = True

    async def _initialize_session(self):
        """Initialize the session with metadata."""
        session_start_data = {
            "event": "session_start",
            "session_id": self.session_id,
            "timestamp": self._get_timestamp(),
            "endpoint": self.config.endpoint,
        }

        try:
            if self.config.is_local:
                # Write to local file
                filename = f"session_{self.session_id}.jsonl"
                await self.manager.write(filename, json.dumps(session_start_data).encode("utf-8"))
            else:
                # Send to remote endpoint
                path = f"sessions/{self.session_id}/events"
                await self.manager.write(path, json.dumps(session_start_data).encode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to initialize replay session {self.session_id}: {e}")

    async def record_message_exchange(self, client_message: Any, agent_message: Any, session_id: str, **kwargs) -> None:
        """Records a client message and the corresponding agent message for a session."""
        await self._ensure_initialized()

        # Create the exchange record
        exchange_data = {
            "event": kwargs.get("exchange_type", "message_exchange"),
            "session_id": session_id,
            "timestamp": self._get_timestamp(),
            "client_message": self._serialize_message(client_message),
            "agent_message": self._serialize_message(agent_message),
            **{k: v for k, v in kwargs.items() if k != "exchange_type"},
        }

        try:
            if self.config.is_local:
                # Append to local file
                filename = f"session_{self.session_id}.jsonl"

                # Check if file exists, if not create it
                if not await self.manager.exists(filename):
                    await self._initialize_session()

                # Read existing content
                existing_content = ""
                try:
                    existing_data = await self.manager.read(filename)
                    existing_content = existing_data.decode("utf-8")
                except Exception:
                    # File doesn't exist or is empty
                    pass

                # Append new line
                new_line = json.dumps(exchange_data) + "\n"
                updated_content = existing_content + new_line

                await self.manager.write(filename, updated_content.encode("utf-8"))
            else:
                # Send to remote endpoint
                path = f"sessions/{session_id}/events"
                await self.manager.write(path, json.dumps(exchange_data).encode("utf-8"))

        except Exception as e:
            logger.error(f"Failed to record message exchange for session {session_id}: {e}")

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Record session end
            session_end_data = {
                "event": "session_end",
                "session_id": self.session_id,
                "timestamp": self._get_timestamp(),
            }

            if self.config.is_local:
                filename = f"session_{self.session_id}.jsonl"

                # Read existing content
                existing_content = ""
                try:
                    existing_data = await self.manager.read(filename)
                    existing_content = existing_data.decode("utf-8")
                except Exception:
                    pass

                # Append session end
                new_line = json.dumps(session_end_data) + "\n"
                updated_content = existing_content + new_line

                await self.manager.write(filename, updated_content.encode("utf-8"))
            else:
                # Send to remote endpoint
                path = f"sessions/{self.session_id}/events"
                await self.manager.write(path, json.dumps(session_end_data).encode("utf-8"))

            # Close manager
            await self.manager.close()

        except Exception as e:
            logger.error(f"Failed to cleanup replay session {self.session_id}: {e}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.utcnow().isoformat() + "Z"


def create_replay_service(session_id: str, config: ReplayConfig | EndpointConfig | None = None) -> Replay:
    """Create a unified replay service.

    Args:
        session_id: Session identifier
        config: Replay configuration (defaults to replay endpoint from env)

    Returns:
        Replay instance
    """
    return Replay(session_id, config)


def create_replay_factory(config: ReplayConfig | EndpointConfig | None = None):
    """Create a replay factory function for use with dispatchers.

    Args:
        config: Replay configuration (defaults to replay endpoint from env)

    Returns:
        Factory function that takes session_id and returns Replay instance
    """

    def factory(session_id: str) -> Replay:
        return Replay(session_id, config)

    return factory
