"""WebSocket handlers for NEMWAS API"""

import json
import logging
import asyncio
import time
from typing import Dict, Set, Optional, Any
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .models import WebSocketMessage, WebSocketCommand, TaskStatus
from .app import app

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        # Active connections by client ID
        self.active_connections: Dict[str, WebSocket] = {}

        # Subscriptions: event_type -> set of client_ids
        self.subscriptions: Dict[str, Set[str]] = {}

        # Client metadata
        self.client_metadata: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> str:
        """Accept new WebSocket connection"""
        await websocket.accept()

        # Generate client ID if not provided
        if not client_id:
            client_id = str(uuid.uuid4())

        # Store connection
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = {
            "connected_at": datetime.now(),
            "subscriptions": set()
        }

        logger.info(f"WebSocket client connected: {client_id}")

        # Send welcome message
        await self.send_personal_message(
            WebSocketMessage(
                type="connection",
                data={
                    "status": "connected",
                    "client_id": client_id,
                    "message": "Welcome to NEMWAS WebSocket API"
                }
            ),
            client_id
        )

        return client_id

    async def disconnect(self, client_id: str):
        """Handle client disconnection"""
        if client_id in self.active_connections:
            # Remove from all subscriptions
            for event_type, subscribers in self.subscriptions.items():
                subscribers.discard(client_id)

            # Remove connection
            del self.active_connections[client_id]
            del self.client_metadata[client_id]

            logger.info(f"WebSocket client disconnected: {client_id}")

    async def send_personal_message(self, message: WebSocketMessage, client_id: str):
        """Send message to specific client"""
        websocket = self.active_connections.get(client_id)
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(message.dict())
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                await self.disconnect(client_id)

    async def broadcast(self, message: WebSocketMessage, event_type: Optional[str] = None):
        """Broadcast message to all connected clients or subscribers"""
        # Determine recipients
        if event_type and event_type in self.subscriptions:
            recipients = self.subscriptions[event_type]
        else:
            recipients = set(self.active_connections.keys())

        # Send to each recipient
        disconnected = []
        for client_id in recipients:
            websocket = self.active_connections.get(client_id)
            if websocket:
                try:
                    await websocket.send_json(message.dict())
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client_id}: {e}")
                    disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

    async def subscribe(self, client_id: str, event_type: str):
        """Subscribe client to event type"""
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = set()

        self.subscriptions[event_type].add(client_id)

        if client_id in self.client_metadata:
            self.client_metadata[client_id]["subscriptions"].add(event_type)

        logger.info(f"Client {client_id} subscribed to {event_type}")

        # Confirm subscription
        await self.send_personal_message(
            WebSocketMessage(
                type="subscription",
                data={
                    "event_type": event_type,
                    "status": "subscribed"
                }
            ),
            client_id
        )

    async def unsubscribe(self, client_id: str, event_type: str):
        """Unsubscribe client from event type"""
        if event_type in self.subscriptions:
            self.subscriptions[event_type].discard(client_id)

        if client_id in self.client_metadata:
            self.client_metadata[client_id]["subscriptions"].discard(event_type)

        logger.info(f"Client {client_id} unsubscribed from {event_type}")

        # Confirm unsubscription
        await self.send_personal_message(
            WebSocketMessage(
                type="subscription",
                data={
                    "event_type": event_type,
                    "status": "unsubscribed"
                }
            ),
            client_id
        )

    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get client metadata"""
        return self.client_metadata.get(client_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "subscriptions": {
                event_type: len(subscribers)
                for event_type, subscribers in self.subscriptions.items()
            },
            "clients": [
                {
                    "client_id": client_id,
                    "connected_at": metadata["connected_at"].isoformat(),
                    "subscriptions": list(metadata["subscriptions"])
                }
                for client_id, metadata in self.client_metadata.items()
            ]
        }


class WebSocketHandler:
    """Handles WebSocket communication for NEMWAS"""

    def __init__(self, nemwas_core):
        self.core = nemwas_core
        self.manager = ConnectionManager()

        # Message handlers
        self.handlers = {
            "subscribe": self.handle_subscribe,
            "unsubscribe": self.handle_unsubscribe,
            "execute": self.handle_execute,
            "status": self.handle_status,
            "list_agents": self.handle_list_agents,
            "ping": self.handle_ping
        }

    async def handle_connection(self, websocket: WebSocket):
        """Handle a WebSocket connection"""
        client_id = await self.manager.connect(websocket)

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()

                try:
                    # Parse command
                    command = WebSocketCommand.parse_raw(data)

                    # Handle command
                    await self.handle_command(client_id, command)

                except json.JSONDecodeError:
                    await self.send_error(
                        client_id,
                        "Invalid JSON",
                        {"raw_data": data[:100]}
                    )
                except Exception as e:
                    logger.error(f"Error handling command: {e}")
                    await self.send_error(
                        client_id,
                        str(e),
                        {"command": data[:100]}
                    )

        except WebSocketDisconnect:
            await self.manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
            await self.manager.disconnect(client_id)

    async def handle_command(self, client_id: str, command: WebSocketCommand):
        """Route command to appropriate handler"""
        handler = self.handlers.get(command.command)

        if handler:
            await handler(client_id, command)
        else:
            await self.send_error(
                client_id,
                f"Unknown command: {command.command}",
                {"available_commands": list(self.handlers.keys())}
            )

    async def handle_subscribe(self, client_id: str, command: WebSocketCommand):
        """Handle subscription request"""
        event_type = command.params.get("event_type") if command.params else None

        if not event_type:
            await self.send_error(client_id, "event_type required")
            return

        valid_events = [
            "task_started",
            "task_completed",
            "task_failed",
            "agent_created",
            "agent_status_changed",
            "performance_update",
            "system_status"
        ]

        if event_type not in valid_events:
            await self.send_error(
                client_id,
                f"Invalid event type: {event_type}",
                {"valid_events": valid_events}
            )
            return

        await self.manager.subscribe(client_id, event_type)

    async def handle_unsubscribe(self, client_id: str, command: WebSocketCommand):
        """Handle unsubscription request"""
        event_type = command.params.get("event_type") if command.params else None

        if not event_type:
            await self.send_error(client_id, "event_type required")
            return

        await self.manager.unsubscribe(client_id, event_type)

    async def handle_execute(self, client_id: str, command: WebSocketCommand):
        """Handle task execution request"""
        if not command.params or "query" not in command.params:
            await self.send_error(client_id, "query required")
            return

        query = command.params["query"]
        agent_id = command.params.get("agent_id")

        # Send acknowledgment
        await self.manager.send_personal_message(
            WebSocketMessage(
                type="task_acknowledged",
                data={
                    "query": query,
                    "status": "processing",
                    "callback_id": command.callback_id
                }
            ),
            client_id
        )

        try:
            # Execute task
            if agent_id and agent_id in self.core.agents:
                agent = self.core.agents[agent_id]
            elif self.core.agents:
                agent = list(self.core.agents.values())[0]
            else:
                await self.send_error(client_id, "No agents available")
                return

            # Track task
            task_id = self.core.performance_tracker.start_task(agent.agent_id, query)

            # Notify subscribers
            await self.manager.broadcast(
                WebSocketMessage(
                    type="task_started",
                    data={
                        "task_id": task_id,
                        "agent_id": agent.agent_id,
                        "query": query
                    }
                ),
                event_type="task_started"
            )

            # Execute
            result = await agent.process(query)

            # Send result
            await self.manager.send_personal_message(
                WebSocketMessage(
                    type="task_result",
                    data={
                        "task_id": task_id,
                        "result": result,
                        "status": "completed",
                        "callback_id": command.callback_id
                    }
                ),
                client_id
            )

            # Notify subscribers
            await self.manager.broadcast(
                WebSocketMessage(
                    type="task_completed",
                    data={
                        "task_id": task_id,
                        "agent_id": agent.agent_id,
                        "status": "completed"
                    }
                ),
                event_type="task_completed"
            )

        except Exception as e:
            logger.error(f"Task execution error: {e}")

            # Send error
            await self.send_error(
                client_id,
                f"Task execution failed: {str(e)}",
                {"query": query, "callback_id": command.callback_id}
            )

            # Notify subscribers
            await self.manager.broadcast(
                WebSocketMessage(
                    type="task_failed",
                    data={
                        "agent_id": agent_id,
                        "error": str(e)
                    }
                ),
                event_type="task_failed"
            )

    async def handle_status(self, client_id: str, command: WebSocketCommand):
        """Handle status request"""
        status = {
            "agents": len(self.core.agents),
            "active_agents": sum(1 for a in self.core.agents.values() if a.context.current_task),
            "plugins": len(self.core.plugin_registry.plugins),
            "websocket_clients": len(self.manager.active_connections),
            "system_metrics": self.core.performance_tracker.get_system_metrics()
        }

        await self.manager.send_personal_message(
            WebSocketMessage(
                type="status",
                data=status
            ),
            client_id
        )

    async def handle_list_agents(self, client_id: str, command: WebSocketCommand):
        """Handle list agents request"""
        agents = []

        for agent_id, agent in self.core.agents.items():
            metrics = self.core.performance_tracker.get_agent_metrics(agent_id)
            agents.append({
                "agent_id": agent_id,
                "name": agent.config.name,
                "status": "busy" if agent.context.current_task else "idle",
                "device": agent.device,
                "total_tasks": metrics.get("total_tasks", 0),
                "success_rate": metrics.get("success_rate", 0.0)
            })

        await self.manager.send_personal_message(
            WebSocketMessage(
                type="agents_list",
                data={"agents": agents}
            ),
            client_id
        )

    async def handle_ping(self, client_id: str, command: WebSocketCommand):
        """Handle ping request"""
        await self.manager.send_personal_message(
            WebSocketMessage(
                type="pong",
                data={
                    "timestamp": datetime.now().isoformat(),
                    "callback_id": command.callback_id
                }
            ),
            client_id
        )

    async def send_error(self, client_id: str, error: str, details: Optional[Dict] = None):
        """Send error message to client"""
        await self.manager.send_personal_message(
            WebSocketMessage(
                type="error",
                data={
                    "error": error,
                    "details": details or {}
                }
            ),
            client_id
        )

    # Event broadcasting methods
    async def broadcast_task_update(self, task_id: str, status: TaskStatus, **kwargs):
        """Broadcast task status update"""
        await self.manager.broadcast(
            WebSocketMessage(
                type=f"task_{status.value}",
                data={
                    "task_id": task_id,
                    "status": status.value,
                    **kwargs
                }
            ),
            event_type=f"task_{status.value}"
        )

    async def broadcast_agent_update(self, agent_id: str, event: str, **kwargs):
        """Broadcast agent update"""
        await self.manager.broadcast(
            WebSocketMessage(
                type=f"agent_{event}",
                data={
                    "agent_id": agent_id,
                    "event": event,
                    **kwargs
                }
            ),
            event_type=f"agent_{event}"
        )

    async def broadcast_system_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast system update"""
        await self.manager.broadcast(
            WebSocketMessage(
                type=f"system_{update_type}",
                data=data
            ),
            event_type="system_status"
        )

async def handle_agent_updates(websocket: WebSocket):
    """Stream agent status updates"""
    await websocket.accept()

    while True:
        # Get tracker data
        agent_data = app.state.core.agent_tracker.get_all_agent_data()

        # Send updates
        await websocket.send_json({
            "type": "agent_status",
            "data": agent_data,
            "timestamp": time.time()
        })

        await asyncio.sleep(1)
