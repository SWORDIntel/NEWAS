"""FastAPI server for NEMWAS"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import prometheus_client

from .background import BackgroundTaskManager

logger = logging.getLogger(__name__)


# Request/Response models
class TaskRequest(BaseModel):
    """Task execution request"""
    query: str
    context: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    priority: str = "normal"


class TaskResponse(BaseModel):
    """Task execution response"""
    task_id: str
    result: str
    execution_time: float
    agent_id: str
    timestamp: datetime


class AgentCreateRequest(BaseModel):
    """Agent creation request"""
    name: str
    purpose: str
    model_path: Optional[str] = None
    device_preference: List[str] = ["NPU", "GPU", "CPU"]


class AgentInfo(BaseModel):
    """Agent information"""
    agent_id: str
    name: str
    status: str
    device: str
    total_tasks: int
    success_rate: float
    last_active: datetime


class SystemStatus(BaseModel):
    """System status information"""
    version: str = "1.0.0"
    uptime: float
    total_agents: int
    active_agents: int
    npu_available: bool
    plugins_loaded: int
    system_metrics: Dict[str, Any]


def create_app(nemwas_core) -> FastAPI:
    """Create FastAPI application"""

    app = FastAPI(
        title="NEMWAS API",
        description="Neural-Enhanced Multi-Workforce Agent System API",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store reference to core
    app.state.core = nemwas_core
    app.state.start_time = datetime.now()

    # Active WebSocket connections
    app.state.websockets = set()

    # Background task manager
    app.state.background_manager = BackgroundTaskManager(nemwas_core)

    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint"""
        return {
            "service": "NEMWAS API",
            "version": "1.0.0",
            "status": "running"
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        try:
            # Check core components
            if not app.state.core.running:
                raise HTTPException(status_code=503, detail="Service not running")

            # Check NPU
            npu_status = "available" if "NPU" in app.state.core.npu_manager.available_devices else "not available"

            return {
                "status": "healthy",
                "timestamp": datetime.now(),
                "components": {
                    "core": "running",
                    "npu": npu_status,
                    "agents": len(app.state.core.agents),
                    "plugins": len(app.state.core.plugin_registry.plugins)
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail=str(e))

    @app.get("/status", response_model=SystemStatus)
    async def get_status():
        """Get system status"""

        uptime = (datetime.now() - app.state.start_time).total_seconds()

        return SystemStatus(
            uptime=uptime,
            total_agents=len(app.state.core.agents),
            active_agents=sum(1 for a in app.state.core.agents.values() if a.context.current_task),
            npu_available="NPU" in app.state.core.npu_manager.available_devices,
            plugins_loaded=len(app.state.core.plugin_registry.plugins),
            system_metrics=app.state.core.performance_tracker.get_system_metrics()
        )

    @app.post("/tasks", response_model=TaskResponse)
    async def execute_task(request: TaskRequest):
        """Execute a task"""

        try:
            # Select agent
            if request.agent_id and request.agent_id in app.state.core.agents:
                agent = app.state.core.agents[request.agent_id]
            elif app.state.core.agents:
                # Use first available agent
                agent = list(app.state.core.agents.values())[0]
            else:
                raise HTTPException(status_code=404, detail="No agents available")

            # Track task
            task_id = app.state.core.performance_tracker.start_task(
                agent.agent_id,
                request.query
            )

            # Execute task
            start_time = datetime.now()
            result = await agent.process(request.query, request.context)
            execution_time = (datetime.now() - start_time).total_seconds()

            # Create response
            response = TaskResponse(
                task_id=task_id,
                result=result,
                execution_time=execution_time,
                agent_id=agent.agent_id,
                timestamp=datetime.now()
            )

            # Broadcast to WebSocket clients
            await broadcast_update({
                "type": "task_completed",
                "data": response.dict()
            })

            return response

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agents", response_model=List[AgentInfo])
    async def list_agents():
        """List all agents"""

        agents = []
        for agent_id, agent in app.state.core.agents.items():
            metrics = app.state.core.performance_tracker.get_agent_metrics(agent_id)

            agents.append(AgentInfo(
                agent_id=agent_id,
                name=agent.config.name,
                status="active" if agent.context.current_task else "idle",
                device=agent.device,
                total_tasks=metrics.get('total_tasks', 0),
                success_rate=metrics.get('success_rate', 0.0),
                last_active=datetime.fromtimestamp(metrics.get('last_active', 0))
            ))

        return agents

    @app.post("/agents", response_model=AgentInfo)
    async def create_agent(request: AgentCreateRequest):
        """Create a new agent"""

        try:
            from src.core.agent import AgentConfig, NEMWASAgent

            # Create agent configuration
            agent_config = AgentConfig(
                name=request.name,
                model_path=request.model_path or app.state.core.config['default_model_path'],
                device_preference=request.device_preference,
                enable_learning=True,
                enable_performance_tracking=True
            )

            # Create agent
            agent = NEMWASAgent(agent_config, app.state.core.npu_manager)
            app.state.core.agents[agent.agent_id] = agent

            # Register plugin tools
            tools = app.state.core.plugin_registry.get_tools()
            for tool_name, tool_def in tools.items():
                from src.core.react import Tool

                plugin_tool = Tool(
                    name=tool_name,
                    description=tool_def['description'],
                    function=tool_def['function'],
                    parameters=tool_def['parameters']
                )

                agent.register_tool(plugin_tool)

            # Create response
            return AgentInfo(
                agent_id=agent.agent_id,
                name=agent.config.name,
                status="idle",
                device=agent.device,
                total_tasks=0,
                success_rate=0.0,
                last_active=datetime.now()
            )

        except Exception as e:
            logger.error(f"Agent creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        """Delete an agent"""

        if agent_id not in app.state.core.agents:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Export context before deletion
        agent = app.state.core.agents[agent_id]
        context = agent.export_context()

        # Remove agent
        del app.state.core.agents[agent_id]

        return {"message": f"Agent {agent_id} deleted", "context": context}

    @app.get("/agents/{agent_id}/metrics")
    async def get_agent_metrics(agent_id: str):
        """Get agent metrics"""

        if agent_id not in app.state.core.agents:
            raise HTTPException(status_code=404, detail="Agent not found")

        metrics = app.state.core.performance_tracker.get_agent_metrics(agent_id)
        agent = app.state.core.agents[agent_id]

        return {
            "agent_id": agent_id,
            "metrics": metrics,
            "capabilities": len(agent.context.capabilities),
            "conversation_history": len(agent.context.conversation_history)
        }

    @app.get("/plugins")
    async def list_plugins():
        """List loaded plugins"""

        return app.state.core.plugin_registry.list_plugins()

    @app.post("/plugins/load")
    async def load_plugin(plugin_path: str):
        """Load a plugin"""

        try:
            success = app.state.core.plugin_registry.load_plugin(plugin_path)

            if success:
                return {"message": f"Plugin loaded: {plugin_path}"}
            else:
                raise HTTPException(status_code=400, detail="Failed to load plugin")

        except Exception as e:
            logger.error(f"Plugin load failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/capabilities")
    async def list_capabilities():
        """List all learned capabilities"""

        all_capabilities = []

        for agent in app.state.core.agents.values():
            if agent.capability_learner:
                capabilities = agent.capability_learner.export_capabilities()
                all_capabilities.extend(capabilities.get('capabilities', []))

        return {
            "total": len(all_capabilities),
            "capabilities": all_capabilities
        }

    @app.get("/performance/analysis")
    async def analyze_performance(agent_id: Optional[str] = None):
        """Analyze performance trends"""

        analysis = app.state.core.performance_tracker.analyze_performance_trends(agent_id)
        return analysis

    @app.get("/metrics")
    async def get_prometheus_metrics():
        """Prometheus metrics endpoint"""

        return prometheus_client.generate_latest()

    @app.get("/background/status")
    async def get_background_status():
        """Get background task status"""

        return app.state.background_manager.get_task_status()

    @app.post("/background/trigger/{task_name}")
    async def trigger_background_task(task_name: str):
        """Manually trigger a background task"""

        try:
            success = await app.state.background_manager.trigger_task(task_name)

            if success:
                return {"message": f"Task {task_name} triggered successfully"}
            else:
                raise HTTPException(status_code=404, detail=f"Unknown task: {task_name}")

        except Exception as e:
            logger.error(f"Failed to trigger task {task_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""

        await websocket.accept()
        app.state.websockets.add(websocket)

        try:
            # Send initial status
            status = await get_status()
            await websocket.send_json({
                "type": "status",
                "data": status.dict()
            })

            # Keep connection alive
            while True:
                # Wait for messages or send periodic updates
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=30)

                    # Process message
                    if message == "ping":
                        await websocket.send_text("pong")
                    else:
                        # Process as command
                        result = await app.state.core.process_command(message)
                        await websocket.send_json({
                            "type": "command_result",
                            "data": {"command": message, "result": result}
                        })

                except asyncio.TimeoutError:
                    # Send periodic status update
                    status = await get_status()
                    await websocket.send_json({
                        "type": "status_update",
                        "data": status.dict()
                    })

        except WebSocketDisconnect:
            app.state.websockets.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            app.state.websockets.remove(websocket)

    async def broadcast_update(message: Dict[str, Any]):
        """Broadcast update to all WebSocket clients"""

        disconnected = set()

        for websocket in app.state.websockets:
            try:
                await websocket.send_json(message)
            except:
                disconnected.add(websocket)

        # Remove disconnected clients
        app.state.websockets -= disconnected

    @app.on_event("startup")
    async def startup_event():
        """Startup event handler"""
        logger.info("NEMWAS API starting up...")

        # Start background tasks
        await app.state.background_manager.start()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown event handler"""
        logger.info("NEMWAS API shutting down...")

        # Stop background tasks
        await app.state.background_manager.stop()

        # Close all WebSocket connections
        for websocket in app.state.websockets:
            try:
                await websocket.close()
            except:
                pass

    # Add __init__.py marker for package
    init_path = Path(__file__).parent / "__init__.py"
    if not init_path.exists():
        init_path.touch()

    return app
