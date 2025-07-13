"""API Routes for NEMWAS"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import prometheus_client

logger = logging.getLogger(__name__)


# Pydantic models for request/response validation
class TaskRequest(BaseModel):
    """Task execution request"""
    query: str = Field(..., min_length=1, max_length=4096, description="Task query or command")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    agent_id: Optional[str] = Field(default=None, description="Specific agent ID to use")
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")
    timeout: Optional[int] = Field(default=None, ge=1, le=300, description="Timeout in seconds")

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class TaskResponse(BaseModel):
    """Task execution response"""
    task_id: str
    result: str
    execution_time: float
    agent_id: str
    timestamp: datetime
    device_used: str
    iterations: int
    success: bool


class AgentCreateRequest(BaseModel):
    """Agent creation request"""
    name: str = Field(..., min_length=1, max_length=64)
    purpose: str = Field(..., max_length=256)
    model_path: Optional[str] = None
    device_preference: List[str] = ["NPU", "GPU", "CPU"]
    max_context_length: Optional[int] = Field(default=4096, ge=512, le=32768)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)


class AgentInfo(BaseModel):
    """Agent information"""
    agent_id: str
    name: str
    status: str
    device: str
    total_tasks: int
    success_rate: float
    last_active: datetime
    capabilities_count: int
    model_info: Dict[str, Any]


class SystemStatus(BaseModel):
    """System status information"""
    version: str = "1.0.0"
    uptime: float
    total_agents: int
    active_agents: int
    npu_available: bool
    npu_utilization: Optional[float]
    plugins_loaded: int
    system_metrics: Dict[str, Any]
    active_tasks: int


class PluginLoadRequest(BaseModel):
    """Plugin load request"""
    plugin_path: str = Field(..., description="Path to plugin file or entry point")
    config: Optional[Dict[str, Any]] = None


class CapabilityInfo(BaseModel):
    """Capability information"""
    id: str
    name: str
    description: str
    success_rate: float
    usage_count: int
    last_used: datetime
    performance_score: float


class PerformanceAnalysis(BaseModel):
    """Performance analysis results"""
    trend: str
    current_avg_time: float
    historical_avg_time: float
    success_rate: float
    device_performance: Dict[str, float]
    optimal_device: Optional[str]
    recommendations: List[str]
    bottlenecks: List[Dict[str, Any]]


# Create router instances
agent_router = APIRouter(prefix="/agents", tags=["agents"])
task_router = APIRouter(prefix="/tasks", tags=["tasks"])
system_router = APIRouter(prefix="/system", tags=["system"])
plugin_router = APIRouter(prefix="/plugins", tags=["plugins"])
metrics_router = APIRouter(prefix="/metrics", tags=["metrics"])


# Dependency to get NEMWAS core
def get_core(request):
    """Get NEMWAS core from app state"""
    return request.app.state.core


# Agent routes
@agent_router.get("", response_model=List[AgentInfo])
async def list_agents(
    core=Depends(get_core),
    status: Optional[str] = Query(None, regex="^(active|idle|all)$"),
    sort_by: str = Query("created", regex="^(created|name|tasks|success_rate)$"),
    limit: int = Query(50, ge=1, le=100)
):
    """List all agents with optional filtering"""
    agents = []

    for agent_id, agent in core.agents.items():
        metrics = core.performance_tracker.get_agent_metrics(agent_id)

        agent_status = "active" if agent.context.current_task else "idle"

        # Apply status filter
        if status and status != "all" and agent_status != status:
            continue

        # Get capability count
        cap_count = len(agent.context.capabilities) if agent.capability_learner else 0

        # Get model info
        model_info = {
            "name": agent.config.name,
            "device": agent.device,
            "context_length": agent.config.max_context_length
        }

        agents.append(AgentInfo(
            agent_id=agent_id,
            name=agent.config.name,
            status=agent_status,
            device=agent.device,
            total_tasks=metrics.get('total_tasks', 0),
            success_rate=metrics.get('success_rate', 0.0),
            last_active=datetime.fromtimestamp(metrics.get('last_active', 0)),
            capabilities_count=cap_count,
            model_info=model_info
        ))

    # Sort results
    if sort_by == "name":
        agents.sort(key=lambda x: x.name)
    elif sort_by == "tasks":
        agents.sort(key=lambda x: x.total_tasks, reverse=True)
    elif sort_by == "success_rate":
        agents.sort(key=lambda x: x.success_rate, reverse=True)

    return agents[:limit]


@agent_router.post("", response_model=AgentInfo)
async def create_agent(request: AgentCreateRequest, core=Depends(get_core)):
    """Create a new agent"""
    try:
        from src.core.agent import AgentConfig, NEMWASAgent

        # Create agent configuration
        agent_config = AgentConfig(
            name=request.name,
            model_path=request.model_path or core.config['models']['default_model_path'],
            device_preference=request.device_preference,
            max_context_length=request.max_context_length,
            temperature=request.temperature,
            enable_learning=True,
            enable_performance_tracking=True
        )

        # Create agent
        agent = NEMWASAgent(agent_config, core.npu_manager)
        core.agents[agent.agent_id] = agent

        # Register plugin tools
        tools = core.plugin_registry.get_tools()
        for tool_name, tool_def in tools.items():
            from src.core.react import Tool

            plugin_tool = Tool(
                name=tool_name,
                description=tool_def['description'],
                function=tool_def['function'],
                parameters=tool_def['parameters']
            )

            agent.register_tool(plugin_tool)

        logger.info(f"Created agent: {agent.agent_id} ({agent.config.name})")

        # Return agent info
        return AgentInfo(
            agent_id=agent.agent_id,
            name=agent.config.name,
            status="idle",
            device=agent.device,
            total_tasks=0,
            success_rate=0.0,
            last_active=datetime.now(),
            capabilities_count=0,
            model_info={
                "name": agent.config.name,
                "device": agent.device,
                "context_length": agent.config.max_context_length
            }
        )

    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@agent_router.get("/{agent_id}", response_model=AgentInfo)
async def get_agent(agent_id: str = Path(...), core=Depends(get_core)):
    """Get specific agent details"""
    if agent_id not in core.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = core.agents[agent_id]
    metrics = core.performance_tracker.get_agent_metrics(agent_id)

    return AgentInfo(
        agent_id=agent_id,
        name=agent.config.name,
        status="active" if agent.context.current_task else "idle",
        device=agent.device,
        total_tasks=metrics.get('total_tasks', 0),
        success_rate=metrics.get('success_rate', 0.0),
        last_active=datetime.fromtimestamp(metrics.get('last_active', 0)),
        capabilities_count=len(agent.context.capabilities),
        model_info={
            "name": agent.config.name,
            "device": agent.device,
            "context_length": agent.config.max_context_length,
            "temperature": agent.config.temperature
        }
    )


@agent_router.delete("/{agent_id}")
async def delete_agent(agent_id: str = Path(...), core=Depends(get_core)):
    """Delete an agent"""
    if agent_id not in core.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Export context before deletion
    agent = core.agents[agent_id]
    context = agent.export_context()

    # Remove agent
    del core.agents[agent_id]

    logger.info(f"Deleted agent: {agent_id}")

    return {
        "message": f"Agent {agent_id} deleted",
        "agent_name": agent.config.name,
        "tasks_completed": context.get('performance_metrics', {}).get('total_tasks', 0),
        "context_exported": True
    }


@agent_router.get("/{agent_id}/metrics")
async def get_agent_metrics(agent_id: str = Path(...), core=Depends(get_core)):
    """Get detailed agent metrics"""
    if agent_id not in core.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = core.agents[agent_id]
    metrics = core.performance_tracker.get_agent_metrics(agent_id)

    # Get capability details if learner is enabled
    capabilities = []
    if agent.capability_learner:
        cap_export = agent.capability_learner.export_capabilities()
        capabilities = cap_export.get('capabilities', [])

    return {
        "agent_id": agent_id,
        "performance_metrics": metrics,
        "capabilities": capabilities,
        "conversation_history_length": len(agent.context.conversation_history),
        "current_task": agent.context.current_task,
        "device_info": {
            "current_device": agent.device,
            "available_devices": agent.npu_manager.available_devices
        }
    }


@agent_router.get("/{agent_id}/context")
async def get_agent_context(
    agent_id: str = Path(...),
    include_history: bool = Query(False),
    core=Depends(get_core)
):
    """Get agent context and state"""
    if agent_id not in core.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = core.agents[agent_id]
    context = agent.export_context()

    # Optionally exclude conversation history for privacy
    if not include_history:
        context.pop('conversation_history', None)

    return context


# Task routes
@task_router.post("", response_model=TaskResponse)
async def execute_task(request: TaskRequest, core=Depends(get_core)):
    """Execute a task"""
    try:
        # Select agent
        if request.agent_id and request.agent_id in core.agents:
            agent = core.agents[request.agent_id]
        elif core.agents:
            # Use least busy agent
            agent = min(
                core.agents.values(),
                key=lambda a: 1 if a.context.current_task else 0
            )
        else:
            raise HTTPException(status_code=404, detail="No agents available")

        # Track task
        task_id = core.performance_tracker.start_task(
            agent.agent_id,
            request.query
        )

        # Execute task with timeout if specified
        start_time = datetime.now()

        if request.timeout:
            result = await asyncio.wait_for(
                agent.process(request.query, request.context),
                timeout=request.timeout
            )
        else:
            result = await agent.process(request.query, request.context)

        execution_time = (datetime.now() - start_time).total_seconds()

        # Get execution details
        agent_metrics = agent.get_metrics()

        # End tracking
        core.performance_tracker.end_task(
            agent.agent_id,
            success=True,
            execution_time=execution_time,
            device_used=agent.device,
            iterations=agent_metrics.get('last_task_iterations', 1)
        )

        # Create response
        return TaskResponse(
            task_id=task_id,
            result=result,
            execution_time=execution_time,
            agent_id=agent.agent_id,
            timestamp=datetime.now(),
            device_used=agent.device,
            iterations=agent_metrics.get('last_task_iterations', 1),
            success=True
        )

    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Task execution timed out")
    except Exception as e:
        logger.error(f"Task execution failed: {e}")

        # Track failure if we have a task_id
        if 'task_id' in locals() and 'agent' in locals():
            core.performance_tracker.end_task(
                agent.agent_id,
                success=False,
                execution_time=0,
                device_used=agent.device,
                iterations=0,
                error=str(e)
            )

        raise HTTPException(status_code=500, detail=str(e))


@task_router.get("/status/{task_id}")
async def get_task_status(task_id: str = Path(...), core=Depends(get_core)):
    """Get task status (for async execution)"""
    # This would be implemented with a task queue in production
    return {
        "task_id": task_id,
        "status": "completed",
        "message": "Task status tracking requires task queue implementation"
    }


# System routes
@system_router.get("/status", response_model=SystemStatus)
async def get_system_status(core=Depends(get_core)):
    """Get comprehensive system status"""
    from src.api.server import app

    uptime = (datetime.now() - app.state.start_time).total_seconds()
    system_metrics = core.performance_tracker.get_system_metrics()

    # Count active tasks
    active_tasks = sum(
        1 for agent in core.agents.values()
        if agent.context.current_task is not None
    )

    # Get NPU utilization if available
    npu_util = None
    if "NPU" in core.npu_manager.available_devices:
        npu_metrics = core.npu_manager.get_device_metrics("NPU")
        npu_util = npu_metrics.get("utilization", 0.0)

    return SystemStatus(
        uptime=uptime,
        total_agents=len(core.agents),
        active_agents=sum(1 for a in core.agents.values() if a.context.current_task),
        npu_available="NPU" in core.npu_manager.available_devices,
        npu_utilization=npu_util,
        plugins_loaded=len(core.plugin_registry.plugins),
        system_metrics=system_metrics,
        active_tasks=active_tasks
    )


@system_router.get("/health")
async def health_check(core=Depends(get_core)):
    """Health check endpoint"""
    try:
        # Check core components
        if not hasattr(core, 'running') or not core.running:
            raise HTTPException(status_code=503, detail="Service not running")

        # Check critical components
        checks = {
            "core": "healthy",
            "npu_manager": "healthy" if core.npu_manager else "unavailable",
            "agents": f"{len(core.agents)} agents" if core.agents else "no agents",
            "plugins": f"{len(core.plugin_registry.plugins)} plugins"
        }

        # Check NPU
        if "NPU" in core.npu_manager.available_devices:
            checks["npu"] = "available"
        else:
            checks["npu"] = "not available (using CPU)"

        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": checks,
            "version": "1.0.0"
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@system_router.post("/shutdown")
async def shutdown_system(core=Depends(get_core)):
    """Gracefully shutdown the system"""
    try:
        # Export metrics
        metrics_file = core.performance_tracker.export_metrics()

        # Save agent contexts
        saved_contexts = {}
        for agent_id, agent in core.agents.items():
            saved_contexts[agent_id] = agent.export_context()

        # Stop core
        core.stop()

        return {
            "status": "shutdown initiated",
            "metrics_exported": metrics_file,
            "agents_saved": len(saved_contexts)
        }

    except Exception as e:
        logger.error(f"Shutdown failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Plugin routes
@plugin_router.get("")
async def list_plugins(
    core=Depends(get_core),
    plugin_type: Optional[str] = Query(None, regex="^(tool|capability|analyzer|all)$")
):
    """List loaded plugins"""
    plugins = core.plugin_registry.list_plugins()

    # Filter by type if specified
    if plugin_type and plugin_type != "all":
        plugins = [p for p in plugins if plugin_type in p['type'].lower()]

    return {
        "total": len(plugins),
        "plugins": plugins
    }


@plugin_router.post("/load")
async def load_plugin(request: PluginLoadRequest, core=Depends(get_core)):
    """Load a plugin"""
    try:
        success = core.plugin_registry.load_plugin(
            request.plugin_path,
            context=request.config
        )

        if success:
            # Register new tools with existing agents
            tools = core.plugin_registry.get_tools()
            for agent in core.agents.values():
                for tool_name, tool_def in tools.items():
                    if tool_name not in [t.name for t in agent.tools.values()]:
                        from src.core.react import Tool

                        plugin_tool = Tool(
                            name=tool_name,
                            description=tool_def['description'],
                            function=tool_def['function'],
                            parameters=tool_def['parameters']
                        )

                        agent.register_tool(plugin_tool)

            return {"message": f"Plugin loaded successfully: {request.plugin_path}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to load plugin")

    except Exception as e:
        logger.error(f"Plugin load failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@plugin_router.delete("/{plugin_name}")
async def unload_plugin(plugin_name: str = Path(...), core=Depends(get_core)):
    """Unload a plugin"""
    try:
        success = core.plugin_registry.unload_plugin(plugin_name)

        if success:
            return {"message": f"Plugin unloaded: {plugin_name}"}
        else:
            raise HTTPException(status_code=404, detail="Plugin not found")

    except Exception as e:
        logger.error(f"Plugin unload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Metrics routes
@metrics_router.get("/performance/analysis", response_model=PerformanceAnalysis)
async def analyze_performance(
    agent_id: Optional[str] = Query(None),
    time_range: str = Query("1h", regex="^(1h|6h|24h|7d|30d)$"),
    core=Depends(get_core)
):
    """Analyze performance trends"""
    # Validate agent_id if provided
    if agent_id and agent_id not in core.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    analysis = core.performance_tracker.analyze_performance_trends(agent_id)

    # Add bottleneck analysis
    bottlenecks = []
    if analysis.get('device_performance'):
        slowest_device = max(
            analysis['device_performance'].items(),
            key=lambda x: x[1]
        )
        if slowest_device[1] > analysis.get('current_avg_time', 0) * 1.5:
            bottlenecks.append({
                "type": "device",
                "description": f"{slowest_device[0]} is significantly slower",
                "impact": "high",
                "recommendation": f"Avoid using {slowest_device[0]} for time-critical tasks"
            })

    return PerformanceAnalysis(
        trend=analysis.get('trend', 'unknown'),
        current_avg_time=analysis.get('current_avg_time', 0.0),
        historical_avg_time=analysis.get('historical_avg_time', 0.0),
        success_rate=analysis.get('success_rate', 0.0),
        device_performance=analysis.get('device_performance', {}),
        optimal_device=analysis.get('optimal_device'),
        recommendations=analysis.get('recommendations', []),
        bottlenecks=bottlenecks
    )


@metrics_router.get("/prometheus")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        prometheus_client.generate_latest(),
        media_type="text/plain"
    )


@metrics_router.get("/capabilities", response_model=List[CapabilityInfo])
async def list_capabilities(
    core=Depends(get_core),
    min_success_rate: float = Query(0.0, ge=0.0, le=1.0),
    sort_by: str = Query("usage", regex="^(usage|success_rate|performance|recent)$")
):
    """List all learned capabilities across agents"""
    all_capabilities = []

    for agent in core.agents.values():
        if agent.capability_learner:
            capabilities = agent.capability_learner.export_capabilities()

            for cap in capabilities.get('capabilities', []):
                # Filter by success rate
                if cap.get('success_rate', 0) >= min_success_rate:
                    all_capabilities.append(CapabilityInfo(
                        id=cap['id'],
                        name=cap['name'],
                        description=cap['description'],
                        success_rate=cap.get('success_rate', 0.0),
                        usage_count=cap.get('usage_count', 0),
                        last_used=datetime.fromtimestamp(cap.get('last_used', 0)),
                        performance_score=cap.get('performance_stats', {}).get('avg_execution_time', 0.0)
                    ))

    # Sort results
    if sort_by == "usage":
        all_capabilities.sort(key=lambda x: x.usage_count, reverse=True)
    elif sort_by == "success_rate":
        all_capabilities.sort(key=lambda x: x.success_rate, reverse=True)
    elif sort_by == "performance":
        all_capabilities.sort(key=lambda x: x.performance_score)
    elif sort_by == "recent":
        all_capabilities.sort(key=lambda x: x.last_used, reverse=True)

    return all_capabilities


@metrics_router.post("/export")
async def export_metrics(
    format: str = Query("json", regex="^(json|csv|prometheus)$"),
    core=Depends(get_core)
):
    """Export metrics in specified format"""
    try:
        if format == "json":
            filepath = core.performance_tracker.export_metrics()
            return {
                "status": "exported",
                "format": "json",
                "filepath": str(filepath)
            }
        else:
            return {
                "status": "not implemented",
                "message": f"Export format '{format}' not yet implemented"
            }

    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket route for real-time updates
@task_router.websocket("/ws")
async def task_websocket(websocket: WebSocket, core=Depends(get_core)):
    """WebSocket endpoint for real-time task updates"""
    await websocket.accept()

    try:
        while True:
            # Receive task request
            data = await websocket.receive_json()

            if data.get("type") == "execute_task":
                # Execute task
                query = data.get("query", "")
                agent_id = data.get("agent_id")

                try:
                    # Select agent
                    if agent_id and agent_id in core.agents:
                        agent = core.agents[agent_id]
                    else:
                        agent = list(core.agents.values())[0]

                    # Execute with streaming updates
                    result = await agent.process(query)

                    await websocket.send_json({
                        "type": "task_complete",
                        "result": result,
                        "agent_id": agent.agent_id
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


# Include all routers in a main router
api_router = APIRouter()
api_router.include_router(agent_router)
api_router.include_router(task_router)
api_router.include_router(system_router)
api_router.include_router(plugin_router)
api_router.include_router(metrics_router)
