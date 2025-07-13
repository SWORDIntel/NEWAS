"""Pydantic models for API request/response validation"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


# Enums
class Priority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AgentStatus(str, Enum):
    """Agent status values"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Base models
class BaseRequest(BaseModel):
    """Base request model"""
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Task models
class TaskRequest(BaseRequest):
    """Task execution request"""
    query: str = Field(..., min_length=1, max_length=10000, description="Task query or command")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    agent_id: Optional[str] = Field(default=None, description="Specific agent to use")
    priority: Priority = Field(default=Priority.NORMAL, description="Task priority")
    timeout: Optional[int] = Field(default=None, ge=1, le=3600, description="Timeout in seconds")

    @validator('query')
    def query_not_empty(cls, v):
        if not v or v.isspace():
            raise ValueError('Query cannot be empty')
        return v.strip()


class TaskResponse(BaseResponse):
    """Task execution response"""
    task_id: str
    agent_id: str
    result: str
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    status: TaskStatus
    iterations: Optional[int] = Field(default=None, ge=0, description="ReAct iterations used")
    device_used: Optional[str] = Field(default=None, description="Compute device used")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "task_id": "task_123456",
                "agent_id": "agent_789",
                "result": "The answer is 42",
                "execution_time": 0.234,
                "status": "completed",
                "iterations": 3,
                "device_used": "NPU"
            }
        }


class TaskStatusResponse(BaseResponse):
    """Task status query response"""
    task_id: str
    status: TaskStatus
    progress: Optional[float] = Field(default=None, ge=0, le=100, description="Progress percentage")
    current_step: Optional[str] = None
    estimated_completion: Optional[datetime] = None


# Agent models
class AgentCreateRequest(BaseRequest):
    """Agent creation request"""
    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    purpose: str = Field(..., min_length=1, max_length=500, description="Agent purpose")
    model_path: Optional[str] = Field(default=None, description="Custom model path")
    device_preference: List[str] = Field(
        default=["NPU", "GPU", "CPU"],
        description="Device preference order"
    )
    max_context_length: Optional[int] = Field(default=4096, ge=512, le=32768)
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    enable_learning: bool = Field(default=True)
    enable_performance_tracking: bool = Field(default=True)

    @validator('device_preference')
    def validate_devices(cls, v):
        valid_devices = {"NPU", "GPU", "CPU", "MYRIAD"}
        for device in v:
            if device not in valid_devices:
                raise ValueError(f"Invalid device: {device}")
        return v


class AgentInfo(BaseModel):
    """Agent information"""
    agent_id: str
    name: str
    purpose: str
    status: AgentStatus
    device: str
    model: str
    created_at: datetime
    last_active: datetime
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    success_rate: float = Field(default=0.0, ge=0, le=1)
    avg_execution_time: Optional[float] = None
    capabilities_count: int = 0


class AgentUpdateRequest(BaseRequest):
    """Agent update request"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_context_length: Optional[int] = Field(default=None, ge=512, le=32768)
    enable_learning: Optional[bool] = None
    enable_performance_tracking: Optional[bool] = None


# System models
class SystemStatus(BaseResponse):
    """System status information"""
    version: str
    uptime: float = Field(..., ge=0, description="Uptime in seconds")
    total_agents: int = Field(..., ge=0)
    active_agents: int = Field(..., ge=0)
    total_tasks_processed: int = Field(..., ge=0)
    npu_available: bool
    npu_utilization: Optional[float] = Field(default=None, ge=0, le=100)
    plugins_loaded: int = Field(..., ge=0)
    system_metrics: Dict[str, Any]


class HealthCheck(BaseResponse):
    """Health check response"""
    status: str = Field(..., regex="^(healthy|degraded|unhealthy)$")
    components: Dict[str, str]
    checks_passed: int
    checks_failed: int


# Performance models
class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    agent_id: Optional[str] = None
    time_period: str = Field(default="1h", description="Time period for metrics")
    total_tasks: int = Field(..., ge=0)
    successful_tasks: int = Field(..., ge=0)
    failed_tasks: int = Field(..., ge=0)
    avg_execution_time: float = Field(..., ge=0)
    median_execution_time: float = Field(..., ge=0)
    p95_execution_time: float = Field(..., ge=0)
    device_distribution: Dict[str, int]
    error_rate: float = Field(..., ge=0, le=1)
    throughput: float = Field(..., ge=0, description="Tasks per minute")


class PerformanceAnalysis(BaseResponse):
    """Performance analysis response"""
    agent_id: Optional[str] = None
    trend: str = Field(..., regex="^(improving|stable|degrading|unknown)$")
    current_performance: PerformanceMetrics
    historical_avg: float
    recommendations: List[str]
    bottlenecks: List[str]
    optimization_opportunities: List[Dict[str, Any]]


# Plugin models
class PluginInfo(BaseModel):
    """Plugin information"""
    name: str
    version: str
    author: str
    description: str
    type: str
    enabled: bool
    npu_compatible: bool
    npu_optimized: bool
    capabilities: List[str]
    dependencies: List[str]


class PluginLoadRequest(BaseRequest):
    """Plugin load request"""
    plugin_path: str = Field(..., description="Path to plugin file or entry point")
    enable: bool = Field(default=True)
    config: Optional[Dict[str, Any]] = None


# Capability models
class CapabilityInfo(BaseModel):
    """Capability information"""
    id: str
    name: str
    description: str
    usage_count: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0, le=1)
    avg_execution_time: float = Field(..., ge=0)
    last_used: datetime
    examples_count: int = Field(..., ge=0)


class CapabilitySearchRequest(BaseRequest):
    """Capability search request"""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)
    min_success_rate: Optional[float] = Field(default=None, ge=0, le=1)
    agent_id: Optional[str] = None


# Batch operations
class BatchTaskRequest(BaseRequest):
    """Batch task execution request"""
    tasks: List[TaskRequest] = Field(..., min_items=1, max_items=100)
    parallel: bool = Field(default=False, description="Execute tasks in parallel")
    stop_on_error: bool = Field(default=False)


class BatchTaskResponse(BaseResponse):
    """Batch task response"""
    batch_id: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    results: List[Union[TaskResponse, Dict[str, str]]]


# WebSocket models
class WebSocketMessage(BaseModel):
    """WebSocket message"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class WebSocketCommand(BaseModel):
    """WebSocket command"""
    command: str = Field(..., min_length=1)
    params: Optional[Dict[str, Any]] = None
    callback_id: Optional[str] = None


# Error models
class ErrorResponse(BaseResponse):
    """Error response"""
    success: bool = False
    error_code: str
    error_type: str
    detail: str
    traceback: Optional[str] = None
    suggestions: Optional[List[str]] = None

    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error_code": "AGENT_NOT_FOUND",
                "error_type": "NotFoundError",
                "detail": "Agent with ID 'agent_123' not found",
                "suggestions": ["Check agent ID", "List available agents"]
            }
        }


# Pagination models
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = None
    sort_order: str = Field(default="asc", regex="^(asc|desc)$")


class PaginatedResponse(BaseResponse):
    """Paginated response wrapper"""
    items: List[Any]
    total_items: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool
