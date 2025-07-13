# API Reference

This document provides a detailed reference for the NEMWAS API. The API is divided into several sections based on functionality.

## Agents

### `GET /agents`

List all agents with optional filtering.

- **Method:** `GET`
- **Path:** `/agents`
- **Query Parameters:**
    - `status` (optional, string): Filter by agent status. Can be `active`, `idle`, or `all`.
    - `sort_by` (optional, string): Sort agents by a specific field. Can be `created`, `name`, `tasks`, or `success_rate`.
    - `limit` (optional, integer): Limit the number of agents returned. Default is `50`.
- **Response Model:** `List[AgentInfo]`

### `POST /agents`

Create a new agent.

- **Method:** `POST`
- **Path:** `/agents`
- **Request Body:** `AgentCreateRequest`
- **Response Model:** `AgentInfo`

### `GET /agents/{agent_id}`

Get specific agent details.

- **Method:** `GET`
- **Path:** `/agents/{agent_id}`
- **Path Parameters:**
    - `agent_id` (required, string): The ID of the agent.
- **Response Model:** `AgentInfo`

### `DELETE /agents/{agent_id}`

Delete an agent.

- **Method:** `DELETE`
- **Path:** `/agents/{agent_id}`
- **Path Parameters:**
    - `agent_id` (required, string): The ID of the agent.
- **Response:** JSON object with a confirmation message.

### `GET /agents/{agent_id}/metrics`

Get detailed agent metrics.

- **Method:** `GET`
- **Path:** `/agents/{agent_id}/metrics`
- **Path Parameters:**
    - `agent_id` (required, string): The ID of the agent.
- **Response:** JSON object with detailed agent metrics.

### `GET /agents/{agent_id}/context`

Get agent context and state.

- **Method:** `GET`
- **Path:** `/agents/{agent_id}/context`
- **Path Parameters:**
    - `agent_id` (required, string): The ID of the agent.
- **Query Parameters:**
    - `include_history` (optional, boolean): Whether to include the conversation history.
- **Response:** JSON object with the agent's context.

## Tasks

### `POST /tasks`

Execute a task.

- **Method:** `POST`
- **Path:** `/tasks`
- **Request Body:** `TaskRequest`
- **Response Model:** `TaskResponse`

### `GET /tasks/status/{task_id}`

Get task status.

- **Method:** `GET`
- **Path:** `/tasks/status/{task_id}`
- **Path Parameters:**
    - `task_id` (required, string): The ID of the task.
- **Response:** JSON object with the task status.

### `WEBSOCKET /tasks/ws`

WebSocket endpoint for real-time task updates.

- **Path:** `/tasks/ws`

## System

### `GET /system/status`

Get comprehensive system status.

- **Method:** `GET`
- **Path:** `/system/status`
- **Response Model:** `SystemStatus`

### `GET /system/health`

Health check endpoint.

- **Method:** `GET`
- **Path:** `/system/health`
- **Response:** JSON object with the system's health status.

### `POST /system/shutdown`

Gracefully shutdown the system.

- **Method:** `POST`
- **Path:** `/system/shutdown`
- **Response:** JSON object with a confirmation message.

## Plugins

### `GET /plugins`

List loaded plugins.

- **Method:** `GET`
- **Path:** `/plugins`
- **Query Parameters:**
    - `plugin_type` (optional, string): Filter by plugin type. Can be `tool`, `capability`, `analyzer`, or `all`.
- **Response:** JSON object with a list of plugins.

### `POST /plugins/load`

Load a plugin.

- **Method:** `POST`
- **Path:** `/plugins/load`
- **Request Body:** `PluginLoadRequest`
- **Response:** JSON object with a confirmation message.

### `DELETE /plugins/{plugin_name}`

Unload a plugin.

- **Method:** `DELETE`
- **Path:** `/plugins/{plugin_name}`
- **Path Parameters:**
    - `plugin_name` (required, string): The name of the plugin.
- **Response:** JSON object with a confirmation message.

## Metrics

### `GET /metrics/performance/analysis`

Analyze performance trends.

- **Method:** `GET`
- **Path:** `/metrics/performance/analysis`
- **Query Parameters:**
    - `agent_id` (optional, string): The ID of the agent to analyze.
    - `time_range` (optional, string): The time range for the analysis. Can be `1h`, `6h`, `24h`, `7d`, or `30d`.
- **Response Model:** `PerformanceAnalysis`

### `GET /metrics/prometheus`

Prometheus metrics endpoint.

- **Method:** `GET`
- **Path:** `/metrics/prometheus`
- **Response:** Prometheus metrics in plain text.

### `GET /metrics/capabilities`

List all learned capabilities across agents.

- **Method:** `GET`
- **Path:** `/metrics/capabilities`
- **Query Parameters:**
    - `min_success_rate` (optional, float): Minimum success rate for a capability to be included.
    - `sort_by` (optional, string): Sort capabilities by a specific field. Can be `usage`, `success_rate`, `performance`, or `recent`.
- **Response Model:** `List[CapabilityInfo]`

### `POST /metrics/export`

Export metrics in specified format.

- **Method:** `POST`
- **Path:** `/metrics/export`
- **Query Parameters:**
    - `format` (optional, string): The format to export the metrics in. Can be `json`, `csv`, or `prometheus`.
- **Response:** JSON object with the export status.
