"""Background task manager for NEMWAS API"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import shutil
import psutil

from ..performance.tracker import PerformanceTracker
from ..capability.learner import CapabilityLearner
from ..plugins.interface import PluginRegistry

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Manages background tasks for NEMWAS system"""

    def __init__(self, nemwas_core):
        self.core = nemwas_core
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        self.task_status: Dict[str, Dict[str, Any]] = {}

        # Task intervals (in seconds)
        self.intervals = {
            "metrics_export": 3600,      # 1 hour
            "model_optimization": 86400,  # 24 hours
            "capability_sync": 1800,      # 30 minutes
            "cache_cleanup": 7200,        # 2 hours
            "health_check": 300,          # 5 minutes
            "plugin_hot_reload": 10,      # 10 seconds
            "performance_analysis": 3600, # 1 hour
        }

        logger.info("Background Task Manager initialized")

    async def start(self):
        """Start all background tasks"""

        if self.running:
            logger.warning("Background tasks already running")
            return

        self.running = True
        logger.info("Starting background tasks...")

        # Start individual tasks
        self.tasks["metrics_export"] = asyncio.create_task(
            self._run_periodic_task("metrics_export", self._export_metrics)
        )

        self.tasks["model_optimization"] = asyncio.create_task(
            self._run_periodic_task("model_optimization", self._optimize_models)
        )

        self.tasks["capability_sync"] = asyncio.create_task(
            self._run_periodic_task("capability_sync", self._sync_capabilities)
        )

        self.tasks["cache_cleanup"] = asyncio.create_task(
            self._run_periodic_task("cache_cleanup", self._cleanup_cache)
        )

        self.tasks["health_check"] = asyncio.create_task(
            self._run_periodic_task("health_check", self._health_check)
        )

        if self.core.config.get("plugins", {}).get("enable_hot_reload", False):
            self.tasks["plugin_hot_reload"] = asyncio.create_task(
                self._run_periodic_task("plugin_hot_reload", self._check_plugin_updates)
            )

        self.tasks["performance_analysis"] = asyncio.create_task(
            self._run_periodic_task("performance_analysis", self._analyze_performance)
        )

        logger.info(f"Started {len(self.tasks)} background tasks")

    async def stop(self):
        """Stop all background tasks"""

        self.running = False
        logger.info("Stopping background tasks...")

        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.tasks.clear()
        logger.info("All background tasks stopped")

    async def _run_periodic_task(self, name: str, func: Callable):
        """Run a task periodically"""

        interval = self.intervals.get(name, 3600)

        # Initialize task status
        self.task_status[name] = {
            "last_run": None,
            "next_run": datetime.now(),
            "run_count": 0,
            "last_error": None,
            "status": "running"
        }

        while self.running:
            try:
                # Update status
                self.task_status[name]["status"] = "executing"
                self.task_status[name]["last_run"] = datetime.now()

                # Execute task
                logger.debug(f"Executing background task: {name}")
                await func()

                # Update success status
                self.task_status[name]["run_count"] += 1
                self.task_status[name]["status"] = "idle"
                self.task_status[name]["last_error"] = None
                self.task_status[name]["next_run"] = datetime.now() + timedelta(seconds=interval)

            except Exception as e:
                logger.error(f"Error in background task {name}: {e}")
                self.task_status[name]["status"] = "error"
                self.task_status[name]["last_error"] = str(e)

            # Wait for next execution
            await asyncio.sleep(interval)

    async def _export_metrics(self):
        """Export performance metrics"""

        try:
            # Export metrics to file
            timestamp = int(time.time())
            export_path = self.core.performance_tracker.export_metrics(
                f"data/metrics/export_{timestamp}.json"
            )

            logger.info(f"Metrics exported to {export_path}")

            # Clean up old exports (keep last 7 days)
            await self._cleanup_old_files("data/metrics", "export_*.json", days=7)

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise

    async def _optimize_models(self):
        """Check and optimize models for NPU if needed"""

        try:
            models_dir = Path("models/original")
            optimized_dir = Path("models/cache")

            for model_file in models_dir.glob("*.xml"):
                # Check if already optimized
                optimized_name = f"{model_file.stem}_npu_optimized.xml"
                optimized_path = optimized_dir / optimized_name

                if not optimized_path.exists():
                    logger.info(f"Optimizing model: {model_file}")

                    # Optimize for NPU
                    self.core.npu_manager.optimize_model_for_npu(
                        str(model_file),
                        model_type="llm",
                        quantization_preset="mixed"
                    )

            logger.info("Model optimization check completed")

        except Exception as e:
            logger.error(f"Failed to optimize models: {e}")
            raise

    async def _sync_capabilities(self):
        """Synchronize learned capabilities across agents"""

        try:
            all_capabilities = []

            # Collect capabilities from all agents
            for agent_id, agent in self.core.agents.items():
                if agent.capability_learner:
                    capabilities = agent.capability_learner.export_capabilities()
                    all_capabilities.extend(capabilities.get('capabilities', []))

            if all_capabilities:
                # Save aggregated capabilities
                output_file = Path("data/capabilities/aggregated_capabilities.json")
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, 'w') as f:
                    json.dump({
                        "timestamp": time.time(),
                        "total_capabilities": len(all_capabilities),
                        "capabilities": all_capabilities
                    }, f, indent=2)

                logger.info(f"Synchronized {len(all_capabilities)} capabilities")

        except Exception as e:
            logger.error(f"Failed to sync capabilities: {e}")
            raise

    async def _cleanup_cache(self):
        """Clean up old cache files"""

        try:
            cache_dirs = [
                "models/cache",
                "data/metrics",
                "data/embeddings"
            ]

            total_cleaned = 0

            for cache_dir in cache_dirs:
                path = Path(cache_dir)
                if not path.exists():
                    continue

                # Clean files older than 7 days
                cutoff_time = time.time() - (7 * 24 * 60 * 60)

                for file in path.iterdir():
                    if file.is_file() and file.stat().st_mtime < cutoff_time:
                        file.unlink()
                        total_cleaned += 1

            if total_cleaned > 0:
                logger.info(f"Cleaned {total_cleaned} old cache files")

            # Check disk space
            disk_usage = psutil.disk_usage('/')
            if disk_usage.percent > 90:
                logger.warning(f"Low disk space: {disk_usage.percent}% used")

        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
            raise

    async def _health_check(self):
        """Perform system health check"""

        try:
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "issues": []
            }

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                health_status["issues"].append(f"High CPU usage: {cpu_percent}%")

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                health_status["issues"].append(f"High memory usage: {memory.percent}%")

            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                health_status["issues"].append(f"Low disk space: {disk.percent}% used")

            # Check agent responsiveness
            unresponsive_agents = []
            for agent_id, agent in self.core.agents.items():
                if agent.context.current_task and time.time() - agent.context.last_active > 300:
                    unresponsive_agents.append(agent_id)

            if unresponsive_agents:
                health_status["issues"].append(f"Unresponsive agents: {unresponsive_agents}")

            # Check NPU status
            if "NPU" in self.core.npu_manager.available_devices:
                npu_metrics = self.core.npu_manager.get_device_metrics("NPU")
                if npu_metrics.get("utilization", 0) > 95:
                    health_status["issues"].append("NPU overloaded")

            # Update overall status
            if health_status["issues"]:
                health_status["status"] = "degraded"
                logger.warning(f"Health check issues: {health_status['issues']}")

            # Save health status
            health_file = Path("data/health_status.json")
            with open(health_file, 'w') as f:
                json.dump(health_status, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to perform health check: {e}")
            raise

    async def _check_plugin_updates(self):
        """Check for plugin updates and hot reload if needed"""

        try:
            plugin_dirs = self.core.config.get("plugins", {}).get("plugin_dirs", [])

            for plugin_dir in plugin_dirs:
                path = Path(plugin_dir)
                if not path.exists():
                    continue

                # Check for new or modified plugins
                for plugin_file in path.glob("*.py"):
                    if plugin_file.name.startswith("_"):
                        continue

                    # Check modification time
                    mtime = plugin_file.stat().st_mtime

                    # Simple check - in production, use more sophisticated tracking
                    plugin_path = f"{plugin_dir}/{plugin_file.stem}:plugin_class"

                    # Check if plugin needs reloading
                    # This is a simplified check - real implementation would track mtimes
                    current_plugins = self.core.plugin_registry.list_plugins()
                    plugin_names = [p['name'] for p in current_plugins]

                    # For now, just log - actual hot reload would require more logic
                    logger.debug(f"Checked plugin: {plugin_file}")

        except Exception as e:
            logger.error(f"Failed to check plugin updates: {e}")
            raise

    async def _analyze_performance(self):
        """Analyze performance trends and generate recommendations"""

        try:
            # Analyze overall system performance
            analysis = self.core.performance_tracker.analyze_performance_trends()

            if analysis.get('trend') == 'degrading':
                logger.warning("Performance degradation detected")

                # Generate alert
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "performance_degradation",
                    "analysis": analysis,
                    "recommendations": analysis.get('recommendations', [])
                }

                # Save alert
                alert_file = Path("data/alerts/performance_alert.json")
                alert_file.parent.mkdir(parents=True, exist_ok=True)

                with open(alert_file, 'w') as f:
                    json.dump(alert, f, indent=2)

            # Analyze per-agent performance
            for agent_id in self.core.agents:
                agent_analysis = self.core.performance_tracker.analyze_performance_trends(agent_id)

                if agent_analysis.get('success_rate', 0) < 0.5:
                    logger.warning(f"Low success rate for agent {agent_id}")

        except Exception as e:
            logger.error(f"Failed to analyze performance: {e}")
            raise

    async def _cleanup_old_files(self, directory: str, pattern: str, days: int):
        """Clean up old files matching pattern"""

        path = Path(directory)
        if not path.exists():
            return

        cutoff_time = time.time() - (days * 24 * 60 * 60)

        for file in path.glob(pattern):
            if file.is_file() and file.stat().st_mtime < cutoff_time:
                file.unlink()
                logger.debug(f"Deleted old file: {file}")

    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all background tasks"""

        return {
            "running": self.running,
            "tasks": self.task_status,
            "active_tasks": len([t for t in self.tasks.values() if not t.done()])
        }

    async def trigger_task(self, task_name: str) -> bool:
        """Manually trigger a background task"""

        if task_name == "metrics_export":
            await self._export_metrics()
        elif task_name == "model_optimization":
            await self._optimize_models()
        elif task_name == "capability_sync":
            await self._sync_capabilities()
        elif task_name == "cache_cleanup":
            await self._cleanup_cache()
        elif task_name == "health_check":
            await self._health_check()
        elif task_name == "performance_analysis":
            await self._analyze_performance()
        else:
            logger.warning(f"Unknown task: {task_name}")
            return False

        logger.info(f"Manually triggered task: {task_name}")
        return True
