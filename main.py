#!/usr/bin/env python3
"""NEMWAS - Neural-Enhanced Multi-Workforce Agent System"""

import asyncio
import signal
import sys
import logging
from pathlib import Path
import click
import uvicorn
from rich.console import Console
from rich.logging import RichHandler

# NEMWAS imports
from src.core.agent import NEMWASAgent, AgentConfig
from src.core.npu_manager import NPUManager
from src.core.react import Tool
from src.nlp.interface import NaturalLanguageInterface
from src.plugins.interface import PluginRegistry, ToolPlugin
from src.performance.tracker import PerformanceTracker
from src.utils.config import load_config
from src.api.server import create_app

# Setup logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("nemwas")


class NEMWASCore:
    """Core NEMWAS system orchestrator"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.running = False
        
        # Initialize components
        logger.info("🚀 Initializing NEMWAS Core...")
        
        # NPU Manager
        self.npu_manager = NPUManager(self.config, cache_dir=self.config.get('cache_dir', './models/cache'))
        
        # Performance Tracker
        self.performance_tracker = PerformanceTracker(
            metrics_dir=self.config.get('metrics_dir', './data/metrics'),
            enable_prometheus=self.config.get('enable_prometheus', True)
        )
        
        # Plugin Registry
        self.plugin_registry = PluginRegistry(
            plugin_dirs=self.config.get('plugin_dirs', ['./plugins/builtin', './plugins/community'])
        )
        self.plugin_registry.set_npu_manager(self.npu_manager)
        
        # Natural Language Interface
        self.nl_interface = NaturalLanguageInterface()
        
        # Agent pool
        self.agents = {}

        # WebSocket Handler
        from src.api.websocket import WebSocketHandler
        self.websocket_handler = WebSocketHandler(self)
        
        # Create default agent
        self._create_default_agent()
        
        logger.info("✅ NEMWAS Core initialized successfully")
    
    def _create_default_agent(self):
        """Create the default agent"""
        
        agent_config = AgentConfig(
            name="Default-Agent",
            model_path=self.config['default_model_path'],
            device_preference=["NPU", "GPU", "CPU"],
            enable_learning=True,
            enable_performance_tracking=True
        )
        
        try:
            agent = NEMWASAgent(agent_config, self.npu_manager)
            self.agents[agent.agent_id] = agent
            logger.info(f"Created default agent: {agent.agent_id}")

            # Register high-priority plugins
            register_high_priority_plugins(agent, self.plugin_registry)

        except Exception as e:
            logger.error(f"Failed to create default agent: {e}")
    
    async def process_command(self, command: str) -> str:
        """Process a natural language command"""
        
        # Parse intent
        intent = self.nl_interface.parse(command)
        logger.debug(f"Parsed intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")
        
        # Route to appropriate handler
        if intent.intent_type.value == "execute_task":
            # Use default agent for task execution
            if self.agents:
                agent = list(self.agents.values())[0]
                result = await agent.process(intent.entities.get('task', command))
                return result
            else:
                return "No agents available. Please create an agent first."
        
        elif intent.intent_type.value == "create_agent":
            # Create new agent
            return await self._handle_create_agent(intent)
        
        elif intent.intent_type.value == "query_status":
            # Get system status
            return self._handle_status_query(intent)
        
        elif intent.intent_type.value == "analyze_performance":
            # Analyze performance
            return self._handle_performance_analysis(intent)
        
        else:
            # Default to using the agent
            if self.agents:
                agent = list(self.agents.values())[0]
                result = await agent.process(command)
                return result
            else:
                return "I'm not sure how to handle that request. Try 'help' for available commands."
    
    async def _handle_create_agent(self, intent):
        """Handle agent creation request"""
        
        try:
            # Extract configuration from intent
            purpose = intent.entities.get('purpose', 'general')
            
            agent_config = AgentConfig(
                name=f"Agent-{purpose[:20]}",
                model_path=self.config['default_model_path'],
                device_preference=["NPU", "GPU", "CPU"],
                enable_learning=True,
                enable_performance_tracking=True
            )
            
            agent = NEMWASAgent(agent_config, self.npu_manager)
            self.agents[agent.agent_id] = agent
            
            return f"Successfully created agent {agent.agent_id} for {purpose}"
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return f"Failed to create agent: {str(e)}"
    
    def _handle_status_query(self, intent):
        """Handle status query"""
        
        status = {
            'system': self.performance_tracker.get_system_metrics(),
            'agents': len(self.agents),
            'active_agents': sum(1 for a in self.agents.values() if a.context.current_task),
            'npu_available': "NPU" in self.npu_manager.available_devices,
            'plugins_loaded': len(self.plugin_registry.plugins)
        }
        
        return self.nl_interface.generate_response(intent, {'system': status})
    
    def _handle_performance_analysis(self, intent):
        """Handle performance analysis request"""
        
        agent_id = intent.entities.get('agent_id')
        
        if agent_id and agent_id in self.agents:
            analysis = self.performance_tracker.analyze_performance_trends(agent_id)
        else:
            analysis = self.performance_tracker.analyze_performance_trends()
        
        return self.nl_interface.generate_response(intent, analysis)
    
    async def start(self):
        """Start the NEMWAS system"""
        
        self.running = True
        logger.info("🚀 Starting NEMWAS system...")
        
        # Load plugins
        self._load_plugins()
        
        # Start API server
        app = create_app(self)
        
        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run server
        config = uvicorn.Config(
            app=app,
            host=self.config.get('api_host', '0.0.0.0'),
            port=self.config.get('api_port', 8080),
            log_level="info"
        )
        server = uvicorn.Server(config)

        # Start background tasks
        asyncio.create_task(self._npu_metrics_broadcaster())
        
        await server.serve()
    
    async def _npu_metrics_broadcaster(self):
        """Periodically broadcast NPU metrics."""
        while self.running:
            if hasattr(self, 'websocket_handler') and "NPU" in self.npu_manager.available_devices:
                metrics = self.npu_manager.get_device_metrics("NPU")
                await self.websocket_handler.broadcast_npu_metrics(metrics)
            await asyncio.sleep(5)  # Broadcast every 5 seconds

    def _load_plugins(self):
        """Load configured plugins"""
        
        logger.info("Loading plugins...")
        
        # Discover available plugins
        available = self.plugin_registry.discover_plugins()
        logger.info(f"Found {len(available)} available plugins")
        
        # Load configured plugins
        for plugin_path in self.config.get('plugins', []):
            if self.plugin_registry.load_plugin(plugin_path):
                logger.info(f"✓ Loaded plugin: {plugin_path}")
            else:
                logger.warning(f"✗ Failed to load plugin: {plugin_path}")
        
        # Register plugin tools with agents
        tools = self.plugin_registry.get_tools()
        for agent in self.agents.values():
            for tool_name, tool_def in tools.items():
                # Create tool wrapper
                from src.core.react import Tool
                
                plugin_tool = Tool(
                    name=tool_name,
                    description=tool_def['description'],
                    function=tool_def['function'],
                    parameters=tool_def['parameters']
                )
                
                agent.register_tool(plugin_tool)
    
    def stop(self):
        """Stop the NEMWAS system"""
        
        logger.info("Shutting down NEMWAS...")
        self.running = False
        
        # Export metrics
        self.performance_tracker.export_metrics()
        
        # Cleanup agents
        for agent in self.agents.values():
            agent.export_context()
        
        logger.info("NEMWAS shutdown complete")


def register_high_priority_plugins(agent: NEMWASAgent, plugin_registry: PluginRegistry):
    """Register the high-priority plugins with the agent"""

    # Load plugins
    plugins_to_load = [
        "plugins/code_architect.py",
        "plugins/security_scanner.py",
        "plugins/memory_consolidator.py",
        "plugins/metrics_exporter.py"
    ]

    for plugin_path in plugins_to_load:
        plugin_id = plugin_registry.load_plugin(plugin_path)
        if plugin_id:
            plugin = plugin_registry.get_plugin(plugin_id)

            # Register as agent tool
            if isinstance(plugin, ToolPlugin):
                tool_def = plugin.get_tool_definition()
                agent.register_tool(Tool(
                    name=tool_def['name'],
                    description=tool_def['description'],
                    function=tool_def['function'],
                    parameters=tool_def['parameters']
                ))

@click.command()
@click.option('--config', '-c', default='config/default.yaml', help='Configuration file path')
@click.option('--interactive', '-i', is_flag=True, help='Start in interactive mode')
@click.option('--command', help='Execute a single command and exit')
def main(config, interactive, command):
    """NEMWAS - Neural-Enhanced Multi-Workforce Agent System"""
    
    console.print("[bold blue]NEMWAS v1.0[/bold blue] - Neural-Enhanced Multi-Workforce Agent System")
    console.print("=" * 60)
    
    # Create core system
    core = NEMWASCore(config)
    
    if command:
        # Execute single command
        async def run_command():
            result = await core.process_command(command)
            console.print(result)
        
        asyncio.run(run_command())
        
    elif interactive:
        # Interactive mode
        console.print("Interactive mode. Type 'help' for commands, 'exit' to quit.")
        console.print()
        
        async def interactive_loop():
            while True:
                try:
                    user_input = console.input("[bold green]nemwas>[/bold green] ")
                    
                    if user_input.lower() in ['exit', 'quit']:
                        break
                    
                    result = await core.process_command(user_input)
                    console.print(result)
                    console.print()
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
        
        asyncio.run(interactive_loop())
        
    else:
        # Server mode
        try:
            asyncio.run(core.start())
        except KeyboardInterrupt:
            pass
    
    core.stop()


if __name__ == "__main__":
    main()
