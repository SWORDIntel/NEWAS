"""Plugin System for NEMWAS Framework"""

import abc
import json
import logging
import importlib
import inspect
from typing import Dict, List, Optional, Any, Callable, Type
from pathlib import Path
from dataclasses import dataclass
import pluggy

logger = logging.getLogger(__name__)

# Plugin hook specifications
hookspec = pluggy.HookspecMarker("nemwas")
hookimpl = pluggy.HookimplMarker("nemwas")


@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    author: str
    description: str
    npu_compatible: bool = False
    requirements: List[str] = None
    capabilities: List[str] = None


class NEMWASPlugin(abc.ABC):
    """Base class for NEMWAS plugins"""

    @abc.abstractmethod
    def __init__(self):
        """Initialize plugin"""
        self.metadata = self.get_metadata()
        self.npu_model = None

    @abc.abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass

    @abc.abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize plugin with system context"""
        pass

    @abc.abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin functionality"""
        pass

    def get_npu_model(self) -> Optional[Any]:
        """Return NPU-optimized model if available"""
        return self.npu_model

    def cleanup(self):
        """Cleanup plugin resources"""
        pass


class ToolPlugin(NEMWASPlugin):
    """Plugin that provides a tool for agents"""

    @abc.abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return tool definition for agent registration"""
        pass


class CapabilityPlugin(NEMWASPlugin):
    """Plugin that provides new capabilities"""

    @abc.abstractmethod
    def get_capability_patterns(self) -> List[Dict[str, Any]]:
        """Return capability patterns this plugin can handle"""
        pass


class AnalyzerPlugin(NEMWASPlugin):
    """Plugin that provides analysis capabilities"""

    @abc.abstractmethod
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze provided data"""
        pass


class PluginRegistry:
    """Manages plugin discovery, loading, and lifecycle"""

    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or ["./plugins/builtin", "./plugins/community"]
        self.pm = pluggy.PluginManager("nemwas")
        self.pm.add_hookspecs(PluginHooks)

        # Registered plugins
        self.plugins: Dict[str, NEMWASPlugin] = {}
        self.tool_plugins: Dict[str, ToolPlugin] = {}
        self.capability_plugins: Dict[str, CapabilityPlugin] = {}
        self.analyzer_plugins: Dict[str, AnalyzerPlugin] = {}

        # NPU manager reference
        self.npu_manager = None

        logger.info(f"Plugin Registry initialized with dirs: {self.plugin_dirs}")

    def set_npu_manager(self, npu_manager):
        """Set NPU manager for plugin optimization"""
        self.npu_manager = npu_manager

    def discover_plugins(self) -> List[str]:
        """Discover available plugins"""
        discovered = []

        for plugin_dir in self.plugin_dirs:
            path = Path(plugin_dir)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                continue

            # Look for Python files
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    # Check if it's a valid plugin
                    module_name = py_file.stem
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Find plugin classes
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and
                                issubclass(obj, NEMWASPlugin) and
                                obj != NEMWASPlugin):
                                discovered.append(f"{plugin_dir}/{module_name}:{name}")

                except Exception as e:
                    logger.warning(f"Error discovering plugin in {py_file}: {e}")

        # Also check for installed packages with entry points
        try:
            import pkg_resources
            for entry_point in pkg_resources.iter_entry_points('nemwas.plugins'):
                discovered.append(f"entry_point:{entry_point.name}")
        except:
            pass

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    def load_plugin(self, plugin_path: str, context: Dict[str, Any] = None) -> bool:
        """Load a specific plugin"""

        try:
            if plugin_path.startswith("entry_point:"):
                # Load from entry point
                plugin_name = plugin_path.split(":", 1)[1]
                import pkg_resources
                for entry_point in pkg_resources.iter_entry_points('nemwas.plugins'):
                    if entry_point.name == plugin_name:
                        plugin_class = entry_point.load()
                        break
                else:
                    logger.error(f"Entry point not found: {plugin_name}")
                    return False
            else:
                # Load from file
                parts = plugin_path.split(":")
                if len(parts) != 2:
                    logger.error(f"Invalid plugin path format: {plugin_path}")
                    return False

                file_path, class_name = parts

                # Load module
                spec = importlib.util.spec_from_file_location("plugin_module", file_path)
                if not spec or not spec.loader:
                    logger.error(f"Could not load module spec: {file_path}")
                    return False

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Get plugin class
                plugin_class = getattr(module, class_name, None)
                if not plugin_class:
                    logger.error(f"Plugin class not found: {class_name}")
                    return False

            # Instantiate plugin
            plugin = plugin_class()

            # Initialize with context
            if not plugin.initialize(context or {}):
                logger.error(f"Plugin initialization failed: {plugin.metadata.name}")
                return False

            # Optimize for NPU if supported
            if plugin.metadata.npu_compatible and self.npu_manager:
                self._optimize_plugin_for_npu(plugin)

            # Register plugin by type
            plugin_name = plugin.metadata.name
            self.plugins[plugin_name] = plugin

            if isinstance(plugin, ToolPlugin):
                self.tool_plugins[plugin_name] = plugin
                self.pm.hook.on_tool_registered(tool_name=plugin_name, plugin=plugin)

            elif isinstance(plugin, CapabilityPlugin):
                self.capability_plugins[plugin_name] = plugin
                self.pm.hook.on_capability_registered(capability_name=plugin_name, plugin=plugin)

            elif isinstance(plugin, AnalyzerPlugin):
                self.analyzer_plugins[plugin_name] = plugin

            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            return False

    def _optimize_plugin_for_npu(self, plugin: NEMWASPlugin):
        """Optimize plugin models for NPU execution"""

        if not self.npu_manager:
            return

        try:
            # Check if plugin has a model to optimize
            if hasattr(plugin, 'get_model_path'):
                model_path = plugin.get_model_path()
                if model_path and Path(model_path).exists():
                    optimized_path = self.npu_manager.optimize_model_for_npu(
                        str(model_path),
                        model_type="plugin",
                        quantization_preset="mixed"
                    )

                    # Load optimized model
                    compiled_model = self.npu_manager.compile_model(optimized_path)
                    plugin.npu_model = compiled_model

                    logger.info(f"Plugin {plugin.metadata.name} optimized for NPU")

        except Exception as e:
            logger.warning(f"Could not optimize plugin for NPU: {e}")

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""

        if plugin_name not in self.plugins:
            logger.warning(f"Plugin not found: {plugin_name}")
            return False

        try:
            plugin = self.plugins[plugin_name]

            # Cleanup
            plugin.cleanup()

            # Remove from registries
            del self.plugins[plugin_name]
            self.tool_plugins.pop(plugin_name, None)
            self.capability_plugins.pop(plugin_name, None)
            self.analyzer_plugins.pop(plugin_name, None)

            logger.info(f"Unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    def get_plugin(self, plugin_name: str) -> Optional[NEMWASPlugin]:
        """Get a loaded plugin by name"""
        return self.plugins.get(plugin_name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins"""

        plugin_list = []
        for name, plugin in self.plugins.items():
            metadata = plugin.metadata
            plugin_list.append({
                'name': name,
                'version': metadata.version,
                'author': metadata.author,
                'description': metadata.description,
                'type': type(plugin).__name__,
                'npu_compatible': metadata.npu_compatible,
                'npu_optimized': plugin.npu_model is not None
            })

        return plugin_list

    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute a plugin"""

        plugin = self.plugins.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")

        try:
            return plugin.execute(*args, **kwargs)
        except Exception as e:
            logger.error(f"Plugin execution error ({plugin_name}): {e}")
            raise

    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all available tools from plugins"""

        tools = {}
        for name, plugin in self.tool_plugins.items():
            try:
                tool_def = plugin.get_tool_definition()
                tools[tool_def['name']] = tool_def
            except Exception as e:
                logger.error(f"Error getting tool from plugin {name}: {e}")

        return tools

    def save_plugin_state(self, filepath: str):
        """Save plugin registry state"""

        state = {
            'loaded_plugins': list(self.plugins.keys()),
            'plugin_metadata': {
                name: {
                    'version': plugin.metadata.version,
                    'npu_optimized': plugin.npu_model is not None
                }
                for name, plugin in self.plugins.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_plugin_state(self, filepath: str, context: Dict[str, Any] = None):
        """Load plugin registry state"""

        with open(filepath, 'r') as f:
            state = json.load(f)

        # Discover available plugins
        available = self.discover_plugins()

        # Load previously loaded plugins
        for plugin_name in state.get('loaded_plugins', []):
            # Find plugin path
            for plugin_path in available:
                if plugin_name in plugin_path:
                    self.load_plugin(plugin_path, context)
                    break


class PluginHooks:
    """Plugin system hooks"""

    @hookspec
    def on_plugin_loaded(self, plugin_name: str, plugin: NEMWASPlugin):
        """Called when a plugin is loaded"""
        pass

    @hookspec
    def on_plugin_unloaded(self, plugin_name: str):
        """Called when a plugin is unloaded"""
        pass

    @hookspec
    def on_tool_registered(self, tool_name: str, plugin: ToolPlugin):
        """Called when a tool plugin is registered"""
        pass

    @hookspec
    def on_capability_registered(self, capability_name: str, plugin: CapabilityPlugin):
        """Called when a capability plugin is registered"""
        pass
