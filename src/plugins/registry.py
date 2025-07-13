"""Plugin Registry for NEMWAS Framework"""

import os
import json
import logging
import importlib
import importlib.util
import inspect
import threading
from typing import Dict, List, Optional, Any, Callable, Type, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import time
import hashlib

import pluggy
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .interface import NEMWASPlugin, ToolPlugin, CapabilityPlugin, AnalyzerPlugin, PluginMetadata

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Information about a loaded plugin"""
    id: str
    name: str
    version: str
    path: str
    class_name: str
    plugin_type: str
    metadata: PluginMetadata
    instance: NEMWASPlugin
    load_time: float
    last_used: float
    usage_count: int = 0
    errors: List[str] = None
    npu_optimized: bool = False


class PluginFileWatcher(FileSystemEventHandler):
    """Watch plugin directories for changes"""

    def __init__(self, registry: 'PluginRegistry'):
        self.registry = registry
        self.debounce_timers = {}

    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith('.py'):
            return

        # Debounce to avoid multiple reloads
        if event.src_path in self.debounce_timers:
            self.debounce_timers[event.src_path].cancel()

        timer = threading.Timer(1.0, self._reload_plugin, args=[event.src_path])
        self.debounce_timers[event.src_path] = timer
        timer.start()

    def _reload_plugin(self, file_path: str):
        """Reload a modified plugin"""
        logger.info(f"Plugin file modified: {file_path}")

        # Find plugin by path
        plugin_info = None
        for info in self.registry.plugin_info.values():
            if info.path == file_path:
                plugin_info = info
                break

        if plugin_info:
            # Unload and reload
            self.registry.unload_plugin(plugin_info.id)
            self.registry.load_plugin_from_file(file_path)


class PluginRegistry:
    """Central registry for managing NEMWAS plugins"""

    def __init__(self,
                 plugin_dirs: List[Union[str, Path]] = None,
                 auto_discover: bool = True,
                 enable_hot_reload: bool = True,
                 cache_dir: str = "./data/plugin_cache"):

        self.plugin_dirs = [Path(d) for d in (plugin_dirs or [])]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Plugin storage
        self.plugins: Dict[str, NEMWASPlugin] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}

        # Categorized plugins
        self.tool_plugins: Dict[str, ToolPlugin] = {}
        self.capability_plugins: Dict[str, CapabilityPlugin] = {}
        self.analyzer_plugins: Dict[str, AnalyzerPlugin] = {}

        # Plugin hooks
        self.pm = pluggy.PluginManager("nemwas")

        # Thread safety
        self.lock = threading.RLock()

        # NPU manager reference
        self.npu_manager = None

        # File watcher for hot reload
        self.observer = None
        if enable_hot_reload:
            self._setup_file_watcher()

        # Plugin cache for faster loading
        self.plugin_cache = self._load_plugin_cache()

        # Statistics
        self.stats = {
            'plugins_loaded': 0,
            'plugins_failed': 0,
            'total_executions': 0,
            'total_errors': 0
        }

        # Auto-discover plugins
        if auto_discover:
            self.discover_and_load_plugins()

        logger.info(f"Plugin Registry initialized with dirs: {self.plugin_dirs}")

    def set_npu_manager(self, npu_manager):
        """Set NPU manager for plugin optimization"""
        self.npu_manager = npu_manager

        # Optimize already loaded plugins
        for plugin_info in self.plugin_info.values():
            if plugin_info.metadata.npu_compatible and not plugin_info.npu_optimized:
                self._optimize_plugin_for_npu(plugin_info)

    def discover_plugins(self) -> List[Dict[str, str]]:
        """Discover available plugins in configured directories"""
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                plugin_dir.mkdir(parents=True, exist_ok=True)
                continue

            # Scan for Python files
            for py_file in plugin_dir.rglob("*.py"):
                if py_file.name.startswith(('_', '.', 'test_')):
                    continue

                # Check cache first
                file_hash = self._get_file_hash(py_file)
                cache_key = f"{py_file.stem}_{file_hash}"

                if cache_key in self.plugin_cache:
                    # Use cached plugin info
                    cached_info = self.plugin_cache[cache_key]
                    discovered.append({
                        'path': str(py_file),
                        'name': cached_info['name'],
                        'class': cached_info['class'],
                        'type': cached_info['type']
                    })
                    continue

                # Discover plugin classes in file
                try:
                    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Find plugin classes
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and
                                issubclass(obj, NEMWASPlugin) and
                                obj not in [NEMWASPlugin, ToolPlugin, CapabilityPlugin, AnalyzerPlugin]):

                                plugin_type = self._get_plugin_type(obj)
                                discovered.append({
                                    'path': str(py_file),
                                    'name': name,
                                    'class': name,
                                    'type': plugin_type
                                })

                                # Update cache
                                self.plugin_cache[cache_key] = {
                                    'name': name,
                                    'class': name,
                                    'type': plugin_type
                                }

                except Exception as e:
                    logger.warning(f"Error discovering plugins in {py_file}: {e}")

        # Check for pip-installed plugins
        discovered.extend(self._discover_entry_point_plugins())

        return discovered

    def load_plugin(self, plugin_path: str, plugin_class: str = None) -> Optional[str]:
        """Load a specific plugin"""

        with self.lock:
            try:
                if plugin_path.startswith("entry_point:"):
                    return self._load_entry_point_plugin(plugin_path)
                else:
                    return self.load_plugin_from_file(plugin_path, plugin_class)

            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_path}: {e}")
                self.stats['plugins_failed'] += 1
                return None

    def load_plugin_from_file(self, file_path: str, plugin_class: str = None) -> Optional[str]:
        """Load plugin from Python file"""

        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Plugin file not found: {file_path}")
            return None

        try:
            # Load module
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if not spec or not spec.loader:
                raise ImportError(f"Cannot load module from {file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin class
            if plugin_class:
                plugin_cls = getattr(module, plugin_class, None)
                if not plugin_cls:
                    raise AttributeError(f"Plugin class '{plugin_class}' not found in {file_path}")
            else:
                # Auto-detect plugin class
                plugin_cls = None
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                        issubclass(obj, NEMWASPlugin) and
                        obj not in [NEMWASPlugin, ToolPlugin, CapabilityPlugin, AnalyzerPlugin]):
                        plugin_cls = obj
                        plugin_class = name
                        break

                if not plugin_cls:
                    # Check for 'plugin_class' attribute
                    if hasattr(module, 'plugin_class'):
                        plugin_cls = module.plugin_class
                        plugin_class = plugin_cls.__name__
                    else:
                        raise ValueError(f"No plugin class found in {file_path}")

            # Instantiate plugin
            plugin_instance = plugin_cls()

            # Get metadata
            metadata = plugin_instance.get_metadata()

            # Generate plugin ID
            plugin_id = f"{metadata.name}_{file_path.stem}_{int(time.time())}"

            # Initialize plugin
            context = self._create_plugin_context()
            if not plugin_instance.initialize(context):
                raise RuntimeError(f"Plugin initialization failed: {metadata.name}")

            # Create plugin info
            plugin_info = PluginInfo(
                id=plugin_id,
                name=metadata.name,
                version=metadata.version,
                path=str(file_path),
                class_name=plugin_class,
                plugin_type=self._get_plugin_type(plugin_cls),
                metadata=metadata,
                instance=plugin_instance,
                load_time=time.time(),
                last_used=time.time(),
                errors=[]
            )

            # Register plugin
            self.plugins[plugin_id] = plugin_instance
            self.plugin_info[plugin_id] = plugin_info

            # Categorize plugin
            if isinstance(plugin_instance, ToolPlugin):
                self.tool_plugins[plugin_id] = plugin_instance
            elif isinstance(plugin_instance, CapabilityPlugin):
                self.capability_plugins[plugin_id] = plugin_instance
            elif isinstance(plugin_instance, AnalyzerPlugin):
                self.analyzer_plugins[plugin_id] = plugin_instance

            # Optimize for NPU if supported
            if metadata.npu_compatible and self.npu_manager:
                self._optimize_plugin_for_npu(plugin_info)

            # Call hook
            self.pm.hook.on_plugin_loaded(plugin_id=plugin_id, plugin=plugin_instance)

            self.stats['plugins_loaded'] += 1
            logger.info(f"Successfully loaded plugin: {metadata.name} (ID: {plugin_id})")

            return plugin_id

        except Exception as e:
            logger.error(f"Error loading plugin from {file_path}: {e}")
            return None

    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""

        with self.lock:
            if plugin_id not in self.plugins:
                logger.warning(f"Plugin not found: {plugin_id}")
                return False

            try:
                plugin = self.plugins[plugin_id]
                plugin_info = self.plugin_info[plugin_id]

                # Call cleanup
                plugin.cleanup()

                # Remove from registries
                del self.plugins[plugin_id]
                del self.plugin_info[plugin_id]

                # Remove from category registries
                self.tool_plugins.pop(plugin_id, None)
                self.capability_plugins.pop(plugin_id, None)
                self.analyzer_plugins.pop(plugin_id, None)

                # Call hook
                self.pm.hook.on_plugin_unloaded(plugin_id=plugin_id)

                logger.info(f"Unloaded plugin: {plugin_info.name} (ID: {plugin_id})")
                return True

            except Exception as e:
                logger.error(f"Error unloading plugin {plugin_id}: {e}")
                return False

    def get_plugin(self, plugin_id: str) -> Optional[NEMWASPlugin]:
        """Get plugin instance by ID"""
        return self.plugins.get(plugin_id)

    def get_plugin_by_name(self, name: str) -> Optional[NEMWASPlugin]:
        """Get plugin instance by name"""
        for plugin_id, info in self.plugin_info.items():
            if info.name == name:
                return self.plugins[plugin_id]
        return None

    def execute_plugin(self, plugin_id: str, *args, **kwargs) -> Any:
        """Execute a plugin with error handling"""

        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_id}")

        plugin_info = self.plugin_info[plugin_id]

        try:
            # Update usage stats
            plugin_info.last_used = time.time()
            plugin_info.usage_count += 1
            self.stats['total_executions'] += 1

            # Execute plugin
            result = plugin.execute(*args, **kwargs)

            return result

        except Exception as e:
            # Record error
            error_msg = f"Plugin execution error: {str(e)}"
            plugin_info.errors = plugin_info.errors or []
            plugin_info.errors.append({
                'timestamp': time.time(),
                'error': error_msg
            })
            self.stats['total_errors'] += 1

            logger.error(f"Plugin {plugin_id} execution failed: {e}")
            raise

    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all available tools from tool plugins"""

        tools = {}

        for plugin_id, plugin in self.tool_plugins.items():
            try:
                tool_def = plugin.get_tool_definition()
                tools[tool_def['name']] = {
                    **tool_def,
                    'plugin_id': plugin_id
                }
            except Exception as e:
                logger.error(f"Error getting tool from plugin {plugin_id}: {e}")

        return tools

    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Get all available capabilities from capability plugins"""

        capabilities = []

        for plugin_id, plugin in self.capability_plugins.items():
            try:
                patterns = plugin.get_capability_patterns()
                for pattern in patterns:
                    capabilities.append({
                        **pattern,
                        'plugin_id': plugin_id
                    })
            except Exception as e:
                logger.error(f"Error getting capabilities from plugin {plugin_id}: {e}")

        return capabilities

    def list_plugins(self, plugin_type: str = None) -> List[Dict[str, Any]]:
        """List all loaded plugins"""

        plugins = []

        for plugin_id, info in self.plugin_info.items():
            if plugin_type and info.plugin_type != plugin_type:
                continue

            plugins.append({
                'id': plugin_id,
                'name': info.name,
                'version': info.version,
                'type': info.plugin_type,
                'author': info.metadata.author,
                'description': info.metadata.description,
                'npu_compatible': info.metadata.npu_compatible,
                'npu_optimized': info.npu_optimized,
                'usage_count': info.usage_count,
                'last_used': info.last_used,
                'errors': len(info.errors) if info.errors else 0
            })

        return plugins

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""

        return {
            **self.stats,
            'plugins_by_type': {
                'tool': len(self.tool_plugins),
                'capability': len(self.capability_plugins),
                'analyzer': len(self.analyzer_plugins),
                'other': len(self.plugins) - len(self.tool_plugins) - len(self.capability_plugins) - len(self.analyzer_plugins)
            },
            'npu_optimized': sum(1 for info in self.plugin_info.values() if info.npu_optimized),
            'total_usage': sum(info.usage_count for info in self.plugin_info.values()),
            'cache_size': len(self.plugin_cache)
        }

    def discover_and_load_plugins(self, auto_load_patterns: List[str] = None):
        """Discover and load plugins based on patterns"""

        discovered = self.discover_plugins()
        loaded = []

        for plugin_desc in discovered:
            # Check if should auto-load
            if auto_load_patterns:
                should_load = any(
                    pattern in plugin_desc['name'] or pattern in plugin_desc['path']
                    for pattern in auto_load_patterns
                )
                if not should_load:
                    continue

            # Load plugin
            plugin_id = self.load_plugin(plugin_desc['path'], plugin_desc['class'])
            if plugin_id:
                loaded.append(plugin_id)

        logger.info(f"Auto-loaded {len(loaded)} plugins")
        return loaded

    def _get_plugin_type(self, plugin_class: Type) -> str:
        """Determine plugin type from class"""
        if issubclass(plugin_class, ToolPlugin):
            return "tool"
        elif issubclass(plugin_class, CapabilityPlugin):
            return "capability"
        elif issubclass(plugin_class, AnalyzerPlugin):
            return "analyzer"
        else:
            return "generic"

    def _create_plugin_context(self) -> Dict[str, Any]:
        """Create context for plugin initialization"""
        return {
            'registry': self,
            'npu_manager': self.npu_manager,
            'cache_dir': self.cache_dir,
            'plugin_dirs': self.plugin_dirs
        }

    def _optimize_plugin_for_npu(self, plugin_info: PluginInfo):
        """Optimize plugin for NPU if possible"""

        if not self.npu_manager or plugin_info.npu_optimized:
            return

        try:
            plugin = plugin_info.instance

            # Check if plugin provides model path
            if hasattr(plugin, 'get_model_path'):
                model_path = plugin.get_model_path()
                if model_path and Path(model_path).exists():
                    # Optimize model
                    optimized_path = self.npu_manager.optimize_model_for_npu(
                        str(model_path),
                        model_type="plugin",
                        quantization_preset="mixed"
                    )

                    # Load optimized model
                    if hasattr(plugin, 'set_optimized_model'):
                        compiled_model = self.npu_manager.compile_model(optimized_path)
                        plugin.set_optimized_model(compiled_model)
                        plugin_info.npu_optimized = True
                        logger.info(f"Plugin {plugin_info.name} optimized for NPU")

        except Exception as e:
            logger.warning(f"Could not optimize plugin {plugin_info.name} for NPU: {e}")

    def _setup_file_watcher(self):
        """Setup file watcher for hot reload"""

        self.observer = Observer()
        handler = PluginFileWatcher(self)

        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                self.observer.schedule(handler, str(plugin_dir), recursive=True)

        self.observer.start()
        logger.info("Plugin hot reload enabled")

    def _load_plugin_cache(self) -> Dict[str, Dict]:
        """Load plugin cache from disk"""

        cache_file = self.cache_dir / "plugin_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load plugin cache: {e}")

        return {}

    def _save_plugin_cache(self):
        """Save plugin cache to disk"""

        cache_file = self.cache_dir / "plugin_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.plugin_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save plugin cache: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for caching"""

        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def _discover_entry_point_plugins(self) -> List[Dict[str, str]]:
        """Discover plugins installed via pip"""

        discovered = []

        try:
            import pkg_resources

            for entry_point in pkg_resources.iter_entry_points('nemwas.plugins'):
                discovered.append({
                    'path': f"entry_point:{entry_point.name}",
                    'name': entry_point.name,
                    'class': entry_point.attrs[0] if entry_point.attrs else entry_point.name,
                    'type': 'unknown'
                })

        except ImportError:
            # pkg_resources not available
            pass

        return discovered

    def _load_entry_point_plugin(self, plugin_path: str) -> Optional[str]:
        """Load plugin from entry point"""

        plugin_name = plugin_path.split(":", 1)[1]

        try:
            import pkg_resources

            for entry_point in pkg_resources.iter_entry_points('nemwas.plugins'):
                if entry_point.name == plugin_name:
                    plugin_cls = entry_point.load()

                    # Create temp info for loading
                    temp_path = f"entry_point_{plugin_name}"
                    return self.load_plugin_from_file(temp_path, plugin_cls.__name__)

            logger.error(f"Entry point plugin not found: {plugin_name}")

        except ImportError:
            logger.error("pkg_resources not available for entry point plugins")

        return None

    def cleanup(self):
        """Cleanup registry resources"""

        # Stop file watcher
        if self.observer and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()

        # Unload all plugins
        plugin_ids = list(self.plugins.keys())
        for plugin_id in plugin_ids:
            self.unload_plugin(plugin_id)

        # Save cache
        self._save_plugin_cache()

        logger.info("Plugin registry cleaned up")
