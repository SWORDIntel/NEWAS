"""Dynamic plugin loader for NEMWAS"""

import os
import sys
import importlib
import importlib.util
import inspect
import logging
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import json
import traceback

from .interface import NEMWASPlugin, ToolPlugin, CapabilityPlugin, AnalyzerPlugin, PluginMetadata

logger = logging.getLogger(__name__)


class PluginLoader:
    """Handles dynamic loading and management of plugins"""

    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or ["./plugins/builtin", "./plugins/community"]
        self.loaded_modules = {}
        self.failed_plugins = {}

    def scan_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Scan a directory for potential plugins"""

        plugins = []
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return plugins

        # Look for Python files
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                plugin_info = self._inspect_plugin_file(py_file)
                if plugin_info:
                    plugins.extend(plugin_info)
            except Exception as e:
                logger.error(f"Error scanning {py_file}: {e}")
                self.failed_plugins[str(py_file)] = str(e)

        # Look for plugin packages (directories with __init__.py)
        for subdir in dir_path.iterdir():
            if subdir.is_dir() and (subdir / "__init__.py").exists():
                try:
                    plugin_info = self._inspect_plugin_package(subdir)
                    if plugin_info:
                        plugins.extend(plugin_info)
                except Exception as e:
                    logger.error(f"Error scanning package {subdir}: {e}")
                    self.failed_plugins[str(subdir)] = str(e)

        return plugins

    def _inspect_plugin_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Inspect a Python file for plugin classes"""

        plugins = []

        # Load module spec
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if not spec or not spec.loader:
            return plugins

        # Create module
        module = importlib.util.module_from_spec(spec)

        # Execute module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"Failed to load module {file_path}: {e}")
            return plugins

        # Find plugin classes
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, NEMWASPlugin) and
                obj not in [NEMWASPlugin, ToolPlugin, CapabilityPlugin, AnalyzerPlugin]):

                plugins.append({
                    'name': name,
                    'class': obj,
                    'module': module,
                    'file_path': str(file_path),
                    'type': self._get_plugin_type(obj)
                })

        # Check for explicit plugin_class attribute
        if hasattr(module, 'plugin_class'):
            plugin_class = getattr(module, 'plugin_class')
            if inspect.isclass(plugin_class) and issubclass(plugin_class, NEMWASPlugin):
                # Check if already added
                if not any(p['class'] == plugin_class for p in plugins):
                    plugins.append({
                        'name': plugin_class.__name__,
                        'class': plugin_class,
                        'module': module,
                        'file_path': str(file_path),
                        'type': self._get_plugin_type(plugin_class)
                    })

        return plugins

    def _inspect_plugin_package(self, package_path: Path) -> List[Dict[str, Any]]:
        """Inspect a package directory for plugins"""

        plugins = []

        # Add package parent to sys.path temporarily
        parent_path = str(package_path.parent)
        if parent_path not in sys.path:
            sys.path.insert(0, parent_path)
            remove_path = True
        else:
            remove_path = False

        try:
            # Import package
            package_name = package_path.name
            module = importlib.import_module(package_name)

            # Look for plugins in __init__.py
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, NEMWASPlugin) and
                    obj not in [NEMWASPlugin, ToolPlugin, CapabilityPlugin, AnalyzerPlugin]):

                    plugins.append({
                        'name': name,
                        'class': obj,
                        'module': module,
                        'file_path': str(package_path),
                        'type': self._get_plugin_type(obj)
                    })

            # Check for plugin registry
            if hasattr(module, 'PLUGINS'):
                for plugin_class in module.PLUGINS:
                    if inspect.isclass(plugin_class) and issubclass(plugin_class, NEMWASPlugin):
                        plugins.append({
                            'name': plugin_class.__name__,
                            'class': plugin_class,
                            'module': module,
                            'file_path': str(package_path),
                            'type': self._get_plugin_type(plugin_class)
                        })

        finally:
            # Clean up sys.path
            if remove_path and parent_path in sys.path:
                sys.path.remove(parent_path)

        return plugins

    def _get_plugin_type(self, plugin_class: Type[NEMWASPlugin]) -> str:
        """Determine plugin type from class inheritance"""

        if issubclass(plugin_class, ToolPlugin):
            return "tool"
        elif issubclass(plugin_class, CapabilityPlugin):
            return "capability"
        elif issubclass(plugin_class, AnalyzerPlugin):
            return "analyzer"
        else:
            return "generic"

    def load_plugin(self, plugin_info: Dict[str, Any]) -> Optional[NEMWASPlugin]:
        """Load and instantiate a plugin"""

        try:
            plugin_class = plugin_info['class']
            plugin_instance = plugin_class()

            # Store module reference
            module_path = plugin_info['file_path']
            self.loaded_modules[module_path] = plugin_info['module']

            logger.info(f"Successfully loaded plugin: {plugin_info['name']} from {module_path}")
            return plugin_instance

        except Exception as e:
            logger.error(f"Failed to instantiate plugin {plugin_info['name']}: {e}")
            logger.debug(traceback.format_exc())
            return None

    def reload_plugin(self, file_path: str) -> List[NEMWASPlugin]:
        """Reload a plugin module (for hot reload)"""

        reloaded_plugins = []

        try:
            # Get existing module
            if file_path in self.loaded_modules:
                module = self.loaded_modules[file_path]

                # Reload module
                importlib.reload(module)

                # Re-scan for plugins
                plugin_infos = self._inspect_plugin_file(Path(file_path))

                # Load new instances
                for plugin_info in plugin_infos:
                    plugin_instance = self.load_plugin(plugin_info)
                    if plugin_instance:
                        reloaded_plugins.append(plugin_instance)

                logger.info(f"Reloaded {len(reloaded_plugins)} plugins from {file_path}")
            else:
                logger.warning(f"Module not loaded, cannot reload: {file_path}")

        except Exception as e:
            logger.error(f"Failed to reload plugin from {file_path}: {e}")

        return reloaded_plugins

    def discover_plugins(self) -> List[Dict[str, Any]]:
        """Discover all available plugins"""

        all_plugins = []

        for directory in self.plugin_dirs:
            plugins = self.scan_directory(directory)
            all_plugins.extend(plugins)

        # Also check for installed packages with entry points
        entry_point_plugins = self._discover_entry_points()
        all_plugins.extend(entry_point_plugins)

        logger.info(f"Discovered {len(all_plugins)} plugins total")
        return all_plugins

    def _discover_entry_points(self) -> List[Dict[str, Any]]:
        """Discover plugins installed as packages with entry points"""

        plugins = []

        try:
            if sys.version_info >= (3, 10):
                from importlib.metadata import entry_points
            else:
                from importlib_metadata import entry_points

            # Look for nemwas.plugins entry points
            discovered = entry_points(group='nemwas.plugins')

            for entry_point in discovered:
                try:
                    plugin_class = entry_point.load()

                    if inspect.isclass(plugin_class) and issubclass(plugin_class, NEMWASPlugin):
                        plugins.append({
                            'name': entry_point.name,
                            'class': plugin_class,
                            'module': inspect.getmodule(plugin_class),
                            'file_path': f"entry_point:{entry_point.name}",
                            'type': self._get_plugin_type(plugin_class)
                        })

                except Exception as e:
                    logger.error(f"Failed to load entry point {entry_point.name}: {e}")

        except ImportError:
            logger.debug("importlib.metadata not available, skipping entry point discovery")

        return plugins

    def validate_plugin(self, plugin_instance: NEMWASPlugin) -> bool:
        """Validate a plugin instance"""

        try:
            # Check metadata
            metadata = plugin_instance.get_metadata()
            if not isinstance(metadata, PluginMetadata):
                logger.error("Plugin metadata is not valid PluginMetadata instance")
                return False

            # Check required attributes
            if not metadata.name or not metadata.version:
                logger.error("Plugin missing required metadata: name or version")
                return False

            # Check required methods
            required_methods = ['initialize', 'execute']
            for method in required_methods:
                if not hasattr(plugin_instance, method) or not callable(getattr(plugin_instance, method)):
                    logger.error(f"Plugin missing required method: {method}")
                    return False

            # Type-specific validation
            if isinstance(plugin_instance, ToolPlugin):
                if not hasattr(plugin_instance, 'get_tool_definition'):
                    logger.error("ToolPlugin missing get_tool_definition method")
                    return False

            elif isinstance(plugin_instance, CapabilityPlugin):
                if not hasattr(plugin_instance, 'get_capability_patterns'):
                    logger.error("CapabilityPlugin missing get_capability_patterns method")
                    return False

            elif isinstance(plugin_instance, AnalyzerPlugin):
                if not hasattr(plugin_instance, 'analyze'):
                    logger.error("AnalyzerPlugin missing analyze method")
                    return False

            return True

        except Exception as e:
            logger.error(f"Plugin validation failed: {e}")
            return False

    def get_plugin_manifest(self, plugin_instance: NEMWASPlugin) -> Dict[str, Any]:
        """Get plugin manifest information"""

        try:
            metadata = plugin_instance.get_metadata()

            manifest = {
                'name': metadata.name,
                'version': metadata.version,
                'author': metadata.author,
                'description': metadata.description,
                'type': self._get_plugin_type(type(plugin_instance)),
                'npu_compatible': metadata.npu_compatible,
                'requirements': metadata.requirements or [],
                'capabilities': metadata.capabilities or [],
                'class_name': type(plugin_instance).__name__,
                'module_name': type(plugin_instance).__module__
            }

            # Add type-specific information
            if isinstance(plugin_instance, ToolPlugin):
                try:
                    tool_def = plugin_instance.get_tool_definition()
                    manifest['tool_name'] = tool_def.get('name')
                    manifest['tool_parameters'] = tool_def.get('parameters', {})
                except:
                    pass

            return manifest

        except Exception as e:
            logger.error(f"Failed to get plugin manifest: {e}")
            return {
                'name': 'unknown',
                'error': str(e)
            }

    def save_plugin_catalog(self, filepath: str):
        """Save discovered plugins to a catalog file"""

        catalog = {
            'version': '1.0',
            'plugins': [],
            'failed': self.failed_plugins,
            'directories': self.plugin_dirs
        }

        # Discover all plugins
        plugin_infos = self.discover_plugins()

        for plugin_info in plugin_infos:
            try:
                # Create temporary instance for metadata
                plugin_instance = plugin_info['class']()
                manifest = self.get_plugin_manifest(plugin_instance)
                manifest['file_path'] = plugin_info['file_path']
                catalog['plugins'].append(manifest)
            except Exception as e:
                logger.error(f"Failed to get manifest for {plugin_info['name']}: {e}")

        # Save catalog
        with open(filepath, 'w') as f:
            json.dump(catalog, f, indent=2)

        logger.info(f"Saved plugin catalog to {filepath}")

    def load_plugin_catalog(self, filepath: str) -> Dict[str, Any]:
        """Load plugin catalog from file"""

        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load plugin catalog: {e}")
            return {}


# Convenience functions
def discover_all_plugins(plugin_dirs: List[str] = None) -> List[Dict[str, Any]]:
    """Discover all available plugins"""
    loader = PluginLoader(plugin_dirs)
    return loader.discover_plugins()


def load_plugin_from_file(file_path: str) -> Optional[NEMWASPlugin]:
    """Load a single plugin from a file"""
    loader = PluginLoader()
    plugin_infos = loader._inspect_plugin_file(Path(file_path))

    if plugin_infos:
        return loader.load_plugin(plugin_infos[0])

    return None
