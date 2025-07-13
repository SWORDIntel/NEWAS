# Plugin Development Guide

This guide explains how to create custom plugins for NEMWAS. Plugins can extend the functionality of the system by adding new tools, capabilities, or analyzers.

## Plugin Structure

A plugin is a single Python file that contains a class that inherits from one of the plugin base classes. The plugin file must be placed in the `plugins` directory to be discovered by the system.

### Base Classes

- `ToolPlugin`: For creating new tools that agents can use.
- `CapabilityPlugin`: For adding new capabilities to the system.
- `AnalyzerPlugin`: For creating new performance analyzers.

### Example: Tool Plugin

Here is an example of a simple tool plugin that adds a `hello` tool:

```python
# plugins/my_plugin.py
from src.plugins.interface import ToolPlugin, PluginMetadata

class MyPlugin(ToolPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_tool",
            version="1.0.0",
            author="Your Name",
            description="A simple example plugin.",
            npu_compatible=False
        )

    def get_tool_definition(self):
        return {
            'name': 'hello',
            'description': 'A simple tool that says hello.',
            'function': self.execute,
            'parameters': {'name': 'str'}
        }

    def execute(self, name: str) -> str:
        return f"Hello, {name}!"

plugin_class = MyPlugin
```

## Metadata

The `get_metadata` method should return a `PluginMetadata` object with the following information:

- `name`: The name of the plugin.
- `version`: The version of the plugin.
- `author`: The author of the plugin.
- `description`: A brief description of the plugin.
- `npu_compatible`: Whether the plugin is compatible with the NPU.

## Tool Definition

The `get_tool_definition` method should return a dictionary with the following information:

- `name`: The name of the tool.
- `description`: A brief description of the tool.
- `function`: The function to execute when the tool is called.
- `parameters`: A dictionary of the tool's parameters, with the parameter name as the key and the type as the value.

## Loading Plugins

Plugins are loaded at startup from the `plugins` directory. You can also load plugins at runtime using the `/plugins/load` API endpoint.

## Best Practices

- Keep plugins small and focused on a single task.
- Use descriptive names for plugins and tools.
- Add clear and concise descriptions for plugins and tools.
- Handle errors gracefully and provide informative error messages.
- Write unit tests for your plugins.
