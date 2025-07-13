"""Example plugin: Web Search Tool for NEMWAS agents"""

import logging
import requests
from typing import Dict, Any, List
from urllib.parse import quote_plus

from src.plugins.interface import ToolPlugin, PluginMetadata

logger = logging.getLogger(__name__)


class WebSearchPlugin(ToolPlugin):
    """Plugin that provides web search capabilities to agents"""
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name="web_search",
            version="1.0.0",
            author="NEMWAS Team",
            description="Provides web search capabilities using DuckDuckGo",
            npu_compatible=False,  # This plugin doesn't use NPU
            requirements=["requests>=2.25.0"],
            capabilities=["search", "web", "information_retrieval"]
        )
    
    def __init__(self):
        """Initialize plugin"""
        super().__init__()
        self.base_url = "https://api.duckduckgo.com/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NEMWAS/1.0 (https://github.com/nemwas)'
        })
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize plugin with system context"""
        try:
            # Test API connectivity
            response = self.session.get(self.base_url, params={'q': 'test', 'format': 'json'})
            response.raise_for_status()
            
            logger.info("Web Search plugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Web Search plugin: {e}")
            return False
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return tool definition for agent registration"""
        return {
            'name': 'web_search',
            'description': 'Search the web for information using DuckDuckGo',
            'function': self.search,
            'parameters': {
                'query': 'str - The search query',
                'max_results': 'int - Maximum number of results (default: 5)'
            }
        }
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin functionality (delegates to search)"""
        return self.search(*args, **kwargs)
    
    def search(self, query: str, max_results: int = 5) -> str:
        """
        Perform web search
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            Formatted search results
        """
        try:
            # Encode query
            encoded_query = quote_plus(query)
            
            # Search using DuckDuckGo Instant Answer API
            params = {
                'q': encoded_query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format results
            results = []
            
            # Abstract (summary)
            if data.get('Abstract'):
                results.append(f"Summary: {data['Abstract']}")
                if data.get('AbstractURL'):
                    results.append(f"Source: {data['AbstractURL']}")
            
            # Instant answer
            if data.get('Answer'):
                results.append(f"Answer: {data['Answer']}")
                if data.get('AnswerType'):
                    results.append(f"Type: {data['AnswerType']}")
            
            # Related topics
            related = data.get('RelatedTopics', [])
            if related:
                results.append("\nRelated information:")
                for i, topic in enumerate(related[:max_results], 1):
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append(f"{i}. {topic['Text']}")
                        if topic.get('FirstURL'):
                            results.append(f"   URL: {topic['FirstURL']}")
            
            # Definition
            if data.get('Definition'):
                results.append(f"\nDefinition: {data['Definition']}")
                if data.get('DefinitionURL'):
                    results.append(f"Source: {data['DefinitionURL']}")
            
            if not results:
                # Try a different search approach or API
                results.append(f"No direct results found for '{query}'.")
                results.append("Consider rephrasing your search or trying more specific terms.")
            
            return "\n".join(results)
            
        except requests.exceptions.Timeout:
            return "Search timed out. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"Search request failed: {e}")
            return f"Search failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return f"An error occurred during search: {str(e)}"
    
    def cleanup(self):
        """Cleanup plugin resources"""
        self.session.close()
        logger.info("Web Search plugin cleaned up")


# Plugin entry point
plugin_class = WebSearchPlugin
