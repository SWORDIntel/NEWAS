#!/usr/bin/env python3
"""
Simple example of using NEMWAS framework

This example demonstrates:
1. Creating an agent
2. Registering custom tools
3. Processing natural language queries
4. Tracking performance
"""

import asyncio
import logging
from pathlib import Path

# Add parent directory to path if running from examples folder
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.agent import NEMWASAgent, AgentConfig
from src.core.npu_manager import NPUManager
from src.core.react import Tool
from src.performance.tracker import PerformanceTracker
from src.nlp.interface import NaturalLanguageInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Custom tool example
def weather_tool(location: str) -> str:
    """Mock weather tool for demonstration"""
    # In a real implementation, this would call a weather API
    weather_data = {
        "london": "Partly cloudy, 18°C",
        "new york": "Sunny, 22°C", 
        "tokyo": "Rainy, 15°C",
        "paris": "Overcast, 16°C"
    }
    
    location_lower = location.lower()
    if location_lower in weather_data:
        return f"The weather in {location} is: {weather_data[location_lower]}"
    else:
        return f"Sorry, I don't have weather data for {location}. Try major cities like London, New York, Tokyo, or Paris."


def stock_price_tool(symbol: str) -> str:
    """Mock stock price tool for demonstration"""
    # In a real implementation, this would call a stock API
    stock_prices = {
        "AAPL": ("Apple Inc.", 185.92, 1.2),
        "GOOGL": ("Alphabet Inc.", 141.80, -0.5),
        "MSFT": ("Microsoft Corp.", 378.85, 2.1),
        "TSLA": ("Tesla Inc.", 238.45, -1.8)
    }
    
    symbol_upper = symbol.upper()
    if symbol_upper in stock_prices:
        name, price, change = stock_prices[symbol_upper]
        direction = "📈" if change > 0 else "📉"
        return f"{name} ({symbol_upper}): ${price:.2f} {direction} {change:+.1f}%"
    else:
        return f"Sorry, I don't have price data for {symbol}. Try symbols like AAPL, GOOGL, MSFT, or TSLA."


async def main():
    """Main example function"""
    
    print("🚀 NEMWAS Simple Example")
    print("=" * 50)
    
    # Initialize NPU Manager
    print("\n1️⃣ Initializing NPU Manager...")
    npu_manager = NPUManager()
    print(f"   Available devices: {npu_manager.available_devices}")
    print(f"   Selected device: {npu_manager.device_preference[0]}")
    
    # Create agent configuration
    print("\n2️⃣ Creating agent...")
    agent_config = AgentConfig(
        name="Example-Agent",
        model_path="./models/original/tinyllama-1.1b-chat.xml",  # Adjust path as needed
        device_preference=["NPU", "GPU", "CPU"],
        temperature=0.7,
        max_iterations=5
    )
    
    # Initialize agent
    try:
        agent = NEMWASAgent(agent_config, npu_manager)
        print(f"   ✅ Agent created: {agent.agent_id}")
        print(f"   Running on: {agent.device}")
    except FileNotFoundError:
        print("   ❌ Model file not found. Please run: python scripts/download_models.py --minimal")
        return
    
    # Register custom tools
    print("\n3️⃣ Registering custom tools...")
    
    # Weather tool
    agent.register_tool(Tool(
        name="check_weather",
        description="Check the weather for a given location",
        function=weather_tool,
        parameters={"location": "str"}
    ))
    print("   ✅ Registered: check_weather")
    
    # Stock price tool
    agent.register_tool(Tool(
        name="check_stock",
        description="Check the stock price for a given symbol",
        function=stock_price_tool,
        parameters={"symbol": "str"}
    ))
    print("   ✅ Registered: check_stock")
    
    # Initialize performance tracker
    print("\n4️⃣ Setting up performance tracking...")
    performance_tracker = PerformanceTracker()
    agent.performance_tracker = performance_tracker
    print("   ✅ Performance tracking enabled")
    
    # Example queries
    queries = [
        "What's 15 factorial?",
        "What's the weather in London?",
        "Check the stock price of AAPL",
        "Calculate the compound interest on $10,000 at 5% annual rate for 10 years",
        "What's the weather in Tokyo and the stock price of MSFT?"
    ]
    
    print("\n5️⃣ Processing example queries...")
    print("=" * 50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 50)
        
        try:
            # Process query
            result = await agent.process(query)
            print(f"✅ Result: {result}")
            
            # Show performance metrics
            metrics = agent.get_metrics()
            print(f"⏱️  Execution time: {metrics.get('avg_execution_time', 0):.3f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Show final statistics
    print("\n" + "=" * 50)
    print("📊 Session Statistics")
    print("=" * 50)
    
    final_metrics = agent.get_metrics()
    print(f"Total queries processed: {final_metrics['total_queries']}")
    print(f"Device used: {final_metrics['device']}")
    
    if performance_tracker:
        system_metrics = performance_tracker.get_system_metrics()
        print(f"Overall success rate: {system_metrics['overall_success_rate']:.1%}")
        print(f"Average execution time: {system_metrics['recent_avg_execution_time']:.3f}s")
    
    # Natural Language Interface example
    print("\n6️⃣ Natural Language Interface Demo...")
    nl_interface = NaturalLanguageInterface()
    
    nl_commands = [
        "create an agent for financial analysis",
        "show me the system status",
        "analyze performance trends"
    ]
    
    for command in nl_commands:
        intent = nl_interface.parse(command)
        print(f"\n📝 Command: '{command}'")
        print(f"   Intent: {intent.intent_type.value} (confidence: {intent.confidence:.2f})")
        print(f"   Entities: {intent.entities}")
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
