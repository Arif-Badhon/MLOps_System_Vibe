import os
import sys
import asyncio
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

from src.agents.specialized.scientist_agent import ScientistAgent

async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is missing! Check your .env file.")

    print("Initializing Scientist Agent...")
    scientist = ScientistAgent()

    # Give it a chaotic prompt
    objective = "Simulate a 3-body system where one body is 100x more massive than the other two, and they start in a tight, high-velocity orbit."

    # Await the asynchronous execution
    result = await scientist.run_experiment(objective)
    
    print("\n=== FINAL EXPERIMENT REPORT ===")
    print(result)

if __name__ == "__main__":
    # Start the async event loop
    asyncio.run(main())