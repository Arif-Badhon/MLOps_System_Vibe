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
from src.agents.specialized.validator_agent import ValidatorAgent

async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is missing! Check your .env file.")

    # Initialize our dual-agent pipeline
    scientist = ScientistAgent()
    validator = ValidatorAgent()

    # --- PHASE 1: GENERATION ---
    objective = "Simulate a 3-body system where one body is 100x more massive than the other two, and they start in a tight, high-velocity orbit."
    scientist_report = await scientist.run_experiment(objective)
    
    print("\n" + "="*50)
    print("🔬 SCIENTIST REPORT (GENERATION)")
    print("="*50)
    print(scientist_report)

    # --- PHASE 2: VALIDATION ---
    # We pipe the output of Agent A directly into the context window of Agent B
    validation_report = await validator.run_validation(scientist_report)

    print("\n" + "="*50)
    print("⚖️ VALIDATOR REPORT (OOD GUARDRAIL)")
    print("="*50)
    print(validation_report)

if __name__ == "__main__":
    asyncio.run(main())