import os
import sys
import asyncio
import uuid
from dotenv import load_dotenv

# --- Path Resolution & Environment ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

# --- Import Agents ---
from src.agents.specialized.scientist_agent import ScientistAgent
from src.agents.specialized.validator_agent import ValidatorAgent

async def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is missing! Check your .env file.")

    # Initialize our dual-agent pipeline
    scientist = ScientistAgent()
    validator = ValidatorAgent()

    # --- THE OBJECTIVE ---
    # We explicitly ask for velocities UNDER the 15.0 hallucination threshold 
    # so it passes the OOD guardrail on the first try.
    # We explicitly ask for a maximum TOTAL vector magnitude under 10.0
    base_objective = "Simulate a stable 3-body system where one body is 100x more massive than the other two. To prevent relativistic hallucinations, ensure the TOTAL magnitude of any initial velocity vector (sqrt(vx^2 + vy^2 + vz^2)) is strictly less than 10.0. For example, use small components like [5.0, 0.0, 0.0]."
    current_prompt = base_objective
    max_retries = 3
    attempt = 1
    success = False

    print("\n" + "="*50)
    print("🚀 INITIATING AUTONOMOUS EXPERIMENT LOOP")
    print("="*50)

    while attempt <= max_retries and not success:
        print(f"\n--- 🔄 ATTEMPT {attempt} OF {max_retries} ---")
        
        # 1. Generation Phase (Scientist)
        scientist_report = await scientist.run_experiment(current_prompt)
        print(f"\n[Scientist Output]\n{scientist_report}\n")

        # 2. Validation Phase (Validator)
        validation_report = await validator.run_validation(scientist_report)
        print(f"\n[Validator Output]\n{validation_report}\n")

        # 3. Semantic Gradient Descent (The Orchestrator)
        if "approved" in validation_report.lower() or "physically sound" in validation_report.lower() or "passed" in validation_report.lower():
            print("\n✅ PIPELINE SUCCESS: The simulation passed all OOD guardrails!")
            success = True
        else:
            print("\n❌ PIPELINE FAILURE: OOD Hallucination detected.")
            if attempt < max_retries:
                print("   -> Feeding Validator critique back to the Scientist for self-correction...")
                current_prompt = f"""
                Your previous simulation failed validation. Here is the strict feedback from the Physics Validator:
                {validation_report}
                
                You pushed the model into an Out-of-Distribution state resulting in an energy hallucination. 
                Please dial back the extreme parameters and strictly ensure velocities do not exceed 14.0 to achieve the original objective safely: 
                {base_objective}
                """
            attempt += 1

    if not success:
        print("\n🛑 FATAL ERROR: Maximum ReAct retries reached. The Scientist could not stabilize the simulation.")

if __name__ == "__main__":
    asyncio.run(main())