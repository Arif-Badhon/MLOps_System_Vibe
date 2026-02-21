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

    scientist = ScientistAgent()
    validator = ValidatorAgent()

    # The inherently chaotic objective
    base_objective = "Simulate a 3-body system where the particles are colliding at extremely high, near-relativistic velocities (e.g., velocity components > 20.0)."
    
    # We maintain the prompt dynamically
    current_prompt = base_objective
    
    max_retries = 3
    attempt = 1
    success = False

    print("\n" + "="*50)
    print("🚀 INITIATING AUTONOMOUS EXPERIMENT LOOP")
    print("="*50)

    while attempt <= max_retries and not success:
        print(f"\n--- 🔄 ATTEMPT {attempt} OF {max_retries} ---")
        
        # 1. The Scientist generates and runs the simulation
        scientist_report = await scientist.run_experiment(current_prompt)
        print(f"\n[Scientist Output]\n{scientist_report}\n")

        # 2. The Validator audits the results
        validation_report = await validator.run_validation(scientist_report)
        print(f"\n[Validator Output]\n{validation_report}\n")

        # 3. The Orchestration Logic (Semantic Gradient Descent)
        # We check the Validator's tone/keywords to see if it passed or failed.
        if "approved" in validation_report.lower() or "physically sound" in validation_report.lower():
            print("\n✅ PIPELINE SUCCESS: The simulation passed all OOD guardrails.")
            success = True
        else:
            print("\n❌ PIPELINE FAILURE: OOD Hallucination detected.")
            if attempt < max_retries:
                print("   -> Feeding Validator critique back to the Scientist for self-correction...")
                # The "Gradient Update": We construct a new prompt with the error feedback
                current_prompt = f"""
                Your previous simulation failed validation. Here is the strict feedback from the Physics Validator:
                {validation_report}
                
                You pushed the model into an Out-of-Distribution state resulting in an energy hallucination. 
                Please dial back the extreme parameters (e.g., reduce the initial velocities) and try to achieve the original objective safely: 
                {base_objective}
                """
            attempt += 1

    if not success:
        print("\n🛑 FATAL ERROR: Maximum ReAct retries reached. The Scientist could not stabilize the simulation.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())