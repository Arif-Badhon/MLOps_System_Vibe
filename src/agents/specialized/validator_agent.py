from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from src.core.tools.physics_tools import validate_conservation_laws
import uuid

class ValidatorAgent:
    """
    The strict OOD Guardrail. Analyzes simulation metrics to ensure 
    fundamental physics (energy, momentum) are not violated.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        # 1. Initialize the Brain
        self.agent = Agent(
            name="ValidatorAgent",
            model=model_name,
            tools=[validate_conservation_laws],
            instruction=self._get_system_prompt()
        )
        
        # 2. Initialize a separate Memory Database
        # We use a separate session service so its memory doesn't bleed into the Scientist's
        self.session_service = InMemorySessionService()
        self.app_name = "ProjectNoether_Validator"
        self.runner = Runner(
            agent=self.agent, 
            app_name=self.app_name, 
            session_service=self.session_service
        )
        print(f"⚖️ Validator Agent initialized with {model_name}.")

    def _get_system_prompt(self) -> str:
        return """
        You are an uncompromising Physics Validation Engine. 
        You will receive the output report and metrics of a generative physics simulation.
        
        RULES OF ENGAGEMENT:
        1. You must extract the JSON metrics from the simulation report.
        2. You must call the `validate_conservation_laws` tool to mathematically verify the drift.
        3. If the tool returns an OOD error, you must aggressively flag the simulation as a failure and explain exactly why the physical laws were violated.
        4. If the tool returns that the trajectory is sound, officially approve the simulation.
        5. Do NOT be polite. You are a strict mathematical guardrail.
        """

    async def run_validation(self, simulation_report: str) -> str:
        print("\n[Validator] Analyzing simulation metrics for OOD violations...")
        
        user_id = "validator_system"
        # GENERATE A DYNAMIC SESSION ID FOR EVERY ATTEMPT
        session_id = f"val_session_{uuid.uuid4().hex[:8]}"
        
        # 3. Create the session in the Validator's memory
        await self.session_service.create_session(
            app_name=self.app_name, 
            user_id=user_id, 
            session_id=session_id
        )
        
        # 4. Pass the Scientist's output as the user prompt
        content = types.Content(
            role="user", 
            parts=[types.Part(text=f"Extract the metrics and validate this simulation report:\n\n{simulation_report}")]
        )
        
        final_text = "Validation failed to generate a report."
        async for event in self.runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = event.content.parts[0].text
                break
                
        return final_text