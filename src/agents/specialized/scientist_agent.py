from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from src.core.tools.physics_tools import run_physics_simulation

class ScientistAgent:
    """
    The orchestrator that generates hypotheses and initial state tensors 
    for the differentiable physics engine.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        # 1. Initialize the ADK Agent (The Brain)
        self.agent = Agent(
            name="ScientistAgent",
            model=model_name,
            tools=[run_physics_simulation],
            instruction=self._get_system_prompt()
        )
        
        # 2. Initialize the Session Service (The Memory Database)
        # In local dev on your M4, we keep this in RAM. 
        # In cloud production, you swap this one line for Firestore or Vertex AI.
        self.session_service = InMemorySessionService()
        self.app_name = "ProjectNoether"
        
        # 3. Initialize the Runner with the Session Service
        self.runner = Runner(
            agent=self.agent, 
            app_name=self.app_name,
            session_service=self.session_service
        )
        print(f"🔬 Scientist Agent initialized with {model_name} and connected to Physics Engine.")

    def _get_system_prompt(self) -> str:
        return """
        You are a Senior Theoretical Physicist controlling a differentiable physics engine.
        Your goal is to test the Out-of-Distribution (OOD) robustness of the generative model 
        by simulating chaotic, high-energy n-body interactions.

        RULES OF ENGAGEMENT:
        1. You must use the `run_physics_simulation` tool to test your hypotheses.
        2. When using the tool, provide strictly valid JSON matching the required schema.
        3. Do NOT set mass <= 0. Do NOT set velocities to astronomical numbers.
        4. If the tool returns a ValidationError, read the error, correct your mathematical parameters, and try again.
        
        Focus on complex boundary cases: extreme mass ratios or high-velocity collisions.
        """

    async def run_experiment(self, objective: str) -> str:
        print(f"\n[Scientist] Beginning experiment: {objective}")
        
        user_id = "research_user"
        session_id = "sim_session_001"
        
        # 4. Explicitly CREATE the session in the memory database before running!
        await self.session_service.create_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        # 5. Construct the strictly typed message payload
        content = types.Content(
            role="user", 
            parts=[types.Part(text=f"Design and execute a simulation for the following objective: {objective}")]
        )
        
        # 6. Execute the async ReAct loop
        final_text = "Simulation concluded without a final text response."
        async for event in self.runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = event.content.parts[0].text
                break
                
        return final_text