import torch
from typing import Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ValidationError

# ==========================================
# 1. Pydantic Schemas (The Contract)
# ==========================================

class ParticleState(BaseModel):
    particle_id: str = Field(..., description="Unique identifier for the particle.")
    mass: float = Field(..., gt=0.0, description="Mass of the particle (must be > 0).")
    position: List[float] = Field(..., min_length=3, max_length=3, description="[x, y, z] coordinates.")
    velocity: List[float] = Field(..., min_length=3, max_length=3, description="[vx, vy, vz] velocity vector.")

    @field_validator('position', 'velocity')
    def check_dimensions(cls, v):
        if len(v) != 3:
            raise ValueError("Vectors must be exactly 3-dimensional.")
        return v

class InitialState(BaseModel):
    simulation_name: str = Field(..., description="Name of the hypothesized scenario.")
    particles: List[ParticleState] = Field(..., min_length=2, description="At least two bodies are required.")
    simulation_steps: int = Field(1000, ge=10, le=10000, description="Number of timesteps to simulate.")
    dt: float = Field(0.01, gt=0.0, description="Time step delta.")

class PhysicsMetrics(BaseModel):
    initial_energy: float = Field(..., description="Total system energy at t=0.")
    final_energy: float = Field(..., description="Total system energy at t=N.")
    energy_drift: float = Field(..., description="Absolute difference between initial and final energy.")
    momentum_conserved: bool = Field(..., description="True if total momentum vector remained constant.")

class SimulationResult(BaseModel):
    simulation_id: str
    status: str
    metrics: PhysicsMetrics
    trajectory_reference_path: str

# ==========================================
# 2. Mock Deep Learning Engine (MPS Optimized)
# ==========================================
class DifferentiablePhysicsEngine:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🧠 Physics Engine initialized on: {self.device}")

    def simulate(self, state: dict) -> dict:
        dummy_tensor = torch.randn((state['simulation_steps'], len(state['particles']), 3), device=self.device)
        initial_e = torch.abs(torch.randn(1, device=self.device)).item() * 100
        drift = 0.05 

        return {
            "simulation_id": "sim_" + str(torch.randint(1000, 9999, (1,)).item()),
            "status": "completed",
            "metrics": {
                "initial_energy": initial_e,
                "final_energy": initial_e + drift,
                "energy_drift": drift,
                "momentum_conserved": True
            },
            "trajectory_reference_path": f"/data/memory/trajectories/sim_mock.pt"
        }

physics_engine = DifferentiablePhysicsEngine()

# ==========================================
# 3. Google ADK Tools (Standard Python Functions)
# ==========================================

def run_physics_simulation(simulation_parameters: str) -> str:
    """
    Executes a high-dimensional physics simulation based on provided JSON parameters.
    
    Args:
        simulation_parameters (str): A JSON string matching the InitialState schema.
    Returns:
        str: A JSON string containing the SimulationResult, or an error message if inputs are non-physical.
    """
    try:
        import json
        raw_dict = json.loads(simulation_parameters)
        validated_state = InitialState(**raw_dict)
        
        result_dict = physics_engine.simulate(validated_state.model_dump())
        
        validated_result = SimulationResult(**result_dict)
        return validated_result.model_dump_json()

    except ValidationError as e:
        return f"PHYSICS VIOLATION OR SCHEMA ERROR: {e.json()}"
    except Exception as e:
        return f"SIMULATION FAILED: {str(e)}"

def validate_conservation_laws(metrics_json: str, tolerance: float = 0.1) -> str:
    """
    Checks if the generative model violated fundamental physics (e.g., energy drift).
    
    Args:
        metrics_json (str): JSON string of PhysicsMetrics.
        tolerance (float): Maximum allowed energy drift (epsilon).
    """
    import json
    metrics = json.loads(metrics_json)
    
    drift = metrics.get("energy_drift", float('inf'))
    if drift > tolerance:
        return f"OOD DETECTED: Energy drift ({drift}) exceeded tolerance ({tolerance})."
    
    if not metrics.get("momentum_conserved"):
        return "OOD DETECTED: Momentum was not conserved."
        
    return "VALIDATION PASSED: Trajectory is physically sound."