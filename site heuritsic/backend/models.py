from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class AlgorithmParams(BaseModel):
    pop_size: int = 30
    # DE
    F: Optional[float] = 0.5
    CR: Optional[float] = 0.9
    # jDE
    tau1: Optional[float] = 0.1
    tau2: Optional[float] = 0.1
    # PSO
    c1: Optional[float] = 2.0
    c2: Optional[float] = 2.0
    # PSO-H
    p_mut: Optional[float] = 0.1
    sigma: Optional[float] = 1.0
    # GA
    mutation_scale: Optional[float] = 1.0
    # JDE Adapted
    p_current_to_best: Optional[float] = 0.1
    ls_interval: Optional[int] = 50
    ls_max_evals: Optional[int] = 30
    # GSA
    G0: Optional[float] = 100.0
    alpha: Optional[float] = 20.0
    # ABC
    limit: Optional[int] = 50

class AlgorithmConfig(BaseModel):
    name: str
    params: AlgorithmParams

class OptimizationRequest(BaseModel):
    function: str
    dimension: int = Field(..., ge=2, le=100)
    max_fes: int = Field(..., ge=1000, le=1000000)
    lb: float = -100.0
    ub: float = 100.0
    seed: int = 0
    algorithms: List[AlgorithmConfig]

class OptimizationResultModel(BaseModel):
    algorithm: str
    best_value: float
    best_solution: List[float]
    history: List[float]
    fes_history: List[int]
    execution_time: float
    parameters: Dict[str, Any]
