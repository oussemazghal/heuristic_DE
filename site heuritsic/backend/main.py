from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
import os

from .models import OptimizationRequest, OptimizationResultModel
from .algorithms import (
    get_function, CEC_FUNCTIONS,
    run_de, run_jde, run_jde_adapted, run_pso, run_pso_hybrid, run_ga, run_gsa, run_abc
)

app = FastAPI(title="Optimization Dashboard API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/functions")
def list_functions():
    """Return available benchmark functions."""
    return {"functions": list(CEC_FUNCTIONS.keys())}

@app.post("/optimize", response_model=List[OptimizationResultModel])
def run_optimization(request: OptimizationRequest):
    """Run selected optimization algorithms."""
    func = get_function(request.function)
    if not func:
        raise HTTPException(status_code=404, detail=f"Function {request.function} not found")

    results = []
    
    for algo_config in request.algorithms:
        name = algo_config.name
        params = algo_config.params
        
        # Create a unique RNG for each algorithm based on seed + index-like offset
        # We use a simple hash of name to offset the seed to ensure different algos get different streams
        # but deterministic for the same seed.
        algo_seed = request.seed + abs(hash(name)) % 10000
        rng = np.random.default_rng(algo_seed)

        try:
            if name == "DE":
                res = run_de(
                    func, request.dimension, request.lb, request.ub, request.max_fes,
                    rng, params.pop_size, params.F, params.CR
                )
            elif name == "jDE":
                res = run_jde(
                    func, request.dimension, request.lb, request.ub, request.max_fes,
                    rng, params.pop_size, params.tau1, params.tau2
                )
            elif name == "jDE-Adapted":
                res = run_jde_adapted(
                    func, request.dimension, request.lb, request.ub, request.max_fes,
                    rng, params.pop_size, params.tau1, params.tau2,
                    params.p_current_to_best, params.ls_interval, params.ls_max_evals
                )
            elif name == "PSO":
                res = run_pso(
                    func, request.dimension, request.lb, request.ub, request.max_fes,
                    rng, params.pop_size, params.c1, params.c2
                )
            elif name == "PSO-H":
                res = run_pso_hybrid(
                    func, request.dimension, request.lb, request.ub, request.max_fes,
                    rng, params.pop_size, params.c1, params.c2, params.p_mut, params.sigma
                )
            elif name == "GA":
                res = run_ga(
                    func, request.dimension, request.lb, request.ub, request.max_fes,
                    rng, params.pop_size, params.mutation_scale
                )
            elif name == "GSA":
                res = run_gsa(
                    func, request.dimension, request.lb, request.ub, request.max_fes,
                    rng, params.pop_size, params.G0, params.alpha
                )
            elif name == "ABC":
                res = run_abc(
                    func, request.dimension, request.lb, request.ub, request.max_fes,
                    rng, params.pop_size, params.limit
                )
            else:
                continue # Skip unknown
            
            results.append(res)
            
        except Exception as e:
            print(f"Error running {name}: {e}")
            # In a real app, we might want to return partial results or specific error objects
            # For now, we skip failed ones or re-raise if critical.
            continue

    return results

# Mount frontend static files
# We assume the frontend files are in ../frontend relative to this file
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
