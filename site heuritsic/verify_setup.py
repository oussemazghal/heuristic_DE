import sys
import os

# Add current directory to path so we can import backend
sys.path.append(os.getcwd())

try:
    print("Testing imports...")
    from backend.main import app
    from backend.algorithms import run_de, sphere
    from backend.models import OptimizationRequest, AlgorithmConfig, AlgorithmParams
    import numpy as np
    print("Imports successful.")

    print("Testing algorithm logic (DE on Sphere)...")
    rng = np.random.default_rng(42)
    res = run_de(sphere, 10, -5.0, 5.0, 1000, rng, 20, 0.5, 0.9)
    print(f"DE Result: {res.best_value}")
    assert res.best_value < 1.0, "DE failed to optimize Sphere"
    print("Algorithm logic verification successful.")

    print("All checks passed!")

except Exception as e:
    print(f"Verification FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
