import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ============================================================================
#  CEC2017 IMPORT LOGIC
# ============================================================================

try:
    from cec2017.functions import (
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
        f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
        f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
    )

    def make_cec_wrapper(func):
        def wrapper(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            values = func(x)
            return float(values[0])
        return wrapper

    CEC_FUNCTIONS = {
        f'f{i}': make_cec_wrapper(func)
        for i, func in enumerate([
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
            f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
            f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
        ], 1)
    }
    CEC_AVAILABLE = True
except ImportError:
    CEC_AVAILABLE = False
    CEC_FUNCTIONS = {}

# Fallback functions
def sphere(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))

def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return float(10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(
        100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2
    ))

if not CEC_AVAILABLE:
    CEC_FUNCTIONS.update({
        "sphere": sphere,
        "rastrigin": rastrigin,
        "rosenbrock": rosenbrock
    })

def get_function(name: str):
    return CEC_FUNCTIONS.get(name)

# ============================================================================
#  ALGORITHMS
# ============================================================================

@dataclass
class OptimizationResult:
    algorithm: str
    best_value: float
    best_solution: List[float]  # Changed to List for JSON serialization
    history: List[float]
    fes_history: List[int]
    execution_time: float
    parameters: Dict[str, Any]

def run_de(func, dim: int, lb: float, ub: float, max_fes: int,
           rng: np.random.Generator,
           pop_size: int, F: float, CR: float) -> OptimizationResult:
    """
    Differential Evolution classique : DE/rand/1/bin
    Adapted from user provided de_classique
    """
    start_time = time.time()

    # Calculate Tmax based on max_fes and pop_size
    # The user's code uses Tmax generations.
    # We approximate Tmax = max_fes // pop_size to respect the budget.
    Tmax = max_fes // pop_size

    # ---- 1) Initialisation ----
    x = rng.uniform(lb, ub, (pop_size, dim))
    fvals = np.array([func(ind) for ind in x])
    
    # Track FES
    fes = pop_size

    best_idx = np.argmin(fvals)
    best_solution = x[best_idx].copy()
    best_value = fvals[best_idx]

    history = [float(best_value)]
    fes_history = [fes]

    for t in range(Tmax):
        new_pop = np.zeros_like(x)

        for i in range(pop_size):
            if fes >= max_fes:
                break
                
            # ------------------ MUTATION : DE/rand/1 ------------------
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = x[rng.choice(candidates, 3, replace=False)]

            v = a + F * (b - c)

            # ---- CLIP DES BORNES ----
            v = np.clip(v, lb, ub)

            # ------------------ CROSSOVER BINOMIAL ------------------
            u = x[i].copy()
            j_rand = rng.integers(0, dim)

            # Vectorized crossover for efficiency/readability with rng
            cross_points = rng.random(dim) < CR
            cross_points[j_rand] = True
            u = np.where(cross_points, v, u)

            # ------------------ SELECTION ------------------
            fu = func(u)
            fes += 1

            if fu < fvals[i]:
                new_pop[i] = u
                fvals[i] = fu
            else:
                new_pop[i] = x[i]

        # Mise à jour population
        x = new_pop

        # Mise à jour du best global
        best_idx = np.argmin(fvals)
        if fvals[best_idx] < best_value:
            best_value = fvals[best_idx]
            best_solution = x[best_idx].copy()

        # Historique du fitness
        history.append(float(best_value))
        fes_history.append(fes)

        if fes >= max_fes:
            break

    execution_time = time.time() - start_time
    return OptimizationResult(
        algorithm="DE",
        best_value=float(best_value),
        best_solution=best_solution.tolist(),
        history=history,
        fes_history=fes_history,
        execution_time=execution_time,
        parameters={"F": F, "CR": CR, "pop_size": pop_size}
    )

def run_jde(func, dim: int, lb: float, ub: float, max_fes: int,
            rng: np.random.Generator,
            pop_size: int, tau1: float, tau2: float) -> OptimizationResult:
    """
    JDE – Self-adaptive Differential Evolution (DE/rand/1/bin)
    Adapted from user provided jde
    """
    start_time = time.time()

    Tmax = max_fes // pop_size

    # ---- 1) Population + paramètres individuels ----
    x = rng.uniform(lb, ub, (pop_size, dim))
    fvals = np.array([func(ind) for ind in x])
    fes = pop_size

    # F_i et CR_i individuels (initially 0.5 and 0.9 as per user code default params, 
    # but user code passed F0=0.5, CR0=0.9)
    F_i = np.full(pop_size, 0.5)
    CR_i = np.full(pop_size, 0.9)

    best_idx = np.argmin(fvals)
    best_solution = x[best_idx].copy()
    best_value = fvals[best_idx]
    
    history = [float(best_value)]
    fes_history = [fes]

    for t in range(Tmax):
        new_pop = np.zeros_like(x)
        new_F = F_i.copy()
        new_CR = CR_i.copy()

        for i in range(pop_size):
            if fes >= max_fes:
                break
                
            # ---- 2) Mutation auto-adaptative des paramètres ----
            # F_i
            if rng.random() < tau1:
                F_i[i] = 0.1 + 0.9 * rng.random()

            # CR_i
            if rng.random() < tau2:
                CR_i[i] = rng.random()

            F = F_i[i]
            CR = CR_i[i]

            # ---------------- MUTATION (rand/1) ----------------
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = x[rng.choice(candidates, 3, replace=False)]

            v = a + F * (b - c)
            v = np.clip(v, lb, ub)

            # ---------------- CROSSOVER BIN -------------------
            u = x[i].copy()
            j_rand = rng.integers(0, dim)
            
            cross_points = rng.random(dim) < CR
            cross_points[j_rand] = True
            u = np.where(cross_points, v, u)

            # ---------------- SELECTION -----------------------
            fu = func(u)
            fes += 1

            if fu < fvals[i]:
                new_pop[i] = u
                fvals[i] = fu

                # garder les paramètres efficaces
                new_F[i] = F
                new_CR[i] = CR
            else:
                new_pop[i] = x[i]

        # maj population et paramètres
        x = new_pop
        F_i = new_F
        CR_i = new_CR

        # best global
        best_idx = np.argmin(fvals)
        if fvals[best_idx] < best_value:
            best_value = fvals[best_idx]
            best_solution = x[best_idx].copy()

        history.append(float(best_value))
        fes_history.append(fes)

        if fes >= max_fes:
            break

    execution_time = time.time() - start_time
    return OptimizationResult(
        algorithm="jDE",
        best_value=float(best_value),
        best_solution=best_solution.tolist(),
        history=history,
        fes_history=fes_history,
        execution_time=execution_time,
        parameters={"tau1": tau1, "tau2": tau2, "pop_size": pop_size}
    )

def run_jde_adapted(func, dim: int, lb: float, ub: float, max_fes: int,
                    rng: np.random.Generator,
                    pop_size: int, tau1: float, tau2: float,
                    p_current_to_best: float, ls_interval: int, ls_max_evals: int) -> OptimizationResult:
    """
    JDE adaptée pour le projet CEC :
    - Base : JDE/rand/1/bin
    - Hybridation :
        * 90% : mutation rand/1 (JDE classique)
        * 10% : mutation current-to-best/1
    - Recherche locale gaussienne sur 'best' toutes les 'ls_interval' générations
    Adapted from user provided jde_adapted
    """
    start_time = time.time()

    Tmax = max_fes // pop_size

    # --- sigma pour la recherche locale (en fonction de la taille du domaine) ---
    # Si LB & UB scalaires : sigma est scalaire, sinon vecteur
    # Here lb and ub are floats, so sigma is float.
    sigma = 0.001 * (ub - lb)

    # ---- 1) Population + paramètres individuels ----
    x = rng.uniform(lb, ub, (pop_size, dim))
    fvals = np.array([func(ind) for ind in x])
    fes = pop_size

    # F0 = 0.5, CR0 = 0.9
    F_i = np.full(pop_size, 0.5)
    CR_i = np.full(pop_size, 0.9)

    best_idx = np.argmin(fvals)
    best_solution = x[best_idx].copy()
    best_value = fvals[best_idx]
    
    history = [float(best_value)]
    fes_history = [fes]

    for t in range(Tmax):
        new_pop = np.zeros_like(x)
        new_F = F_i.copy()
        new_CR = CR_i.copy()

        for i in range(pop_size):
            if fes >= max_fes:
                break
                
            # ---- 2) Mutation auto-adaptative des paramètres (JDE) ----
            if rng.random() < tau1:
                F_i[i] = 0.1 + 0.9 * rng.random()
            if rng.random() < tau2:
                CR_i[i] = rng.random()

            F = F_i[i]
            CR = CR_i[i]

            # ---------------- MUTATION : rand/1 OU current-to-best/1 ----------------
            candidates = [j for j in range(pop_size) if j != i]

            if rng.random() < p_current_to_best:
                # ---- DE/current-to-best/1 ----
                # v = x_i + F*(best - x_i) + F*(x_a - x_b)
                a, b = x[rng.choice(candidates, 2, replace=False)]
                v = x[i] + F * (best_solution - x[i]) + F * (a - b)
            else:
                # ---- DE/rand/1 classique ----
                a, b, c = x[rng.choice(candidates, 3, replace=False)]
                v = a + F * (b - c)

            # Clip des bornes
            v = np.clip(v, lb, ub)

            # ---------------- CROSSOVER BIN -------------------
            u = x[i].copy()
            j_rand = rng.integers(0, dim)
            
            cross_points = rng.random(dim) < CR
            cross_points[j_rand] = True
            u = np.where(cross_points, v, u)

            # ---------------- SELECTION -----------------------
            fu = func(u)
            fes += 1

            if fu < fvals[i]:
                new_pop[i] = u
                fvals[i] = fu
                new_F[i] = F
                new_CR[i] = CR
            else:
                new_pop[i] = x[i]

        # maj population + paramètres
        x = new_pop
        F_i = new_F
        CR_i = new_CR

        # best global après la génération
        best_idx = np.argmin(fvals)
        if fvals[best_idx] < best_value:
            best_value = fvals[best_idx]
            best_solution = x[best_idx].copy()

        # =========================
        #   GAUSSIAN LOCAL SEARCH
        # =========================
        if (t + 1) % ls_interval == 0 and fes < max_fes:
            evals_used = 0
            while evals_used < ls_max_evals and fes < max_fes:
                # petite perturbation gaussienne autour de best
                noise = rng.normal(0, 1, dim) * sigma
                candidate = best_solution + noise
                candidate = np.clip(candidate, lb, ub)

                f_candidate = func(candidate)
                evals_used += 1
                fes += 1

                if f_candidate < best_value:
                    best_value = f_candidate
                    best_solution = candidate.copy()
                
                # Check if we've exceeded max_fes
                if fes >= max_fes:
                    break
            
            # Optionnel : réinjecter le best dans la population
            if fes < max_fes:  # Only update if we haven't exceeded
                worst_idx = np.argmax(fvals)
                x[worst_idx] = best_solution.copy()
                fvals[worst_idx] = best_value

        history.append(float(best_value))
        fes_history.append(fes)

        if fes >= max_fes:
            break

    execution_time = time.time() - start_time
    return OptimizationResult(
        algorithm="jDE-Adapted",
        best_value=float(best_value),
        best_solution=best_solution.tolist(),
        history=history,
        fes_history=fes_history,
        execution_time=execution_time,
        parameters={
            "tau1": tau1, "tau2": tau2, "pop_size": pop_size,
            "p_current_to_best": p_current_to_best,
            "ls_interval": ls_interval, "ls_max_evals": ls_max_evals
        }
    )

def run_pso(func, dim: int, lb: float, ub: float, max_fes: int,
            rng: np.random.Generator,
            pop_size: int, c1: float, c2: float) -> OptimizationResult:
    start_time = time.time()
    particles = rng.uniform(lb, ub, (pop_size, dim))
    velocities = rng.uniform(-1, 1, (pop_size, dim))

    fitness = np.array([func(p) for p in particles])
    fes = pop_size

    pbest = particles.copy()
    pbest_fitness = fitness.copy()

    gbest_idx = np.argmin(fitness)
    gbest = particles[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]

    history = [float(gbest_fitness)]
    fes_history = [fes]

    w_max, w_min = 0.9, 0.4
    max_iter = max_fes // pop_size
    iteration = 0

    while fes < max_fes:
        w = w_max - (w_max - w_min) * iteration / max_iter

        for i in range(pop_size):
            if fes >= max_fes:
                break
                
            r1 = rng.random(dim)
            r2 = rng.random(dim)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )

            particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

            fitness[i] = func(particles[i])
            fes += 1

            if fitness[i] < pbest_fitness[i]:
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = fitness[i]

                if fitness[i] < gbest_fitness:
                    gbest = particles[i].copy()
                    gbest_fitness = fitness[i]

            history.append(float(gbest_fitness))
            fes_history.append(fes)

            if fes >= max_fes:
                break

        iteration += 1

    execution_time = time.time() - start_time
    return OptimizationResult(
        algorithm="PSO",
        best_value=float(gbest_fitness),
        best_solution=gbest.tolist(),
        history=history,
        fes_history=fes_history,
        execution_time=execution_time,
        parameters={"c1": c1, "c2": c2, "pop_size": pop_size}
    )

def run_pso_hybrid(func, dim: int, lb: float, ub: float, max_fes: int,
                   rng: np.random.Generator,
                   pop_size: int, c1: float, c2: float,
                   p_mut: float, sigma: float) -> OptimizationResult:
    start_time = time.time()

    particles = rng.uniform(lb, ub, (pop_size, dim))
    velocities = rng.uniform(-1, 1, (pop_size, dim))

    fitness = np.array([func(p) for p in particles])
    fes = pop_size

    pbest = particles.copy()
    pbest_fitness = fitness.copy()

    gbest_idx = np.argmin(fitness)
    gbest = particles[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]

    history = [float(gbest_fitness)]
    fes_history = [fes]

    w_max, w_min = 0.9, 0.4
    max_iter = max_fes // pop_size
    iteration = 0

    while fes < max_fes:
        w = w_max - (w_max - w_min) * iteration / max_iter

        for i in range(pop_size):
            if fes >= max_fes:
                break
                
            r1 = rng.random(dim)
            r2 = rng.random(dim)

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )

            particles[i] = np.clip(particles[i] + velocities[i], lb, ub)

            if rng.random() < p_mut:
                mutation = rng.normal(0, sigma, dim)
                particles[i] = np.clip(particles[i] + mutation, lb, ub)

            fitness[i] = func(particles[i])
            fes += 1

            if fitness[i] < pbest_fitness[i]:
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = fitness[i]

                if fitness[i] < gbest_fitness:
                    gbest = particles[i].copy()
                    gbest_fitness = fitness[i]

            history.append(float(gbest_fitness))
            fes_history.append(fes)

            if fes >= max_fes:
                break

        iteration += 1

    execution_time = time.time() - start_time
    return OptimizationResult(
        algorithm="PSO-H",
        best_value=float(gbest_fitness),
        best_solution=gbest.tolist(),
        history=history,
        fes_history=fes_history,
        execution_time=execution_time,
        parameters={
            "c1": c1, "c2": c2,
            "p_mut": p_mut, "sigma": sigma,
            "pop_size": pop_size
        }
    )

def run_ga(func, dim: int, lb: float, ub: float, max_fes: int,
           rng: np.random.Generator,
           pop_size: int, mutation_scale: float) -> OptimizationResult:
    start_time = time.time()

    pop = rng.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([func(ind) for ind in pop])
    fes = pop_size

    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx].copy()
    best_value = fitness[best_idx]

    history = [float(best_value)]
    fes_history = [fes]

    while fes < max_fes:
        new_pop = []
        for _ in range(pop_size):
            i, j = rng.choice(pop_size, 2, replace=False)
            winner = i if fitness[i] < fitness[j] else j
            new_pop.append(pop[winner].copy())
        new_pop = np.array(new_pop)

        for i in range(0, pop_size - 1, 2):
            if rng.random() < 0.8:
                point = rng.integers(1, dim)
                new_pop[i, point:], new_pop[i + 1, point:] = \
                    new_pop[i + 1, point:].copy(), new_pop[i, point:].copy()

        for i in range(pop_size):
            if rng.random() < 0.2:
                mutation = rng.normal(0, mutation_scale, dim)
                new_pop[i] = np.clip(new_pop[i] + mutation, lb, ub)

        pop = new_pop
        fitness = np.array([func(ind) for ind in pop])
        fes += pop_size

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_value:
            best_value = fitness[current_best_idx]
            best_solution = pop[current_best_idx].copy()

        history.append(float(best_value))
        fes_history.append(fes)

        if fes >= max_fes:
            break

    execution_time = time.time() - start_time
    return OptimizationResult(
        algorithm="GA",
        best_value=float(best_value),
        best_solution=best_solution.tolist(),
        history=history,
        fes_history=fes_history,
        execution_time=execution_time,
        parameters={"mutation_scale": mutation_scale, "pop_size": pop_size}
    )

def run_gsa(func, dim: int, lb: float, ub: float, max_fes: int,
            rng: np.random.Generator,
            pop_size: int, G0: float, alpha: float) -> OptimizationResult:
    """
    Gravitational Search Algorithm (GSA)
    """
    start_time = time.time()

    X = rng.uniform(lb, ub, (pop_size, dim))
    V = np.zeros((pop_size, dim))
    
    fvals = np.array([func(ind) for ind in X])
    fes = pop_size

    best_idx = np.argmin(fvals)
    best_solution = X[best_idx].copy()
    best_value = fvals[best_idx]

    history = [float(best_value)]
    fes_history = [fes]

    t = 0
    while fes < max_fes:
        t += 1
        # Calculate gravitational constant
        G = G0 * np.exp(-alpha * t / (max_fes / pop_size))

        # Calculate masses
        worst = np.max(fvals)
        best = np.min(fvals)
        if worst == best:
            M = np.ones(pop_size) / pop_size
        else:
            m_raw = (worst - fvals) / (worst - best + 1e-12)
            M = m_raw / (np.sum(m_raw) + 1e-12)

        # Calculate forces (vectorized)
        F = np.zeros((pop_size, dim))
        for i in range(pop_size):
            diffs = X - X[i]
            dists = np.linalg.norm(diffs, axis=1) + 1e-12
            rand = rng.random((pop_size, dim))
            forces = rand * G * M[i] * M[:, None] * diffs / dists[:, None]
            F[i] = np.sum(forces, axis=0)

        # Update acceleration, velocity, and position
        a = F / (M[:, None] + 1e-12)
        rand2 = rng.random((pop_size, dim))
        V = rand2 * V + a
        X = X + V
        X = np.clip(X, lb, ub)

        # Evaluate new positions
        fvals = np.array([func(ind) for ind in X])
        fes += pop_size

        # Update best
        current_best_idx = np.argmin(fvals)
        if fvals[current_best_idx] < best_value:
            best_value = fvals[current_best_idx]
            best_solution = X[current_best_idx].copy()

        history.append(float(best_value))
        fes_history.append(fes)

        if fes >= max_fes:
            break

    execution_time = time.time() - start_time
    return OptimizationResult(
        algorithm="GSA",
        best_value=float(best_value),
        best_solution=best_solution.tolist(),
        history=history,
        fes_history=fes_history,
        execution_time=execution_time,
        parameters={"G0": G0, "alpha": alpha, "pop_size": pop_size}
    )

def run_abc(func, dim: int, lb: float, ub: float, max_fes: int,
            rng: np.random.Generator,
            pop_size: int, limit: int) -> OptimizationResult:
    """
    Artificial Bee Colony (ABC) Algorithm
    """
    start_time = time.time()

    X = rng.uniform(lb, ub, (pop_size, dim))
    fvals = np.array([func(ind) for ind in X])
    fes = pop_size

    trials = np.zeros(pop_size, dtype=int)

    best_idx = np.argmin(fvals)
    best_solution = X[best_idx].copy()
    best_value = fvals[best_idx]

    history = [float(best_value)]
    fes_history = [fes]

    while fes < max_fes:
        # Employed bees phase
        k_indices = np.array([rng.choice([j for j in range(pop_size) if j != i]) 
                              for i in range(pop_size)])
        j_indices = rng.integers(0, dim, pop_size)
        phi = rng.uniform(-1, 1, pop_size)
        
        V = X.copy()
        for i in range(pop_size):
            V[i, j_indices[i]] = X[i, j_indices[i]] + phi[i] * (X[i, j_indices[i]] - X[k_indices[i], j_indices[i]])
        V = np.clip(V, lb, ub)
        
        fv = np.array([func(ind) for ind in V])
        fes += pop_size
        if fes >= max_fes:
            break
            
        improved = fv < fvals
        X[improved] = V[improved]
        fvals[improved] = fv[improved]
        trials[improved] = 0
        trials[~improved] += 1

        # Onlooker bees phase
        fit_vals = 1 / (1 + fvals)
        probs = fit_vals / np.sum(fit_vals)
        
        selected = rng.choice(pop_size, size=pop_size, p=probs)
        k_indices = np.array([rng.choice([j for j in range(pop_size) if j != s]) 
                              for s in selected])
        j_indices = rng.integers(0, dim, pop_size)
        phi = rng.uniform(-1, 1, pop_size)
        
        V = X[selected].copy()
        for idx, i in enumerate(selected):
            V[idx, j_indices[idx]] = X[i, j_indices[idx]] + phi[idx] * (X[i, j_indices[idx]] - X[k_indices[idx], j_indices[idx]])
        V = np.clip(V, lb, ub)
        
        fv = np.array([func(ind) for ind in V])
        fes += pop_size
        if fes >= max_fes:
            break
            
        for idx, i in enumerate(selected):
            if fv[idx] < fvals[i]:
                X[i] = V[idx]
                fvals[i] = fv[idx]
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout bees phase
        scouts = trials > limit
        n_scouts = np.sum(scouts)
        if n_scouts > 0:
            X[scouts] = rng.uniform(lb, ub, (n_scouts, dim))
            fs = np.array([func(ind) for ind in X[scouts]])
            fes += n_scouts
            if fes >= max_fes:
                break
            fvals[scouts] = fs
            trials[scouts] = 0

        # Update best
        current_best_idx = np.argmin(fvals)
        if fvals[current_best_idx] < best_value:
            best_value = fvals[current_best_idx]
            best_solution = X[current_best_idx].copy()

        history.append(float(best_value))
        fes_history.append(fes)

        if fes >= max_fes:
            break

    execution_time = time.time() - start_time
    return OptimizationResult(
        algorithm="ABC",
        best_value=float(best_value),
        best_solution=best_solution.tolist(),
        history=history,
        fes_history=fes_history,
        execution_time=execution_time,
        parameters={"limit": limit, "pop_size": pop_size}
    )
