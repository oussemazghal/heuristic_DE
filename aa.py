import numpy as np

# =========================================================
#  DE CLASSIQUE
# =========================================================
def de_classique(func, D=30, N=100, LB=-100, UB=100, F=0.8, CR=0.9, Tmax=300):
    """
    Differential Evolution classique : DE/rand/1/bin
    - func : fonction à minimiser (batch: reçoit matrice (N,D))
    - D : dimension
    - N : taille population
    - LB, UB : bornes (scalaires ou tableaux)
    - F : facteur de mutation
    - CR : taux de croisement
    - Tmax : générations max
    """

    LB = np.asarray(LB)
    UB = np.asarray(UB)

    # ---- 1) Initialisation ----
    x = np.random.uniform(LB, UB, (N, D))
    fvals = func(x)

    best_idx = np.argmin(fvals)
    best = x[best_idx].copy()
    best_val = fvals[best_idx]

    history = np.zeros(Tmax)

    for t in range(Tmax):

        new_pop = np.zeros_like(x)

        for i in range(N):

            # ------------------ MUTATION : DE/rand/1 ------------------
            candidates = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)

            v = x[a] + F * (x[b] - x[c])

            # ---- CLIP DES BORNES ----
            v = np.clip(v, LB, UB)

            # ------------------ CROSSOVER BINOMIAL ------------------
            u = x[i].copy()
            j_rand = np.random.randint(D)

            for j in range(D):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            # ------------------ SELECTION ------------------
            fu = func(u[None, :])[0]   # évaluation plus propre

            if fu < fvals[i]:
                new_pop[i] = u
                fvals[i] = fu
            else:
                new_pop[i] = x[i]

        # Mise à jour population
        x = new_pop

        # Mise à jour du best global
        best_idx = np.argmin(fvals)
        if fvals[best_idx] < best_val:
            best_val = fvals[best_idx]
            best = x[best_idx].copy()

        # Historique du fitness
        history[t] = best_val

    return history, best, best_val


# =========================================================
#  JDE SIMPLE (self-adaptatif, rand/1 uniquement)
# =========================================================
def jde(func, D=30, N=100, LB=-100, UB=100, 
        F0=0.5, CR0=0.9, tau1=0.1, tau2=0.1, Tmax=300):
    """
    JDE – Self-adaptive Differential Evolution (DE/rand/1/bin)
    - chaque individu possède son F_i et CR_i
    - F_i et CR_i mutent avec probabilités tau1, tau2
    """

    LB = np.asarray(LB)
    UB = np.asarray(UB)

    # ---- 1) Population + paramètres individuels ----
    x = np.random.uniform(LB, UB, (N, D))
    fvals = func(x)

    # F_i et CR_i individuels
    F_i = np.full(N, F0)
    CR_i = np.full(N, CR0)

    best_idx = np.argmin(fvals)
    best = x[best_idx].copy()
    best_val = fvals[best_idx]
    history = np.zeros(Tmax)

    for t in range(Tmax):

        new_pop = np.zeros_like(x)
        new_F = F_i.copy()
        new_CR = CR_i.copy()

        for i in range(N):

            # ---- 2) Mutation auto-adaptative des paramètres ----
            # F_i
            if np.random.rand() < tau1:
                F_i[i] = 0.1 + 0.9 * np.random.rand()

            # CR_i
            if np.random.rand() < tau2:
                CR_i[i] = np.random.rand()

            F = F_i[i]
            CR = CR_i[i]

            # ---------------- MUTATION (rand/1) ----------------
            candidates = [j for j in range(N) if j != i]
            a, b, c = np.random.choice(candidates, 3, replace=False)

            v = x[a] + F * (x[b] - x[c])
            v = np.clip(v, LB, UB)

            # ---------------- CROSSOVER BIN -------------------
            u = x[i].copy()
            j_rand = np.random.randint(D)

            for j in range(D):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            # ---------------- SELECTION -----------------------
            fu = func(u[None, :])[0]

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
        if fvals[best_idx] < best_val:
            best_val = fvals[best_idx]
            best = x[best_idx].copy()

        history[t] = best_val

    return history, best, best_val


# =========================================================
#  JDE ADAPTÉE : hybridation + Gaussian Local Search
# =========================================================
def jde_adapted(func, D=30, N=30, LB=-100, UB=100,
                F0=0.5, CR0=0.9, tau1=0.1, tau2=0.1,
                Tmax=1000,
                p_current_to_best=0.05,
                ls_interval=100,
                ls_max_evals=10):
    """
    jDE-Adapted optimisée pour FESmax = 30000.

    Améliorations par rapport à ta version AVANT :
    ------------------------------------------------
    ✓ N réduit 50 → 30  → +67% plus de générations
    ✓ sigma 0.001 → 0.01  → recherche locale utile
    ✓ p_current_to_best 0.1 → 0.05 → meilleure diversité
    ✓ ls_interval 50 → 100 → LS moins fréquente = moins de gaspillage FES
    ✓ ls_max_evals 30 → 10 → LS plus légère
    """

    LB = np.asarray(LB)
    UB = np.asarray(UB)

    # --- sigma pour la recherche locale (10x plus efficace qu'avant) ---
    sigma = 0.01 * (UB - LB)

    # ---- 1) Population + paramètres individuels ----
    x = np.random.uniform(LB, UB, (N, D))
    fvals = func(x)

    F_i = np.full(N, F0)
    CR_i = np.full(N, CR0)

    best_idx = np.argmin(fvals)
    best = x[best_idx].copy()
    best_val = fvals[best_idx]
    history = np.zeros(Tmax)

    for t in range(Tmax):

        new_pop = np.zeros_like(x)
        new_F = F_i.copy()
        new_CR = CR_i.copy()

        for i in range(N):

            # ---- 2) Mutation auto-adaptative des paramètres JDE ----
            if np.random.rand() < tau1:
                F_i[i] = 0.1 + 0.9 * np.random.rand()
            if np.random.rand() < tau2:
                CR_i[i] = np.random.rand()

            F = F_i[i]
            CR = CR_i[i]

            # ---------------- MUTATION : rand/1 OU current-to-best/1 ----------------
            candidates = [j for j in range(N) if j != i]

            if np.random.rand() < p_current_to_best:
                # ---- DE/current-to-best/1 ----
                a, b = np.random.choice(candidates, 2, replace=False)
                v = x[i] + F * (best - x[i]) + F * (x[a] - x[b])
            else:
                # ---- DE/rand/1 ----
                a, b, c = np.random.choice(candidates, 3, replace=False)
                v = x[a] + F * (x[b] - x[c])

            # Clip bornes
            v = np.clip(v, LB, UB)

            # ---------------- CROSSOVER BIN -------------------
            u = x[i].copy()
            j_rand = np.random.randint(D)
            for j in range(D):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]

            # ---------------- SÉLECTION -----------------------
            fu = func(u[None, :])[0]

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

        # best global
        best_idx = np.argmin(fvals)
        if fvals[best_idx] < best_val:
            best_val = fvals[best_idx]
            best = x[best_idx].copy()

        # ============================
        #   GAUSSIAN LOCAL SEARCH
        # ============================
        if (t + 1) % ls_interval == 0:
            evals_used = 0
            while evals_used < ls_max_evals:
                noise = np.random.randn(D) * sigma
                candidate = np.clip(best + noise, LB, UB)

                f_candidate = func(candidate[None, :])[0]
                evals_used += 1

                if f_candidate < best_val:
                    best_val = f_candidate
                    best = candidate.copy()

            # Réinjection du best
            worst_idx = np.argmax(fvals)
            x[worst_idx] = best.copy()
            fvals[worst_idx] = best_val

        history[t] = best_val

    return history, best, best_val


# -------- TEST : Sphere Function --------
def sphere(X):
    return np.sum(X**2, axis=1)

if __name__ == "__main__":
    hist, best, best_val = de_classique(sphere)
    hist2, best2, best_val2 = jde(sphere)
    hist3, best3, best_val3 = jde_adapted(sphere)

    print("DE classique   Best =", best_val)
    print("JDE simple     Best =", best_val2)
    print("JDE adapted    Best =", best_val3)
from cec2017.functions import f1
import numpy as np

# Ceci DOIT retourner une valeur ~1e11
print(f1(np.zeros((1,30))))
