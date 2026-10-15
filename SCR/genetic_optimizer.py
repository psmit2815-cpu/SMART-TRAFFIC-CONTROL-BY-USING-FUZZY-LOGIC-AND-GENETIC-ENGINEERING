"""
genetic_optimizer.py
====================
Real-coded Genetic Algorithm that tunes the 14 membership-function genes of
the Mamdani FIS (Units 4-6: hybrid Fuzzy-GA system).

    Encoding   : real-valued chromosome (14 genes, see fuzzy_system.GENE_NAMES)
    Fitness    : mean average-vehicle-delay over training scenarios (MINIMISED)
    Selection  : tournament (k = 3)
    Crossover  : arithmetic (blend) crossover
    Mutation   : Gaussian, sigma = fraction of each gene's range
    Elitism    : best individuals copied unchanged
    Repair     : fuzzy_system.repair() keeps breakpoints ordered & in bounds

The fitness is computed by running the traffic simulator, so the GA optimises
the real objective (delay), not just a curve fit.  Training uses different
random seeds than testing to avoid over-fitting to one traffic realisation.
"""
import time
import numpy as np

from .fuzzy_system import MamdaniFIS, DEFAULT_PARAMS, GENE_BOUNDS, repair
from .controllers import FISController
from .traffic_simulator import run_simulation

# 7 of the 10 scenarios are used for training; S01, S05 and S07 stay UNSEEN
# so that main.py can report generalisation to new traffic patterns.
TRAIN_SCENARIOS = ["S02_moderate_balanced", "S03_heavy_balanced", "S04_ns_arterial_heavy",
                   "S06_morning_peak_ns", "S08_rush_hour_surge", "S09_fluctuating",
                   "S10_one_sided_heavy"]
TRAIN_SEEDS = (101, 102)


def traffic_fitness(params, scenarios=TRAIN_SCENARIOS, seeds=TRAIN_SEEDS, duration=1800):
    """Mean average delay (s/veh) achieved by a FIS built from `params`."""
    ctrl = FISController(MamdaniFIS(params), "GA-candidate")
    delays = [run_simulation(s, ctrl, seed=sd, duration=duration).avg_delay
              for s in scenarios for sd in seeds]
    return float(np.mean(delays))


class RealCodedGA:
    def __init__(self, fitness_fn, bounds=GENE_BOUNDS, pop_size=24, generations=30,
                 crossover_rate=0.9, mutation_rate=0.15, mutation_sigma=0.08,
                 tournament_k=3, elitism=2, seed=42, repair_fn=repair, seed_individuals=None):
        self.f = fitness_fn
        self.bounds = np.asarray(bounds, dtype=float)
        self.n = len(self.bounds)
        self.pop_size, self.generations = pop_size, generations
        self.pc, self.pm, self.sigma = crossover_rate, mutation_rate, mutation_sigma
        self.k, self.elitism = tournament_k, elitism
        self.rng = np.random.default_rng(seed)
        self.repair = repair_fn or (lambda x: np.clip(x, self.bounds[:, 0], self.bounds[:, 1]))
        self.seed_individuals = seed_individuals or []
        self.history = {"best": [], "mean": [], "worst": []}
        self.evaluations = 0

    # ---------------------------------------------------------------- operators
    def _init_population(self):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        pop = [self.repair(self.rng.uniform(lo, hi)) for _ in range(self.pop_size)]
        for i, s in enumerate(self.seed_individuals[:self.pop_size]):
            pop[i] = self.repair(s)
        return np.array(pop)

    def _tournament(self, pop, fit):
        idx = self.rng.integers(0, len(pop), self.k)
        return pop[idx[np.argmin(fit[idx])]].copy()

    def _crossover(self, p1, p2):
        if self.rng.random() > self.pc:
            return p1.copy(), p2.copy()
        a = self.rng.random()
        return a * p1 + (1 - a) * p2, a * p2 + (1 - a) * p1

    def _mutate(self, x):
        span = self.bounds[:, 1] - self.bounds[:, 0]
        mask = self.rng.random(self.n) < self.pm
        x = x + mask * self.rng.normal(0.0, self.sigma * span)
        return x

    # --------------------------------------------------------------------- run
    def run(self, verbose=True):
        t0 = time.time()
        pop = self._init_population()
        fit = np.array([self.f(x) for x in pop])
        self.evaluations += len(pop)
        for gen in range(self.generations + 1):
            order = np.argsort(fit)
            pop, fit = pop[order], fit[order]
            self.history["best"].append(float(fit[0]))
            self.history["mean"].append(float(fit.mean()))
            self.history["worst"].append(float(fit[-1]))
            if verbose:
                print(f"  gen {gen:3d} | best {fit[0]:7.3f} | mean {fit.mean():7.3f} | worst {fit[-1]:7.3f}")
            if gen == self.generations:
                break
            new_pop = list(pop[:self.elitism])
            new_fit = list(fit[:self.elitism])
            while len(new_pop) < self.pop_size:
                c1, c2 = self._crossover(self._tournament(pop, fit), self._tournament(pop, fit))
                for c in (c1, c2):
                    c = self.repair(self._mutate(c))
                    new_pop.append(c)
                    new_fit.append(self.f(c))
                    self.evaluations += 1
                    if len(new_pop) >= self.pop_size:
                        break
            pop, fit = np.array(new_pop[:self.pop_size]), np.array(new_fit[:self.pop_size])
        self.best_x, self.best_f = pop[0].copy(), float(fit[0])
        self.runtime = time.time() - t0
        return self.best_x, self.best_f, self.history


def optimise_membership_functions(pop_size=24, generations=30, seed=42, verbose=True):
    """Convenience wrapper used by main.py and the Streamlit app."""
    base = traffic_fitness(DEFAULT_PARAMS)
    ga = RealCodedGA(traffic_fitness, pop_size=pop_size, generations=generations,
                     seed=seed, seed_individuals=[DEFAULT_PARAMS])
    best_x, best_f, hist = ga.run(verbose=verbose)
    return {"best_params": best_x, "best_fitness": best_f, "baseline_fitness": base,
            "history": hist, "evaluations": ga.evaluations, "runtime": ga.runtime}
