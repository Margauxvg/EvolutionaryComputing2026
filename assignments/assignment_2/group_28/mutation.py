from collections import deque

import numpy as np

from ariel.ec import Individual, Population

import config


def mutate(genotype: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian mutation: add noise with std sigma to every weight."""
    return genotype + rng.normal(scale=sigma, size=genotype.shape)


class MutationVariant:
    """Mutates every offspring and measures how often that made it better than its parent."""

    sigma: float

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._pending: list[Individual] = []
        self._window: deque[tuple[int, int]] = deque(maxlen=config.ADAPTIVE_WINDOW)
        self.last_success_rate: float | None = None
        self.last_num_scored: int = 0
        self.last_num_mutated: int = 0

    def mutate(self, offspring: Population) -> Population:
        for individual in offspring:
            genotype = np.asarray(individual.genotype, dtype=np.float64)
            individual.genotype = mutate(genotype, self.sigma, self.rng).tolist()

        self._pending = list(offspring)
        self.last_num_mutated = len(self._pending)
        return offspring

    def _measure(self) -> None:
        """Call after evaluate() has scored the mutated offspring."""
        successes = sum(
            1 for individual in self._pending
            if individual.fitness_ is not None and individual.fitness_ < individual.tags["parent_fitness"]
        )
        self._window.append((successes, len(self._pending)))

        window_successes = sum(s for s, _ in self._window)
        window_total = sum(n for _, n in self._window)

        self.last_num_scored = window_total
        self.last_success_rate = window_successes / window_total if window_total else None

    def adapt(self, population: Population) -> Population:
        self._measure()
        return population


class StaticMutation(MutationVariant):
    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)
        self.sigma = config.STATIC_SIGMA


class AdaptiveMutation(MutationVariant):
    """Rechenberg's 1/5 rule: sigma grows if mutated offspring beat their parent more than
    1/5 of the time, and shrinks otherwise."""

    def __init__(self, rng: np.random.Generator, max_sigma: float) -> None:
        super().__init__(rng)
        self.sigma = config.ADAPTIVE_INITIAL_SIGMA
        self.max_sigma = max_sigma

    def adapt(self, population: Population) -> Population:
        self._measure()

        # Do not adapt if there are too few scored mutations in the window.
        if self.last_success_rate is None or self.last_num_scored < config.ADAPTIVE_MIN_SAMPLES:
            return population

        if self.last_success_rate > config.ADAPTIVE_TARGET_SUCCESS:
            self.sigma = min(self.max_sigma, self.sigma * config.ADAPTIVE_FACTOR)
        elif self.last_success_rate < config.ADAPTIVE_TARGET_SUCCESS:
            self.sigma = max(config.ADAPTIVE_MIN_SIGMA, self.sigma / config.ADAPTIVE_FACTOR)

        return population


def make_mutation(variant: str, rng: np.random.Generator) -> MutationVariant | None:
    if variant == "static":
        return StaticMutation(rng)
    if variant == "adaptive":
        return AdaptiveMutation(rng, max_sigma=config.ADAPTIVE_MAX_SIGMA)
    if variant == "adaptive_cap1":
        return AdaptiveMutation(rng, max_sigma=config.ADAPTIVE_CAP1_MAX_SIGMA)
    if variant == "baseline":
        return None
    raise ValueError(f"Unknown variant: {variant}")
