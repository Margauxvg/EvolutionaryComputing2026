from collections import deque

import numpy as np
import numpy.typing as npt

from ariel.ec import Individual, Population

import config


def mutate(
    genotype: npt.NDArray[np.float64],
    sigma: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Add Gaussian noise of standard deviation `sigma` to every weight."""
    return genotype + rng.normal(scale=sigma, size=genotype.shape)


class MutationVariant:
    """Applies Gaussian mutation and measures how often it helps."""

    sigma: float

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._pending: list[Individual] = []
        self._window: deque[tuple[int, int]] = deque(maxlen=config.ADAPTIVE_WINDOW)
        self.last_success_rate: float | None = None
        self.last_num_scored: int = 0
        self.last_num_mutated: int = 0

    def mutate(self, offspring: Population) -> Population:
        self._pending = []
        self.last_num_mutated = 0

        for individual in offspring:
            genotype = np.asarray(individual.genotype, dtype=np.float64)
            individual.genotype = mutate(genotype, self.sigma, self.rng).tolist()
            self.last_num_mutated += 1
            self._pending.append(individual)

        return offspring

    def _measure(self) -> None:
        """Success rate over the pooled window. Call after evaluate() has scored `_pending`."""
        successes = sum(
            1
            for individual in self._pending
            if individual.fitness_ is not None
            and individual.fitness_ < individual.tags["parent_fitness"]
        )
        self._window.append((successes, len(self._pending)))

        window_successes = sum(s for s, _ in self._window)
        window_total = sum(n for _, n in self._window)

        self.last_num_scored = window_total
        self.last_success_rate = (
            window_successes / window_total if window_total else None
        )

    def adapt(self, population: Population) -> Population:
        """Measure, and do nothing else. AdaptiveMutation overrides this."""
        self._measure()
        return population


class StaticMutation(MutationVariant):
    """Fixed step size. The control."""

    def __init__(
        self,
        rng: np.random.Generator,
        sigma: float = config.STATIC_SIGMA,
    ) -> None:
        super().__init__(rng)
        self.sigma = sigma


class AdaptiveMutation(MutationVariant):
    """Rechenberg's 1/5 rule: sigma grows when mutated offspring beat their parent more than
    one time in five, and shrinks otherwise."""

    def __init__(
        self,
        rng: np.random.Generator,
        initial_sigma: float = config.ADAPTIVE_INITIAL_SIGMA,
        target_success: float = config.ADAPTIVE_TARGET_SUCCESS,
        factor: float = config.ADAPTIVE_FACTOR,
        min_sigma: float = config.ADAPTIVE_MIN_SIGMA,
        max_sigma: float = config.ADAPTIVE_MAX_SIGMA,
        min_samples: int = config.ADAPTIVE_MIN_SAMPLES,
    ) -> None:
        super().__init__(rng)
        self.sigma = initial_sigma
        self.target_success = target_success
        self.factor = factor
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_samples = min_samples

    def adapt(self, population: Population) -> Population:
        self._measure()

        # Too few scored mutations in the window to trust the rate: hold sigma.
        if self.last_success_rate is None or self.last_num_scored < self.min_samples:
            return population

        if self.last_success_rate > self.target_success:
            self.sigma = min(self.max_sigma, self.sigma * self.factor)
        elif self.last_success_rate < self.target_success:
            self.sigma = max(self.min_sigma, self.sigma / self.factor)

        return population


def make_mutation(variant: str, rng: np.random.Generator) -> MutationVariant | None:
    """The mutation strategy for a variant. None for the baseline, which does not mutate."""
    if config.STATIC_SIGMA != config.ADAPTIVE_INITIAL_SIGMA:
        msg = (
            "STATIC_SIGMA and ADAPTIVE_INITIAL_SIGMA differ, so the variants would not "
            "start identical. Set them to the same value in config.py."
        )
        raise ValueError(msg)

    if variant == "static":
        return StaticMutation(rng, sigma=config.STATIC_SIGMA)
    if variant == "adaptive":
        return AdaptiveMutation(
            rng,
            initial_sigma=config.ADAPTIVE_INITIAL_SIGMA,
            max_sigma=config.ADAPTIVE_MAX_SIGMA,
        )
    if variant == "adaptive_cap1":
        return AdaptiveMutation(
            rng,
            initial_sigma=config.ADAPTIVE_INITIAL_SIGMA,
            max_sigma=config.ADAPTIVE_CAP1_MAX_SIGMA,
        )
    if variant == "baseline":
        return None
    msg = f"Unknown variant: {variant!r}"
    raise ValueError(msg)


# --------------------------------------------------------------------------- #
#  Self-test: python mutation.py (checks the 1/5 logic with fake fitnesses, no simulator)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    def fake_offspring(n: int, parent_fitness: float, rng) -> Population:
        """n unevaluated children of a parent with a known fitness."""
        people = []
        for _ in range(n):
            individual = Individual()
            individual.genotype = rng.normal(size=8).tolist()
            individual.tags = {"parent_fitness": parent_fitness}
            people.append(individual)
        return Population(people)

    rng = np.random.default_rng(config.DEFAULT_SEED)

    print(f"start sigma           : {config.ADAPTIVE_INITIAL_SIGMA}")
    print(f"target success        : {config.ADAPTIVE_TARGET_SUCCESS}")
    print(f"factor                : {config.ADAPTIVE_FACTOR}")
    print(f"window / min samples  : {config.ADAPTIVE_WINDOW} / {config.ADAPTIVE_MIN_SAMPLES}\n")

    for label, success_fraction in (("succeeding 60% of the time", 0.6),
                                    ("succeeding 0% of the time", 0.0)):
        adaptive = AdaptiveMutation(np.random.default_rng(1))
        static = StaticMutation(np.random.default_rng(1))
        print(f"{label}:")
        print(f"{'gen':>5} {'adaptive sigma':>16} {'static sigma':>14} {'success rate':>14}")

        for generation in range(1, 9):
            for variant in (adaptive, static):
                offspring = fake_offspring(20, parent_fitness=1.0, rng=rng)
                variant.mutate(offspring)
                # Stand in for evaluate(): a fixed fraction of children beat the parent.
                for i, individual in enumerate(offspring):
                    individual.fitness = 0.9 if i < 20 * success_fraction else 1.1
                variant.adapt(offspring)

            rate = adaptive.last_success_rate
            print(f"{generation:>5} {adaptive.sigma:>16.4f} {static.sigma:>14.4f} "
                  f"{'-' if rate is None else f'{rate:>14.2f}'}")
        print()

    print("Expected: sigma grows while success is above 1/5, shrinks below it, and the static "
          "variant never moves\nwhile still reporting a success rate - that is what makes the "
          "two curves comparable.")
