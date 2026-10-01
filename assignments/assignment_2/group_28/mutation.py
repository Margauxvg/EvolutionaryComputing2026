"""Mutation, and the 1/5 rule. This file IS the research question.

Everything else in the project is machinery for asking one thing: does adapting the mutation
step size by Rechenberg's 1/5 success rule beat holding it fixed? The two variants differ here
and nowhere else - same body, world, SIM_DURATION, fitness, population, budget, seeds and
selection. There is no crossover: mutation is the EA's only variation operator, which is what
makes every offspring attributable to it (stage1_findings.md section 9).

A PORT OF A1's mutation.py, WITH ONE DELIBERATE DIFFERENCE

    A1                                  A2
    MutationVariant / Static / Adaptive same three classes
    _measure() / adapt() split          same
    window + min-samples guard          same
    static logs but does not act        same
    mutation PROBABILITY adapts         mutation STEP SIZE (sigma) adapts
    discrete tree operators             Gaussian perturbation of a real vector
    mutation fires with probability p   mutation ALWAYS fires; sigma sets its size

That last row matters. A1's report noted that applying the 1/5 rule to a discrete mutation
probability was "an analogy rather than a transfer of its underlying analysis, and our results
give some reason to doubt that the analogy holds". Rechenberg derived the rule for a continuous
step size on a real-valued genome, which is exactly what sigma is here. So A2 tests the rule in
its native setting, and the Introduction can say why that is worth doing.

WHAT ea.py CALLS

    make_mutation(variant, rng) -> MutationVariant | None     None for "baseline"
    mutation.sigma                 read BEFORE offspring are made, so the logged value is the
                                   one that produced them
    mutation.mutate(offspring)     perturb, and remember which children are scorable
    mutation.adapt(population)     called AFTER evaluate; measures success, and for the
                                   adaptive variant updates sigma

WHAT THE STAGE 1 PROBE SAYS TO WATCH FOR

The quick search showed seven consecutive generations with no improvement at a fixed sigma
(report/stage1_findings.md section 4). The 1/5 rule shrinks sigma when success falls below one
in five - correct when steps are overshooting a smooth optimum, wrong on a flat plateau, where
smaller steps find no improvement either and the rule shrinks itself into stagnation. A1 saw
exactly that. Which kind of plateau this landscape has is what the experiment should reveal,
and it is only visible if sigma is logged every generation from generation 0.
"""

# Standard library
from collections import deque

# Third-party libraries
import numpy as np
import numpy.typing as npt

from ariel.ec import Individual, Population

import config


def mutate(
    genotype: npt.NDArray[np.float64],
    sigma: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Add Gaussian noise of standard deviation `sigma` to every weight.

    A1's equivalent returned (genome, was_mutated) and could no-op: its tree operators
    sometimes failed to produce a valid, changed genome, so it retried up to
    MAX_MUTATION_ATTEMPTS and gave up. Nothing here can fail - every gene moves, every time -
    so there is no retry loop and no was_mutated flag.

    Weights are unbounded on purpose. There is no clipping, because clipping would interact
    with sigma in a way that muddies the comparison: a large sigma against a bound behaves
    differently from a large sigma in open space. simulate.py catches the consequence (a
    controller that overflows to inf or nan scores WORST_FITNESS) and experiments.py nan shows
    that finite weights, however large, stay finite through tanh.
    """
    return genotype + rng.normal(scale=sigma, size=genotype.shape)


class MutationVariant:
    """Applies Gaussian mutation and measures how often it helps.

    The measuring half runs for BOTH variants. The static variant logs its success rate and
    ignores it; only the adaptive one acts. That is A1's trick and it is what makes the
    comparison interpretable: the two curves are measured identically, so any difference in
    the sigma column is the controller and nothing else.
    """

    sigma: float

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._pending: list[Individual] = []
        self._window: deque[tuple[int, int]] = deque(maxlen=config.ADAPTIVE_WINDOW)
        self.last_success_rate: float | None = None
        self.last_num_scored: int = 0
        self.last_num_mutated: int = 0

    def mutate(self, offspring: Population) -> Population:
        """Perturb every child in place, and remember which ones can be scored.

        Mutating in place keeps each Individual's `parent_fitness` tag, set by ea.reproduction,
        and leaves requires_eval True, so ea.evaluate picks them up.

        Every child goes into `_pending`. A1 had to exclude recombined children here - an
        improvement over a parent cannot be attributed to mutation if crossover also touched the
        child - which cost it roughly 70% of its samples. With mutation as the only variation
        operator there is nothing to exclude.
        """
        self._pending = []
        self.last_num_mutated = 0

        for individual in offspring:
            genotype = np.asarray(individual.genotype, dtype=np.float64)
            individual.genotype = mutate(genotype, self.sigma, self.rng).tolist()
            self.last_num_mutated += 1
            self._pending.append(individual)

        return offspring

    def _measure(self) -> None:
        """Success rate over the pooled window. Call after evaluate() has scored `_pending`.

        A single generation is a noisy estimate of the success rate, so A1 pooled
        ADAPTIVE_WINDOW generations before comparing to the 1/5 target, and that carries over.

        An individual whose controller blew up scores WORST_FITNESS, which is worse than any
        real parent fitness, so it counts as a failure. That is the right behaviour: mutation
        that overflows the weights has not helped.
        """
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
    one time in five, and shrinks otherwise.

    The intuition is that a one-in-five success rate marks the balance point between steps too
    small to make progress and steps so large they mostly land worse. Above it, the search is
    being too cautious; below it, too bold.

    On the sigma floor. A1 floored the mutation probability at 0.1 and it pinned there once
    success hit zero, which the report read as a failure of the rule. In continuous space a
    shrinking sigma near an optimum is the rule working exactly as designed, so the floor here
    is set low enough to be effectively off (config.ADAPTIVE_MIN_SIGMA = 1e-4) and only the
    ceiling really guards anything. If sigma bottoms out anyway, that is a finding about the
    landscape - see the plateau note in the module docstring - and not a floor artefact.
    """

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

        # Too few scored mutations in the window to trust the rate - hold sigma rather than
        # react to noise. Without this guard a couple of unlucky generations can send sigma
        # somewhere it takes many generations to come back from.
        if self.last_success_rate is None or self.last_num_scored < self.min_samples:
            return population

        if self.last_success_rate > self.target_success:
            self.sigma = min(self.max_sigma, self.sigma * self.factor)
        elif self.last_success_rate < self.target_success:
            self.sigma = max(self.min_sigma, self.sigma / self.factor)

        return population


def make_mutation(variant: str, rng: np.random.Generator) -> MutationVariant | None:
    if config.STATIC_SIGMA != config.ADAPTIVE_INITIAL_SIGMA:
        msg = (
            "STATIC_SIGMA and ADAPTIVE_INITIAL_SIGMA differ, so the variants would not "
            "start identical. Set them to the same value in config.py."
        )
        raise ValueError(msg)
    
    """The mutation strategy for a variant. None for the baseline, which does not mutate.

    Both variants start at the same sigma - config.STATIC_SIGMA and
    config.ADAPTIVE_INITIAL_SIGMA must be equal. A1 nearly shipped a confound here, with the
    static variant at 0.6 and the adaptive one starting at 0.3, which would have made any
    divergence partly a difference in starting point rather than in the controller.
    """
    if variant == "static":
        return StaticMutation(rng, sigma=config.STATIC_SIGMA)
    if variant == "adaptive":
        return AdaptiveMutation(rng, initial_sigma = config.ADAPTIVE_INITIAL_SIGMA)
    if variant == "baseline":
        return None
    msg = f"Unknown variant: {variant!r}"
    raise ValueError(msg)


# --------------------------------------------------------------------------- #
#  Self-test: python mutation.py
#  Checks the 1/5 logic without touching the simulator, by feeding it fake fitnesses.
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
