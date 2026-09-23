import random
from collections import deque

import config
from ariel.ec.genotypes.tree.operators import (
    get_tree_depth,
    mutate_replace_node,
    mutate_subtree_replacement,
    mutate_shrink,
    mutate_hoist,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from helpers import make_individual


def mutate(genome: TreeGenome, mutation_probability: float, operator_name: str) -> tuple[TreeGenome, bool]:
    """Returns (new_genome, was_mutated). was_mutated is False if the mutation didn't fire,
    or if every attempt failed to create a valid new genome (shrink/hoist can
    silently no-op on a tree with nothing left to remove or change)."""
    if random.random() >= mutation_probability:
        return genome, False

    # TreeGenome.to_dict()/from_dict() (ariel/ec/genotypes/tree/tree_genome.py) used as a "copy" here
    original = genome.to_dict()

    for _ in range(config.MAX_MUTATION_ATTEMPTS):  # retry until valid and changed
        candidate = TreeGenome.from_dict(original)

        if operator_name == "replace_node":
            mutate_replace_node(candidate)
        elif operator_name == "subtree_replacement":
            mutate_subtree_replacement(candidate, max_modules=config.NUM_OF_MODULES)
        elif operator_name == "shrink":
            mutate_shrink(candidate)
        elif operator_name == "hoist":
            mutate_hoist(candidate)
        else:
            raise ValueError(f"Unknown mutation operator: {operator_name}")

        if candidate.to_dict() == original:
            continue

        if len(candidate.nodes) <= config.MAX_TOTAL_MODULES and get_tree_depth(candidate) <= config.MAX_TREE_DEPTH:
            return candidate, True

    return genome, False


class MutationVariant:
    """Applies one mutation operator and measures its success rate."""

    probability: float
    operator: str

    def __init__(self) -> None:
        self._pending: list[dict] = []
        self._window: deque[tuple[int, int]] = deque(maxlen=config.ADAPTIVE_WINDOW)
        self.last_success_rate: float | None = None
        self.last_num_scored: int = 0
        self.last_num_mutated: int = 0

    def mutate(self, offspring: list[dict]) -> list[dict]:
        mutated: list[dict] = []
        self._pending = []
        self.last_num_mutated = 0

        for individual in offspring:
            genome = TreeGenome.from_dict(individual["genotype"])
            new_genome, was_mutated = mutate(genome, self.probability, self.operator)

            entry = make_individual(
                new_genome.to_dict(),
                parent_fitness=individual["parent_fitness"],
                crossed=individual.get("crossed", False),
            )
            mutated.append(entry)

            if not was_mutated:
                continue
            self.last_num_mutated += 1

            # Score offsprings that were only affected by mutation
            if entry["crossed"]:
                continue
            self._pending.append(entry)

        return mutated

    def _measure(self) -> None:
        """Call after evaluate() has scored the mutated offspring."""
        successes = sum(
            1 for entry in self._pending
            if entry["fitness"] is not None and entry["fitness"] < entry["parent_fitness"]
        )
        self._window.append((successes, len(self._pending)))

        window_successes = sum(s for s, _ in self._window)
        window_total = sum(n for _, n in self._window)

        self.last_num_scored = window_total
        self.last_success_rate = window_successes / window_total if window_total else None

    def adapt(self, population: list[dict]) -> list[dict]:
        self._measure()
        return population


class StaticMutation(MutationVariant):
    def __init__(self, operator: str, probability: float = config.STATIC_MUTATION_PROBABILITY):
        super().__init__()
        self.operator = operator
        self.probability = probability


class AdaptiveMutation(MutationVariant):
    """Rechenberg's 1/5 rule: the probability grows if mutated offspring beat their parent
    more than 1/5 of the time, and shrinks otherwise."""

    def __init__(
        self,
        operator: str,
        initial_probability: float = config.ADAPTIVE_INITIAL_PROBABILITY,
        target_success: float = config.ADAPTIVE_TARGET_SUCCESS,
        factor: float = config.ADAPTIVE_FACTOR,
        min_probability: float = config.ADAPTIVE_MIN_PROBABILITY,
        max_probability: float = config.ADAPTIVE_MAX_PROBABILITY,
        min_samples: int = config.ADAPTIVE_MIN_SAMPLES,
    ):
        super().__init__()
        self.operator = operator
        self.probability = initial_probability
        self.target_success = target_success
        self.factor = factor
        self.min_probability = min_probability
        self.max_probability = max_probability
        self.min_samples = min_samples

    def adapt(self, population: list[dict]) -> list[dict]:
        self._measure()

        # Do not adapt if there are too few scored mutations in the window.
        if self.last_success_rate is None or self.last_num_scored < self.min_samples:
            return population

        if self.last_success_rate > self.target_success:
            self.probability = min(self.max_probability, self.probability * self.factor)
        elif self.last_success_rate < self.target_success:
            self.probability = max(self.min_probability, self.probability / self.factor)

        return population


def make_mutation(variant: str, operator: str | None) -> MutationVariant | None:
    if variant == "static":
        assert operator is not None
        return StaticMutation(operator)
    if variant == "adaptive":
        assert operator is not None
        return AdaptiveMutation(operator)
    if variant == "baseline":
        return None
    raise ValueError(f"Unknown variant: {variant}")
