import random

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


# ---------------------------------------------------------------------------
# Mutation operator
# ---------------------------------------------------------------------------

def mutate(genome: TreeGenome, mutation_probability: float) -> tuple[TreeGenome, bool]:
    """Returns (genome, was_mutated); False if the coin flip missed or no
    valid candidate was found within MAX_MUTATION_ATTEMPTS."""
    if random.random() >= mutation_probability:
        return genome, False

    for _ in range(config.MAX_MUTATION_ATTEMPTS):
        candidate = TreeGenome.from_dict(genome.to_dict())

        operator_name = random.choices(
            list(config.MUTATION_OPERATOR_WEIGHTS.keys()),
            weights=list(config.MUTATION_OPERATOR_WEIGHTS.values()),
        )[0]

        if operator_name == "replace_node":
            mutate_replace_node(candidate)
        elif operator_name == "subtree_replacement":
            mutate_subtree_replacement(candidate, max_modules=config.NUM_OF_MODULES)
        elif operator_name == "shrink":
            mutate_shrink(candidate)
        elif operator_name == "hoist":
            mutate_hoist(candidate)

        if len(candidate.nodes) <= config.MAX_TOTAL_MODULES and get_tree_depth(candidate) <= config.MAX_TREE_DEPTH:
            return candidate, True

    return genome, False


# ---------------------------------------------------------------------------
# Mutation variants
# ---------------------------------------------------------------------------

class MutationVariant:
    probability: float

    def mutate(self, offspring: list[dict]) -> list[dict]:
        mutated: list[dict] = []
        self._pending: list[tuple[dict, float]] = []

        for individual in offspring:
            genome = TreeGenome.from_dict(individual["genotype"])
            new_genome, was_mutated = mutate(genome, self.probability)
            entry = make_individual(new_genome.to_dict())
            mutated.append(entry)

            if was_mutated:
                self._pending.append((entry, individual["parent_fitness"]))

        return mutated

    def adapt(self, population: list[dict]) -> list[dict]:
        """Call after evaluate() has scored the mutated offspring."""
        return population


class StaticMutation(MutationVariant):
    """Fixed mutation probability for the whole run (no adaptation)."""

    def __init__(self, probability: float = config.STATIC_MUTATION_PROBABILITY):
        self.probability = probability


class AdaptiveMutation(MutationVariant):
    """Rechenberg's 1/5 rule: probability grows if mutated offspring beat
    their parent's fitness more than 1/5 of the time, shrinks otherwise."""

    def __init__(
        self,
        initial_probability: float = config.ADAPTIVE_INITIAL_PROBABILITY,
        target_success: float = config.ADAPTIVE_TARGET_SUCCESS,
        factor: float = config.ADAPTIVE_FACTOR,
        min_probability: float = config.ADAPTIVE_MIN_PROBABILITY,
        max_probability: float = config.ADAPTIVE_MAX_PROBABILITY,
    ):
        self.probability = initial_probability
        self.target_success = target_success
        self.factor = factor
        self.min_probability = min_probability
        self.max_probability = max_probability

    def adapt(self, population: list[dict]) -> list[dict]:
        if not self._pending:
            return population

        successes = sum(
            1 for entry, parent_fitness in self._pending
            if entry["fitness"] is not None and entry["fitness"] < parent_fitness  # minimisation
        )
        success_rate = successes / len(self._pending)

        if success_rate > self.target_success:
            self.probability = min(self.max_probability, self.probability * self.factor)
        elif success_rate < self.target_success:
            self.probability = max(self.min_probability, self.probability / self.factor)

        return population


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_mutation(variant: str) -> MutationVariant | None:
    if variant == "static":
        return StaticMutation()
    if variant == "adaptive":
        return AdaptiveMutation()
    if variant == "baseline":
        return None
    raise ValueError(f"Unknown variant: {variant}")
