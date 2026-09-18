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


# ---------------------------------------------------------------------------
# Mutation operator
# ---------------------------------------------------------------------------
def weights_for_strength(strength: float) -> dict[str, float]:
    """strength bestuurt puur de balans tussen conservatief (replace_node)
    en explorerend-met-nieuw-materiaal (subtree_replacement/shrink).
    hoist staat hier los van, met een klein constant aandeel als
    complexiteit-reducerende tegenkracht tegen bloat."""
    hoist_share = 0.15  # vast, niet gestuurd door strength

    remaining = 1 - hoist_share
    return {
        "replace_node": remaining * (1 - strength),
        "subtree_replacement": remaining * strength * 0.5,
        "shrink": remaining * strength * 0.5,
        "hoist": hoist_share,
    }

def mutate(genome: TreeGenome, mutation_probability: float, operator_weights: dict[str, float], subtree_max_modules: int) -> tuple[TreeGenome, bool]:
    """Returns (genome, was_mutated); False if the coin flip missed or no
    valid candidate was found within MAX_MUTATION_ATTEMPTS."""
    if random.random() >= mutation_probability:
        return genome, False

    for _ in range(config.MAX_MUTATION_ATTEMPTS):
        candidate = TreeGenome.from_dict(genome.to_dict())

        operator_name = random.choices(
            list(operator_weights.keys()),
            weights=list(operator_weights.values()),
        )[0]

        if operator_name == "replace_node":
            mutate_replace_node(candidate)
        elif operator_name == "subtree_replacement":
            mutate_subtree_replacement(candidate, max_modules=subtree_max_modules)
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
    strength: float | None = None

    def _operator_weights(self) -> dict[str, float]:
        raise NotImplementedError

    def _subtree_max_modules(self) -> int:
        raise NotImplementedError

    def mutate(self, offspring: list[dict]) -> list[dict]:
        mutated: list[dict] = []
        self._pending: list[tuple[dict, float]] = []
        weights = self._operator_weights()
        max_modules = self._subtree_max_modules()

        for individual in offspring:
            genome = TreeGenome.from_dict(individual["genotype"])
            new_genome, was_mutated = mutate(genome, self.probability, weights, max_modules)
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

    def __init__(self, probability: float = config.MUTATION_PROBABILITY,
        operator_weights: dict[str, float] = config.MUTATION_OPERATOR_WEIGHTS,
    ):
        self.probability = probability
        self._weights = operator_weights

    def _operator_weights(self) -> dict[str, float]:
        return self._weights

    def _subtree_max_modules(self) -> int:
        return config.NUM_OF_MODULES


class AdaptiveMutation(MutationVariant):
    """Rechenberg's 1/5 rule: probability grows if mutated offspring beat
    their parent's fitness more than 1/5 of the time, shrinks otherwise."""

    def __init__(
        self,
        probability: float = config.MUTATION_PROBABILITY,
        initial_strength: float = config.ADAPTIVE_INITIAL_STRENGTH,
        target_success: float = config.ADAPTIVE_TARGET_SUCCESS,
        factor: float = config.ADAPTIVE_STRENGTH_FACTOR,
        min_strength: float = config.ADAPTIVE_MIN_STRENGTH,
        max_strength: float = config.ADAPTIVE_MAX_STRENGTH,
        window_size: int = config.ADAPTIVE_WINDOW_SIZE,
    ):
        self.probability = probability
        self.strength = initial_strength
        self.target_success = target_success
        self.factor = factor
        self.min_strength = min_strength
        self.max_strength = max_strength
        self._history: deque[bool] = deque(maxlen=window_size)

    def _operator_weights(self) -> dict[str, float]:
        return weights_for_strength(self.strength)

    def _subtree_max_modules(self) -> int:
        return config.NUM_OF_MODULES

    def adapt(self, population: list[dict]) -> list[dict]:
        for entry, parent_fitness in self._pending:
            if entry["fitness"] is not None:
                self._history.append(entry["fitness"] < parent_fitness)  # minimisation

        if not self._history:
            return population

        success_rate = sum(self._history) / len(self._history)

        if success_rate > self.target_success:
            self.strength = min(self.max_strength, self.strength * self.factor)
        elif success_rate < self.target_success:
            self.strength = max(self.min_strength, self.strength / self.factor)

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
