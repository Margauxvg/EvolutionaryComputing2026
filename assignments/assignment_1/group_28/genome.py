"""Tree genome: sampling, the operators ariel supplies, and JSON (de)serialisation.

`Individual.genotype_` is a JSON column, so genotypes are stored in
`TreeGenome.to_dict()` form. `to_native`/`from_native` convert to and from the
`TreeGenome` ariel's operators expect.
"""

import random

import networkx as nx

from ariel.ec.genotypes.tree.operators import (
    add_node,
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from . import config

# variants.py picks from these to isolate "the one aspect" it varies.
SUPPLIED_MUTATIONS = {
    "replace_node": mutate_replace_node,
    "subtree_replacement": mutate_subtree_replacement,
    "shrink": mutate_shrink,
    "hoist": mutate_hoist,
}
SUPPLIED_CROSSOVER = {"subtree": crossover_subtree}
# add_node needs a parent/face/id/type/rotation choice, unlike the above -
# a building block, not a drop-in operator.
PRIMITIVES = {"add_node": add_node}


def sample(rng: random.Random) -> dict:
    """One random genotype, in stored (JSON) form.

    ariel's tree operators draw from the global `random` module, not from an
    injected generator; `rng` is accepted only for interface symmetry.
    """
    return random_tree(max_modules=config.NUM_OF_MODULES).to_dict()


def to_native(genotype: dict) -> TreeGenome:
    return TreeGenome.from_dict(genotype)


def from_native(native: TreeGenome) -> dict:
    return native.to_dict()


def decode(genotype: dict) -> nx.DiGraph:
    return to_native(genotype).to_networkx()


def size(genotype: dict) -> int:
    return len(to_native(genotype).nodes)
