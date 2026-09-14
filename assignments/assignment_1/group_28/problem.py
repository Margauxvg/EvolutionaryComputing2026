"""Targets and fitness, as defined in A1_template_2026.py."""

from pathlib import Path

import networkx as nx

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)

from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
)

from . import config


def load_targets(target_dir: Path = config.TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


def fitness(body: nx.DiGraph, targets: list[nx.DiGraph]) -> float:
    """Mean + 1 std of tree edit distance to every target. Lower is better."""
    return mean_plus_std_tree_edit_distance(body, targets)


def per_target_distances(body: nx.DiGraph, targets: list[nx.DiGraph]) -> list[float]:
    return list(distances_to_targets(body, targets))
