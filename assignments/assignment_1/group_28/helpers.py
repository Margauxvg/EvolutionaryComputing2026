from pathlib import Path

import config
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json


def load_targets(target_dir: Path = config.TARGET_DIR) -> list:
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No target bodies found in {target_dir}")
    return [load_graph_from_json(path) for path in paths]


def make_individual(genotype: dict, **extra) -> dict:
    return {"genotype": genotype, "fitness": None, "alive": True, **extra}
