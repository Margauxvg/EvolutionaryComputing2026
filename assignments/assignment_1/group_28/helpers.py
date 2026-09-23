import csv
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


def variant_dir(variant: str, operator: str | None = None) -> Path:
    path = config.RESULTS_DIR / variant
    if operator is not None:
        path = path / operator
    return path


def write_csv(rows: list[dict], output_path: Path, fieldnames: tuple[str, ...] | None = None) -> Path:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path
