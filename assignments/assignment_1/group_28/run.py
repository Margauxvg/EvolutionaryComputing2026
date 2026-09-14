"""Run a variant across seeds.

    python -m group_28.run --variant A --seeds 1 2 3 4 5
    python -m group_28.run --variant baseline
"""

import argparse
import time

from . import baseline, config, variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one EA variant")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(config.SEEDS))
    args = parser.parse_args()

    for seed in args.seeds:
        db_path = config.db_path(args.variant, seed)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if args.variant == "baseline":
            baseline.run(seed, db_path)
        else:
            variants.run(variants.VARIANTS[args.variant], seed, db_path)
        print(f"{args.variant} seed={seed}  {time.perf_counter() - started:.1f}s  {db_path}")


if __name__ == "__main__":
    main()
