"""Create partial descriptions by randomly removing sentences from CTD."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicad_data import (
    HiCADDataset,
    read_json,
    relative_path,
    remove_random_sentences,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=relative_path, default=Path("HiCAD_Dataset"))
    parser.add_argument("--output", type=relative_path)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or args.dataset_dir / "Partial Textual Description.json"
    try:
        dataset = HiCADDataset(args.dataset_dir)
        ids = dataset.sample_ids(args.split, args.sample_ids, args.limit)
        results = read_json(output) if output.exists() else {}
        if not isinstance(results, dict):
            raise ValueError(f"Expected an ID-to-description mapping in {output}.")
        rng = random.Random(args.seed)
        count = 0
        for sample_id in ids:
            partial = remove_random_sentences(dataset.description(sample_id, "complete"), rng)
            if sample_id not in results or args.overwrite:
                results[sample_id] = partial
                count += 1
        if count:
            write_json(output, results)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {count} partial descriptions to {output}.")


if __name__ == "__main__":
    main()
