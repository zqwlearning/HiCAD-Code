"""Dataset access and original task prompts shared by HiCAD scripts."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path, PureWindowsPath

TASK_PROMPTS = {
    "only_HDS": (
        "Using only the hand-drawn CAD sketch provided in <image>, generate clean, "
        "correct, and executable CadQuery code that reconstructs the geometry as "
        "faithfully as possible."
    ),
    "only_CTD": (
        "Using only the complete CAD model description provided in <text>, generate "
        "clean, correct, and executable CadQuery code that reconstructs the geometry "
        "as faithfully as possible."
    ),
    "only_HDS_PTD": (
        "Using the hand-drawn CAD sketch in <image> together with the partial CAD "
        "model description in <text>, generate clean, correct, and executable "
        "CadQuery code that reconstructs the geometry as faithfully as possible."
    ),
    "only_HDS_CTD": (
        "Using the hand-drawn CAD sketch in <image> together with the complete CAD "
        "model description in <text>, generate clean, correct, and executable "
        "CadQuery code that reconstructs the geometry as faithfully as possible."
    ),
}
TASKS = tuple(TASK_PROMPTS)
DESCRIPTION_FILES = {
    "complete": "Complete Textual Description.json",
    "partial": "Partial Textual Description.json",
}


def relative_path(value: str) -> Path:
    """Interpret CLI paths relative to the current working directory."""
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).drive or value.startswith(("/", "\\")):
        raise argparse.ArgumentTypeError("Use a relative path from the repository root.")
    return path


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        try:
            return json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, data) -> None:
    """Write atomically so an interrupted batch cannot corrupt the JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_description(value, sample_id: str = "unknown") -> str:
    """Accept released plain text and the original Qwen JSON response format."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            if not match:
                raise ValueError(f"Malformed description fence for {sample_id}.")
            text = match.group(1).strip()
        if text.startswith("{"):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid description JSON for {sample_id}.") from exc
        else:
            value = text
    if isinstance(value, dict):
        value = value.get("description")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a non-empty description for {sample_id}.")
    return value.strip()


def remove_random_sentences(text: str, rng: random.Random) -> str:
    """Remove random sentences while keeping at least one in the original order."""
    sentences = [s.strip() for s in re.findall(r"[^.!?]*[.!?]", text) if s.strip()]
    if len(sentences) <= 1:
        return text
    count = rng.randint(1, len(sentences) - 1)
    removed = set(rng.sample(range(len(sentences)), count))
    return " ".join(sentence for index, sentence in enumerate(sentences) if index not in removed)


def build_prompt(task: str, description: str | None = None) -> str:
    """Build the task prompt with its textual input."""
    prompt = TASK_PROMPTS[task]
    if "<text>" in prompt:
        if description is None:
            raise ValueError(f"Task {task} requires a textual description.")
        prompt = prompt.replace("<text>", f"```description\n{description}\n```")
    return prompt


class HiCADDataset:
    """Read aligned inputs by ID without loading ground-truth code for inference."""

    def __init__(self, root: Path = Path("HiCAD_Dataset")):
        self.root = Path(root)
        self.splits = read_json(self.root / "train_val_test_splits.json")
        self._descriptions: dict[str, dict] = {}

    def sample_ids(
        self, split: str, requested: list[str] | None = None, limit: int | None = None
    ) -> list[str]:
        ids = (
            [sample_id for values in self.splits.values() for sample_id in values]
            if split == "all"
            else self.splits[split]
        )
        if requested is not None:
            unknown = sorted(set(requested) - set(ids))
            if unknown:
                raise ValueError(f"IDs not in the {split} split: {', '.join(unknown[:5])}")
            ids = list(dict.fromkeys(requested))
        if limit is not None:
            if limit <= 0:
                raise ValueError("The sample limit must be positive.")
            ids = ids[:limit]
        if not ids:
            raise ValueError(f"No samples selected from the {split} split.")
        for sample_id in ids:
            if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
                raise ValueError(f"Invalid sample ID: {sample_id!r}")
        return list(ids)

    def description(self, sample_id: str, kind: str) -> str:
        if kind not in self._descriptions:
            self._descriptions[kind] = read_json(self.root / DESCRIPTION_FILES[kind])
        values = self._descriptions[kind]
        if sample_id not in values:
            raise ValueError(
                f"Missing {kind} description for {sample_id} in {DESCRIPTION_FILES[kind]}."
            )
        return parse_description(values[sample_id], sample_id)

    def input(self, sample_id: str, task: str, sketch_dir: str = "handdraw"):
        """Use the published PTD directly; never remask evaluation inputs."""
        description = None
        if task == "only_HDS_PTD":
            description = self.description(sample_id, "partial")
        elif task in ("only_CTD", "only_HDS_CTD"):
            description = self.description(sample_id, "complete")
        image_path = None
        if "<image>" in TASK_PROMPTS[task]:
            image_path = self.root / sketch_dir / f"{sample_id}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing sketch: {image_path}")
        return build_prompt(task, description), image_path
