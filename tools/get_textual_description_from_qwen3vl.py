"""Generate CTD from CadQuery code, CAD renders, and geometric attributes."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hicad_data import HiCADDataset, parse_description, read_json, relative_path, write_json

LOGGER = logging.getLogger(__name__)


def parse_annotation(response: str) -> str:
    """Enforce Figure 7's single-field JSON and single-paragraph output format."""
    text = response.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"description"}:
        raise ValueError("Expected a single JSON field named description.")
    description = parse_description(data)
    if "\n" in description or "\r" in description:
        raise ValueError("Expected a single-paragraph description without newline characters.")
    return description


def build_messages(
    sample_id: str, dataset_dir: Path, images_dir: Path, geometry: dict, template: str
) -> list:
    """Fill Figure 7's placeholders without executing the ground-truth CadQuery."""
    code = (dataset_dir / "cadquery" / f"{sample_id}.py").read_text(encoding="utf-8")
    image_path = images_dir / f"{sample_id}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing CAD render: {image_path}")
    attributes = geometry.get(sample_id)
    if not isinstance(attributes, list) or len(attributes) != 4:
        raise ValueError(f"Expected [length, width, height, through_holes] for {sample_id}.")
    length, width, height, holes = attributes
    replacements = {
        "<cadquery>": f"```python\n{code}```",
        "<length>": str(length),
        "<width>": str(width),
        "<height>": str(height),
        "<through-holes>": str(holes),
    }
    prompt = template
    for placeholder, value in replacements.items():
        if placeholder not in template:
            raise ValueError(f"Prompt template is missing {placeholder}.")
        prompt = prompt.replace(placeholder, value)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.as_posix()},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_inputs_for_vllm(messages: list, processor) -> dict:
    """Use Qwen3-VL's image patch size when preparing multimodal inputs."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)
    return {
        "prompt": text,
        "multi_modal_data": {"image": images},
        # qwen-vl-utils already resizes the image; avoid a second resize.
        "mm_processor_kwargs": {"do_resize": False},
    }


def generate_descriptions(args) -> int:
    dataset = HiCADDataset(args.dataset_dir)
    ids = dataset.sample_ids(args.split, args.sample_ids, args.limit)
    results = read_json(args.output) if args.output.exists() else {}
    if not isinstance(results, dict):
        raise ValueError(f"Expected an ID-to-description mapping in {args.output}.")
    pending = []
    for sample_id in ids:
        if sample_id in results and not args.overwrite and not args.dry_run:
            parse_description(results[sample_id], sample_id)
        else:
            pending.append(sample_id)
    LOGGER.info("%d pending / %d selected samples", len(pending), len(ids))
    if not pending:
        return 0
    geometry = read_json(args.geometry_file)
    template = args.prompt_file.read_text(encoding="utf-8")
    # Validate all source files before allocating GPU memory.
    for sample_id in pending:
        build_messages(sample_id, args.dataset_dir, args.images_dir, geometry, template)
    if args.dry_run:
        LOGGER.info("Annotation inputs passed validation; no model loaded.")
        return 0
    if not (args.model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"Place the Qwen3-VL-32B checkpoint in {args.model_dir} (missing config.json)."
        )

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("Description generation requires a CUDA-capable GPU.")
    tp_size = args.tensor_parallel_size or torch.cuda.device_count()
    if tp_size > torch.cuda.device_count():
        raise ValueError("Tensor parallel size exceeds the number of visible GPUs.")
    processor = AutoProcessor.from_pretrained(args.model_dir.as_posix())
    llm = LLM(
        model=args.model_dir.as_posix(),
        max_model_len=32768,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
        tensor_parallel_size=tp_size,
        seed=args.seed,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(temperature=0, max_tokens=1024, top_k=-1, stop_token_ids=[])
    args.responses_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        messages = [
            build_messages(sample_id, args.dataset_dir, args.images_dir, geometry, template)
            for sample_id in batch
        ]
        inputs = [prepare_inputs_for_vllm(message, processor) for message in messages]
        outputs = llm.generate(inputs, sampling_params=sampling, use_tqdm=True)
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM returned an unexpected number of annotation outputs.")
        for sample_id, output in zip(batch, outputs):
            response = output.outputs[0].text
            (args.responses_dir / f"{sample_id}.txt").write_text(response, encoding="utf-8")
            try:
                # Released CTD files store plain text, while Qwen outputs JSON strings.
                results[sample_id] = parse_annotation(response)
            except ValueError as exc:
                failures += 1
                results.pop(sample_id, None)
                LOGGER.error("%s: %s; raw response retained.", sample_id, exc)
        write_json(args.output, results)
        LOGGER.info(
            "Completed %d / %d annotations; checkpoint saved to %s",
            start + len(batch),
            len(pending),
            args.output,
        )
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=relative_path, default=Path("HiCAD_Dataset"))
    parser.add_argument(
        "--model-dir", type=relative_path, default=Path("HiCAD_Model/Qwen3-VL-32B-Instruct-FP8")
    )
    parser.add_argument(
        "--images-dir", type=relative_path, help="CAD renders; default: <dataset-dir>/images."
    )
    parser.add_argument(
        "--geometry-file", type=relative_path, help="Default: <dataset-dir>/cad_geometry.json."
    )
    parser.add_argument(
        "--prompt-file", type=relative_path, default=Path("tools/textual_description_prompt.txt")
    )
    parser.add_argument(
        "--output",
        type=relative_path,
        help="Default: <dataset-dir>/Complete Textual Description.json.",
    )
    parser.add_argument(
        "--responses-dir",
        type=relative_path,
        default=Path("results/textual_descriptions/responses"),
    )
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite", action="store_true", help="Regenerate descriptions for selected IDs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without loading the annotation model.",
    )
    args = parser.parse_args()
    args.images_dir = args.images_dir or args.dataset_dir / "images"
    args.geometry_file = args.geometry_file or args.dataset_dir / "cad_geometry.json"
    args.output = args.output or args.dataset_dir / "Complete Textual Description.json"
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("Batch size and sample limit must be positive.")
    if args.tensor_parallel_size is not None and args.tensor_parallel_size <= 0:
        parser.error("Tensor parallel size must be positive.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("GPU memory utilization must be in (0, 1].")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        status = generate_descriptions(args)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
