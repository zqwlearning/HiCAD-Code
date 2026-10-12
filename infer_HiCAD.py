"""Generate CadQuery code with HiCAD-SFT or HiCAD-RL using vLLM."""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

from hicad_data import TASKS, HiCADDataset, relative_path, write_json

LOGGER = logging.getLogger(__name__)


def make_request(prompt: str, image_path: Path | None = None) -> dict:
    """Preserve the original Qwen chat tokens and image placeholder ordering."""
    request = {}
    if "<image>" in prompt:
        if image_path is None:
            raise ValueError("A prompt containing <image> requires a sketch.")
        from PIL import Image

        prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        with Image.open(image_path) as image:
            request["multi_modal_data"] = {"image": image.convert("RGB")}
    request["prompt"] = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return request


def extract_code(response: str) -> str:
    """Save final code separately from the unmodified model response."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = response.strip()
    fence = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n?```", response, re.DOTALL)
    if fence:
        response = fence.group(1).strip()
    if not response:
        raise ValueError("The model returned empty CadQuery code.")
    return response + "\n"


def run_inference(args) -> int:
    """Validate inputs before loading weights, then generate in bounded batches."""
    dataset = HiCADDataset(args.dataset_dir)
    ids = dataset.sample_ids(args.split, args.sample_ids, args.limit)
    pending = {}
    for task in args.tasks:
        selected = []
        for sample_id in ids:
            names = (
                [sample_id]
                if args.num_samples == 1
                else [f"{sample_id}_{index:03d}" for index in range(args.num_samples)]
            )
            code_dir = args.output_dir / task / "cadquery"
            if (
                not args.overwrite
                and not args.dry_run
                and all((code_dir / f"{name}.py").is_file() for name in names)
            ):
                continue
            dataset.input(sample_id, task, args.sketch_dir)
            selected.append(sample_id)
        pending[task] = selected
        LOGGER.info("%s: %d pending / %d selected samples", task, len(selected), len(ids))
    if not any(pending.values()):
        return 0
    if args.dry_run:
        LOGGER.info("Dataset and task inputs passed validation; no model loaded.")
        return 0
    if not (args.model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"Place a Hugging Face checkpoint in {args.model_dir} (missing config.json)."
        )

    # --help and --dry-run remain usable without the GPU dependencies.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Inference requires a CUDA-capable GPU and a compatible vLLM installation."
        )
    tp_size = args.tensor_parallel_size or torch.cuda.device_count()
    if tp_size > torch.cuda.device_count():
        raise ValueError("Tensor parallel size exceeds the number of visible GPUs.")
    llm = LLM(
        model=args.model_dir.as_posix(),
        tensor_parallel_size=tp_size,
        mm_processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 1280 * 28 * 28, "fps": 1},
        limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_samples,
    )
    metadata = {
        key: value.as_posix() if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    write_json(args.output_dir / "run_config.json", metadata)
    failures = 0
    for task, selected in pending.items():
        code_dir = args.output_dir / task / "cadquery"
        response_dir = args.output_dir / task / "responses"
        code_dir.mkdir(parents=True, exist_ok=True)
        response_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start : start + args.batch_size]
            requests = [
                make_request(*dataset.input(sample_id, task, args.sketch_dir))
                for sample_id in batch
            ]
            try:
                outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=True)
            finally:
                for request in requests:
                    for image in request.get("multi_modal_data", {}).values():
                        image.close()
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned an unexpected number of sample outputs.")
            for sample_id, output in zip(batch, outputs):
                if len(output.outputs) != args.num_samples:
                    raise RuntimeError(f"Unexpected number of candidates for {sample_id}.")
                for index, candidate in enumerate(output.outputs):
                    name = sample_id if args.num_samples == 1 else f"{sample_id}_{index:03d}"
                    code_path = code_dir / f"{name}.py"
                    if code_path.exists() and not args.overwrite:
                        continue
                    (response_dir / f"{name}.txt").write_text(candidate.text, encoding="utf-8")
                    try:
                        code = extract_code(candidate.text)
                    except ValueError as exc:
                        failures += 1
                        if args.overwrite:
                            code_path.unlink(missing_ok=True)
                        LOGGER.error("%s/%s: %s; raw response retained.", task, name, exc)
                        continue
                    code_path.write_text(code, encoding="utf-8")
            LOGGER.info("%s: completed %d / %d samples", task, start + len(batch), len(selected))
    return 1 if failures else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("SFT", "RL"), default="RL")
    parser.add_argument(
        "--model-dir", type=relative_path, help="Default: HiCAD_Model/HiCAD-<stage>."
    )
    parser.add_argument("--dataset-dir", type=relative_path, default=Path("HiCAD_Dataset"))
    parser.add_argument("--output-dir", type=relative_path, help="Default: results/HiCAD-<stage>.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    parser.add_argument("--sketch-dir", choices=("handdraw", "humanhanddraw"), default="handdraw")
    parser.add_argument(
        "--sample-ids", nargs="+", help="Select IDs belonging to --split, in the given order."
    )
    parser.add_argument("--limit", type=int, help="Use only the first N selected samples.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Candidates per input.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, help="Default: all visible CUDA GPUs.")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing code files.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs without loading a model."
    )
    args = parser.parse_args(argv)
    args.model_dir = args.model_dir or Path(f"HiCAD_Model/HiCAD-{args.stage}")
    args.output_dir = args.output_dir or Path(f"results/HiCAD-{args.stage}")
    for name in (
        "batch_size",
        "num_samples",
        "max_tokens",
        "max_model_len",
        "limit",
        "tensor_parallel_size",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error(
            "Require temperature >= 0, 0 < top-p <= 1, and 0 < GPU memory utilization <= 1."
        )
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        status = run_inference(args)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        status = 1
    raise SystemExit(status)


if __name__ == "__main__":
    main()
