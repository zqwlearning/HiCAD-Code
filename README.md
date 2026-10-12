# Rethinking Human Intent-to-CAD: Parametric CAD Model Generation via Cooperative Multi-Task Alignment and Spatial-Aware Reinforcement Learning

<p align="center">
  <img src="images/HiCAD-logo.png" alt="HiCAD logo" width="240">
</p>
<p align="center">
  📄 <a href="https://icml.cc/virtual/2026/poster/66611">Paper</a> &nbsp;｜&nbsp;
  💻 <a href="https://zqwlearning.github.io/HiCAD">Project</a> &nbsp;｜&nbsp;
  📊 <a href="https://drive.google.com/file/d/1vkaGsnvEKfc_sZF5qzL0LBRNErTgIV8u/view?usp=sharing">Dataset</a> &nbsp;｜&nbsp;
  🤗 <a href="https://huggingface.co/zhangqingwang/HiCAD_Model">Model</a> &nbsp;｜&nbsp;
  🖼️ <a href="https://icml.cc/media/PosterPDFs/ICML%202026/66611.png?t=1783269649.354715">Poster</a>
</p>


<div style="text-align: center;">
    <p style="font-weight: bold; font-size: 1.2em;">
        📢🎉 It has been accepted by ICML-26 🎉📢
    </p>
    <img src="images/ICML-logo.svg" style="width:300px; height:auto; display:inline-block;">
</div>
<div style="text-align: center; font-weight: bold; font-size: 1.1em;">
    This is a PyTorch implementation of the “Rethinking Human Intent-to-CAD: Parametric CAD Model Generation via Cooperative Multi-Task Alignment and Spatial-Aware Reinforcement Learning”.
</div>

HiCAD turns sketches and textual design intent into executable **CadQuery Python code**. It supports four tasks within one model:

| Task argument | Input | Design setting |
| --- | --- | --- |
| `only_HDS` | Hand-drawn sketch (HDS) | Sketch-based reconstruction |
| `only_CTD` | Complete textual description (CTD) | Text-based modeling |
| `only_HDS_PTD` | Sketch + partial textual description (PTD) | Completion of incomplete intent |
| `only_HDS_CTD` | Sketch + complete textual description | Fusion of visual and textual constraints |

**HiCAD-SFT** applies Cooperative Multi-Task Alignment (CMTA) to Qwen3-VL-4B-Instruct. **HiCAD-RL** adds Spatial-Aware Reinforcement Learning (SARL), using Group Sequence Policy Optimization (GSPO) and rewards for topology and volumetric overlap. See the [project overview](https://zqwlearning.github.io/HiCAD/) and paper Sections 3–5 for the method and experimental results.

## Repository contents

```text
HiCAD-Code/
├── infer_HiCAD.py
├── hicad_data.py
├── requirements.txt
├── HiCAD_Dataset/
│   ├── train_val_test_splits.json
│   ├── Complete Textual Description.json
│   ├── Partial Textual Description.json
│   ├── cad_geometry.json
│   ├── cadquery/
│   ├── handdraw/
│   └── humanhanddraw/
├── HiCAD_Model/
│   ├── HiCAD-SFT/
│   └── HiCAD-RL/
└── tools/
    ├── step2handdrawn.py
    ├── get_textual_description_from_qwen3vl.py
    ├── create_partial_descriptions.py
    └── textual_description_prompt.txt
```

The code supports dataset construction and inference with SFT/RL checkpoints. The supplied CAD programs and images are example subsets; full-split inference requires the complete dataset. Model weights must be placed in the corresponding `HiCAD_Model` directory in Hugging Face format, including the tokenizer and processor files.

## Installation

Use Python 3.10+ and run commands from `HiCAD-Code`. All path arguments are relative to this directory. GPU inference requires Linux or WSL2 with a CUDA-compatible PyTorch/vLLM installation.

```bash
python -m pip install -r requirements.txt
```

For STEP sketch generation, use a separate geometry environment:

```bash
conda create -n hicad-sketch -c conda-forge python=3.10 pythonocc-core=7.8.1 numpy pillow
conda activate hicad-sketch
```

## Inference

One entry point loads either checkpoint:

```bash
python infer_HiCAD.py --stage SFT
python infer_HiCAD.py --stage RL
```

Both stages use the four task prompts and decoding defaults from the supplied scripts: temperature `0.1`, top-p `0.9`, a `4096`-token budget, and one prediction per input. The default stage is `RL`, and the default split is `test`.

Select tasks or samples as needed:

```bash
python infer_HiCAD.py --stage RL --tasks only_CTD --limit 8
python infer_HiCAD.py --stage SFT --split train --sample-ids 00000061 --batch-size 1
```

Validate all four tasks using the supplied synthetic examples without loading weights:

```bash
python infer_HiCAD.py --stage RL --split train --sample-ids 00000061 00000070 00000073 --dry-run
```

For the supplied human sketches:

```bash
python infer_HiCAD.py --stage RL --sketch-dir humanhanddraw --tasks only_HDS only_HDS_CTD --sample-ids 00003877 00005082 00005083 --dry-run
```

Predictions are saved to `results/HiCAD-<stage>/<task>/cadquery/<sample_id>.py`, with raw responses in the adjacent `responses` directory. Both stages remove `<think>` blocks and outer Python code fences. Existing predictions are skipped; `--overwrite` regenerates them. Use a separate `--output-dir` when changing the checkpoint or decoding settings.

Use `--model-dir` for another local checkpoint and `--batch-size` or `--tensor-parallel-size` to control GPU usage. See `python infer_HiCAD.py --help` for all options.

## Dataset construction

CAD programs originate from DeepCAD models converted to CadQuery by [CAD-Recode](https://github.com/filaPro/cad-recode). Samples share an ID across code, sketches, and descriptions. The dataset contains 82,075 training, 4,536 validation, and 4,015 test samples.

### Hand-drawn sketches

Place STEP files in `HiCAD_Dataset/step`. The renderer projects OpenCascade visible edges onto a 512 × 512 canvas; hidden edges contribute to framing but are not drawn.

```bash
python tools/step2handdrawn.py --step-dir HiCAD_Dataset/step
python tools/step2handdrawn.py --step-dir HiCAD_Dataset/step --jitter --seed 0
```

`--jitter` enables subdivision and noise from the paper's sketch algorithm. The default retains the supplied renderer's unjittered behavior. The nominal camera requires an OpenGL display; `--view-direction 1 1 1` selects a fixed view without opening a viewer.

### Complete textual descriptions

Place CAD renders in `HiCAD_Dataset/images` and Qwen3-VL-32B-Instruct-FP8 in `HiCAD_Model/Qwen3-VL-32B-Instruct-FP8`. The tool combines each render with its CadQuery code, `cad_geometry.json` attributes, and the prompt in `tools/textual_description_prompt.txt`.

```bash
python tools/get_textual_description_from_qwen3vl.py --batch-size 32
```

Descriptions are saved to `HiCAD_Dataset/Complete Textual Description.json`. Existing annotations are skipped; `--overwrite` regenerates selected entries. Use `--dry-run` to check inputs before loading the model.

### Partial textual descriptions

Generate PTD by removing random sentences from CTD while retaining at least one sentence in its original order:

```bash
python tools/create_partial_descriptions.py --seed 0
```

The output is `HiCAD_Dataset/Partial Textual Description.json`. Existing entries are preserved unless `--overwrite` is set. The supplied PTD file contains training and test entries; this tool can also create missing validation entries. Inference reads the saved PTD without remasking it.

All construction tools accept relative input/output paths. Run each tool with `--help` for sample selection and output options.

## Acknowledgments and related references

We thank the authors and maintainers of the following projects:

- [DeepCAD](https://github.com/rundiwu/DeepCAD): Rundi Wu, Chang Xiao, and Changxi Zheng. *DeepCAD: A Deep Generative Network for Computer-Aided Design Models*. ICCV 2021.
- [CAD-Recode](https://github.com/filaPro/cad-recode): Danila Rukhovich et al. *CAD-Recode: Reverse Engineering CAD Code from Point Clouds*. ICCV 2025.
- [Cadrille](https://github.com/col14m/cadrille): Maksim Kolodiazhnyi et al. *cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning*. 2025.
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [DINOv3](https://github.com/facebookresearch/dinov3), [CadQuery](https://github.com/CadQuery/cadquery), and [pythonocc-core](https://github.com/tpaviot/pythonocc-core) for model and geometry tooling.
- [GSPO](https://arxiv.org/abs/2507.18071), [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), [EasyR1](https://github.com/hiyouga/EasyR1), and [vLLM](https://github.com/vllm-project/vllm) for training and inference infrastructure.

## Citation

If HiCAD supports your research, please cite:

```bibtex
@inproceedings{zhang2026hicad,
  title     = {Rethinking Human Intent-to-CAD: Parametric CAD Model Generation via Cooperative Multi-Task Alignment and Spatial-Aware Reinforcement Learning},
  author    = {Zhang, Qingwang and Li, Jiahao and Zhou, Xiangdong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {PMLR 306},
  year      = {2026},
  address   = {Seoul, South Korea},
  url       = {https://zqwlearning.github.io/HiCAD/}
}
```
