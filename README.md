# zeroshot-VL

Code for the paper: **[Open-Source Image Editing Models Are Zero-Shot Vision Learners](https://arxiv.org/pdf/2605.04566)**

Zero-shot evaluation of open-source image-editing models on dense visual prediction tasks — depth estimation, surface normal estimation, and semantic segmentation — without any fine-tuning.

## Supported Models

| Backend | Model | HuggingFace |
|---------|-------|-------------|
| `qwen` | Qwen-Image-Edit-2511 | [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) |
| `firered` | FireRed-Image-Edit-1.1 | [FireRedTeam/FireRed-Image-Edit-1.1](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1) |
| `longcat` | LongCat-Image-Edit | [meituan-longcat/LongCat-Image-Edit](https://huggingface.co/meituan-longcat/LongCat-Image-Edit) |

## Installation

```bash
git clone https://github.com/liuwei1206/zeroshot-VL.git
cd zeroshot-VL
pip install -r requirements.txt
```

Requires CUDA GPU with **16+ GB VRAM**.

## Data Preparation

### NYUv2 (Depth + Normals)

1. Download the official labeled dataset `nyu_depth_v2_labeled.mat` and `splits.mat`.
2. Download MarrRevisited normals (Ladicky et al.) containing `nm_*.mat` files.
3. Prepare data:

```bash
python scripts/prepare_data.py --dataset nyuv2 --output data/
python scripts/extract_normals.py --marr data/marr --nyuv2 data/nyuv2 --splits data/splits.mat
```

### DIODE (Depth)

Download DIODE validation data from http://diode-dataset.s3.amazonaws.com/val.tar.gz and unpack:

```bash
# Expected layout:
# data/DIODE/indoors/scene_XXXXX/scan_XXXXX/
# data/DIODE/outdoor/scene_XXXXX/scan_XXXXX/
```

### Cityscapes (Semantic Segmentation)

Register at https://www.cityscapes-dataset.com/ and download `leftImg8bit_trainvaltest.zip` + `gtFine_trainvaltest.zip`.

```bash
# Expected layout:
# data/Cityscapes/leftImg8bit/val/
# data/Cityscapes/gtFine/val/
```

## Quick Start

### Single-Image Visualization

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Depth (grayscale prompt, luminance decoding)
python scripts/visualize_single.py \
  --task depth \
  --model /path/to/Qwen-Image-Edit-2511 \
  --backend qwen \
  --dataset nyuv2 \
  --data data/nyuv2 \
  --index 0 \
  -o output/depth_vis

# Surface normals
python scripts/visualize_single.py \
  --task normals \
  --model /path/to/Qwen-Image-Edit-2511 \
  --backend qwen \
  --dataset nyuv2 \
  --data data/nyuv2 \
  --index 0 \
  -o output/normals_vis
```

## Full Evaluation

### NYUv2 (8-GPU parallel)

```bash
bash scripts/8gpu_nyuv2.sh          # depth
bash scripts/8gpu_nyuv2_normal.sh   # normals
```

### DIODE

```bash
bash scripts/8gpu_diode_indoor.sh   # indoor depth
bash scripts/8gpu_diode_outdoor.sh  # outdoor depth
```

### Cityscapes

```bash
bash scripts/8gpu_cityscapes.sh     # semantic segmentation (19-class + 7-category)
```

### Single-GPU Evaluation

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# NYUv2 depth
python scripts/eval_nyuv2.py \
  --model /path/to/model \
  --backend qwen \
  --data data/nyuv2 \
  --output output/eval_nyuv2 \
  --task depth

# NYUv2 normals
python scripts/eval_nyuv2.py \
  --model /path/to/model \
  --backend qwen \
  --data data/nyuv2 \
  --output output/eval_nyuv2 \
  --task normals

# Cityscapes segmentation
python scripts/eval_cityscapes.py \
  --model /path/to/model \
  --backend qwen \
  --data data/Cityscapes \
  --output output/eval_cityscapes
```

## Metrics

| Task | Metrics | Alignment |
|------|---------|-----------|
| Depth | δ₁ ↑, AbsRel ↓, RMSE ↓ | Affine-aligned (scale + offset per image) |
| Normals | Mean/Median angular error ↓, A11/A22/A30 ↑ | Auto axis convention (48 combinations) |
| Segmentation | mIoU ↑, Pixel Accuracy ↑ | Oracle class list, nearest-color decode |

## Project Structure

```
vision_learner/
├── __init__.py          Package exports
├── pipeline.py          VisionLearner + backend registry
├── depth.py             DepthCodec — luminance decoding (ITU-R BT.709)
├── normals.py           NormalCodec — standard normal-map encoding
├── segmentation.py      SegmentationDecoder — nearest-color matching
└── prompt_loader.py     Prompt template loader (Jinja2)

prompts/
├── depth.txt                    Grayscale depth prompt
├── normals.txt                  Surface normal prompt
├── semantic_segmentation_19.txt 19-class Cityscapes segmentation prompt
└── semantic_segmentation_7.txt  7-category Cityscapes segmentation prompt

scripts/
├── eval_nyuv2.py        NYUv2 evaluation (depth + normals)
├── eval_cityscapes.py   Cityscapes evaluation (semantic segmentation)
├── visualize_single.py  Single-image visualization for figures
├── prepare_data.py      Data preparation
├── extract_normals.py   Extract Ladicky normals from MarrRevisited
└── 8gpu_*.sh            Multi-GPU evaluation launchers

merge.sh                 Merge multi-GPU shard results
zero.sh                  Quick single-sample visualization
test.sh                  Evaluation launch helper
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{liu2026opensourceimageeditingmodels,
      title={Open-Source Image Editing Models Are Zero-Shot Vision Learners}, 
      author={Wei Liu and Jiaxin Lin and Rui Chen},
      year={2026},
      eprint={2605.04566},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.04566}, 
}
```

## License

Apache 2.0
