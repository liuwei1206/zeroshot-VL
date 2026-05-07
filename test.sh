export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# <<"COMMENT"
python scripts/eval_nyuv2.py \
  --model /your_model_path/Qwen-Image-Edit-2511 \
  --data data/nyuv2 \
  --task normals \
  --max_samples 5 \
  -o output/eval_nyuv2 \
# COMMENT



<<"COMMENT"
python scripts/eval_nyuv2.py \
  --model /your_model_path/Qwen-Image-Edit-2511 \
  --data data/nyuv2 \
  --task depth \
  --max_samples 20 \
  -o output/eval_nyuv2 \
COMMENT


<<"COMMENT"
python scripts/eval_nyuv2.py \
    --model /your_model_path/Qwen-Image-Edit-2511 \
    --data data/DIODE/indoors \
    --task depth \
    --output output/eval_diode_indoor \
    --max_samples 2 \
    --device cuda
COMMENT


<<"COMMENT"
PYTHONPATH=. python3 scripts/eval_nyuv2.py \
    --model /your_model_path/Qwen-Image-Edit-2511 \
    --data data/DIODE/outdoor \
    --task depth \
    --output output/eval_diode_outdoor \
    --max_depth 350 \
    --max_samples 2 \
    --device cuda
COMMENT


<<"COMMENT"
python scripts/eval_cityscapes.py \
    --model /your_model_path/Qwen-Image-Edit-2511 \
    --data data/Cityscapes \
    --max_samples 48 \
    --mode 19 \
    --device cuda
COMMENT


# 先只跑 7 类，快速看效果（不需要校准，不需要清缓存）
<<"COMMENT"
python scripts/eval_cityscapes.py \
  --model /your_model_path/Qwen-Image-Edit-2511 \
  --backend qwen \
  --data data/Cityscapes \
  --max_samples 5 \
  --mode 7 \
  --device cuda
COMMENT



