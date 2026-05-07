export PYTHONPATH="${PYTHONPATH}:$(pwd)"

model_path="/your_model_path/Qwen-Image-Edit-2511"
backend="qwen"

# model_path="/your_model_path/FireRed-Image-Edit-1.1"
# backend="firered"

# model_path="/your_model_path/LongCat-Image-Edit"
# backend="longcat"

<<"COMMENT"
task="normals"        # depth | normals
data="data/nyuv2"

python3 scripts/eval_nyuv2.py \
    --model ${model_path} \
    --backend ${backend} \
    --data ${data} \
    --task ${task} \
    --output output/eval_nyuv2 \
    --n_shards 8 \
    --merge

COMMENT



<<"COMMENT"
task="depth"
data="data/DIODE/indoor"

python3 scripts/eval_nyuv2.py \
    --model ${model_path} \
    --backend ${backend} \
    --data ${data} \
    --task ${task} \
    --output output/eval_diode_indoor_grayscale_v2 \
    --n_shards 8 \
    --merge

COMMENT



<<"COMMENT"
task="depth"
data="data/DIODE/outdoor"

python3 scripts/eval_nyuv2.py \
    --model ${model_path} \
    --backend ${backend} \
    --data ${data} \
    --task ${task} \
    --output output/eval_diode_outdoor_grayscale_v2 \
    --max_depth 350 \
    --n_shards 8 \
    --merge

COMMENT


<<"COMMENT"
data="data/Cityscapes"

python3 scripts/eval_cityscapes.py \
    --model ${model_path} \
    --backend ${backend} \
    --data ${data} \
    --output output/eval_Cityscapes_new \
    --mode="19" \
    --n_shards 8 \
    --merge \
    --miou "per_image"

COMMENT
