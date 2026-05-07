# model="/your_model_path/Qwen-Image-Edit-2511"
# backend="qwen"

# model="/your_model_path/FireRed-Image-Edit-1.1"
# backend="firered"

# model="/your_model_path/LongCat-Image-Edit"
# backend="longcat"


# <<"COMMENT"
# Depth on NYUv2 sample #36 with Qwen
python scripts/visualize_single.py \
    --task depth \
    --model ${model} \
    --backend ${backend} \
    --dataset nyuv2 \
    --data data/nyuv2 \
    --index 36
# COMMENT


# <<"COMMENT"
# Normals on NYUv2 sample #5
python scripts/visualize_single.py \
    --task normals \
    --model ${model} \
    --backend ${backend} \
    --dataset nyuv2 \
    --data data/nyuv2 \
    --index 5
# COMMENT


<<"COMMENT"
# 19-class segmentation on Cityscapes sample #0
python scripts/visualize_single.py \
    --task seg19 \
    --model ${model} \
    --backend ${backend} \
    --dataset cityscapes \
    --data data/Cityscapes \
    --index 14
COMMENT


<<"COMMENT"
# 7-category segmentation on Cityscapes sample #0
python scripts/visualize_single.py \
    --task seg7 \
    --model ${model} \
    --backend ${backend} \
    --dataset cityscapes \
    --data data/Cityscapes \
    --index 11
COMMENT
