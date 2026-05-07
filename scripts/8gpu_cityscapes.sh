#!/bin/bash
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

MODEL="/your_model_path/Qwen-Image-Edit-2511"
BACKEND="qwen"

# MODEL="/your_model_path/FireRed-Image-Edit-1.1"
# BACKEND="firered"

# MODEL="/your_model_path/LongCat-Image-Edit"
# BACKEND="longcat"

DATA="data/Cityscapes"
OUTPUT="output/eval_Cityscapes"
N_GPUS=8
SEED=0
MODE="7"  # pass "19", "7", or "both" as first argument

mkdir -p logs/cityscapes

echo "=== Launching $N_GPUS GPU workers (Cityscapes, ${BACKEND}, mode=${MODE}) ==="

pids=()
for i in $(seq 0 $((N_GPUS - 1))); do
    echo "  Starting shard $i on GPU $i ..."
    CUDA_VISIBLE_DEVICES=$i python3 scripts/eval_cityscapes.py \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --data "$DATA" \
        --output "$OUTPUT" \
        --seed "$SEED" \
        --device "cuda" \
        --shard_id "$i" \
        --n_shards "$N_GPUS" \
        --mode "$MODE" \
        > "logs/cityscapes/shard_${i}_${BACKEND}_${MODE}.log" 2>&1 &
    pids+=($!)
done

echo "=== Waiting for all shards to finish ==="
echo "  Monitor: tail -f logs/cityscapes/shard_*_${BACKEND}_${MODE}.log"

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  Shard $i finished successfully"
    else
        echo "  Shard $i FAILED (exit code $?)"
        failed=1
    fi
done

if [ "$failed" -eq 1 ]; then
    echo "ERROR: Some shards failed. Check logs/cityscapes/"
    exit 1
fi

echo "=== Merging results ==="
python3 scripts/eval_cityscapes.py \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --n_shards "$N_GPUS" \
    --mode "$MODE" \
    --merge \
    --miou "per_image"

echo "=== Done ==="
