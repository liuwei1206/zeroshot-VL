#!/bin/bash
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

MODEL="/your_model_path/Qwen-Image-Edit-2511"
BACKEND="qwen"

MODEL="/your_model_path/FireRed-Image-Edit-1.1"
BACKEND="firered"

MODEL="/your_model_path/LongCat-Image-Edit"
BACKEND="longcat"


DATA="data/DIODE/indoors"
OUTPUT="output/eval_diode_indoor"
TASK="depth"
N_GPUS=8
SEED=0

mkdir -p logs/diode_indoor

echo "=== Launching $N_GPUS GPU workers (DIODE Indoor, ${BACKEND}, ${TASK}) ==="

pids=()
for i in $(seq 0 $((N_GPUS - 1))); do
    echo "  Starting shard $i on GPU $i ..."
    CUDA_VISIBLE_DEVICES=$i python3 scripts/eval_nyuv2.py \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --data "$DATA" \
        --output "$OUTPUT" \
        --task "$TASK" \
        --seed "$SEED" \
        --device "cuda" \
        --shard_id "$i" \
        --n_shards "$N_GPUS" \
        > "logs/diode_indoor/shard_${i}_${TASK}_${BACKEND}.log" 2>&1 &
    pids+=($!)
done

echo "=== Waiting for all shards to finish ==="
echo "  Monitor: tail -f logs/diode_indoor/shard_*_${TASK}_${BACKEND}.log"

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
    echo "ERROR: Some shards failed. Check logs/diode_indoor/"
    exit 1
fi

echo "=== Merging results ==="
python3 scripts/eval_nyuv2.py \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --n_shards "$N_GPUS" \
    --merge

echo "=== Done ==="
