#!/bin/bash
set -x
set -e

# Common variables
OUT_DIR=exp_outs

export CONFIG="llama3_1/8B_full_single_device"
export MAX_NUM_BATCHES=1000
export WITH_WARMUP=1
export TORCH_COMPILE_BACKEND=inductor

mkdir -p $OUT_DIR

# Baseline
TORCH_COMPILE_MODE=default \
tune run full_finetune_single_device --config llama3_1/8B_full_single_device \
&> $OUT_DIR/default_no_padding.log

# CudaGraphs + Padding
multiplier=512
while [ $multiplier -ge 1 ]; do
    TORCH_COMPILE_MODE=reduce-overhead PAD_MULTIPLE=$multiplier \
    tune run full_finetune_single_device --config llama3_1/8B_full_single_device \
    &> $OUT_DIR/cudagraphs_pad$multiplier.log

    multiplier=$((multiplier / 2))
done

# CudaGraphs + No Padding
TORCH_COMPILE_MODE=reduce-overhead PAD_MULTIPLE=64 \
tune run full_finetune_single_device --config llama3_1/8B_full_single_device \
&> $OUT_DIR/cudagraphs_nopad.log
