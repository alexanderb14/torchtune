#!/bin/bash
set -x
set -e

# Common variables
OUT_DIR=exp_outs

export CONFIG="llama3_1/8B_full_single_device"
export MAX_NUM_BATCHES=10
export WITH_WARMUP=1
export TORCH_COMPILE_BACKEND=inductor

mkdir -p $OUT_DIR

# Default
TORCH_COMPILE_MODE=default \
tune run full_finetune_single_device --config llama3_1/8B_full_single_device \
&> $OUT_DIR/default.log

# Default without padding
TORCHINDUCTOR_COMPREHENSIVE_PADDING=0 \
TORCHINDUCTOR_SHAPE_PADDING=0 \
TORCH_COMPILE_MODE=default \
tune run full_finetune_single_device --config llama3_1/8B_full_single_device \
&> $OUT_DIR/default_disabled_padding.log

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
