#!/bin/bash
set -x
set -e

# Common variables
OUT_DIR=exp_outs

mkdir -p $OUT_DIR

mkdir -p $OUT_DIR/inductor-default
mkdir -p $OUT_DIR/inductor-reduce-overhead

# Log Metrics
# ##################
for i in {1..10}; do
    # Default
    tune run full_finetune_single_device --config llama3_1/8B_full_single_device \
    &> $OUT_DIR/inductor-default/$i.log

    # Reduce overhead
    TORCH_COMPILE_MODE=reduce-overhead \
    tune run full_finetune_single_device --config llama3_1/8B_full_single_device \
    &> $OUT_DIR/inductor-reduce-overhead/$i.log
done

# Profile
# ##################
PROFILE_FILENAME=$OUT_DIR/trace_inductor-default.json \
tune run full_finetune_single_device --config llama3_1/8B_full_single_device

PROFILE_FILENAME=$OUT_DIR/trace_inductor-reduce-overhead.json \
TORCH_COMPILE_MODE=reduce-overhead \
tune run full_finetune_single_device --config llama3_1/8B_full_single_device
