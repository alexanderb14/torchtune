#!/bin/bash
set -x
set -e

# Common variables
OUT_DIR=exp_outs

mkdir -p $OUT_DIR

# Log Metrics
# ##################
# Default
tune run full_finetune_single_device --config llama3_1/8B_full_single_device \
&> $OUT_DIR/inductor-default.log

# Reduce overhead
TORCH_COMPILE_MODE=reduce-overhead \
tune run full_finetune_single_device --config llama3_1/8B_full_single_device \
&> $OUT_DIR/inductor-reduce-overhead.log

# Profile
# ##################
PROFILE_FILENAME=$OUT_DIR/trace_inductor-default.json \
tune run full_finetune_single_device --config llama3_1/8B_full_single_device

PROFILE_FILENAME=$OUT_DIR/trace_inductor-reduce-overhead.json \
TORCH_COMPILE_MODE=reduce-overhead \
tune run full_finetune_single_device --config llama3_1/8B_full_single_device
