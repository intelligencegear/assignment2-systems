uv run python cs336_systems/benchmark_memory.py \
    --context_length 128 \
    --sequence_length 128 \
    --use_bf16 \
    --profile_memory \
    --measure_steps 1

# uv run python cs336_systems/benchmark_memory.py \
#     --context_length 128 \
#     --sequence_length 128 \
#     --use_bf16 \
#     --profile_memory \
#     --include_backward \
#     --include_optimizer \
#     --measure_steps 1
