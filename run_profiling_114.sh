# 前向传播分析
nsys profile -o nsys_profiles/model_l2_d256_ctx128_forward --trace=cuda,nvtx \
uv run python benchmark_nvtx.py --use_nvtx --num_layers 2 --d_model 256 --context_length 128 --sequence_length 128


# 前向 + 反向传播分析
nsys profile -o nsys_profiles/model_l8_d1024_ctx1024_forward_backward --trace=cuda,nvtx \
uv run python benchmark_nvtx.py --use_nvtx --num_layers 8 --d_model 1024 --context_length 1024 --sequence_length 1024 --include_backward

# 前向 + 反向传播分析
nsys profile -o nsys_profiles/model_l8_d1024_ctx1024_full_training --trace=cuda,nvtx \
uv run python benchmark.py --use_nvtx --num_layers 8 --d_model 1024 --context_length 1024 --sequence_length 1024 --include_backward --include_optimizer

