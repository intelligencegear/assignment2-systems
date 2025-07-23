#!/bin/bash

# 测试前向传播（默认参数）
echo "=== 前向传播测试 ==="
uv run python cs336_systems/benchmark.py --num_layers 8 --d_model 768 --warmup_steps 5 --measure_steps 10
echo -e "\n"

# 测试前向+反向传播（自定义模型大小）
echo "=== 测试前向+反向传播测试 ==="
uv run python cs336_systems/benchmark.py --num_layers 8 --d_model 768 --warmup_steps 5 --measure_steps 10 --include_backward
echo -e "\n"

# 无热身步骤测试
echo "=== 无热身步骤测试 ==="
uv run python cs336_systems/benchmark.py --num_layers 8 --d_model 768 --warmup_steps 0 --measure_steps 10 --include_backward
echo -e "\n"

# 1次热身步骤测试
echo "=== 1次热身步骤测试 ==="
uv run python cs336_systems/benchmark.py --num_layers 8 --d_model 768 --warmup_steps 1 --measure_steps 10 --include_backward
echo -e "\n"

# 2次热身步骤测试
echo "=== 2次热身步骤测试 ==="
uv run python cs336_systems/benchmark.py --num_layers 8 --d_model 768 --warmup_steps 2 --measure_steps 10 --include_backward