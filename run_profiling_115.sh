uv run python cs336_systems/mixed_precision_accumulation.py
uv run python cs336_systems/benchmark_mixed_precision.py --num_layers 2 --d_model 256 --include_backward
uv run python cs336_systems/benchmark_mixed_precision.py --num_layers 2 --d_model 256 --include_backward --use_bf16
uv run python cs336_systems/benchmark_mixed_precision.py --num_layers 4 --d_model 256 --include_backward
uv run python cs336_systems/benchmark_mixed_precision.py --num_layers 4 --d_model 256 --include_backward --use_bf16
