import argparse
import timeit
import torch
import numpy as np
from cs336_basics.model import BasicsTransformerLM  # 导入模型类


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Transformer 前向/反向传播性能基准测试")
    # 模型超参数
    parser.add_argument("--vocab_size", type=int, default=50257, help="词汇表大小")
    parser.add_argument("--context_length", type=int, default=1024, help="上下文长度")
    parser.add_argument("--d_model", type=int, default=512, help="模型维度")
    parser.add_argument("--num_layers", type=int, default=6, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=2048, help="前馈网络隐藏层维度")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE的theta参数")
    # 数据参数
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--sequence_length", type=int, default=1024, help="序列长度")
    # 基准测试参数
    parser.add_argument("--warmup_steps", type=int, default=5, help="热身步骤数")
    parser.add_argument("--measure_steps", type=int, default=10, help="测量步骤数")
    parser.add_argument("--include_backward", action="store_true", help="是否包含反向传播")
    
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    # 生成随机批次数据 (batch_size, sequence_length)
    input_ids = torch.randint(
        0, args.vocab_size, 
        (args.batch_size, args.sequence_length), 
        device=device
    )

    # 定义单次步骤函数
    def step():
        # 前向传播
        outputs = model(input_ids)
        if args.include_backward:
            # 计算损失（随机目标，仅用于反向传播）
            loss = outputs.mean()
            # 清零梯度
            model.zero_grad(set_to_none=True)
            # 反向传播
            loss.backward()
        # 同步CUDA确保计时准确
        torch.cuda.synchronize(device)
        return outputs  # 仅为避免优化器删除计算图

    # 热身步骤（不计时）
    print(f"执行 {args.warmup_steps} 次热身步骤...")
    for _ in range(args.warmup_steps):
        step()

    # 测量步骤
    print(f"执行 {args.measure_steps} 次测量步骤...")
    times = []
    for _ in range(args.measure_steps):
        start = timeit.default_timer()
        step()
        end = timeit.default_timer()
        times.append(end - start)

    # 计算统计结果
    times_np = np.array(times)
    avg_time = times_np.mean()
    std_time = times_np.std()

    # 输出结果
    step_type = "前向+反向传播" if args.include_backward else "前向传播"
    print(f"\n{step_type}性能统计:")
    print(f"平均耗时: {avg_time:.6f} 秒")
    print(f"标准差: {std_time:.6f} 秒")
    print(f"单次步骤耗时范围: [{times_np.min():.6f}, {times_np.max():.6f}] 秒")


if __name__ == "__main__":
    main()