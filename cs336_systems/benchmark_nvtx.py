import argparse
import timeit
import torch
import numpy as np
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM, scaled_dot_product_attention

# 创建带NVTX标注的注意力函数
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    with nvtx.range("scaled dot product attention"):
        d_k = K.shape[-1]
        
        with nvtx.range("compute attention scores"):
            attention_scores = torch.einsum('...qd,...kd->...qk', Q, K) / np.sqrt(d_k)
        
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float('-inf'))
        
        with nvtx.range("compute softmax"):
            attention_weights = torch.softmax(attention_scores, dim=-1)
        
        with nvtx.range("compute output"):
            output = torch.einsum('...qk,...kd->...qd', attention_weights, V)
        
        return output

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Transformer性能基准测试")
    
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
    parser.add_argument("--include_optimizer", action="store_true", help="是否包含优化器步骤")
    
    # 新增：NVTX和批量测试参数
    parser.add_argument("--use_nvtx", action="store_true", help="使用NVTX标注进行性能分析")
    parser.add_argument("--batch_test", action="store_true", help="批量测试不同模型配置")
    
    args = parser.parse_args()
    
    # 批量测试模式
    if args.batch_test:
        # 定义要测试的模型配置
        test_configs = [
            {"num_layers": 2, "d_model": 256, "context_length": 128},
            {"num_layers": 4, "d_model": 512, "context_length": 256},
            {"num_layers": 6, "d_model": 768, "context_length": 512},
            {"num_layers": 8, "d_model": 1024, "context_length": 1024},
        ]
        
        for config in test_configs:
            print(f"\n=== 测试配置: {config} ===")
            
            # 使用配置参数覆盖默认参数
            for key, value in config.items():
                setattr(args, key, value)
            
            # 执行测试
            run_benchmark(args)
    else:
        # 单配置测试
        run_benchmark(args)

def run_benchmark(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 替换原始注意力函数为带标注的版本
    if args.use_nvtx:
        scaled_dot_product_attention = annotated_scaled_dot_product_attention
        print("已启用NVTX标注")
    
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
    
    # 初始化优化器（仅在需要时）
    optimizer = None
    if args.include_optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        print("已启用优化器步骤")
    
    # 生成随机批次数据
    input_ids = torch.randint(
        0, args.vocab_size, 
        (args.batch_size, args.sequence_length), 
        device=device
    )
    
    # 定义单次步骤函数
    def step():
        if args.use_nvtx:
            nvtx.range_push("training_step")
        
        # 前向传播
        if args.use_nvtx:
            nvtx.range_push("forward")
        outputs = model(input_ids)
        if args.use_nvtx:
            nvtx.range_pop()
        
        if args.include_backward:
            # 反向传播
            if args.use_nvtx:
                nvtx.range_push("backward")
            loss = outputs.mean()
            model.zero_grad(set_to_none=True)
            loss.backward()
            if args.use_nvtx:
                nvtx.range_pop()
            
            # 优化器步骤
            if args.include_optimizer:
                if args.use_nvtx:
                    nvtx.range_push("optimizer_step")
                optimizer.step()
                if args.use_nvtx:
                    nvtx.range_pop()
        
        # 同步CUDA确保计时准确
        torch.cuda.synchronize(device)
        
        if args.use_nvtx:
            nvtx.range_pop()
        
        return outputs
    
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
    step_type = "前向传播"
    if args.include_backward:
        step_type += "+反向传播"
    if args.include_optimizer:
        step_type += "+优化器"
    
    print(f"\n{step_type}性能统计:")
    print(f"平均耗时: {avg_time:.6f} 秒")
    print(f"标准差: {std_time:.6f} 秒")
    print(f"单次步骤耗时范围: [{times_np.min():.6f}, {times_np.max():.6f}] 秒")


if __name__ == "__main__":
    main()