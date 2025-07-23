import argparse  
import timeit  
import torch  
import numpy as np  
from contextlib import nullcontext  
from cs336_basics.model import BasicsTransformerLM  


def main():  
    parser = argparse.ArgumentParser(description="Transformer 混合精度内存分析脚本")  
    # 模型超参数  
    parser.add_argument("--vocab_size", type=int, default=50257, help="词汇表大小")  
    parser.add_argument("--context_length", type=int, default=128, help="上下文长度（测试128/256/512）")  
    parser.add_argument("--d_model", type=int, default=2048, help="2.7B模型典型维度")  # 2.7B模型参数适配  
    parser.add_argument("--num_layers", type=int, default=32, help="2.7B模型典型层数")  # 2.7B模型参数适配  
    parser.add_argument("--num_heads", type=int, default=32, help="注意力头数")  
    parser.add_argument("--d_ff", type=int, default=8192, help="前馈网络维度")  
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE参数")  
    # 数据与测试参数  
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小（避免OOM）")  
    parser.add_argument("--sequence_length", type=int, default=128, help="序列长度，与context_length一致")  
    parser.add_argument("--warmup_steps", type=int, default=2, help="热身步骤（减少内存干扰）")  
    parser.add_argument("--measure_steps", type=int, default=1, help="测量步骤（单次运行便于内存分析）")  
    parser.add_argument("--include_backward", action="store_true", help="包含反向传播（完整训练步骤）")  
    parser.add_argument("--include_optimizer", action="store_true", help="包含优化器步骤")  
    # 混合精度与内存分析选项  
    parser.add_argument("--use_bf16", action="store_true", help="启用BF16混合精度")  
    parser.add_argument("--profile_memory", action="store_true", help="启用内存分析并保存快照")  # 新增内存分析开关  

    args = parser.parse_args()  

    # 设备设置  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"使用设备: {device}")  

    # 初始化2.7B模型（适配参数）  
    model = BasicsTransformerLM(  
        vocab_size=args.vocab_size,  
        context_length=args.context_length,  
        d_model=args.d_model,  
        num_layers=args.num_layers,  
        num_heads=args.num_heads,  
        d_ff=args.d_ff,  
        rope_theta=args.rope_theta,  
    ).to(device)  

    # 生成输入数据  
    input_ids = torch.randint(  
        0, args.vocab_size,  
        (args.batch_size, args.sequence_length),  
        device=device  
    )  

    # 优化器（完整训练步骤需要）  
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if args.include_optimizer else None  

    # 混合精度上下文  
    precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.use_bf16 else nullcontext()  

    # 定义步骤函数（含前向/反向/优化器）  
    def step():  
        # 前向传播（混合精度）  
        with precision_context:  
            outputs = model(input_ids)  

        # 反向传播  
        if args.include_backward:  
            loss = outputs.mean()  
            model.zero_grad(set_to_none=True)  
            loss.backward()  

        # 优化器步骤  
        if args.include_optimizer and args.include_backward:  
            optimizer.step()  
            optimizer.zero_grad(set_to_none=True)  

        torch.cuda.synchronize(device)  
        return outputs  

    # 热身步骤  
    print(f"执行 {args.warmup_steps} 次热身步骤...")  
    for _ in range(args.warmup_steps):  
        step()  

    # 内存分析：启动记录（若启用）  
    if args.profile_memory:  
        print("启动内存记录...")  
        torch.cuda.memory._record_memory_history(max_entries=1000000)  # 记录内存历史  

    # 测量步骤（仅1次，便于聚焦内存变化）  
    print(f"执行 {args.measure_steps} 次测量步骤...")  
    for _ in range(args.measure_steps):  
        step()  

    # 内存分析：保存快照并停止记录  
    if args.profile_memory:  
        # 生成快照文件名（区分模式）  
        run_type = "forward_only" if not args.include_backward else "full_training"  
        precision_type = "bf16" if args.use_bf16 else "fp32"  
        snapshot_name = f"memory_snapshot_{run_type}_{precision_type}_ctx{args.context_length}.pickle"  
        torch.cuda.memory._dump_snapshot(snapshot_name)  # 保存快照  
        torch.cuda.memory._record_memory_history(enabled=None)  # 停止记录  
        print(f"内存快照已保存至: {snapshot_name}")  

    print("测试完成")  


if __name__ == "__main__":  
    main()  