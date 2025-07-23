import torch  
import timeit  
import numpy as np  

# 朴素缩放点积注意力（单头）  
def scaled_dot_product_attention(Q, K, V):  
    d_k = K.shape[-1]  
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)  # (batch, seq_len, seq_len)  
    attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch, seq_len, seq_len)  
    output = torch.matmul(attention_weights, V)  # (batch, seq_len, d_model)  
    return output  


def benchmark_attention():  
    # 配置参数  
    batch_size = 8  
    dmodel_list = [16, 32, 64, 128]  
    seq_len_list = [256, 1024, 4096, 8192, 16384]  
    warmup_steps = 10  # 热身次数  
    measure_steps = 100  # 测量次数  

    # 存储结果：(dmodel, seq_len, forward_time, backward_time, memory_before_backward, oom)  
    results = []  

    for dmodel in dmodel_list:  
        for seq_len in seq_len_list:  
            print(f"测试配置: dmodel={dmodel}, seq_len={seq_len}")  
            try:  
                # 生成随机输入 Q, K, V: (batch_size, seq_len, dmodel)，并启用梯度  
                Q = torch.randn(batch_size, seq_len, dmodel, device="cuda", requires_grad=True)  
                K = torch.randn(batch_size, seq_len, dmodel, device="cuda", requires_grad=True)  
                V = torch.randn(batch_size, seq_len, dmodel, device="cuda", requires_grad=True)  


                # 定义前向+反向步骤（用于热身和测量）  
                def forward_step():  
                    output = scaled_dot_product_attention(Q, K, V)  
                    return output  

                def backward_step():  
                    output = forward_step()  
                    loss = output.sum()  # 构造简单损失  
                    loss.backward()  
                    return output  

                # 热身：前向传播  
                print("  热身前向传播...")  
                for _ in range(warmup_steps):  
                    forward_step()  
                torch.cuda.synchronize()  # 同步GPU  

                # 热身：反向传播  
                print("  热身反向传播...")  
                for _ in range(warmup_steps):  
                    backward_step()  
                    Q.grad = None  # 清零梯度  
                    K.grad = None  
                    V.grad = None  
                torch.cuda.synchronize()  

                # 测量前向传播时间  
                print("  测量前向传播...")  
                forward_times = []  
                for _ in range(measure_steps):  
                    start = timeit.default_timer()  
                    forward_step()  
                    torch.cuda.synchronize()  # 确保GPU完成  
                    end = timeit.default_timer()  
                    forward_times.append(end - start)  
                forward_avg = np.mean(forward_times) * 1000  # 转换为毫秒  

                # 测量反向传播前的内存使用（单位：MB）  
                torch.cuda.synchronize()  
                memory_before_backward = torch.cuda.memory_allocated() / (1024 **2)  

                # 测量反向传播时间  
                print("  测量反向传播...")  
                backward_times = []  
                for _ in range(measure_steps):  
                    start = timeit.default_timer()  
                    backward_step()  
                    Q.grad = None  # 清零梯度  
                    K.grad = None  
                    V.grad = None  
                    torch.cuda.synchronize()  # 确保GPU完成  
                    end = timeit.default_timer()  
                    backward_times.append(end - start)  
                backward_avg = np.mean(backward_times) * 1000  # 转换为毫秒  

                # 记录结果（无OOM）  
                results.append({  
                    "dmodel": dmodel,  
                    "seq_len": seq_len,  
                    "forward_time_ms": forward_avg,  
                    "backward_time_ms": backward_avg,  
                    "memory_before_backward_mb": memory_before_backward,  
                    "oom": False  
                })  
                print("  测试完成\n")  

            except RuntimeError as e:  
                if "out of memory" in str(e):  
                    # 记录OOM错误  
                    results.append({  
                        "dmodel": dmodel,  
                        "seq_len": seq_len,  
                        "forward_time_ms": None,  
                        "backward_time_ms": None,  
                        "memory_before_backward_mb": None,  
                        "oom": True  
                    })  
                    print("  测试失败：显存不足（OOM）\n")  
                else:  
                    # 其他错误  
                    raise e  

    # 输出结果表格  
    print("\n===== 基准测试结果 =====")  
    print(f"{'dmodel':<6} {'seq_len':<6} {'前向时间(ms)':<12} {'反向时间(ms)':<12} {'反向前内存(MB)':<16} {'状态'}")  
    for res in results:  
        oom_flag = "OOM" if res["oom"] else "正常"  
        print(f"{res['dmodel']:<6} {res['seq_len']:<6} "  
              f"{res['forward_time_ms']:.2f}<12" if res['forward_time_ms'] else "N/A".ljust(12),  
              f"{res['backward_time_ms']:.2f}<12" if res['backward_time_ms'] else "N/A".ljust(12),  
              f"{res['memory_before_backward_mb']:.2f}<16" if res['memory_before_backward_mb'] else "N/A".ljust(16),  
              oom_flag)  

    return results  


if __name__ == "__main__":  
    torch.cuda.empty_cache()  # 清空缓存  
    results = benchmark_attention()  
