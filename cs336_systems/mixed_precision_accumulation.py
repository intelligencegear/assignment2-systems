import torch
import torch.nn as nn
from contextlib import nullcontext

s = torch.tensor(0, dtype=torch.float32)  
for i in range(1000):  
    s += torch.tensor(0.01, dtype=torch.float32)  
print(s)  

s = torch.tensor(0, dtype=torch.float16)  
for i in range(1000):  
    s += torch.tensor(0.01, dtype=torch.float16)  
print(s)  

s = torch.tensor(0, dtype=torch.float32)  
for i in range(1000):  
    s += torch.tensor(0.01, dtype=torch.float16)  
print(s)  

s = torch.tensor(0, dtype=torch.float32)  
for i in range(1000):  
    x = torch.tensor(0.01, dtype=torch.float16)  
    s += x.type(torch.float32)  
print(s)  

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

def test_toy_model_mixed_precision():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToyModel(in_features=5, out_features=3).to(device)
    input_tensor = torch.randn(2, 5, device=device)
    
    # 测试FP32全精度
    print("\n=== FP32全精度测试 ===")
    with nullcontext():  # 空上下文（不使用混合精度）
        test_forward_and_check_types(model, input_tensor)
    
    # 测试FP16混合精度
    print("\n=== FP16混合精度测试 ===")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        test_forward_and_check_types(model, input_tensor)
    
    # 测试BF16混合精度
    print("\n=== BF16混合精度测试 ===")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        test_forward_and_check_types(model, input_tensor)

def test_forward_and_check_types(model, input_tensor):
    # 验证模型参数类型
    print("模型参数类型:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.dtype}")
    
    # 前向传播并捕获中间输出
    fc1_output = model.relu(model.fc1(input_tensor))
    ln_output = model.ln(fc1_output)
    logits = model.fc2(ln_output)
    
    # 计算损失并反向传播
    loss = logits.mean()
    loss.backward()
    
    # 验证各组件类型
    print("\n前向传播组件类型:")
    print(f"  fc1输出: {fc1_output.dtype}")
    print(f"  ln输出: {ln_output.dtype}")
    print(f"  logits: {logits.dtype}")
    print(f"  损失: {loss.dtype}")
    
    # 验证梯度类型
    print("\n梯度类型:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}.grad: {param.grad.dtype}")
        else:
            print(f"  {name}.grad: None")
    
    # 清零梯度以便下次测试
    model.zero_grad()

if __name__ == "__main__":
    test_toy_model_mixed_precision()

