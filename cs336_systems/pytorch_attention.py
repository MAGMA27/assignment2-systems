import torch
import itertools
import time
import pandas as pd
import math

# 设置随机种子以确保可重现性
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 固定参数
batch_size = 8
num_heads = 1  # 不使用多头注意力

# 参数范围
d_model_values = [16, 32, 64, 128]
seq_len_values = [256, 1024, 4096, 8192, 16384]

# 用于存储结果
results = []

# 生成随机输入
def generate_random_inputs(batch_size, seq_len, d_model):
    Q = torch.randn(batch_size, seq_len, d_model).cuda()
    K = torch.randn(batch_size, seq_len, d_model).cuda()
    V = torch.randn(batch_size, seq_len, d_model).cuda()
    return Q, K, V

# 前向传播
def forward_pass(Q, K, V):
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    # 应用softmax
    attn = torch.softmax(scores, dim=-1)
    # 计算输出
    output = torch.matmul(attn, V)
    return output

# 反向传播
def backward_pass(output):
    loss = output.sum()
    loss.backward()

# 主测试函数
def run_benchmark():
    for d_model, seq_len in itertools.product(d_model_values, seq_len_values):
        try:
            # 生成随机输入
            Q, K, V = generate_random_inputs(batch_size, seq_len, d_model)
            
            # 预热
            for _ in range(10):
                output = forward_pass(Q, K, V)
                torch.cuda.synchronize()
            
            # 前向传播计时
            start_time = time.time()
            for _ in range(100):
                output = forward_pass(Q, K, V)
                torch.cuda.synchronize()
            forward_time = time.time() - start_time
            
            # 测量内存占用
            torch.cuda.synchronize()
            mem_before_backward = torch.cuda.memory_allocated()
            
            # 反向传播计时
            start_time = time.time()
            for _ in range(100):
                backward_pass(output)
                torch.cuda.synchronize()
            backward_time = time.time() - start_time
            
            # 记录结果
            results.append({
                'd_model': d_model,
                'seq_len': seq_len,
                'forward_time': forward_time,
                'backward_time': backward_time,
                'mem_before_backward': mem_before_backward
            })
            
            print(f"Completed: d_model={d_model}, seq_len={seq_len}")
        
        except RuntimeError as e:
            # 处理内存不足错误
            if "out of memory" in str(e):
                results.append({
                    'd_model': d_model,
                    'seq_len': seq_len,
                    'error': 'Out of Memory'
                })
                print(f"Error: d_model={d_model}, seq_len={seq_len} - Out of Memory")
            else:
                raise e

# 运行测试
run_benchmark()

# 保存结果到CSV
df = pd.DataFrame(results)
df.to_csv('attention_benchmark_results.csv', index=False)