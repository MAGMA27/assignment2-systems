import torch
import torch.nn as nn
import itertools
import timeit
import pandas as pd
import math
import numpy as np

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

class atten_layer(nn.Module):
    def __init__(self, Q, K, V):
        super().__init__()
        self.Q = Q
        self.K = K
        self.V = V

    # 前向传播
    def forward(self):
        # 计算注意力分数
        scores = torch.matmul(self.Q, self.K.transpose(-2, -1)) / math.sqrt(self.Q.size(-1))
        # 应用softmax
        attn = torch.softmax(scores, dim=-1)
        # 计算输出
        output = torch.matmul(attn, self.V)
        return output

    # 反向传播
    def backward_pass(self, output):
        loss = output.sum()
        loss.backward()

# 生成随机输入
def generate_random_inputs(batch_size, seq_len, d_model):
    Q = torch.randn(batch_size, seq_len, d_model, requires_grad=True).cuda()
    K = torch.randn(batch_size, seq_len, d_model, requires_grad=True).cuda()
    V = torch.randn(batch_size, seq_len, d_model, requires_grad=True).cuda()
    return Q, K, V

# 主测试函数
def run_benchmark():
    for d_model, seq_len in itertools.product(d_model_values, seq_len_values):
        try:
            # 生成随机输入
            Q, K, V = generate_random_inputs(batch_size, seq_len, d_model)

            atten = atten_layer(Q, K, V)
            atten = torch.compile(atten)
            
            # 预热
            for _ in range(5):
                output = atten()
                atten.backward_pass(output)

            torch.cuda.synchronize()

            forward_time_lst = []
            mem_before_backward_lst = []
            backward_time_lst = []
            
            for _ in range(10):
                # 前向传播计时
                start_time = timeit.default_timer()
                output = atten()
                torch.cuda.synchronize()
                forward_time = timeit.default_timer() - start_time
                forward_time_lst.append(forward_time)

                # 测量内存占用
                torch.cuda.synchronize()
                mem_before_backward = torch.cuda.memory_allocated()
                mem_before_backward_lst.append(mem_before_backward)
                
                # 反向传播计时
                start_time = timeit.default_timer()
                atten.backward_pass(output)
                torch.cuda.synchronize()
                backward_time = timeit.default_timer() - start_time
                backward_time_lst.append(backward_time)
            
            # 记录结果
            results.append({
                'd_model': d_model,
                'seq_len': seq_len,
                'forward_time': np.mean(forward_time_lst),
                'backward_time': np.mean(backward_time_lst),
                'mem_before_backward': np.mean(mem_before_backward_lst)
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


if __name__ == "__main__":
    # 运行测试
    run_benchmark()

    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv('attention_benchmark_results_compiled.csv', index=False)