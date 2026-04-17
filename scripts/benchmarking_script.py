import torch
import timeit
import cs336_basics.model as model
import cs336_basics.nn_utils as nn_utils
from argparse import Namespace
import pandas as pd

config = Namespace(
    size='',
    vocab_size=10000,
    context_length=256,
    d_model=768,
    d_ff=3072,
    num_layers=12,
    num_heads=12,
    rope_theta=10000,
    device='cuda',
    precise='fp32',
    batch_size=32,
    warmup_steps=5,
    test_steps=1
)


if config.size:
    size_data = {
    'd_model': [768, 1024, 1280, 2560, 4608],
    'd_ff': [3072, 4096, 5120, 10240, 12288],
    'num_layers': [12, 24, 36, 32, 50],
    'num_heads': [12, 16, 20, 32, 36],
    }

    config_df = pd.DataFrame(data=size_data, index=['small', 'medium', 'large', 'xl', '10B'])

    config_row = config_df.loc[config.size]

    config.d_model = config_row['d_model']
    config.d_ff = config_row['d_ff']
    config.num_layers = config_row['num_layers']
    config.num_heads = config_row['num_heads']


model = model.BasicsTransformerLM(
    config.vocab_size, config.context_length, config.d_model, config.num_layers, 
    config.num_heads, config.d_ff, config.rope_theta
)

# print(config.d_model, config.d_ff, config.num_layers, config.num_heads)

model = model.to(config.device)

data = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length)).to(config.device)
target = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length)).to(config.device)

for i in range(config.warmup_steps):
    lm_head = model(data)
    loss = nn_utils.cross_entropy(lm_head, target)
    loss.backward()
    torch.cuda.synchronize()

start = timeit.default_timer()

for i in range(config.test_steps):
    lm_head = model(data)
    loss = nn_utils.cross_entropy(lm_head, target)
    loss.backward()
    torch.cuda.synchronize()

end = timeit.default_timer()
elapsed = (end - start) / config.test_steps

print(f"代码执行耗时: {elapsed} 秒")
