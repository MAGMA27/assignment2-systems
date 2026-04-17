import torch
import cs336_basics
from argparse import Namespace

config = Namespace(
    vocab_size=10000,
    context_length=256,
    d_model=768,
    d_ff=3072,
    num_layers=12,
    num_heads=12,
    rope_theta=10000,
    device='cuda',
    # precise='fp32',
    warmup_steps=5,
    batch_size=32
)


model = cs336_basics.model.BasicsTransformerLM(
    config.vocab_size, config.context_length, config.num_layers, config.d_model, 
    config.num_heads, config.d_ff, config.rope_theta
)

model = model.to(config.device)

data = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
print(data)