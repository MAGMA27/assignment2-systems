import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(f'fc1 weights : {self.fc1.weight.dtype}')
        print(f'fc1 output : {x.dtype}')
        x = self.ln(x)
        print(f'ln weights : {self.ln.weight.dtype}')
        print(f'ln output : {x.dtype}')
        x = self.fc2(x)
        print(f'fc2 weights : {self.fc2.weight.dtype}')
        print(f'fc2 output : {x.dtype}')
        return x
    
model = ToyModel(10, 10).to('cuda')
dtype = torch.float16
x = torch.randint(0, 10, (1, 10), dtype=torch.float32).to('cuda')

with torch.autocast(device_type="cuda", dtype=dtype):
    y = model(x)
    print(f'output : {x.dtype}')
    loss = torch.mean(y**2)
    print(f'loss : {loss.dtype}')
    loss.backward()
    print(f'grad : {model.fc1.weight.grad.dtype}')
    