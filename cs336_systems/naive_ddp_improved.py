import torch
import torch.nn as nn

import os
import torch
import timeit

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

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

def setup(rank, world_size, backend="gloo"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(0)

    dist.init_process_group(backend, rank=rank, world_size=world_size)

def distributed_demo(rank, data, chunk_size, state_dict, world_size, backend="gloo"):
    setup(rank, world_size, backend=backend)

    start = rank * chunk_size
    end = start + chunk_size
    
    my_data = data[start:end, ...].to('cuda')

    # initial model
    model = ToyModel(10, 10).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

    if rank == 0:
        model.load_state_dict(state_dict)

    params = [p.data for p in model.parameters()]
    for param in params:
        dist.broadcast(param, src=0)

    optimizer.zero_grad()
    # forward
    y = model(my_data)
    # loss
    loss = torch.mean(y**2)
    # backward
    loss.backward()
    # all reduce
    # for param in model.parameters():
    #     dist.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG, async_op=False)
    #====================================================================================
    grads = []
    for param in model.parameters():
        grads.append(param.grad)

    flatten_grads = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flatten_grads, op=torch.distributed.ReduceOp.AVG, async_op=False)
    recons_grads = torch._utils._unflatten_dense_tensors(flatten_grads, grads)

    for param, grad in zip(model.parameters(), recons_grads):
        param.grad = grad
    #====================================================================================
    # step
    optimizer.step()

    torch.save(model.state_dict(), f"ckpt/ddp_model_weights_{rank}.pt")


if __name__ == '__main__':
    world_size = 4
    
    # initial data
    data = 10 * torch.rand(40, 10, device='cuda')
    chunk_size = 40 // world_size
    # data = data.share_memory_()
    # initial model
    test_model = ToyModel(10, 10)
    test_model.fc1.weight = nn.Parameter(torch.randn(10, 10))
    test_model.ln.weight = nn.Parameter(torch.randn(10))
    test_model.ln.bias = nn.Parameter(torch.randn(10))
    test_model.fc2.weight = nn.Parameter(torch.randn(10, 10))
    state_dict = test_model.state_dict()
    # for key in state_dict:
    #     state_dict[key] = state_dict[key].share_memory_()
    test_model = test_model.to('cuda')
    # initial optimizer
    optimizer = optim.Adam(test_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    # training
    optimizer.zero_grad()
    ## forward
    y = test_model(data)
    ## loss
    loss = torch.mean(y**2)
    ## backward
    loss.backward()
    ## step
    optimizer.step()

    torch.save(test_model.state_dict(), "ckpt/ddp_model_weights.pt")

    # ddp
    mp.spawn(fn=distributed_demo, args=(data, chunk_size, state_dict, world_size), 
             nprocs=world_size, join=True)

    # check result
    ddp_model = []
    ddp_model.append(torch.load('ckpt/ddp_model_weights_0.pt'))
    ddp_model.append(torch.load('ckpt/ddp_model_weights_1.pt'))
    ddp_model.append(torch.load('ckpt/ddp_model_weights_2.pt'))
    ddp_model.append(torch.load('ckpt/ddp_model_weights_3.pt'))

    single = torch.load('ckpt/ddp_model_weights.pt')

    print(single)
    print(ddp_model[0])
    print(ddp_model[1])

    for dicts in ddp_model:
        for param1, param2 in zip(single, dicts):
            assert torch.allclose(single[param1], dicts[param2], atol=1e-6)