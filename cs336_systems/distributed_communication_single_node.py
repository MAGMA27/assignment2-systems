import os
import torch
import timeit

import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend="gloo"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(0)

    dist.init_process_group(backend, rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, warm_up, backend="gloo"):
    setup(rank, world_size, backend=backend)

    for i in range(warm_up):
        data = torch.randint(0, 10, (3,)).cuda()
        dist.all_reduce(data, async_op=False)

    # print(f"rank {rank} data (before all-reduce): {data}")
    data = torch.randint(0, 10, (3,)).cuda()
    torch.cuda.synchronize()
    # gathered_list = [torch.zeros_like(data) for _ in range(world_size)]

    start = timeit.default_timer()
    dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    
    # print(f"rank {rank} data (after all-reduce): {data}")
    elapsed = (end - start)
    # print(f"代码执行耗时: {elapsed} 秒")
    elapsed_lst = [None]*world_size
    dist.all_gather_object(elapsed_lst, elapsed)
    print(f'rank{rank}: 耗时{elapsed_lst[rank]}s')

    
if __name__ == "__main__":
    world_size = 4
    warm_up = 5

    mp.spawn(fn=distributed_demo, args=(world_size, warm_up, "gloo"), nprocs=world_size, join=True)
