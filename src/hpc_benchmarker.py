import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_hpc(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def run_large_scale_benchmarking(rank, world_size, dataset):
    """
    Project 2306 - HPC Skills.
    Executes VLM interrogation across multiple GPUs/Nodes.
    """
    setup_hpc(rank, world_size)
    
    # Load model to specific GPU
    model = LoadVLM().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

    results = []
    for frames in dataloader:
        # Perform large-scale inference
        scores = ddp_model(frames.to(rank))
        results.append(scores.detach().cpu())
    
    # Gather results from all nodes
    dist.destroy_process_group()
    return results