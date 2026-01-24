import torch

def check_gpu_status():
    # 1. Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("No GPU found. The model will run on the CPU.")
        return []

    # 2. Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # 3. Get the list of GPU names and their current memory status
    gpu_list = []
    print("\n--- GPU Details ---")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        
        # Get memory info (in Gigabytes)
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
        
        gpu_list.append(gpu_name)
        print(f"GPU {i}: {gpu_name}")
        print(f"  - Total Memory: {total_memory:.2f} GB")
        print(f"  - Currently Allocated: {allocated_memory:.2f} GB")

    return gpu_list

# Run the check
available_gpus = check_gpu_status()