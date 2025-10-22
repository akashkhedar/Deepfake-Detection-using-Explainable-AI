import torch


def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  CUDA capability: {torch.cuda.get_device_capability(i)}")
            print(
                f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB"
            )
    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    check_cuda()
