import torch

def gpu_test():
    try:
        print("CUDA available:", torch.cuda.is_available())
        if not torch.cuda.is_available():
            print("❌ No CUDA device detected. Check drivers and CUDA installation.")
            return

        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_devices}")

        for i in range(num_devices):
            props = torch.cuda.get_device_properties(i)
            print(f"\n--- GPU {i} ---")
            print("Name:", props.name)
            print("Total memory (MB):", props.total_memory // (1024 ** 2))
            print("Memory allocated (MB):", torch.cuda.memory_allocated(i) // (1024 ** 2))
            print("Memory reserved  (MB):", torch.cuda.memory_reserved(i) // (1024 ** 2))

        # Try allocating a small tensor on GPU 0
        try:
            x = torch.randn(1024, 1024, device="cuda:0")
            print("\n✅ Allocation test succeeded on cuda:0")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print("\n❌ Allocation test failed:", e)

    except Exception as e:
        print("Error while testing GPU:", e)

gpu_test()
