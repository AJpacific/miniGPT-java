# config.py

class CFG:
    # data
    data_path = r"data/java_dataset_100mb.txt"
    out_dir = "out_large"

    # tokenizer
    level = "char"  # char-level tokenizer

    # model (fits 2GB VRAM - Upscaled)
    block_size = 256     # Increased context length
    n_layer = 6          # More layers (was 4)
    n_head = 8           # More heads (was 4)
    n_embd = 256         # Larger embedding (was 128)
    dropout = 0.1

    # training
    batch_size = 4       # Reduced batch size to save VRAM
    max_iters = 30000    # Train longer (was 12000)
    eval_interval = 1000
    eval_iters = 200

    learning_rate = 3e-4
    weight_decay = 0.1
    grad_clip = 1.0

    # device
    device = "cuda"
    dtype = "float16"    # fp16 helps 2GB VRAM

    # generation (Clean Mode)
    temperature = 0.45
    top_k = 30
