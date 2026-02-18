
from datasets import load_dataset
import random

TARGET_MB = 100
TARGET_BYTES = TARGET_MB * 1024 * 1024
output_path = r"mini-gpt-java/data/java_dataset_100mb.txt"

# Setting seed for reproducibility
random.seed(42)

print("Loading dataset in streaming mode...")
ds = load_dataset("ajibawa-2023/Java-Code-Large", split="train", streaming=True)

current_bytes = 0
seen = set()

print(f"Target size: {TARGET_MB} MB")

with open(output_path, "w", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        if i % 10000 == 0:
            print(f"Processed {i} rows. Current size: {current_bytes / (1024 * 1024):.2f} MB")
        
        # User requested to use 'content', but dataset has 'code'
        s = row.get("code", "")
        if not s:
            s = row.get("content", "")
        
        if not isinstance(s, str):
            continue

        s = s.strip()
        # Filter: Length < 50
        if len(s) < 50:
            continue

        # Filter: Must look like Java
        if not ("class " in s or "interface " in s or "enum " in s or "public " in s):
            continue

        # Filter: Too long
        if s.count("\n") > 500:
            continue

        if s in seen:
            continue
        seen.add(s)

        to_write = s + "\n\n"
        b = len(to_write.encode("utf-8"))

        if current_bytes + b > TARGET_BYTES:
            break

        f.write(to_write)
        current_bytes += b

print(f"Done. Final Size: {current_bytes / (1024 * 1024):.2f} MB")
