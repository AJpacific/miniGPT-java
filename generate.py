# generate.py
import os
import torch

from config import CFG
from model import GPT


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(CFG.out_dir, "ckpt.pt")
    vocab_path = os.path.join(CFG.out_dir, "vocab.pt")

    if not os.path.exists(ckpt_path):
        print("No checkpoint found. Train first.")
        return

    vocab = torch.load(vocab_path)
    stoi, itos = vocab["stoi"], vocab["itos"]

    ckpt = torch.load(ckpt_path, map_location=device)
    vocab_size = ckpt["vocab_size"]

    model = GPT(
        vocab_size=vocab_size,
        block_size=CFG.block_size,
        n_layer=CFG.n_layer,
        n_head=CFG.n_head,
        n_embd=CFG.n_embd,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = input("Enter Java prompt: ")

    idx = torch.tensor([[stoi.get(ch, 0) for ch in prompt]], dtype=torch.long).to(device)

    out = model.generate(
        idx,
        max_new_tokens=200,
        temperature=CFG.temperature,
        top_k=CFG.top_k
    )[0].tolist()

    text = "".join([itos[i] for i in out])
    print("\n--- Generated ---\n")
    print(text)


if __name__ == "__main__":
    main()
