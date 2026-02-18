# train.py
import os
import time
import torch
import numpy as np
from tqdm import tqdm

from config import CFG
from model import GPT


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text, stoi):
    return np.array([stoi[c] for c in text], dtype=np.int32)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    os.makedirs(CFG.out_dir, exist_ok=True)

    print("Loading text...")
    text = read_text(CFG.data_path)
    print("Chars:", len(text))

    print("Building vocab...")
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    print("Vocab size:", vocab_size)

    # save vocab for generation
    torch.save({"stoi": stoi, "itos": itos}, os.path.join(CFG.out_dir, "vocab.pt"))

    print("Encoding...")
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    # train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    device = torch.device(CFG.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = GPT(
        vocab_size=vocab_size,
        block_size=CFG.block_size,
        n_layer=CFG.n_layer,
        n_head=CFG.n_head,
        n_embd=CFG.n_embd,
        dropout=CFG.dropout,
    ).to(device)

    # fp16 helps on 2GB VRAM
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and CFG.dtype == "float16"))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay
    )

    start_iter = 0
    ckpt_path = os.path.join(CFG.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint["iter"] + 1
        print(f"Resuming from iter {start_iter}")

    print("Training...")
    t0 = time.time()

    for it in tqdm(range(start_iter, CFG.max_iters)):
        # evaluate
        if it % CFG.eval_interval == 0:
            losses = estimate_loss(
                model, train_data, val_data,
                CFG.block_size, CFG.batch_size,
                device, CFG.eval_iters
            )
            print(f"\nstep {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": it,
                "cfg": {k: v for k, v in vars(CFG).items() if not k.startswith("__")},
                "vocab_size": vocab_size
            }
            torch.save(ckpt, os.path.join(CFG.out_dir, "ckpt.pt"))

        xb, yb = get_batch(train_data, CFG.block_size, CFG.batch_size, device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and CFG.dtype == "float16")):
            _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)

        scaler.step(optimizer)
        scaler.update()

    t1 = time.time()
    print("Done. Total time:", (t1 - t0) / 60, "minutes")


if __name__ == "__main__":
    main()
