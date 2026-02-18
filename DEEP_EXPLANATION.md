# MiniGPT-Java — Deep Explanation (Interview Prep)

> A complete reference for confidently explaining every aspect of the project.

---

## Table of Contents

1. [30-Second Interview Pitch](#1-30-second-interview-pitch)
2. [End-to-End System Explanation](#2-end-to-end-system-explanation)
3. [Transformer Architecture Breakdown](#3-transformer-architecture-breakdown)
4. [Training Details](#4-training-details)
5. [Generation Details](#5-generation-details)
6. [Metrics + Evaluation](#6-metrics--evaluation)
7. [Limitations (Honest)](#7-limitations-honest)
8. [Improvements / Next Steps](#8-improvements--next-steps)

---

## 1. 30-Second Interview Pitch

> "I built a GPT-style decoder-only Transformer **from scratch** in PyTorch, trained on 100 MB of real Java code. The entire model was designed to fit within **2 GB of VRAM** on a laptop NVIDIA MX570 GPU. I implemented every component myself — multi-head causal self-attention, positional embeddings, residual connections, layer norm, weight tying — and trained it using mixed-precision fp16 with AdamW and gradient clipping. Over 30,000 training steps (~6 hours), the model's loss dropped from 7.92 to 0.79, and it learned to generate syntactically plausible Java code including class declarations, method signatures, Javadoc comments, and proper indentation. This project gave me deep, hands-on understanding of how large language models actually work under the hood."

**Key numbers to remember:**

| Metric | Value |
|---|---|
| Architecture | GPT decoder-only, 6 layers, 8 heads, 256 embedding |
| Parameters | ~6M |
| Dataset | 100 MB Java code (character-level) |
| GPU | NVIDIA MX570 A (2 GB VRAM) |
| Training | 30,000 steps, ~6 hours, fp16 |
| Final Loss | 0.79 (train), 0.82 (val) |

---

## 2. End-to-End System Explanation

Think of this as a story — data flows in from the left, and generated code comes out on the right.

### Step 1 — Dataset

```
Raw Java code (100 MB .txt file)
├── Sourced from HuggingFace dataset: ajibawa-2023/Java-Code-Large
├── Streamed (not fully downloaded) to avoid memory issues
├── Filtered: min 50 chars, valid Java syntax, max 500 lines
└── Deduplicated by hash to remove exact copies
```

The dataset is a single `.txt` file where thousands of Java code snippets are concatenated together, separated by delimiters. The model sees it as one long stream of characters.

### Step 2 — Character-Level Tokenizer

```python
text = "public class Solution {"
chars = sorted(set(text))          # → [' ', 'S', 'a', 'b', 'c', 'i', 'l', 'n', 'o', 'p', 's', 't', 'u', '{']
stoi  = {ch: i for i, ch in enumerate(chars)}   # char → integer
itos  = {i: ch for i, ch in enumerate(chars)}   # integer → char
```

**Why character-level?** It's the simplest possible tokenizer — every unique character (a-z, A-Z, 0-9, `{`, `}`, `;`, etc.) becomes one token. Our vocab size is ~98 characters.

**Trade-off:** Simple but inefficient. The word `public` requires 6 tokens instead of 1 with BPE. This means the model needs longer context windows to see the same amount of "meaning."

### Step 3 — Encoding the Data

```python
encoded = [stoi[c] for c in text]
# "public class" → [54, 75, 40, 52, 48, 42, 0, 42, 52, 38, 64, 64]
data = torch.tensor(encoded, dtype=torch.long)
# Shape: (total_chars,)  → e.g., (104_857_600,) for 100 MB
```

The entire 100 MB file becomes one giant 1D tensor of integers.

### Step 4 — Train/Val Split

```python
n = int(0.9 * len(data))       # 90% for training
train_data = data[:n]           # first 90%
val_data   = data[n:]           # last 10%
```

We split **sequentially** (not randomly), because the data is already shuffled at the snippet level. Validation data is used to detect overfitting.

### Step 5 — Batching (`get_batch`)

```python
def get_batch(data, block_size=256, batch_size=4, device='cuda'):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size]     for i in ix])   # input
    y = torch.stack([data[i+1 : i + block_size+1] for i in ix])   # target (shifted by 1)
    return x.to(device), y.to(device)
```

**What's happening:**

1. Pick 4 random starting positions in the data
2. For each, grab a window of 256 characters → this is `x` (input)
3. Shift by 1 → this is `y` (target)
4. The model learns: given characters 0..255, predict characters 1..256

```
x = "public class Sol"     ← the model sees this
y = "ublic class Solu"     ← the model must predict this
```

**Shapes:**
- `x`: `(B, T)` = `(4, 256)` — 4 sequences of 256 tokens each
- `y`: `(B, T)` = `(4, 256)` — shifted by 1 position

### Step 6 — Model Forward Pass

```
Input tokens (4, 256)
    │
    ├─→ Token Embedding     → (4, 256, 256)   [lookup table]
    ├─→ Position Embedding  → (1, 256, 256)   [lookup table]
    │
    ├─→ Add + Dropout       → (4, 256, 256)
    │
    ├─→ Block 1 (LayerNorm → Attention → Add → LayerNorm → MLP → Add)
    ├─→ Block 2 ...
    ├─→ Block 3 ...
    ├─→ Block 4 ...
    ├─→ Block 5 ...
    ├─→ Block 6 ...
    │
    ├─→ Final LayerNorm     → (4, 256, 256)
    ├─→ Linear Head         → (4, 256, vocab_size)   [~98]
    │
    └─→ Logits (raw scores for each character at each position)
```

### Step 7 — Loss Calculation

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),   # (4*256, 98)
    targets.view(-1)                     # (4*256,)
)
```

Cross-entropy loss measures: "How surprised is the model by the correct next character?" Lower = better predictions.

### Step 8 — Backward Pass (Backpropagation)

```python
optimizer.zero_grad(set_to_none=True)   # clear old gradients
scaler.scale(loss).backward()            # compute gradients (in fp16)
```

PyTorch's autograd walks backward through every operation, computing ∂loss/∂weight for every parameter. These gradients tell us how to nudge each weight to reduce the loss.

### Step 9 — Optimizer Update

```python
scaler.unscale_(optimizer)                             # convert gradients back to fp32
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent explosion
scaler.step(optimizer)                                 # AdamW updates weights
scaler.update()                                        # adjust fp16 loss scaling
```

AdamW adjusts every weight using its gradient, plus momentum (past gradients) and adaptive learning rates (per-parameter).

### Step 10 — Checkpointing

```python
ckpt = {
    "model": model.state_dict(),        # all weights
    "optimizer": optimizer.state_dict(), # optimizer momentum states
    "iter": it,                          # current step number
    "cfg": {...},                        # hyperparameters
    "vocab_size": vocab_size             # needed to reconstruct model
}
torch.save(ckpt, "out_large/ckpt.pt")
```

Saved every 1,000 steps. Allows resuming training if interrupted.

### Step 11 — Inference / Generation

```python
prompt = "public class Solution {"
idx = encode(prompt)                           # (1, len(prompt))

for _ in range(200):                            # generate 200 new chars
    idx_cond = idx[:, -256:]                    # only last 256 chars (context window)
    logits, _ = model(idx_cond)                # forward pass
    logits = logits[:, -1, :] / temperature    # take last position, scale
    # top-k filtering
    probs = softmax(logits)                    # convert to probabilities
    next_char = multinomial(probs)             # sample one character
    idx = cat([idx, next_char])                # append to sequence
```

The model generates **one character at a time**, autoregressively.

---

## 3. Transformer Architecture Breakdown

### Overview Diagram

```
┌─────────────────────────────────────────────────────┐
│                    GPT Model                        │
│                                                     │
│   Input: (B=4, T=256) integer token IDs             │
│                                                     │
│   ┌─────────────────┐   ┌──────────────────┐        │
│   │ Token Embedding │ + │ Position Embedding│        │
│   │ (vocab, 256)    │   │ (256, 256)        │        │
│   └────────┬────────┘   └────────┬─────────┘        │
│            └─────────┬───────────┘                   │
│                      ▼                               │
│              Dropout(0.1)                            │
│                      │                               │
│            ┌─────────▼──────────┐                    │
│            │   Block × 6        │                    │
│            │ ┌────────────────┐ │                    │
│            │ │ LayerNorm      │ │                    │
│            │ │ ↓              │ │                    │
│            │ │ CausalSelfAttn │ │ ← + residual      │
│            │ │ (8 heads)      │ │                    │
│            │ ├────────────────┤ │                    │
│            │ │ LayerNorm      │ │                    │
│            │ │ ↓              │ │                    │
│            │ │ MLP (4×expand) │ │ ← + residual      │
│            │ └────────────────┘ │                    │
│            └─────────┬──────────┘                    │
│                      ▼                               │
│              Final LayerNorm                         │
│                      │                               │
│              Linear Head (256 → vocab)               │
│              [weight tied with token embedding]      │
│                      │                               │
│              Logits: (B=4, T=256, vocab=~98)         │
└─────────────────────────────────────────────────────┘
```

### 3.1 Token Embedding

```python
self.token_emb = nn.Embedding(vocab_size, n_embd)   # (~98, 256)
```

**What it does:** Converts each integer token ID into a dense vector of 256 numbers.

```
Input:  [54, 75, 40, ...]     → shape (B, T)
Output: [[0.12, -0.3, ...],   → shape (B, T, C) = (4, 256, 256)
          [0.05, 0.8, ...],
          ...]
```

**Why?** Neural networks can't process raw integers. Embeddings give each token a learnable "meaning" vector. Similar tokens (like `a` and `A`) end up with similar vectors after training.

### 3.2 Positional Embedding

```python
self.pos_emb = nn.Embedding(block_size, n_embd)   # (256, 256)
```

**What it does:** Adds position information. Position 0 gets one vector, position 1 gets another, etc.

```
pos = torch.arange(0, T)              # [0, 1, 2, ..., 255]
pos_vectors = self.pos_emb(pos)        # (256, 256)
```

**Why?** Without this, the model cannot distinguish `"AB"` from `"BA"` — it would see the same set of embeddings. Position embeddings tell it **where** each token is in the sequence.

**Combined:**

```python
x = dropout(token_emb(idx) + pos_emb(positions))   # (B, T, C)
```

### 3.3 Self-Attention — The Core Mechanism

Self-attention lets each token look at all previous tokens and decide which ones are relevant.

#### Query, Key, Value (Q, K, V)

```python
self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)   # (256 → 768)
qkv = self.qkv(x)                                        # (B, T, 768)
q, k, v = qkv.split(C, dim=2)                            # each (B, T, 256)
```

**Analogy:**
- **Query (Q):** "I'm the semicolon at position 15. What should I pay attention to?"
- **Key (K):** "I'm the `if` keyword at position 10. Here's my label."
- **Value (V):** "I'm the `if` keyword. Here's the information I carry."

Attention = "How much does my **query** match each token's **key**? Then, take a weighted average of **values**."

#### Reshape for Multi-Head

```python
q = q.view(B, T, n_head, head_dim).transpose(1, 2)   # (B, 8, T, 32)
k = k.view(B, T, n_head, head_dim).transpose(1, 2)   # (B, 8, T, 32)
v = v.view(B, T, n_head, head_dim).transpose(1, 2)   # (B, 8, T, 32)
```

The 256-dim vector is split into 8 heads of 32 dimensions each.

**Shape journey:**
```
(B, T, C)      →  split into heads  →  (B, n_head, T, head_dim)
(4, 256, 256)  →                    →  (4, 8, 256, 32)
```

#### Scaled Dot-Product Attention

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
# (B, 8, 256, 32) @ (B, 8, 32, 256) = (B, 8, 256, 256)
```

This produces a **(T × T) attention matrix** for each head. Entry `[i, j]` says "how much should token `i` attend to token `j`."

**Why scale by √head_dim?** Without scaling, the dot products become very large as dimensions grow. Large values push softmax into regions where gradients are tiny (vanishing gradients). Dividing by √32 ≈ 5.66 keeps values in a reasonable range.

```
Before scaling: att values might be [-50, 80, 120]  → softmax → [0.0, 0.0, 1.0]  (too peaked)
After scaling:  att values become    [-8.8, 14.1, 21.2] → softmax → [0.0, 0.001, 0.999]  (better)
```

#### Causal Masking

```python
att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
att = F.softmax(att, dim=-1)
```

**The mask is a lower-triangular matrix:**

```
mask = [[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]
```

**Why?** This is what makes GPT a *decoder-only* model. Token at position 3 can only "see" tokens at positions 0, 1, 2, 3 — never future tokens. This is essential because during generation, future tokens don't exist yet! If we trained without masking, the model would cheat by looking ahead, and then fail at generation time.

**After masking + softmax:**

```
Before:  [2.1,  3.4,  -inf, -inf]
Softmax: [0.21, 0.79, 0.0,  0.0]    ← future tokens get zero weight
```

#### Weighted Sum of Values

```python
y = att @ v   # (B, 8, 256, 256) @ (B, 8, 256, 32) = (B, 8, 256, 32)
```

Each token's output is a weighted combination of all previous tokens' value vectors.

#### Concatenate Heads and Project

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)   # (B, 256, 256)
y = self.resid_drop(self.proj(y))                    # (B, 256, 256)
```

### 3.4 Multi-Head Attention — Why Multiple Heads?

With 8 heads, the model can attend to 8 different "aspects" simultaneously:

- **Head 1** might learn: "pay attention to the matching opening brace"
- **Head 2** might learn: "pay attention to the variable type declaration"
- **Head 3** might learn: "pay attention to the method name pattern"
- **Head 4** might learn: "pay attention to indentation level"
- etc.

Each head has its own Q, K, V matrices (32 dims each), so each head can specialize.

### 3.5 Feedforward Network (MLP)

```python
class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        self.fc   = nn.Linear(n_embd, 4 * n_embd)    # 256 → 1024 (expand)
        self.proj = nn.Linear(4 * n_embd, n_embd)     # 1024 → 256 (contract)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)        # expand to 4× wider
        x = F.gelu(x)         # non-linear activation
        x = self.proj(x)      # compress back
        x = self.drop(x)
        return x
```

**Shape journey:**
```
(B, T, 256) → Linear → (B, T, 1024) → GELU → Linear → (B, T, 256)
```

**Why 4× expansion?** This gives the network more capacity to transform information. Attention mixes information *between* tokens; the MLP processes information *within* each token. The expansion ratio of 4 is a convention from the original Transformer paper.

**GELU activation:** A smooth version of ReLU. Unlike ReLU (which is exactly 0 for negative values), GELU allows small negative values through, which helps training stability.

### 3.6 Residual Connections

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))   # residual around attention
    x = x + self.mlp(self.ln2(x))    # residual around MLP
    return x
```

**What it does:** Instead of `x = f(x)`, we do `x = x + f(x)`. The output is the original input *plus* whatever the layer learned.

**Why?** Two critical reasons:
1. **Gradient flow:** In deep networks, gradients can vanish as they pass through many layers. The residual connection provides a "highway" for gradients to flow directly backward.
2. **Easier learning:** The layer only needs to learn the **delta** (what to add), not reconstruct the entire representation from scratch.

### 3.7 Layer Normalization

```python
class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, 1e-5)
```

**What it does:** For each token independently, normalizes its 256-dimensional vector to have mean=0, std=1, then applies learned scale and shift.

```
Before: [102.5, -3.2, 88.1, ...]   (arbitrary scale)
After:  [1.21, -0.89, 0.95, ...]   (normalized, then scaled by learnable params)
```

**Why?** Keeps activations in a stable range throughout the network. Without it, values could grow or shrink layer by layer, causing training instability. We use **Pre-LayerNorm** (normalize *before* attention/MLP), which is more stable than the original Transformer's post-norm.

### 3.8 Dropout

```python
self.attn_drop = nn.Dropout(0.1)   # randomly zero out 10% of attention weights
self.resid_drop = nn.Dropout(0.1)  # randomly zero out 10% of residual values
self.drop = nn.Dropout(0.1)        # in MLP
```

**What it does:** During training, randomly sets 10% of values to zero. During inference, does nothing.

**Why?** Prevents overfitting. Forces the network to be redundant — it can't rely on any single neuron, so it learns more robust features.

### 3.9 Weight Tying

```python
self.head = nn.Linear(n_embd, vocab_size, bias=False)
self.head.weight = self.token_emb.weight   # shared weights!
```

**What it does:** The output projection layer and the token embedding layer share the **same weight matrix**.

**Why?**
1. **Saves parameters:** Instead of two (vocab × 256) matrices, we only store one. _(Saves ~25K parameters.)_
2. **Better learning:** Intuitively, the embedding of token "a" should be related to how the model predicts "a". Tying forces this consistency.
3. **Used by GPT-2, GPT-3, and most modern LLMs.**

---

## 4. Training Details

### 4.1 Objective: Next-Token Prediction

The model is trained on a single task: **given a sequence of tokens, predict the next token.**

```
Input:   p u b l i c   c l a s s   S
Target:  u b l i c   c l a s s   S o
```

Every position simultaneously predicts its next character. This is a form of **self-supervised learning** — we don't need manually labeled data.

### 4.2 Why Cross-Entropy Loss?

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

The model outputs a probability distribution over ~98 characters. Cross-entropy measures the "distance" between the model's predicted distribution and the true answer (one-hot).

```
Model predicts: { 'a': 0.01, 'b': 0.02, ..., ';': 0.70, ... }
True answer:    { ';': 1.0, everything else: 0.0 }
Loss = -log(0.70) = 0.36   ← low loss, good prediction

Model predicts: { 'a': 0.01, 'b': 0.02, ..., ';': 0.02, ... }
True answer:    { ';': 1.0 }
Loss = -log(0.02) = 3.91   ← high loss, bad prediction
```

**Why not MSE?** Cross-entropy is designed for classification (picking 1 of N categories). MSE treats all outputs as continuous values, which doesn't match the discrete nature of token prediction.

### 4.3 Why AdamW?

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
```

**Adam** = SGD + momentum + per-parameter adaptive learning rates.

- **Momentum (β₁=0.9):** Uses exponential moving average of past gradients. Smooths out noisy updates.
- **Adaptive LR (β₂=0.999):** Parameters with big gradients get smaller learning rates; parameters with small gradients get bigger ones.
- **"W" = Weight Decay:** Adds `0.1 × weight` to the gradient. This is L2 regularization — it gently pushes weights toward zero to prevent overfitting.

**Why not plain SGD?** Transformers have thousands of parameters with wildly different gradient scales. Adam handles this automatically. Every modern LLM uses Adam or a variant.

### 4.4 Why fp16 + GradScaler?

```python
scaler = torch.cuda.amp.GradScaler(enabled=True)

with torch.cuda.amp.autocast(enabled=True):   # forward pass in fp16
    _, loss = model(xb, yb)

scaler.scale(loss).backward()    # backward in fp16
scaler.step(optimizer)           # weights updated in fp32
scaler.update()
```

**Mixed Precision (fp16):**
- **fp32** (32-bit float): Full precision, 4 bytes per number
- **fp16** (16-bit float): Half precision, 2 bytes per number

**Benefits:**
- **2× less VRAM** for activations and gradients → critical for 2 GB GPU
- **Faster compute** on modern GPUs (tensor cores operate in fp16)

**GradScaler — Why needed?**
fp16 can only represent numbers down to ~6×10⁻⁸. Gradients can be smaller than this and become zero ("underflow"). The scaler:
1. Multiplies the loss by a large factor (e.g., 65536) before backward → gradients are larger → no underflow
2. Divides gradients by the same factor before the optimizer step
3. Dynamically adjusts the scale factor to avoid overflow too

### 4.5 Why Gradient Clipping?

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the model hits a bad batch, gradients can become enormous ("gradient explosion"). Without clipping:

```
Normal gradient:    [0.01, -0.03, 0.02]     → healthy update
Exploded gradient:  [500.0, -800.0, 1200.0] → catastrophic weight change → loss goes to infinity
```

Clipping scales all gradients so the total norm ≤ 1.0. It preserves direction but limits magnitude.

### 4.6 Why block_size Matters

`block_size = 256` means the model can see at most 256 characters of context at a time.

```
"public class Solution {\n    public static void main(String[] args) {\n        ..."
 ↑──── if this is 300 chars long, the model only sees the last 256 ────↑
```

**Larger block_size:**
- ✅ More context → better understanding of long-range patterns
- ❌ Quadratic attention cost: O(T²) → 512 is 4× more compute than 256
- ❌ More VRAM: attention matrix is (T × T) per head per batch

### 4.7 Why Batch Size Impacts VRAM

```
VRAM usage ≈ Model weights + Activations + Gradients + Optimizer states
           ≈ Fixed          + batch_size × block_size × n_embd × ...
```

**batch_size = 4 vs 8:**
- Activations (intermediate values stored for backward pass) scale **linearly** with batch size
- With 2 GB VRAM, batch_size = 8 would cause Out of Memory (OOM) errors

### 4.8 Why VRAM is the Bottleneck, Not RAM

| Resource | Size | Usage |
|---|---|---|
| System RAM | 8-16 GB | Holds dataset, Python, OS. Plenty. |
| GPU VRAM | **2 GB** | Holds model + activations + gradients + optimizer. Tight. |

During training, **everything involved in the forward/backward pass** must be on the GPU. The GPU's 2 GB VRAM becomes the limiting factor for model size, batch size, and context length. System RAM is only used for loading data, which is much less demanding.

---

## 5. Generation Details

### 5.1 Greedy vs Sampling

**Greedy decoding:** Always pick the highest-probability token.

```
probs = [0.7, 0.2, 0.05, 0.05]   → always picks token 0
```

- ✅ Deterministic, consistent output
- ❌ Boring, repetitive. Gets stuck in loops like `}\n}\n}\n}\n`

**Sampling:** Randomly pick a token according to the probability distribution.

```
probs = [0.7, 0.2, 0.05, 0.05]   → usually picks token 0, sometimes 1, rarely 2 or 3
```

- ✅ Creative, varied output
- ❌ Can pick unlikely tokens, producing nonsense

**Our implementation uses sampling** (via `torch.multinomial`).

### 5.2 Temperature

```python
logits = logits / temperature
probs  = softmax(logits)
```

Temperature controls the "sharpness" of the probability distribution:

| Temperature | Effect | Probabilities (example) |
|---|---|---|
| 0.1 (very low) | Almost greedy. Picks the top token almost every time. | [0.99, 0.01, 0.00, 0.00] |
| 0.45 (our setting) | Clean, focused. Mostly picks likely tokens. | [0.85, 0.12, 0.02, 0.01] |
| 1.0 (neutral) | Uses the model's raw probabilities. | [0.70, 0.20, 0.05, 0.05] |
| 2.0 (high) | Flattened. All tokens become more equally likely. | [0.40, 0.30, 0.15, 0.15] |

**Mathematical intuition:** Dividing logits by a number < 1 makes big values bigger and small values smaller → more peaked distribution. Dividing by a number > 1 compresses the range → flatter distribution.

### 5.3 Top-k Filtering

```python
if top_k is not None:
    v, _ = torch.topk(logits, top_k)               # get top 30 values
    logits[logits < v[:, [-1]]] = -float("inf")     # zero out the rest
probs = F.softmax(logits, dim=-1)
```

With `top_k = 30`: Only the top 30 most likely characters are considered. All others get probability = 0.

```
Before top-k: probs = [0.25, 0.20, 0.15, ..., 0.001, 0.0005, ...]   (98 options)
After top-k:  probs = [0.27, 0.22, 0.16, ..., 0.0,   0.0,    ...]   (only 30 non-zero)
```

**Why?** Even trained models assign small but non-zero probability to absurd tokens. Top-k prevents sampling from the "long tail" of bad options.

### 5.4 Why High Temperature Causes Nonsense

With temperature = 2.0:
```
Original logits: [5.0, 3.0, 1.0, -1.0, -3.0]
After /2.0:      [2.5, 1.5, 0.5, -0.5, -1.5]
Softmax:         [0.39, 0.24, 0.15, 0.09, 0.03]
```

Tokens that the model is ~95% sure are wrong now have 15-30% chance of being picked. Over 200 characters, even a few bad choices compound — the model drifts into a region of "token space" it's never seen during training, and output becomes gibberish.

### 5.5 How Prompts Influence Output

The model generates based on the **last `block_size` (256) characters** of context. The prompt sets the initial context:

```
Prompt: "public class Solution {"
→ The model has seen thousands of examples starting like this
→ It knows what typically follows: method declarations, fields, comments
→ Output is Java-like

Prompt: "x"
→ Very little context. 'x' could be anything.
→ Output is more random and less structured
```

### 5.6 Why Longer Prompts Work Better

1. **More context = more constraints.** A 100-character prompt narrows down what can follow much more than a 5-character prompt.
2. **Pattern activation.** Longer prompts are more likely to activate specific patterns the model learned (e.g., the pattern of `public static void main` triggers Android boilerplate).
3. **Closer to training data.** The model saw 256-character chunks during training. A longer prompt mimics that context length, putting the model in a familiar situation.

---

## 6. Metrics + Evaluation

### 6.1 What Train Loss and Val Loss Mean

| Metric | Computed on | Meaning |
|---|---|---|
| **Train loss** | 90% of data (used for weight updates) | "How wrong is the model on data it's actively learning from?" |
| **Val loss** | 10% of data (never used for training) | "How wrong is the model on data it's never seen?" |

Both are **cross-entropy loss.** Lower = better.

### 6.2 What Overfitting Looks Like

```
Normal (healthy):
  Train loss: 0.79 ↓     Val loss: 0.82 ↓     (both decreasing, close together)

Overfitting:
  Train loss: 0.30 ↓     Val loss: 1.50 ↑     (train drops, val rises = memorization!)
```

**Our model:** Train loss = 0.79, Val loss = 0.82. The gap is only 0.03 — **no significant overfitting.** The model is generalizing well.

### 6.3 What Perplexity Is

```
Perplexity = e^(loss)
```

Perplexity = "How many characters is the model equally confused between?"

| Loss | Perplexity | Interpretation |
|---|---|---|
| 7.92 (start) | e^7.92 ≈ 2,745 | "I'm equally confused between nearly all 98 characters" |
| 2.39 (step 1k) | e^2.39 ≈ 10.9 | "I've narrowed it down to about 11 candidates" |
| 1.17 (Phase 1 end) | e^1.17 ≈ 3.22 | "I'm choosing between about 3 characters" |
| 0.79 (Phase 2 end) | e^0.79 ≈ 2.20 | "I'm usually choosing between about 2 characters" |

At perplexity ~2.2, the model is quite confident about its predictions on average — it's only uncertain between ~2 options per character.

### 6.4 Why Loss Improved from 1.17 to 0.79

Four factors contributed:

1. **5× more data (20 MB → 100 MB):** More diverse Java patterns to learn from. Reduces overfitting.
2. **Larger model (4L/128E → 6L/256E):** More parameters means more capacity to memorize patterns. ~4× more parameters.
3. **Longer training (11K → 30K steps):** More passes over the data. The model sees more examples.
4. **Bigger context window (128 → 256):** Can see longer-range dependencies like matching braces and method patterns.

### 6.5 What Quality Improvements Should Be Expected

| Loss Range | Expected Output Quality |
|---|---|
| >4.0 | Random characters, no structure |
| 2.0–4.0 | Some Java keywords appear, basic patterns |
| 1.0–2.0 | Recognizable Java structure: `class`, `void`, braces, indentation |
| 0.5–1.0 | Correct syntax patterns, realistic boilerplate, proper formatting |
| <0.5 | Near-perfect syntax, meaningful variable names (needs much more data/params) |

Our model at 0.79 correctly generates class structures, method signatures, field declarations, Javadoc-style comments, and proper indentation — but the **semantic logic** is still meaningless.

---

## 7. Limitations (Honest)

### 7.1 Why It Doesn't Behave Like ChatGPT

| Aspect | Our MiniGPT | ChatGPT (GPT-4) |
|---|---|---|
| Parameters | ~6 million | ~1.8 **trillion** |
| Training data | 100 MB | ~13 **trillion** tokens |
| Tokenizer | Character-level (~98) | BPE (~100,000) |
| Training | Next-token only | Next-token + RLHF + instruction tuning |
| Context window | 256 chars | 128K tokens (~500K chars) |
| Capabilities | Pattern mimicry | Reasoning, instruction following |

Our model is **300,000× smaller** than GPT-4. It's like comparing a toy airplane to a Boeing 747 — same fundamental principles (wings, lift), vastly different capabilities.

### 7.2 Why It Cannot Reason

Our model is a **statistical pattern matcher**, not a reasoning engine.

```
What it does:     "After 'public static void', the character 'm' is very likely"
What it doesn't:  "I need a main method here because this class needs an entry point"
```

Reasoning requires:
- **World knowledge** (we only trained on raw code, no explanations)
- **Instruction tuning** (RLHF teaches following commands)
- **Scale** (emergent reasoning appears at ~100B+ parameters)

### 7.3 Why Character-Level Tokenization is Weaker

```
BPE:  "public" → [1 token]            → model budget: 1 of 4096 context positions
Char: "public" → [p, u, b, l, i, c]   → model budget: 6 of 256 context positions
```

**Problems:**
- **Wastes context window:** 256 chars ≈ 40 words. With BPE, 256 tokens ≈ 200 words.
- **Harder learning:** The model must learn that `p-u-b-l-i-c` is a word, then that the word is a keyword. BPE handles this automatically.
- **Longer sequences:** More tokens per example → more compute, slower training.

### 7.4 Why Dataset Quality Affects Output

The model can only output patterns it has seen. If training data contains:
- Commented-out dead code → model generates dead code
- Spanish/Portuguese variable names → model generates mixed-language identifiers
- Android boilerplate → model is biased toward Android patterns
- Duplicate snippets → model memorizes and over-represents those patterns

"Garbage in, garbage out" — the model is a mirror of its training data.

---

## 8. Improvements / Next Steps

### 8.1 BPE Tokenizer (Byte Pair Encoding)

**What:** Replace character-level with BPE (e.g., using `tiktoken` or `sentencepiece`).

**Impact:**
- Vocab size: ~98 → ~8,000–32,000
- Context: 256 tokens ≈ 40 words → 256 tokens ≈ 200+ words
- Model sees 5× more "meaning" per forward pass
- **Expected improvement: significant** — this alone could be the biggest single improvement

### 8.2 Top-p Sampling (Nucleus Sampling)

**What:** Instead of keeping the top-k tokens, keep the smallest set of tokens whose cumulative probability exceeds `p` (e.g., 0.9).

```
top-k = 30:   Always keeps exactly 30 tokens (even if 5 would suffice)
top-p = 0.9:  Keeps only enough tokens to cover 90% of probability mass
```

**Impact:** More adaptive than top-k. When the model is confident, it picks from fewer options. When uncertain, it considers more.

### 8.3 Repetition Penalty

**What:** Reduce the logit score of tokens that already appeared in the recent context.

```python
for token_id in recent_tokens:
    logits[token_id] /= repetition_penalty   # e.g., 1.2
```

**Impact:** Eliminates loops like `formatter formatter formatter...`

### 8.4 Dataset Cleaning

**What:** Better filtering of training data:
- Remove non-Java content (comments in other languages, HTML in strings)
- Filter by code quality (e.g., only files with Javadoc)
- Better deduplication (semantic, not just hash-based)
- Balance between different Java patterns (not just Android)

**Impact:** Cleaner training data → cleaner outputs. Currently our model generates Android-flavored code because the dataset is biased that way.

### 8.5 Training Longer / Learning Rate Schedule

**What:**
- Train for 100K+ steps instead of 30K
- Add cosine learning rate schedule (start high, decay to near zero)
- Add warmup (start at very low LR, ramp up over first 1K steps)

```
LR schedule:
Step 0-1000:    warmup 0 → 3e-4
Step 1000-30K:  cosine decay 3e-4 → 3e-5
```

**Impact:** Better convergence. The current constant learning rate means we might oscillate around the minimum instead of settling into it.

### 8.6 LoRA Fine-Tuning an Instruction Model

**What:** Instead of training from scratch, take a pre-trained model (e.g., CodeLlama-7B) and fine-tune it using **LoRA** (Low-Rank Adaptation).

**How LoRA works:**
- Freeze all original weights
- Add small trainable "adapter" matrices (rank 4-16) to attention layers
- Only train ~0.1% of parameters

```
Original:  W (frozen, millions of params)
LoRA:      W + A × B (A and B are tiny, trainable)
```

**Impact:** Could get ChatGPT-level Java generation on the same hardware. LoRA reduces VRAM needs by 10-100× compared to full fine-tuning.

---

## Quick-Reference Cheat Sheet

### Architecture Summary

```
GPT(
  token_emb:  Embedding(98, 256)           ← shared with head
  pos_emb:    Embedding(256, 256)
  drop:       Dropout(0.1)
  blocks:     6 × Block(
                ln1:  LayerNorm(256)
                attn: CausalSelfAttention(256, 8 heads, 32 dim/head)
                ln2:  LayerNorm(256)
                mlp:  MLP(256 → 1024 → 256, GELU)
              )
  ln_f:       LayerNorm(256)
  head:       Linear(256, 98)              ← weight tied
)
```

### Key Shapes

| Tensor | Shape | Description |
|---|---|---|
| Input tokens | (B, T) = (4, 256) | Batch of token IDs |
| After embedding | (B, T, C) = (4, 256, 256) | Token + position vectors |
| Q, K, V (per head) | (B, nh, T, hd) = (4, 8, 256, 32) | Attention inputs |
| Attention matrix | (B, nh, T, T) = (4, 8, 256, 256) | Token-to-token scores |
| MLP hidden | (B, T, 4C) = (4, 256, 1024) | Expanded representation |
| Logits | (B, T, V) = (4, 256, 98) | Raw prediction scores |

### Loss Trajectory

```
Step     0: loss 7.92  (random, perplexity ~2745)
Step  1000: loss 2.39  (learning basic characters)
Step  5000: loss 1.38  (learning keywords and structure)
Step 10000: loss 1.00  (learning Java patterns)
Step 20000: loss 0.85  (refining predictions)
Step 30000: loss 0.79  (final, perplexity ~2.2)
```

### Interview Questions You Might Face

| Question | Key Answer |
|---|---|
| "What is self-attention?" | Each token computes a weighted sum of all previous tokens' representations, where weights are based on query-key similarity. |
| "Why causal masking?" | Prevents the model from seeing future tokens during training, matching the autoregressive generation scenario. |
| "What does temperature do?" | Scales logits before softmax. Lower = more deterministic. Higher = more random. |
| "Why weight tying?" | Saves parameters and enforces consistency between how the model represents tokens and how it predicts them. |
| "Why not just use CPU?" | GPU parallelism. Matrix multiplications (the core of Transformers) are 10-100× faster on GPU. |
| "What would you improve first?" | BPE tokenizer — it would instantly give the model 5× more context per forward pass. |
| "How is this different from ChatGPT?" | Same architecture fundamentals, but 300,000× smaller, no instruction tuning, no RLHF, character-level instead of BPE. |
