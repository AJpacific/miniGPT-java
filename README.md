# MiniGPT-Java 🧠☕

A GPT-style decoder-only Transformer built **from scratch** in PyTorch, trained on 100 MB of real Java code.

## Highlights

| Spec | Value |
|---|---|
| Architecture | GPT decoder-only, 6 layers, 8 heads, 256-dim embedding |
| Parameters | ~6M |
| Dataset | 100 MB Java code (character-level tokenizer) |
| GPU | NVIDIA GeForce MX570 A (2 GB VRAM) |
| Training | 30,000 steps, ~6 hours, mixed-precision fp16 |
| Final Loss | 0.79 (train) / 0.82 (val) |

## Features

- **Custom Transformer** — Multi-head causal self-attention, residual connections, pre-LayerNorm, GELU MLP, weight tying
- **Mixed Precision (fp16)** — Fits within 2 GB VRAM using `GradScaler`
- **Checkpoint Resume** — Training can be interrupted and resumed seamlessly
- **Configurable** — All hyperparameters in `config.py`
- **Clean Generation** — Temperature and top-k sampling for controlled output

## Setup

```bash
pip install -r requirements.txt
```

### Prepare Dataset

```bash
# Extract 100 MB of Java code from HuggingFace (streaming mode)
python extract_java_100mb.py
```

### Train

```bash
python train.py
```

Checkpoints saved every 1,000 steps to `out_large/ckpt.pt`.

### Generate

```bash
python generate.py
```

Enter a Java prompt (e.g., `public class Solution {`) and the model will autocomplete.

## Architecture

```
Input (B, T) → Token Emb + Pos Emb → Dropout
  → [LayerNorm → CausalSelfAttn → +residual → LayerNorm → MLP → +residual] × 6
  → Final LayerNorm → Linear Head (weight-tied) → Logits (B, T, vocab)
```

## Project Structure

```
mini-gpt-java/
├── config.py          # All hyperparameters
├── model.py           # GPT model (Attention, MLP, Block, GPT)
├── train.py           # Training loop with fp16, checkpointing, eval
├── generate.py        # Interactive code generation
├── requirements.txt   # PyTorch dependency
└── data/              # Dataset (not tracked in git)
```

## Training Progress

```
Step     0: loss 7.92  →  random noise
Step  5000: loss 1.38  →  learning keywords
Step 10000: loss 1.00  →  learning Java patterns
Step 20000: loss 0.85  →  refining predictions
Step 30000: loss 0.79  →  generating valid syntax
```

## Sample Output

**Prompt:** `public class Solution {`

```java
public class Solution {
    public static final String COKSER = "*";
    public DefaultPlugin(ServletRequest request) {
        return Plugin.showWidget();
    }
    public void showWidget(Boundle request, final String context) {
        super(r...
```

## Configuration

Key settings in `config.py`:

```python
block_size   = 256      # Context window (characters)
n_layer      = 6        # Transformer blocks
n_head       = 8        # Attention heads
n_embd       = 256      # Embedding dimension
batch_size   = 4        # Sequences per step
temperature  = 0.45     # Generation creativity
top_k        = 30       # Top-k sampling
```

## Tech Stack

- Python 3.12
- PyTorch 2.10.0 + CUDA 12.8
- Mixed precision fp16
- AdamW optimizer with gradient clipping
