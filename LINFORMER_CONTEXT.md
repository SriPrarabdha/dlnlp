# Linformer experiments: complete context document

## 0. How to use this document (read first)

You are helping **Tejash**, a student who is responsible for the **Linformer** part of a four-student course project on long-document classification with efficient Transformers. Tejash will ask you questions about the Linformer concepts, the code, the experiments, the settings, how to run them on the cluster, and later how to interpret the results.

Treat this document as the ground truth about the project. The facts come from four kinds of sources, labelled in the text:

- **[code]**: read directly from the repository's source code (commit `5703d41`).
- **[measured]**: produced by actually running the code on an RTX 3090 machine.
- **[plan]**: from the team's planning document `Presentation.md`, written before the code. The code decides what will actually run; where the plan and the code differ, this document says so.
- **[paper]**: general knowledge about the original Linformer paper.
- **[estimate]**: a calculation or projection, not a direct measurement.

Rules for answering:

1. Prefer simple words and small examples. Tejash asked for plain explanations.
2. If something is not covered here, say so instead of guessing, and point to the file that would answer it (section 3 lists them).
3. **No real experimental results exist yet.** Only a short "plumbing" test with tiny settings was run; its accuracies were at chance level and mean nothing.
4. This document covers Linformer only. The repository also contains Longformer, Long-Short Transformer, Compressive Transformer and RoFormer experiments that teammates run; they are out of scope here.

Written on 15 October 2026.

---

## 1. The project in one paragraph

The repository (`focused-longdoc`, Python package `longdoc`, GitHub `SriPrarabdha/dlnlp`) trains small Transformer encoders **from scratch** under identical conditions and compares how well different "efficient attention" designs classify long documents (up to 4,096 tokens). Every model first learns language by masked-language-model (MLM) pretraining, then is fine-tuned on classification tasks. Tejash runs only the Linformer suite, started with:

```bash
run_experiment --experiment linformer ...
```

This suite trains four Linformer variants (a 2×2 design), fine-tunes them on real and synthetic tasks with three random seeds each, and runs speed benchmarks, a "quality vs. pretraining amount" curve, and a few inference-only analyses. It is 29 jobs in total.

---

## 2. Concepts

### 2.1 Self-attention and why it is expensive

In a Transformer layer, each token produces three vectors: a **query** (what am I looking for?), a **key** (what do I contain?) and a **value** (what do I pass on?). For each token, attention compares its query with **every** token's key, turns the scores into weights with a softmax, and takes the weighted average of all values.

For a document of n tokens that is an n × n table of scores per head per layer. At n = 4,096 that is about 16.8 million scores, so time and memory grow with n², which is why long documents are expensive.

### 2.2 The Linformer idea [paper]

*"Linformer: Self-Attention with Linear Complexity"*, Wang, Li, Khabsa, Fang and Ma (Facebook AI), 2020, arXiv 2006.04768.

- **Observation.** The n × n attention matrix is approximately **low-rank**: a few directions carry most of the information.
- **Mechanism.** Learn two projection matrices, **E** and **F**, each of shape k × n (k ≪ n). They compress the *sequence* axis of the keys and values:
  - K′ = E·K and V′ = F·V, which reduces n rows to k rows.
  - The attention is softmax(Q·K′ᵀ / √d)·V′.
  - The score table is n × k instead of n × n, so the cost is O(n·k): linear in n.
- **Sharing options in the paper:**
  - headwise: one E and one F per layer, shared by all heads;
  - key-value: E = F;
  - layerwise: one matrix for the whole model.
- **Other ideas the paper mentions:** a different k per layer, and pooling or convolution instead of a learned E.
- **Limitations:**
  - E has exactly n columns, so the maximum length is fixed.
  - Each column belongs to an absolute position.
  - The projection ignores content.
  - It cannot do causal (left-to-right) masking, because each compressed slot mixes all positions.

**Theory [plan, summarising the paper]:**
- Theorem 1: attention can be ε-approximated by a matrix of rank Θ(log n), proved with a Johnson–Lindenstrauss random-projection argument.
- Theorem 2: k = Θ(d log d / ε²), independent of n, is enough for suitable E and F to exist.
- **The "vacuous bound" slide.** The paper's constant, k = 5 ln n / (ε² − ε³), gives k ≈ 4,620 for n = 4,096 and ε = 0.1. That is larger than n itself. These are existence results; in practice the *learned* E and F do the work.

**Results reported for the paper [plan]:**
- At n = 512, k = 256 with layerwise sharing, the GLUE+IMDB average was 92.30, against 92.25 for RoBERTa.
- Pretraining perplexity at k = 256 stayed flat from n = 512 to 4,096.
- At n = 4,096 and k = 128: 3.4× faster inference and 14× less memory.
- Downstream tasks were only tested at n = 512 or 1,024.
- On LRA-Text (byte-level IMDB at 4K), others measured Linformer at **56.1% vs 65.4%** for full attention.

### 2.3 The project's main hypothesis about Linformer

In plain words: **E's columns are tied to positions, and documents have different lengths.**

- A 300-token review only ever touches E's first 300 columns.
- A 4,000-token paper touches almost all of them.

So short and long documents are summarised by effectively different recipes, and most of what E learned is unused for short documents. This may hurt classification when lengths vary. The experiment tests two fixes: **stretch E to each document's length** (`resampled`), and **let the classifying token read every word directly** (`exact_cls`).

### 2.4 The analogy used in earlier discussion

- **64 note-takers.** Linformer has 64 note-takers (k = 64).
- **Instruction sheets.** Each note-taker has a sheet with 4,096 slots, one per word position, saying how much of each word to put into their summary. That sheet is one row of E for keys, and the matching row of F for values.
- **Summaries.** Every word then looks at the 64 summaries instead of all 4,096 words.
- **The judge.** The `[CLS]` token (written `<s>` in this tokenizer) is the "judge": the classifier reads only its final vector.
- **`fixed` (the paper's way).** Slot 1 is always word 1, so short documents only use the start of the sheet.
- **`resampled`.** The sheet is stretched or squeezed like a rubber band to fit each document.
- **`exact_cls`.** The judge alone reads every real word directly; everyone else still reads summaries.

---

## 3. Where things are in the code

| What | File |
|---|---|
| Linformer attention layer (`LinformerAttention`) | `src/longdoc/models/attention.py` (lines ~291–364) |
| Encoder, Transformer block, classifier and MLM heads | `src/longdoc/models/modeling.py` |
| The four Linformer variants ("cells") | `src/longdoc/registry.py` (lines ~44–58) |
| Which tasks and analyses the Linformer suite runs | `src/longdoc/suites.py` (lines ~107–123) |
| Job scheduler, resume and checklist | `src/longdoc/orchestrator.py`, `src/longdoc/status.py` |
| Training, fine-tuning and evaluation loops | `src/longdoc/training/trainer.py` |
| Metrics | `src/longdoc/training/metrics.py` |
| Inference-only analyses (prepend, truncation, spectrum) | `src/longdoc/training/analysis.py` |
| Speed and memory benchmark | `src/longdoc/training/benchmark.py` |
| Datasets, tokenizer, batching, synthetic tasks | `src/longdoc/data/{catalog,download,tokenizer,batching,synthetic}.py` |
| Reports and plots | `src/longdoc/plotting.py` (Linformer plots at lines ~336–346), `src/longdoc/reporting.py` |
| Settings | `configs/base.yaml` (real runs), `configs/quick.yaml` (plumbing test only) |
| Checkpoint cleanup for small disks (added for Tejash) | `scripts/compact_finished.py` |
| Cluster job script for KAIC (Slurm), with Tejash's paths | `scripts/linformer_kaic.sbatch` |
| One-time cluster check job (GPU, data checksums, unit tests, 10-minute smoke run) | `scripts/linformer_kaic_check.sbatch` |
| Command-line entry points | `run_experiment` (= `src/longdoc/cli.py`), `longdoc-data` / `download_data.py` |

---

## 4. The model as implemented [code]

### 4.1 Shared encoder (identical for every model in the project)

- **Input:** token embeddings (vocabulary 32,000; `<pad>` = id 0 has a zero vector) plus **learned absolute position embeddings** (4,096 positions). Then LayerNorm and dropout 0.1.
- **4 Transformer blocks, pre-LayerNorm style:**
  - x ← x + Dropout(Attention(LayerNorm(x)))
  - x ← x + FFN(LayerNorm(x))
  - FFN = Linear(512→2048), GELU, Dropout, Linear(2048→512), Dropout.
- **Size:** width 512, 8 attention heads of size 64, maximum length 4,096, final LayerNorm.
- **Initialisation:** Linear and embedding weights N(0, 0.02), biases 0.
- **Classification head:** take the final vector at position 0 (`[CLS]`), apply dropout 0.1, then Linear(512 → number of labels).
  - Loss is cross-entropy for binary and multiclass tasks (binary uses 2 outputs).
  - Loss is binary cross-entropy with logits for multi-label tasks (ECtHR-A).
- **MLM head (pretraining only):** LayerNorm then a separate (untied) Linear(512 → 32,000) without bias. During training the vocabulary projection is computed only at the masked positions, to save compute.

### 4.2 `LinformerAttention` step by step

Shapes: B documents in the batch, L = padded length of the batch, H = 8 heads, d = 64 per head, k = 64 summary slots.

1. One Linear(512 → 1,536) produces Q, K and V, each split into 8 heads of 64 dimensions.
2. K and V at padding positions are multiplied by 0, so padding can never leak into the summaries.
3. Build the key projection matrix M_E (k × L) and the value projection matrix M_F (k × L) from the learned parameters E and F, according to `projection_type` (section 4.3).
4. Compute K′ = M_E·K (64 summary keys per head) and V′ = M_F·V (64 summary values per head).
5. Compute softmax(Q·K′ᵀ / 8)·V′ using PyTorch's `scaled_dot_product_attention`.
   - There is no mask, because all 64 slots are valid.
   - Dropout 0.1 is applied to the attention weights during training.
6. If `exact_cls` is on: the row for position 0 is replaced by ordinary full attention of q₀ over **all real tokens'** keys and values (padding masked). The same Q, K and V are used; there are no new weights.
7. Output rows at padding positions are set to 0. The heads are merged, then an output Linear(512 → 512) is applied.

Details of E and F:
- Each is an `nn.Parameter` of shape **64 × 4,096**, initialised from N(0, std = 1/√4096 = 1/64).
- Each layer has its own E and its own F (no layerwise or key-value sharing).
- One E and one F are shared by all 8 heads of the layer (the paper's **headwise** sharing).
- That is 2 × 64 × 4,096 = 524,288 parameters per layer, about 2.1M across the 4 layers.
- E and F get weight decay like other 2-D weights.

### 4.3 The three projection types (`model.projection_type`)

**`fixed`: the paper's way.** M_E = E[:, :L], the first L columns. Because padded keys are zero, a document with r real tokens effectively uses only columns 0…r−1. The result does not depend on what else is in the batch.

**`resampled`: the new idea.** For each document, every 64 × 4,096 row of E (and of F) is resized to exactly r columns, where r is that document's real length:

1. **Stretch.** Linear interpolation (`F.interpolate`, mode `linear`, `align_corners=False`) takes r evenly spaced readings across the whole 4,096-column row. Each reading is a blend of the two neighbouring columns.
2. **Size fix.** Each resized row is multiplied by one factor, so its length (L2 norm) equals the norm of the original full 4,096-long row.
3. Zeros are added for the padding positions.

More details:
- **Computed fresh every forward pass** for every document; nothing extra is stored.
- **Gradients flow back** through the interpolation to the original E, which is the only thing learned.
- **For a document of exactly 4,096 tokens, `resampled` equals `fixed`.** Resizing 4,096 → 4,096 is the identity.
- **[measured] for a 300-token document:**
  - `fixed` row norm is 0.27 and `resampled` row norm is 1.00.
  - Under `resampled`, token 0 reads E's column ~6 and token 299 reads column ~4,089; under `fixed`, tokens 0–299 read columns 0–299.
- **Tiny worked example [measured]:** a sheet row [1, 2, 3, 4, 5, 6, 7, 8] and a 4-word document.
  - `fixed` uses [1, 2, 3, 4].
  - `resampled` step 1, stretch: [1.5, 3.5, 5.5, 7.5], halfway values across the whole row.
  - `resampled` step 2, size fix: [2.13, 4.97, 7.82, 10.66], the same shape scaled to the full row's size.

**`mean`: implemented but never run.** This is a parameter-free adaptive mean-pool: the r real positions are split into 64 equal contiguous bins, and each slot is the average of its bin.
- **No experiment uses it.** None of the registered cells select it.
- **`--set` can't turn it on** in single-cell runs, because the cell's own setting overwrites it (section 13). Running it needs a new registry cell.

### 4.4 `exact_cls`

- **A fixed on/off switch** (`model.exact_cls`, True or False), set by which cell is run.
  - **Not learnable.** It never changes during training and adds **no parameters**. The parameter count is identical to the cell without it: 33,190,914 for a 2-label classifier [measured].
- **The same for every document.** It is either on or off for all documents. But *what* `[CLS]` attends to differs per document, because the weights are computed from that document's words each time.
- **What is learned:** the ordinary Q, K and V weights that decide which words `[CLS]` pays attention to.
- **Cost:** one extra attention row per layer, about 4,096 comparisons, against 4,096 × 64 for the rest. That is still linear in n.
- **Never seen in pretraining.** The `exact_cls` cells **reuse the pretrained weights of their parent cell**, which was pretrained *without* `exact_cls`. The switch is turned on only for fine-tuning and testing.
- **Verified [measured]:** with `exact_cls`, row 0 equals true full attention exactly, and all other rows are unchanged.

### 4.5 Model size and cost [code / measured]

- **Parameter counts:**
  - Classifier model: 33.19M parameters (31.1M shared encoder plus 2.1M for E and F).
  - MLM model: 49.57M parameters (the extra is mostly the 32,000 × 512 vocabulary head).
- **Attention cost** grows as n·k. The benchmark's estimated attention FLOPs are: batch × layers × 4 × heads × head_dim × n × k.
- **Peak GPU memory** for batch 4 × 4,096, bf16 [measured]:
  - 1.1–1.7 GiB with gradient checkpointing;
  - 2.1–3.1 GiB without it.

### 4.6 Differences from the paper

| Aspect | Paper | This repo |
|---|---|---|
| Core mechanism (learned E, F compress K and V along the sequence) | yes | yes |
| Sharing | headwise, key-value or layerwise | headwise only; separate E and F per layer |
| k | 128 or 256 in main results | 64 |
| Model | RoBERTa-scale, pretrained on large corpora | 4 layers, width 512, pretrained from scratch on 200M tokens |
| Max length | 512 to 4,096 studied | 4,096 |
| Other projections (pooling, convolution, different k per layer) | mentioned | only mean-pooling coded (unused) |
| Length-adaptive projection (`resampled`) | not in paper | **new** |
| Exact attention for the CLS token (`exact_cls`) | not in paper | **new** |

---

## 5. The experiment design: a 2×2 grid [code]

| | `[CLS]` sees summaries only | `[CLS]` reads every word |
|---|---|---|
| **Paper's position-locked E** | `linformer.fixed` (original Linformer; the suite's baseline) | `linformer.fixed_exact_cls` |
| **Stretched E** | `linformer.resampled` | `linformer.resampled_exact_cls` |

Settings and pretraining source for each cell:

| Cell | `projection_type` | `exact_cls` | Pretraining |
|---|---|---|---|
| `linformer.fixed` | fixed | False | its own (fresh) |
| `linformer.fixed_exact_cls` | fixed | True | **reuses** `linformer.fixed` |
| `linformer.resampled` | resampled | False | its own (fresh) |
| `linformer.resampled_exact_cls` | resampled | True | **reuses** `linformer.resampled` |

Tasks run for each cell (from `suites.py`):

| Cell | Real tasks | Synthetic stress tasks |
|---|---|---|
| `linformer.fixed` | arXiv, SCOTUS, ECtHR-A, IMDB | retrieve, combine, order, distract |
| `linformer.fixed_exact_cls` | IMDB | retrieve |
| `linformer.resampled` | IMDB | retrieve |
| `linformer.resampled_exact_cls` | IMDB | retrieve |

So the **full 2×2 comparison exists only on IMDB and `stress_retrieve`**. The other tasks run only for `linformer.fixed`, which also stands in for "Linformer" in the team's cross-paper comparisons.

The two questions:
- **Question A, fixed vs resampled.** Does tying E's columns to positions hurt when document lengths vary?
- **Question B, without vs with `exact_cls`.** Is the bottleneck that the classifying token only sees compressed summaries?

---

## 6. Data

### 6.1 Tokenizer [code / measured]

- **Training data:** a byte-level BPE with 32,000 tokens, trained only on the **training splits** of the four real datasets, taking documents round-robin so arXiv cannot dominate.
- **Normalisation:** NFC.
- **Special tokens:** `<pad>`=0, `<s>`=1 (the `[CLS]` token), `</s>`=2, `<unk>`=3, `<mask>`=4.
- **Every document** is encoded as `<s> … </s>`.

### 6.2 Real datasets [code / measured on the prepared data]

They are loaded from pinned Hugging Face Parquet snapshots and cleaned as follows:
- Exact duplicate documents shared between splits are removed from the lower-priority split (priority: test > validation > train).
- IMDB additionally drops training reviews that also appear in its test set and repeated training reviews. It then carves a stratified 2,500-review validation set out of training.
- Only training splits are used for the tokenizer and for MLM pretraining.

| Task | What it is | Labels | Train / val / test | Median length (tokens) | > 512 tokens | > 4,096 tokens | Main metric |
|---|---|---|---|---|---|---|---|
| **arXiv** (`ccdv/arxiv-classification`, `no_ref`) | Topic of a full scientific paper | 11 single-label classes: math.AC, cs.CV, cs.AI, cs.SY, math.GR, cs.CE, cs.PL, cs.IT, cs.DS, cs.NE, math.ST (fairly balanced, 2.1K–3.5K train docs each) | 28,044 / 2,479 / 2,500 (21 val and 344 train removed as cross-split duplicates) | 13,737 | 100% | 96.7% | accuracy (+ macro-F1) |
| **SCOTUS** (LexGLUE) | Issue area of a US Supreme Court opinion | 13 classes present in the data (code wrongly configures 14; section 13). Very imbalanced: from 1,043 train docs down to 3 | 5,000 / 1,400 / 1,400 | 6,139 | 93.7% | 64.9% | accuracy (+ macro-F1) |
| **ECtHR-A** (LexGLUE) | Which European Convention articles a court found violated | Multi-label over 10 articles (2, 3, 5, 6, 8, 9, 10, 11, 14, P1-1); 153 of 1,000 test cases have **no** label | 9,000 / 1,000 / 1,000 | 1,294 | 81.7% | 12.9% | micro-F1 (+ macro-F1) |
| **IMDB** | Movie review sentiment | 2 classes (neg/pos), balanced | 22,281 / 2,500 / 25,000 (123 test-overlap and 96 duplicate training reviews removed) | 244 | 16.3% | 0% | accuracy |

How long documents are handled:
- **Head + tail truncation [code].** A document longer than 4,096 tokens keeps its **first 2,048 and last 2,048 tokens**, including `<s>` and `</s>`.
- **arXiv:** the model sees about 30% of a typical paper.
- **IMDB:** never truncated at 4,096, and mostly short. That makes IMDB a good test of "short documents use only the first columns of E".
- **arXiv quirk:** about 7% of arXiv *training* rows (2,066) are the same paper repeated with different labels (papers in several categories). They are kept on purpose. This is label noise, but the test set is barely affected (best possible test accuracy is about 99.8%).
- **Data preparation [measured]:** about 2 min 40 s, 8.4 GB peak RAM, internet access needed.
- **Folder size:** 6.8 GB in total. Only `datasets/` (2.3 GB), `tokenizer/` (2.2 MB) and `manifest.json` are needed to run experiments; the 4.6 GB `cache/` is only used while downloading. Preparation also writes about 1.3 GB into `~/.cache/huggingface`.

### 6.3 Synthetic "stress" tasks [code]

These are artificial binary tasks where **we know exactly where the evidence is**, so we can see which positions and lengths a model can "see".

- **Construction.** Documents are built directly as token IDs of **exactly 1,024, 2,048 or 4,096 tokens** (one third of examples each), so there is no padding.
- **Filler.** The background is a fixed, repeating cycle of 20 words: "amber river quiet cedar stone paper orbit lantern meadow copper winter garden silver canvas harbor gentle valley cloud timber violet".
- **Markers.** Marker phrases are tokenised and written over the filler at a relative position p, at token index ≈ p × length.
- **Sizes:** train 20,000; validation 2,000; test 2,000. The same synthetic data (data seed 17) is used for all fine-tuning seeds.
- **Class words.** A marker containing **"cobalt"** means class 0; **"saffron"** means class 1. The surrounding words differ by split, so exact phrases never repeat across splits:
  - train: "archive flag cobalt 100–123" / "archive flag saffron 124–147";
  - validation: "sealed memorandum …" with numbers 200+;
  - test: "verdict token …" with numbers 300+.

| Task | What the document contains | Label | Train positions | Test positions | Run by |
|---|---|---|---|---|---|
| `stress_retrieve` | one class marker | its class | uniform 0–1 | fixed grid 5%, 25%, 50%, 75%, 95% | all 4 cells |
| `stress_combine` | two class markers, the second at least half the document after the first (first in 3–24%, second up to 97%) | XOR of the two classes (1 if they differ) | random | random, same rule | `fixed` only |
| `stress_order` | an "alpha beacon" marker and a "beta beacon" marker | 1 if alpha comes first | early in 5–25%, late in 75–95% | pairs (5%, 55%), (15%, 65%), (30%, 80%), (45%, 95%) | `fixed` only |
| `stress_distract` | one class marker plus 0, 8 or 32 irrelevant "distractor" phrases ("noise ledger/irrelevant docket/decoy record + colour + instance N") at least 2.5% away from it and 1.2% from each other | the marker's class | random | grid of 5 positions × {0, 8, 32} distractors | `fixed` only |

Test results are broken down by length, position, region (oldest < 25% ≤ middle < 50% ≤ recent < 75% ≤ current), distractor count, and which 512-token segments the two markers fall in.

The tasks mainly test whether a position is **visible** to the model, not language understanding. Spotting "cobalt" vs "saffron" anywhere is enough for retrieve and distract.

---

## 7. Training and evaluation pipeline with every setting

### 7.1 Settings from `configs/base.yaml` [code]

| Group | Setting | Value |
|---|---|---|
| general | seed | 17 |
| model | vocab_size / hidden_size / intermediate_size | 32,000 / 512 / 2,048 |
| model | num_layers / num_heads / max_length / dropout | 4 / 8 / 4,096 / 0.1 |
| model | projection_rank (k) | **64** |
| model | other fields (attention_window, segment_size, memory sizes…) | used only by other architectures |
| train | lengths (MLM pretraining) | [512, 1024, 2048, 4096] |
| train | mlm_tokens | 200,000,000 |
| train | batch_tokens / eval_batch_tokens | 16,384 / 32,768 |
| train | learning_rate / weight_decay / warmup_ratio | 1e-4 / 0.01 / 0.06 |
| train | epochs (fine-tuning) | 3 |
| train | gradient_checkpointing | true by default (**Tejash sets false**, section 12) |
| train | compile | false |
| train | num_workers | 4 by default (**Tejash sets 8**) |
| train | log_every / save_every | 20 / 1,000 steps |
| train | checkpoint_tokens (pretraining snapshots) | 25M, 50M, 100M, 200M |
| train | amp | auto (bf16 on Ampere, Ada and Hopper GPUs) |
| train | finetune_seeds | 17, 29, 43 |
| train | mask_probability | 0.15 |
| monitoring | W&B project / mode | `longdoc-five-paper` / online (**Tejash uses offline**) |
| runtime | resume | auto |
| runtime | checkpoint_every_minutes | 15 |
| runtime | keep_last_checkpoints | 3 (**Tejash sets 0**) |
| runtime | heartbeat_seconds / min_free_disk_gb / fail_fast | 60 / 2 / false |

Optimiser and schedule (same for pretraining and fine-tuning):
- **Optimiser:** AdamW (betas 0.9/0.999, eps 1e-8; fused on GPU).
  - Weight decay 0.01 on 2-D weights, including E and F.
  - No decay on biases and LayerNorm weights.
- **Learning rate:** linear warmup from 0 over the first 6% of steps, then cosine decay to 0.
- **Gradient clipping** at norm 1.0.
- **Precision:** bf16 autocast (fp16 with a gradient scaler only on GPUs without bf16).
- **Out-of-memory handling:** the batch is split into smaller micro-batches with gradient accumulation. This never triggers for Linformer.

### 7.2 MLM pretraining [code / measured]

- **Data.** Training text of all four real datasets, tokenised and concatenated with `</s>` between documents, then packed into rows of exactly L tokens: `<s>` + (L − 2) tokens + `</s>`. There is no padding except the last batch before each snapshot.
- **Corpus balance.** Rows alternate between the four corpora, so each gets 25% of rows. IMDB's small training text is therefore repeated many times, while arXiv is seen less than once.
- **Length cycling.** L cycles 512 → 1,024 → 2,048 → 4,096, one length per step. Every step has 16,384 tokens: 32×512, 16×1,024, 8×2,048 or 4×4,096.
  - Consequence for `fixed`: E's first 512 columns are trained on every step, but columns 2,048–4,095 only on the 4,096-token steps (a quarter of steps).
  - Consequence for `resampled`: E is squeezed to L on every step.
- **Masking.** 15% of non-special tokens are selected; of those, 80% become `<mask>`, 10% a random token and 10% stay unchanged.
- **Length of training:** 200M non-padding tokens = **12,208 steps** (warmup 732).
- **Snapshots.** `checkpoint_tokens_<N>.pt` is saved at 25M, 50M, 100M and 200M tokens, plus a `.json` with step, tokens and training time. The 200M snapshot has the same weights as the final `checkpoint_last.pt`.
- **Cells pretrained:** only `linformer.fixed` and `linformer.resampled` are pretrained; the two `exact_cls` cells reuse them.

### 7.3 Fine-tuning [code]

For each cell, task and seed (17, 29, 43):

1. **Set up the model.**
   - Set the seed. It controls the classifier's initialisation, dropout and the order of training batches.
   - Load the pretrained **encoder** weights; the MLM head is discarded.
   - Create a new classifier head.
2. **Build batches.**
   - Real tasks: 16,384 // 4,096 = **4 documents per batch**, padded to the longest document in the batch (rounded up to a multiple of 8).
   - Stress tasks: batches never mix lengths: 16 documents at 1K, 8 at 2K, 4 at 4K.
3. **Train** for 3 epochs, with the steps per run in the table below.
4. **Pick the best epoch.** After each epoch, evaluate on the full validation set. Keep `checkpoint_best.pt` whenever the validation score strictly improves. The score is accuracy, or micro-F1 for ECtHR-A.
5. **Test once** with the best checkpoint:
   - real tasks at 4,096 tokens (head + tail truncation);
   - stress tasks at all three lengths (1K, 2K, 4K).
6. **Save outputs:** metrics, raw logits and labels (`test_predictions.npz`), and `completion.json`.

Steps per fine-tuning run:

| Task | Steps per epoch | Steps per run (3 epochs) | Warmup steps |
|---|---|---|---|
| arXiv | 7,011 | 21,033 | 1,261 |
| SCOTUS | 1,250 | 3,750 | 225 |
| ECtHR-A | 2,250 | 6,750 | 405 |
| IMDB | 5,571 | 16,713 | 1,002 |
| each stress task | 2,918 | 8,754 | 525 |

### 7.4 Metrics [code]

- **accuracy:** fraction of correct argmax predictions.
- **macro-F1** (multiclass): average F1 over **all configured classes**, with a class that has no true and no predicted examples counting as 0. This is what the SCOTUS bug affects.
- **micro-F1 and macro-F1** (multi-label): a label is predicted when its logit is ≥ 0, i.e. probability ≥ 0.5.
- **`confidence_95`:** a 95% bootstrap interval for each metric (1,000 resamples of the test set; 2.5th and 97.5th percentiles).
- **`length_buckets`:** metrics by original document length: 0–512, 513–1,024, 1,025–2,048, 2,049–4,096 and 4,097+ tokens.
- **`retained_token_coverage`:** the average of min(1, 4,096 / original length), i.e. how much of each document the model saw.
- **Stress tasks only:**
  - `evidence_heatmap`: accuracy per length and position.
  - `stress_breakdown`: accuracy per length, position, region, distractor count, segment pair, and their combinations.
- **Suite summaries** report the mean ± standard deviation over the 3 seeds. Different metrics are never averaged together.

---

## 8. The full Linformer suite: 29 jobs, run one after another on one GPU [code]

1. `pretrain:linformer.fixed`
2. `pretrain:linformer.resampled`
3. For each of the 4 cells, one `benchmark` job plus its `finetune` jobs: 8 for `fixed` and 2 for each other cell (14 fine-tuning jobs × 3 seeds = **42 fine-tuning runs**).
4. `curve:linformer.fixed:{arxiv, scotus, ecthr_a, imdb}`: fine-tune each of the 4 pretraining snapshots on that task with 3 seeds. That is 4 × 4 × 3 = **48 more fine-tuning runs**.
5. Four analyses, each run once per seed on that seed's `checkpoint_best.pt`:
   - `prepend_128` on `resampled_exact_cls`, IMDB;
   - `prepend_512` on `resampled_exact_cls`, IMDB;
   - `projection_spectrum` on `resampled`, IMDB;
   - `truncation` on `resampled_exact_cls`, IMDB.
6. `report`: builds the summaries and plots.

Total: **90 fine-tuning runs.** A job waits for the jobs it depends on; if a dependency fails, the dependent jobs are marked "blocked" and the others continue.

### 8.1 What each analysis and benchmark does [code]

- **`prepend_128` / `prepend_512` (the text-shift test):**
  - At test time it inserts 128 or 512 copies of the token " the" right after `<s>`, which pushes the review to the right. It then cuts to 4,096 tokens and measures accuracy.
  - For comparison with no shift, use the same seed's normal `test_metrics.json`.
  - Why it matters: under `fixed`, every word then meets a different column of E. Under `resampled`, the document just gets longer and E is stretched a bit differently. Under `exact_cls`, `[CLS]` still sees every word.
- **`projection_spectrum`:**
  - Computes the 64 singular values of each learned projection matrix (layers 0–3 × key/value = 8 matrices of 64 × 4,096) of the fine-tuned `resampled` IMDB model.
  - It shows whether E really uses 64 independent directions, or whether a few dominate (low effective rank).
  - It does **not** measure the rank of the attention matrices themselves.
- **`truncation`:** re-tests IMDB with documents cut (head + tail) to 512, 1,024, 2,048 and 4,096 tokens. Since 84% of IMDB reviews are ≤ 512 tokens, the curve is expected to be nearly flat above 512.
- **`benchmark` (one per cell):**
  - Uses random weights and random tokens without padding, in inference mode with bf16. The batch is 32,768 // L documents, with 2 warm-up passes and 10 timed passes at L = 512, 1,024, 2,048 and 4,096.
  - It records documents per second, milliseconds per batch, peak GPU memory, parameter count and estimated attention FLOPs.
- **Compute curve:** the quality of `linformer.fixed` after 25M, 50M, 100M and 200M pretraining tokens, on the 4 real tasks.
  - It answers "how much pretraining does Linformer need?"
  - It's not part of the fixed-vs-resampled question.
  - The 200M point repeats the main `fixed` runs exactly.

---

## 9. Output folder and how to read results

The suite writes everything under `<output-dir>/linformer/<run-id>/` (Tejash uses run-id `main`):

```text
manifest.json                      # state of every job, errors, durations (authoritative)
resolved_config.yaml, suite_plan.yaml
status/CHECKLIST.md                # live human-readable checklist (running/completed/pending/failed)
status/status_summary.json
logs/run.log, logs/events.jsonl
cells/<cell>/pretrain/             # config.yaml, train.jsonl (loss per 20 steps), checkpoint_last.pt,
                                   # checkpoint_tokens_<N>.pt + .json, completion.json, heartbeat.json
cells/<cell>/finetune/<task>/seed_<s>/
    config.yaml, train.jsonl, validation.jsonl (one line per epoch)
    test_metrics.json              # all metrics for this seed
    test_predictions.npz           # ids, logits, labels, coverage, lengths
    checkpoint_best.pt, completion.json, heartbeat.json, logs/
cells/<cell>/benchmark/metrics.json (+ plots/benchmark.svg)
curves/linformer.fixed/<task>/tokens_<N>/finetune/<task>/seed_<s>/...   + complete.json
analyses/<cell>/<analysis>/seed_<s>/result.json (+ plot.svg)
summaries/cells/<cell>.json        # per task: metric, mean, std, seeds, coverage
summaries/suite_summary.json, compute_curves.json, efficiency.json, analyses.json
plots/                             # task_scores, validation_curves, pretraining_loss,
                                   # accuracy_vs_pretraining_compute, throughput_by_length (.svg)
presentation/experiment_comparison.svg   # heatmap: 4 cells × tasks, mean ± std
presentation/length_buckets.svg          # stress_retrieve accuracy at 1K/2K/4K per cell  ← main plot for Question A
presentation/position_shift.svg          # stress_retrieve accuracy vs marker position per cell
```

The name `position_shift.svg` is misleading: it plots accuracy by marker position, not the prepend shift test. The prepend results are in `analyses/linformer.resampled_exact_cls/prepend_*/`.

Important keys in `test_metrics.json`:
- `accuracy`, `macro_f1`, `micro_f1`;
- `confidence_95`;
- `length_buckets`;
- `retained_token_coverage`;
- for stress tasks, `stress_breakdown` with keys such as `length_1024`, `position_0.95`, `length_4096/position_0.05`, `distractors_32` and `region_oldest`.

---

## 10. Measured performance (RTX 3090, bf16, real code) [measured]

### 10.1 Speed and memory per cell

Training throughput at batch 4 × 4,096 with random tokens:

| Cell | Stage | Checkpointing on: tokens/s (peak GiB) | Checkpointing off: tokens/s (peak GiB) |
|---|---|---|---|
| fixed | pretrain | 258,361 (1.7) | 320,323 (3.1) |
| fixed | fine-tune | 295,485 (1.1) | 379,079 (2.1) |
| fixed_exact_cls | pretrain | 204,710 (1.7) | 248,228 (3.1) |
| fixed_exact_cls | fine-tune | 227,295 (1.1) | 282,187 (2.1) |
| resampled | pretrain | 209,072 (1.7) | 282,040 (3.1) |
| resampled | fine-tune | 233,254 (1.1) | 326,209 (2.2) |
| resampled_exact_cls | pretrain | 172,018 (1.7) | 224,372 (3.1) |
| resampled_exact_cls | fine-tune | 187,377 (1.1) | 252,412 (2.2) |

The suite never pretrains the two `exact_cls` cells; their pretraining rows are for reference only.

For context, in the same harness (fine-tuning, checkpointing on), full attention trained at about 89,500 tokens/s, against about 283,000 for Linformer `fixed`. Linformer is roughly **3× faster** than full attention here.

### 10.2 With the real data pipeline (checkpointing off)

**Pretraining:** `fixed` 229,571 tokens/s; `resampled` 149,597 tokens/s. `resampled` builds a separate matrix per row, which costs more with the 32-row batches at length 512.

**Fine-tuning and evaluation:**

| Task | Seconds per step | Validation pass (s) | Test pass (s) |
|---|---|---|---|
| arXiv | 0.067 (4 workers), **0.047 (8 workers)**, 0.047 (12 workers) | 35.5 | 35.5 |
| SCOTUS | 0.048 | 17.7 | 18.7 |
| ECtHR-A | 0.038 | 3.5 | 3.5 |
| IMDB | 0.014 | 2.0 | 17.9 (25,000 reviews) |
| stress task | 0.046 | 3.9 | 3.9 |

Tokenising long arXiv papers on the CPU is a bottleneck with 4 data-loader workers; 8 workers fix it.

### 10.3 Time estimate for the whole suite on one RTX 3090 [estimate]

With checkpointing off and 8 workers:

| Part | Time |
|---|---|
| Pretraining (`fixed` + `resampled`) | 0.6 h |
| Main fine-tuning (42 runs) | 5.5 h (biggest: `fixed` arXiv about 1.0 h for 3 seeds) |
| Compute curves (48 runs) | 6.6 h (arXiv 3.8 h; the duplicate 200M point is 1.7 h of it) |
| Analyses + benchmarks | 0.2 h |
| **Total** | **≈ 13 h** (roughly 16–18 h with the default settings) |

Per seed, a fine-tuning run takes about 19 min on arXiv, 4 min on SCOTUS, 5 min on ECtHR-A, 4 min on IMDB and 7 min per stress task.

### 10.4 Other verified facts

- **End-to-end plumbing test.** The whole Linformer suite ran with `configs/quick.yaml` (29/29 jobs completed, plots produced).
- **Unit tests.** All 55 pass.
- **Resume.** A pretraining run killed with SIGTERM and resumed gave the same losses as an uninterrupted run.
- **Padding.** Masked padding tokens cannot change the classification; there is a unit test for this.

---

## 11. Checkpoints and storage [measured]

Checkpoint sizes:
- **Full checkpoint** (weights + optimiser state): 0.60 GB for pretraining, 0.40 GB for fine-tuning. The optimiser state is only needed to continue an *unfinished* run.
- **Weights only:** 0.20 GB for pretraining, 0.13 GB for fine-tuning.

The default code never deletes old checkpoints:
- Every finished fine-tuning run keeps `checkpoint_best.pt`, `checkpoint_last.pt` (a copy of best with optimiser state) and up to 3 `checkpoint_step_*.pt`.
- Each pretraining run keeps `checkpoint_last.pt`, 4 snapshots and 3 step files.
- In total that is about **205 GB** for the Linformer suite.

---

## 12. Running on the KAIC cluster (decided plan)

### 12.1 Hardware and choice

KAIC has 10 nodes:
- A5000 24 GB (18 cards)
- A6000 48 GB (19 cards)
- A100 80 GB (8 cards)
- ADA6000 48 GB (24 cards)
- H200 140 GB (8 cards)

Jobs may run 48 h, except on the A100 and H200 (24 h).

| GPU | Estimated suite time [estimate] | Verdict |
|---|---|---|
| **ADA6000 48 GB** | ~8–12 h | **chosen**: fastest of the 48-hour cards, most cards available |
| A6000 48 GB | ~13 h | fine |
| A5000 24 GB | ~14–17 h | fine; 24 GB is far more than 3.1 GB |
| A100 80 GB | ~7–11 h | not worth it: 24 h limit, scarce, little gain for a tiny model |
| H200 140 GB | ~7–11 h | not worth it, same reasons |

The other GPUs' times are scaled from the 3090 measurement. Part of every step is CPU work (data loading, Python overhead) that a bigger GPU does not speed up.

**Request:** 1 GPU (the suite uses one GPU, one job at a time), 10–16 CPU cores, 32 GB RAM, 48 h of wall time, and PyTorch ≥ 2.5.

### 12.2 Storage plan: "Option B" (chosen; Tejash has a 75 GB quota)

1. **The data is prepared on Tejash's local machine and copied to the cluster.** Tejash cannot run Python on the cluster login node, so the cluster never downloads anything.
   - **On the local machine** (fustionstor3), the transfer-ready folder is `/home/tejashmore/dlnlp/data/` (git-ignored, 2.3 GB, 51 files). It holds `datasets/`, `tokenizer/`, `manifest.json` and `SHA256SUMS`, a checksum list. The 4.6 GB download `cache/` is left out, because training never reads it.
   - **On the cluster**, it is copied with `rsync` into `/scratch/users/rajivs/datasets_rajivs/tejash_temp`, the `--data-dir`. Only data lives there.
   - **The copy is verified** with `sha256sum -c SHA256SUMS`, which needs no Python.
   - **The folder is relocatable.** A copy at a new path was verified with the checksums, the unit tests and the full 29-job smoke run [measured].
   - **Everything else** (code, environment, results, logs) lives under `/home/rajivs/tejashmore/` (section 12.3).
2. **Add `--set runtime.keep_last_checkpoints=0`.** Step checkpoints are deleted as soon as the next one is written; the newest state survives as `checkpoint_latest.pt`, so resume still works (tested).
3. **Run `scripts/compact_finished.py` every 10 minutes alongside the suite.** For runs whose `completion.json` says "completed":
   - **Fine-tuning folders:** it keeps only `checkpoint_best.pt` as weights only, and deletes the last, latest and step copies.
   - **Pretraining folders:** it keeps `checkpoint_last.pt` and the 4 snapshots as weights only, and deletes latest and step copies.
   - **Unfinished runs** are never touched.
   - **Tested:** after compaction, re-running the suite skipped all 27 completed jobs, and a deliberately removed analysis re-ran correctly from a slimmed checkpoint.

Expected final storage with Option B [estimate]:

| Location | Item | Size |
|---|---|---|
| `/home/rajivs/tejashmore/results` | Fine-tuning checkpoints, 90 × 0.13 GB | 12.0 GB |
| `/home/rajivs/tejashmore/results` | Pretraining checkpoints, 2 × 5 × 0.20 GB | 2.0 GB |
| `/home/rajivs/tejashmore/results` | Files of the run in progress | ~1.5 GB |
| `/home/rajivs/tejashmore/results`, `logs` | Logs, predictions, plots, W&B files | < 1 GB |
| `/home/rajivs/tejashmore/envs` | Python environment (torch + CUDA libraries + packages) | ~5.5 GB |
| `/home/rajivs/tejashmore/dlnlp` | Code | ~2 MB |
| **home total** | | **≈ 22 GB final, ≈ 25 GB peak** |
| scratch `tejash_temp` | Datasets, tokenizer, manifest, checksums | 2.3 GB |
| **scratch total** | | **≈ 2.3 GB** |

Without any of this it would be about 205 GB. With only `keep_last_checkpoints=0`, about 86 GB.

### 12.3 Cluster layout and job script

Data and everything else are kept apart:

| Path | Contents |
|---|---|
| `DATA=/scratch/users/rajivs/datasets_rajivs/tejash_temp` | **data only** (`--data-dir`): `datasets/`, `tokenizer/`, `manifest.json`, `SHA256SUMS`, copied from the local machine |
| `WORK=/home/rajivs/tejashmore` | everything else |
| `$WORK/dlnlp` | the code (all commands are run from here) |
| `$WORK/envs/longdoc` | the conda environment (Python 3.10, torch 2.5.1 + cu121) |
| `$WORK/results` | `--output-dir`; the suite writes to `$WORK/results/linformer/main/` |
| `$WORK/logs` | Slurm job logs |

The job script is `scripts/linformer_kaic.sbatch`. Its Slurm settings:
- one GPU, 12 CPU cores, 32 GB RAM, 48 h;
- `--signal=B:USR1@180` (a warning 3 minutes before the time limit);
- `--partition` must be set to KAIC's ADA6000 partition.

It does the following:
1. Activates the environment and `cd`s into the code folder. This is required, because `configs/base.yaml` and `scripts/` are relative paths.
2. Starts the Option B cleaner loop (`scripts/compact_finished.py` every 10 minutes).
3. Runs:
   ```bash
   run_experiment --experiment linformer --run-id main --data-dir $DATA --output-dir $WORK/results \
     --device cuda --wandb-mode offline --set train.gradient_checkpointing=false \
     --set train.num_workers=8 --set runtime.keep_last_checkpoints=0
   ```
4. **Forwards Slurm's warning signal** to the experiment. With `B:`, Slurm signals only the batch script, so the script traps USR1/TERM and sends USR1 to the Python process. The experiment then saves a checkpoint and exits.
5. Runs a final cleanup and prints the checklist.

Tested by simulation [measured]: sending USR1 to the script produced "Received SIGUSR1; checkpoint saved" and a manifest marked `interrupted`. Re-running the same script resumed pretraining "at step 31".

The code also handles SIGTERM, and Ctrl-C (SIGINT), the same way.

**One-time check job.** Before the real run, `scripts/linformer_kaic_check.sbatch` (1 h limit, same partition) runs inside Slurm, so nothing runs on the login node. It checks four things:
- the GPU and PyTorch;
- the transferred data's checksums and split sizes;
- the 55 unit tests;
- a ~10-minute smoke run of the whole Linformer suite with `configs/quick.yaml`. It should end with "Progress: 29/29"; its accuracies are meaningless.

Verified locally against a fresh copy of the data folder [measured].

### 12.4 Resuming after a time limit or crash

Re-run **exactly the same command**. The manifest skips finished jobs, and the running job continues from `checkpoint_latest.pt`: model, optimiser, scheduler, random-number state, data position and token counters are all restored.

- **Safe to change on resume:** `train.gradient_checkpointing`, any `runtime.*` setting, W&B mode, `--device`.
- **Must NOT change:**
  - `train.num_workers` (the resume is rejected as a "different experiment");
  - the `--data-dir` path (part of the experiment identity too);
  - any model, training or data setting;
  - the `--run-id` and `--output-dir`, which together decide which folder the run lives in. Changing either starts a new run instead of resuming.
- **Useful flags:**
  - `--resume required` refuses to start fresh;
  - `--new-run` deliberately creates a separate run;
  - `--fail-fast` stops at the first failure.
- **Progress:** `cat $OUT/linformer/main/status/CHECKLIST.md`, or `run_experiment --experiment linformer --run-id main --stage status ...` with the same paths.
- **Lock:** a `.suite.lock` file prevents two processes writing to the same run.
- **W&B offline runs** can be uploaded later with `wandb sync` from a machine with internet.

---

## 13. Known bugs and caveats that affect the Linformer experiment

### 13.1 Bugs and pitfalls in the code

1. **SCOTUS has 13 classes, but the code is configured with 14** (`src/longdoc/data/catalog.py`, `num_labels=14`).
   - Accuracy, the selection metric and the main SCOTUS metric, is not affected.
   - **Macro-F1 is deflated:** even perfect predictions give macro-F1 = 0.929 instead of 1.0 [measured].
   - Fix: `num_labels=13`. This must happen before training, because it changes the classifier size.
   - It affects `linformer.fixed` on SCOTUS, both the main runs and the curve runs.
2. **PyTorch older than 2.5 breaks error handling.** `pyproject.toml` allows torch ≥ 2.2, but the trainer catches `torch.OutOfMemoryError`, which only exists from 2.5. On older versions any training error is replaced by a misleading `AttributeError`. Use torch ≥ 2.5.
3. **The defaults are slow for Linformer:**
   - gradient checkpointing (unnecessary at 3 GB; ~1.3× slower);
   - 4 data-loader workers (arXiv becomes CPU-bound; 8 workers are ~16% faster).
   - `num_workers` cannot be changed on resume, so set it from the first submission.
4. **Checkpoints are never cleaned up** (~205 GB). Option B solves this.
5. **The built-in low-disk check looks at the wrong number.** `min_free_disk_gb` checks the filesystem's free space, not the user's quota, so it will not warn before a quota is hit.
6. **W&B is online by default.** On compute nodes without internet, each of the ~110 runs can stall waiting for W&B. Use `--wandb-mode offline` or `--no-wandb`.
7. **Data preparation needs internet** and writes 1.3 GB into `~/.cache/huggingface` in addition to the data folder. In Tejash's setup it is done on the local machine and the prepared folder is copied to the cluster (section 12.2).
8. **The compute curve's 200M point repeats the main `fixed` runs** (wasted ~1.7 h). Removing it needs a small code change.
9. **Never use `--cell` mode for Linformer variants; use `--experiment linformer`.** In `--cell` mode:
   - (a) `--set` values for model settings that a cell defines (`projection_type`, `exact_cls`) are silently overwritten by the cell's own values. For example, `--set model.projection_type=mean` just runs `resampled`.
   - (b) If an `exact_cls` cell runs before its parent has a checkpoint, the code pretrains the parent *with `exact_cls=True`*. It saves it under the parent's name, and later runs reuse that wrong model.

### 13.2 Design limitations (nothing crashes; they affect what can be concluded)

- **`resampled` changes two things at once:** it stretches E *and* rescales its rows. If it wins, the experiment cannot tell which change helped; a "fixed + rescale only" control does not exist.
- **The prepend test is limited:**
  - It runs only on `resampled_exact_cls`, so it cannot show that `fixed` breaks while the fix does not.
  - It uses repeated " the" tokens, not random text.
  - It only uses m = 128 and 512, not a sweep.
- **`exact_cls` is never pretrained;** it is switched on only at fine-tuning.
- **`projection_spectrum` measures E and F,** not the effective rank of the attention matrices (the plan's N2).
- **Only one projection rank is tested.** `mean` pooling is coded but never run, DCT is not implemented, and k is fixed at 64 (no k × n sweep).
- **The 2×2 comparison exists only on IMDB and `stress_retrieve`.** arXiv, SCOTUS, ECtHR-A and the other stress tasks run only for `fixed`, and only `fixed` has the compute curve.
- **At exactly 4,096 tokens, `fixed` and `resampled` perform the same operation** (resizing 4,096 → 4,096 is the identity). Any difference at 4K comes from their separately pretrained weights. The informative comparisons are at 1K and 2K in `stress_retrieve`, and on IMDB, where every document is shorter than 4,096.
- **IMDB is mostly short** (84% ≤ 512 tokens). That is good for Question A, but the IMDB truncation curve will be nearly flat.
- **arXiv shows little of each paper.** The model sees only the first and last 2,048 tokens of most papers, and 7% of arXiv training rows are duplicates with conflicting labels.
- **The stress tasks are easy to read.** They reduce to spotting the class words "cobalt"/"saffron", so they test *visibility of a position*, not language understanding. For `stress_order`, the test positions (pairs reaching the middle of the document) differ from the training positions (first and last quarter).

---

## 14. The team plan (`Presentation.md`) vs what the code runs

The plan proposed these Linformer experiments [plan]:

| ID | Planned experiment | Hypothesis | Priority | In the code? |
|---|---|---|---|---|
| N1 | Length-adaptive projection: (a) adaptive mean-pooling of the real length into k bins, (b) learned E resampled by linear interpolation to the real length, (c) the k lowest DCT basis vectors on the real length | Closes most of the 56 → 65 gap on LRA-Text, because the failure is length variance, not low rank | Must | (b) **yes**, as `linformer.resampled`; (a) coded as `mean` but **not run**; (c) **not implemented** |
| N2 | Effective rank (singular values needed for 90% energy) of a trained full-attention model's attention at n = 1K, 2K, 4K per layer | Effective rank grows with n, which limits how small k can be | Must | **No** (`projection_spectrum` measures E and F instead) |
| N3 | Shift robustness: insert m random bytes at the start at test time, m = 0…512; compare Linformer, N1(a), a dynamic projection, full attention | Linformer degrades with m; pooled and dynamic versions stay flat | Should | **Partly:** `prepend_128`/`prepend_512` with " the" tokens, only on `resampled_exact_cls` |
| N4 | k × n sweep: k ∈ {32, 64, 128, 256}, n ∈ {1K, 2K, 4K} | At fixed k, accuracy drops as n grows, against the theorem's spirit | Should | **No** (k fixed at 64; n varies only through the 1K/2K/4K stress lengths) |
| F5 | Effective rank of Llama-3.1-8B attention at 4K, 32K, 128K (frontier track) | Rank grows with length | Should | **No** (not in this repo) |
| — | `exact_cls` variants | — | — | **Yes**, a code addition not in the plan |

Other plan details:
- **LRA-Text is not in the code.** IMDB (BPE tokens) and the synthetic stress tasks play its role.
- **Related work [plan]:**
  - DCT-Former (Scribano et al., 2022) is the closest prior work for N1(c).
  - Transformer-LS made the same "length variance" diagnosis without testing a fix.
- **Signal-processing view [plan]:**
  - learned E is a position-locked filter bank;
  - mean-pooling is a box-filter decimator;
  - DCT keeps the k lowest-frequency components of the key/value sequence.
- **Other uses of N1 [plan]:**
  - RoFormer's planned R3 ("binned RoPE") builds on N1(a).
  - The team's proposed hybrid model uses a length-adaptive projection (N1) in its global branch.
- **"Where the field went" [plan]:**
  - A fixed projection along the sequence did not reach frontier models.
  - The cheap-attention line survives as linear attention interleaved with softmax layers (MiniMax-01's lightning attention).
  - It also survives as learned compression of token blocks inside sparse attention (DeepSeek's Native Sparse Attention).
- **Timeline [plan]:**
  - Must results frozen on **18 October**; dry run on **21 October**.
  - Linformer talk on **22 October**: about 25 minutes covering the paper, the "vacuous bound" slide, and the N1 and N2 results.
- **Talk template:** problem; mechanism (figure, key equations, complexity); an implementation detail; what the paper reports with caveats; our experiment (hypothesis slide first, then setup, result with error bars, meaning); a one-line verdict.
- **Running results table:** model, accuracy on the long-range benchmark, evidence-task accuracy, peak memory and documents/s at 4K.
- **Rehearsed question:** "If Theorem 2 says k is independent of n, why does Linformer fail on LRA-Text?" Answer: the theorem is an existence result with a random projection and constants that exceed n at ε = 0.1; the failure is length variance, which N1 tests.

---

## 15. How to interpret the results when they arrive

Where to look:
- **Main plot for Question A:** `presentation/length_buckets.svg`, i.e. `stress_retrieve` accuracy at 1K, 2K and 4K for the four cells.
  - If `resampled` beats `fixed` at 1K and 2K but they are similar at 4K, that supports "position-locked columns hurt shorter documents".
- **IMDB accuracy** (`summaries/cells/*.json`) compares all four cells on a real task where every document is shorter than 4,096 tokens.
- **Question B:** compare `fixed` vs `fixed_exact_cls` and `resampled` vs `resampled_exact_cls`. A gain suggests the classifier's compressed view is a bottleneck.
  - If `exact_cls` helps a lot even for `resampled`, the summaries themselves lose information.
- **Positions:** `presentation/position_shift.svg` (accuracy by marker position) shows whether some positions are "invisible", for example late positions in long documents for `fixed`.
- **The shift test:** compare `analyses/.../prepend_128` and `prepend_512` accuracy with the same seed's normal test accuracy. Small drops mean the model is robust to text moving.
- **`projection_spectrum`:** a few large singular values followed by a fast drop means the learned E uses only a few directions; a flat spectrum means it uses all 64.
- **The compute curve** shows whether `fixed` keeps improving with more pretraining (25M → 200M tokens).
- **The benchmarks** show Linformer's speed and memory across lengths. Cost should grow roughly linearly with length.

Statistics:
- Always look at **mean ± standard deviation over 3 seeds** and the bootstrap intervals.
- A difference smaller than the seed-to-seed spread is not evidence.
- A clean negative result counts as evidence in this course.

---

## 16. Quick answers to likely questions

- **What is k?** The number of summary slots (rows of E and F): 64 here.
- **Why does `fixed` hurt short documents?** Because the columns belong to positions: a 300-token review uses only 300 of the 4,096 columns, and long documents use different columns, so documents of different lengths get very different summaries.
- **How does resampling work?** Each 4,096-number row of E is linearly interpolated to the document's real length, rescaled to the original row's norm, and padded with zeros. It is redone for every document on every forward pass, and only the original E is learned.
- **Is `exact_cls` learnable? Is it per document?** No and no. It is an on/off setting for the whole model. The learned Q/K/V weights decide which words `[CLS]` attends to, and that differs per document.
- **Why are the `exact_cls` cells not pretrained separately?** To save compute: no weights change, so they reuse the parent's encoder and switch on at fine-tuning.
- **Do `fixed` and `resampled` differ on a 4,096-token document?** Not in the operation, only in their (separately) pretrained weights.
- **Is Linformer faster than full attention here?** Yes, about 3× faster in training at 4K on the same harness, and it uses ≤ 3.1 GB.
- **How long does the whole suite take?** About 13 h on an RTX 3090 with the recommended settings; roughly 8–12 h estimated on the ADA6000.
- **How much storage?** About 205 GB if nothing is deleted; about 24 GB with Option B.
- **What happens at the 48 h limit?** It shouldn't be reached. If it is, resubmit the identical command and it resumes.
- **Why is SCOTUS macro-F1 low?** Partly the 13-vs-14-class bug (at most 0.929 even when perfect), and SCOTUS is very imbalanced.
- **Why is the IMDB truncation curve flat?** 84% of reviews are ≤ 512 tokens and none exceed 4,096, so cutting at 1K/2K/4K changes almost nothing.
- **Why is the theory "vacuous"?** With the paper's constant, the required k (≈ 4,620 at n = 4,096, ε = 0.1) is larger than n, so the theorem guarantees nothing useful at this length; it only shows existence.
- **What did the paper not test that we test?** Long-document classification, variable document lengths, and a length-adaptive projection.

---

## 17. Glossary

- **Token:** a piece of text from the BPE tokenizer (a word or part of a word).
- **Query / key / value:** per-token vectors used by attention: what a token looks for, what it offers for matching, and what it passes on.
- **Head:** one of 8 parallel attention computations per layer, each with 64 dimensions.
- **Projection E / F:** Linformer's learned 64 × 4,096 matrices that compress keys / values along the sequence.
- **Rank / low-rank:** how many independent directions a matrix really uses; low-rank means few.
- **Singular values:** numbers measuring how much each independent direction of a matrix contributes.
- **`[CLS]` / `<s>`:** the first token of every document, whose final vector the classifier reads.
- **Padding:** filler (`<pad>`, id 0) added so documents in a batch have equal length; it is masked out.
- **MLM pretraining:** learning language by predicting randomly hidden (masked) tokens.
- **Fine-tuning:** further training of the pretrained encoder plus a new classifier on a labelled task.
- **Seed:** a number fixing randomness. Three seeds = three independent fine-tuning runs, to measure variability.
- **Epoch:** one pass over the training set.
- **Warmup / cosine schedule:** the learning rate rises linearly at first, then decreases smoothly to zero.
- **bf16:** a 16-bit number format that makes GPU training faster with little accuracy loss.
- **Gradient checkpointing:** saves GPU memory by recomputing parts of the forward pass during backpropagation, which makes training slower.
- **Checkpoint:** a saved model file. "Full" includes the optimiser state needed to continue training; "weights only" is enough for evaluation.
- **Head + tail truncation:** keeping the first half and last half of a too-long document.
- **Bootstrap interval:** a range of plausible metric values obtained by re-sampling the test set many times.
- **Micro-F1 / macro-F1:** F1 pooled over all labels (micro) vs averaged over labels equally (macro).
- **Suite / cell / job:** a suite (`linformer`) is the whole experiment; a cell is one model variant; a job is one scheduled step (pretrain, fine-tune, benchmark, analysis, curve, report).
