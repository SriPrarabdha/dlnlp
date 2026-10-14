Long Document Classification — Presentation & Experiment Plan
Oct 11, 2026 · @Prararabdha Srivastava 
TL;DR
Tell one story: five different bets on which attention entries matter, tested on one question — can the model use evidence wherever it sits in a long document? Put every architectural novelty in one shared from-scratch harness, and every claim about real documents in a small fine-tuning track against truncation baselines.
• One Must experiment per paper. Longformer: sentence-level global tokens (L1). Linformer: a length-adaptive projection aimed at its 56% on LRA-Text (N1). Long-Short: DualLN on classification and whether LRA-Text needs long range at all (S1, S2). Compressive: its first classification adaptation, with full backprop through memory (C1, C2). RoFormer: length extrapolation and frequency design for long-document encoders (R1, R2).
• One shared closer. An evidence-position task where you plant the label-bearing text at known positions (plus an XOR variant that forces combining distant evidence), and a hybrid built from the winners.
• Biggest risk. LRA-Text may not test long range: Long-Short's best configuration there uses a window of 1 and a projection rank of 1. The evidence task exists so your conclusions do not rest on it.
• Budget and dates. About 100 GPU-hours; harness running by Oct 13; Must results frozen Oct 18; dry run Oct 21.
Added: Track C, the frontier track. Test each paper's descendant at 128K–1M tokens on the benchmarks labs report (RULER, LongBench v2, HELMET, MRCR v2, BABILong), inference-only on open 7–12B models plus a small API budget for 1M-token probes. Must: F1 (RoPE scaling), F3 (KV-cache compression as the Compressive Transformer's heir), F6 (long-context classification today). Tracks A and B stay: they are the only fair test of the mechanisms themselves and the only way to reproduce the papers.
Assumptions and the story that ties the papers together
All five papers attack one bottleneck: full self-attention costs O(n²) time and memory in sequence length n. Each bets on a different assumption about which attention entries matter. Make that the spine of the talk, so five summaries become one argument tested on long document classification.
Paper
Bet about attention
Mechanism
Cost per layer
What it gives up
Longformer (Beltagy et al., 2020)
Most useful attention is local; a few tokens must see everything
Sliding window w, dilated windows, task-chosen global tokens
O(n·w)
Long-range links must route through global tokens or many layers
Linformer (Wang et al., 2020)
The attention matrix is low-rank
Learned k×n projections E, F compress keys and values along the sequence
O(n·k)
Fixed max length; projection is tied to positions, blind to content; no causal mask
Long-Short Transformer (Zhu et al., 2021)
Attention = local band + low-rank remainder
Segment-wise sliding window + content-dependent (dynamic) projection, joined with DualLN
O(n·(w + r))
Two branches to tune; still approximate
Compressive Transformer (Rae et al., 2019)
Old context can be stored coarsely
Segment recurrence with a FIFO memory; evicted memories compressed at rate c
O(nₛ·(nₛ + nₘ + n_cm)) per segment
Sequential over segments; no gradient into memory
RoFormer (Su et al., 2021)
Position should enter as relative phase
Rotate each 2-D pair of q, k by position-dependent angles, so q·k depends on m − n
No change; composes with the others
Going past the trained length needs extra tricks
The narrative in one line: Longformer and Linformer are the two pure strategies (sparse vs low-rank). Long-Short Transformer is their sum, with a fix for each one's failure. Compressive Transformer moves compression from the attention matrix into time. RoFormer is the position axis that modern long-context encoders adopted.
Assumptions this plan makes (change them and the experiment list shrinks or grows):
• Compute: roughly one 24–48 GB GPU shared by the team for about a week. With only Colab/Kaggle T4s, run the items marked Must and drop the rest.
• Time: 11 days to Oct 22. The talk splits over two sessions of about 90 minutes (Oct 22 and Oct 23).
• Grading rewards three things: depth on each paper, one honest novel idea per paper, and evidence. A clean negative result counts as evidence.
Presentation structure (3 hours over Oct 22–23)
Order the papers by idea, not by date: sparse (Longformer), then low-rank (Linformer), then their combination (Long-Short) on day 1; memory over time (Compressive) and position (RoFormer) on day 2, followed by the shared experiment and the hybrid. Day 2 then pays off everything day 1 set up. The owner split below is a suggestion; swap names freely, but keep one owner per experiment.
Session
Segment
Minutes
Suggested owner
What it covers
Oct 22
Opening
15
Rohit
Why 512 tokens is a wall; truncation, chunking and hierarchical baselines; the taxonomy table; datasets; what "novel" means in this talk
Oct 22
Longformer
25
Rohit
Section above; L1 and L5 results
Oct 22
Linformer
25
Tejash
Section above; N1 and N2 results; the vacuous-bound slide
Oct 22
Long-Short Transformer
25
Tejash
Section above as the synthesis of the two; S1 and S2 results
Oct 23
Recap of day 1
5
Aryan
The running results table so far
Oct 23
Compressive Transformer
25
Aryan
Section above; C1 and C2 results
Oct 23
RoFormer
20
Prarabdha
Section above; R1 and R2 results; the frequency view
Oct 23
Evidence-position experiment
20
Aryan + Prarabdha
The heatmap across all models; variant B
Oct 23
Hybrid and where the field went
10
Prarabdha
Component table, hybrid result, ModernBERT, FlashAttention, honest limits
Oct 23
Questions
10
All
See likely questions below
Use one template for every paper so the audience learns the rhythm:
1. The problem it attacks, in one sentence and one picture of the attention matrix (2 min).
2. Mechanism: one figure, the key equations, the complexity (8 min).
3. Implementation detail that matters in practice (2 min).
4. What the paper reports, with your caveats (4 min).
5. Our experiment: hypothesis on its own slide first, then setup, result with error bars, and what it means (7 min).
6. One-line verdict and one row added to the running results table (2 min).
Keep a single running results table on screen at the end of each paper section: model, LRA-Text accuracy, evidence-task accuracy, peak memory and documents per second at 4K. By the closing segment the audience has watched it fill in.
Fitting Track C into the 3 hours. Add a seventh step to the per-paper template, "At 1M tokens today" (3 min): the descendant in current models and its Track C result. Fund it by trimming the mechanism step from 8 to 6 minutes and the experiment step from 7 to 6. On Oct 23, extend the evidence-position segment from 20 to 25 minutes to show F7 (the same task at 128K–1M on frontier APIs) beside the small-model heatmap, taking the 5 minutes from Questions. Keep a second running table for Track C: model, RULER at 128K, LongBench v2 accuracy, and MRCR 8-needle at the longest bin you ran.
Shared experimental setup
Run two tracks. Track A trains tiny models from scratch in one harness, so every architectural change is a fair, controlled comparison. Track B fine-tunes pretrained checkpoints, which exist only for Longformer and ModernBERT (the English RoPE + local/global descendant); Linformer, Long-Short and Compressive have no comparable English long-document checkpoints, so their novel variants live in Track A.
Track A harness. Use the lra folder of NVIDIA/transformer-ls (PyTorch, MIT). It already switches between softmax, Linformer, Nyström, Performer, Reformer and long-short attention via one attn_type flag. Add four things: window + global tokens (Longformer), a position-encoding switch (none, learned, sinusoidal, RoPE), Linformer projection variants, and a segment-recurrent wrapper with memory (Compressive). Keep the paper's LRA-Text config so your numbers are comparable to theirs: 2 layers, d = 64, 2 heads, FFN 128, batch 32, learning rate 1e-4, 8K warmup, 20K steps, max length 4,096.
Dataset
Track
Task
Size
Length
Why it is in
LRA-Text (byte-level IMDB)
A
Binary sentiment
IMDB 25K train / 25K test
1,296 ± 893 bytes, padded to 4,096
Published numbers for Linformer, Long-Short and full attention to reproduce
Evidence-position task (you build it; synthesis section)
A
Binary, label set by a planted span
Choose; 20K is plenty
1K–4K
The only test where you know long-range use is required
LRA-ListOps (optional)
A
10-way, nested operations
96K train
up to 2K
Order-sensitive; needed for R3
Hyperpartisan
B
Binary
516 / 64 / 65 docs
744 ± 678 BERT tokens; 53.5% over 512
Longformer's headline result; tiny, so 5 seeds
Paired Book Summary
B
Multi-label, 227 genres
5,115 / 639 / 639
1,149 ± 934 tokens; 75.5% over 512
Two separate evidence blocks; truncation fails here
Inverted EURLEX (stretch)
B
Multi-label, 4,271 labels
45,000 train
708 ± 539 tokens
Key text moved to the end; expensive
Baselines that appear in every results table:
• Track A: full softmax attention at 4K, and a character n-gram TF-IDF + logistic regression model. If the linear model is close, the benchmark is not testing long-range modelling.
• Track B: RoBERTa-base on the first 512 tokens; head + tail truncation (first 128 + last 382 tokens); BERT + TextRank and BERT + Random from Park et al. (their code is public).
Reporting rules: mean ± std over 3 seeds (Track A) or 5 seeds (Hyperpartisan); identical splits, steps and seeds across variants; accuracy for binary and multi-class, micro-F1 for multi-label; and for every model, peak GPU memory and documents per second at 4K. Efficiency claims without wall-clock numbers are the most common weakness in these papers, so make yours concrete.
Compute: the Must experiments total roughly 30 configurations × 3 seeds in Track A, plus about 30 fine-tuning runs in Track B. Budget around 100 GPU-hours, and time one full-attention LRA-Text run on day 1 before committing to the grid. If runs take longer than an hour, cut to 2 seeds and drop Should items before touching Must items.
Code starting points: allenai/longformer-base-4096 and answerdotai/ModernBERT-base on Hugging Face; amazon-science/efficient-longdoc-classification for Hyperpartisan, Paired Book Summary and the TextRank baseline; lucidrains/compressive-transformer-pytorch for the memory logic.
Track C: the frontier track (128K to 1M+ tokens)
Add a third track that tests each paper's living descendant on the benchmarks frontier releases actually report: multi-needle retrieval (MRCR v2), synthetic stress tests (RULER) and realistic long-document reasoning (LongBench v2), at 128K to 1M tokens. Run it inference-only on open checkpoints, plus a small API budget for 1M-token probes and one training experiment (BABILong) that genuinely reaches millions of tokens.
Why Tracks A and B stay (put this on a slide). No student budget can pretrain a 1M-token model, so the only fair test of the five mechanisms themselves is small models trained from scratch, where 4K is the honest length. And LRA-Text, IMDB and Hyperpartisan are what these papers report, so they are the only way to reproduce and extend them. Track C answers a different question: did the idea survive at 1M?
Benchmark
What it tests
Lengths
Size and metric
Role here
MRCR v2 (OpenAI, MIT licence)
Return the i-th of 2, 4 or 8 near-identical requests hidden in a long synthetic chat
8 bins, from 4K–8K up to 512K–1M
2,400 items, 100 per bin; SequenceMatcher ratio, required hash prefix
The headline long-context number in frontier releases (8-needle); F7
RULER (NVIDIA)
13 synthetic tasks: retrieval, multi-hop variable tracking, aggregation, QA
Configurable; usually 4K–128K
Accuracy; "effective length" = longest length still above Llama-2-7B's 4K score of 85.6
Main metric for F1–F3; reported in Qwen, MiniMax and Cohere reports
LongBench v2 (THUDM)
Multiple-choice deep reading: documents, multi-doc, long in-context learning, dialogue, code, tables
8K to 2M words, most under 128K
503 questions; accuracy; random = 25%
Realistic check; small enough to run fully
HELMET (Princeton, ICLR 2025)
7 application categories, including many-shot classification (TREC, NLU, BANKING77, CLINC150)
8K–128K
Task metrics; model-judged QA and summaries
Classification at 128K keeps the course topic; F6
BABILong (NeurIPS 2024)
20 bAbI reasoning tasks with facts scattered through PG-19 books
Any length; published up to 50M
Exact match; training data provided
The one benchmark where you can train a model that runs at 1M+; F4
Reference points to quote, all from the benchmarks' own pages:
• Claimed is not usable. Apply RULER's own 85.6 threshold to its published table: Llama-3.1-8B-Instruct (claims 128K) drops below it after 32K, Qwen3-8B after 32K, and two models that claim 1M, Qwen2.5-7B-Instruct-1M and GLM-4-9B-1M, after 64K. Some "1M" models are worse: InternLM2.5-7B-1M is usable only to 4K.
• Realistic reading is hard. On LongBench v2, human experts score 53.7% under a 15-minute limit; the best entry on the leaderboard page is 63.3% (Gemini 2.5 Pro with reasoning); Llama-3.1-8B and Qwen2.5-7B score 30.0% without chain-of-thought, against 25% for random guessing.
• Memory beats windows at extreme length. BABILong reports that 137M-parameter recurrent-memory models (RMT, ARMT on GPT-2) fine-tuned on the task handle QA1 into the millions of tokens, with ARMT reported up to 50M, trained on 16K-token inputs.
• Frontier numbers move monthly. Closed-model MRCR scores differ by source because of reasoning-effort settings, length binning and needle count. Pull them from model cards or contextarena.ai the week of the talk, same variant (8-needle) and same bin, and label the date on the slide.
Use this as the first frontier slide: every line starts near 95% at 4K, and none of the four stays above the bar at 128K, including both models sold as 1M-token models.
Open models to run:
Model
Claimed context
Long-context design
Why it is in
Memory note
Llama-3.1-8B-Instruct
128K
Full attention, RoPE with frequency rescaling
Dense baseline, well documented on RULER and LongBench v2
KV cache about 128 KB per token in bf16, so about 17 GB at 128K plus 16 GB of weights: one 40–80 GB GPU
Qwen2.5-7B-Instruct-1M
1M
Dual Chunk Attention and sparse attention in a custom vLLM
The only open 1M model you can push to 256K+ on student hardware
About 56 KB of KV per token, so about 15 GB at 256K; the full 1M needs at least 120 GB across GPUs
Gemma 3 (4B or 12B)
128K
5 local layers (1,024-token window) per global layer; RoPE base 10K local, 1M global
Longformer + RoFormer in production
Local layers keep the cache small
Qwen3-8B
128K via YaRN
Full attention + YaRN
Comparison point for F1 and F3
As Llama-3.1-8B
Proposed experiments:
ID
Extends
Experiment
Hypothesis
Cost
Priority
F1
RoFormer (R1, R2)
Training-free RoPE extension: Qwen2.5-7B-Instruct (native 32K) on a RULER subset at 32K, 64K and 128K with no scaling, linear interpolation, dynamic NTK and YaRN, set through Hugging Face rope_scaling
YaRN > NTK > linear > none beyond 32K, matching how each treats high vs low frequencies
Inference only: 4 RULER tasks × 50 items × 4 settings × 3 lengths
Must
F2
Longformer (L1, L3)
Fewer global layers at inference: in Gemma 3, switch every second global layer to the 1,024-token local window; measure RULER retrieval vs aggregation and KV memory at 32K–128K
Retrieval collapses while aggregation degrades gracefully, which prices what the global layers buy
Per-layer attention-type config change + inference
Should
F3
Compressive (C3)
KV-cache compression as test-time compressive memory, with NVIDIA kvpress on Llama-3.1-8B or Qwen3-8B: evict-only presses (StreamingLLM's sink + recent window, SnapKV, Expected Attention) vs a merging press that folds evicted tokens into survivors, at 25/50/75% compression on RULER and LongBench v2
The paper's thesis, compress rather than discard, holds: merging beats eviction at 75% on aggregation and multi-hop tasks
Inference only; the library implements the presses
Must
F4
Compressive (C1, C2)
Train a small recurrent-memory model on BABILong QA1–QA2 at 4K–16K (your compressive classifier with full backprop, or the RMT/ARMT reference code), then evaluate at 128K, 1M and beyond; compare with an 8B LLM at 128K
A ~100M model with learned memory beats 8B full-attention models beyond their usable length
GPU-hours, with a length curriculum
Should
F5
Linformer (N2)
The low-rank claim at frontier length: sample 1,024 query rows of Llama-3.1-8B attention over 4K, 32K and 128K contexts and compute effective rank per layer
Effective rank grows with length, which is why frontier models moved to selection and sparsity, not fixed projection
Analysis only (forward hooks)
Should
F6
Course topic
Long-context classification in 2026: (a) HELMET many-shot classification at 8K–128K for Llama-3.1-8B, Qwen3-8B and Gemma 3; (b) zero-shot classification of Paired Book Summary and Inverted EURLEX by prompting an 8B model with the whole document, against Track B's fine-tuned Longformer and ModernBERT
Many-shot accuracy saturates or falls before the claimed length; the fine-tuned 150M encoder still wins on cost, probably on accuracy
Inference only
Must
F7
Synthesis
The evidence-position task (variant A and the XOR variant B) plus an 8-needle MRCR subset at 128K, 512K and 1M on 1–3 frontier APIs, about 10 items per cell
Single-span retrieval is close to solved at 1M by the best models; combining two spans 500K tokens apart is not — a small new benchmark at frontier scale
Roughly 20–40M input tokens; ask the course for API credits
Should
Hardware for Track C: one 80 GB GPU covers 128K for 8B models and about 256K for Qwen2.5-7B-1M. With 24–48 GB, cap at 32K–64K or use Gemma 3 4B. Budget roughly 30–50 extra GPU-hours (a rough estimate; time one 128K RULER pass first). RULER and HELMET ship runners for Hugging Face, vLLM and the major APIs; MRCR's dataset card includes a grading script.
Longformer (Beltagy, Peters, Cohan, 2020)
Longformer's lasting contribution is a recipe: stretch a pretrained RoBERTa to 4,096 tokens with a local window plus a few task-chosen global tokens, for 65K steps of extra pretraining. Its weak spot for classification is that only [CLS] is global, so every long-range signal must squeeze through one token. That is where your novel experiment should push.
What to present (about 25 minutes):
• Pattern. Window w, w/2 tokens each side; receptive field ℓ·w after ℓ layers (12 × 512 = 6,144, so the top layer can see all 4,096 tokens). Dilation d stretches it to ℓ·d·w. Global attention is symmetric: a global token attends to everything and everything attends to it, through separate projections Q_g, K_g, V_g initialised from the local ones.
• Implementation. Three kernels: loop (slow), chunks (vectorised, about 2× the ideal memory, used for fine-tuning) and a TVM CUDA kernel (dilation, char-LM). One slide on why "linear" did not mean "fast" in 2020: banded matmul is not native in PyTorch.
• Pretraining recipe. Window 512; the 512 RoBERTa position embeddings are copied 8 times to reach 4,096. MLM bits-per-character: 10.299 with random new positions, 1.957 after copying, 1.753 after 2K steps, 1.705 after 65K steps (RoBERTa at 512 tokens: 1.846).
• Results that matter here. IMDB accuracy 95.3 → 95.7; Hyperpartisan F1 87.4 → 94.8. Say the caveat out loud: Hyperpartisan has 645 documents averaging 705 wordpieces, and IMDB averages 300.
• Ablations worth a slide. On WikiHop, removing global attention and its projections costs 8.3 points; removing only the separate projections costs 1.6. In char-LM, windows that grow with depth beat shrinking ones (1.21 vs 1.24 BPC), and dilation on 2 heads helps slightly (1.20).
• Independent check. Park et al. (2022) found Longformer beat 512-token BERT truncation on Hyperpartisan (95.69 vs 92.00) and Paired Book Summary (57.76 vs 52.24 micro-F1), lost badly on EURLEX (54.53 vs 73.09), and trained about 12× slower. They needed an unusual learning rate of 0.005 for Longformer, which may explain part of that gap.
Proposed experiments:
ID
Experiment
What is new vs the paper
Hypothesis
Cost
Priority
L1
Global-token placement: [CLS] only vs [CLS] + first token of every sentence vs [CLS] + every 256th token vs [CLS] + the same number of random tokens
The paper makes only [CLS] global for classification; sentence-level globals appear only in its HotpotQA model
Sentence globals act as a built-in hierarchical summary and help more as documents get longer; the random control separates "more globals" from "the right globals"
Only the global_attention_mask changes; 4 variants × 5 seeds on longformer-base-4096
Must
L2
Learned global selection: a linear scorer picks the top-k tokens per document to be global (straight-through top-k)
The paper hand-picks global tokens per task
Learned selection beats the sentence heuristic when evidence is sparse (Paired Book Summary)
One extra layer; tricky to train
Stretch
L3
How much window does classification need: window 512 (pretrained) vs 256 vs 128, and a 128 → 512 schedule over layers
Window schedules were tested only for char-LM
Accuracy saturates well below 512; the growing schedule keeps most of it at lower cost
Per-layer attention_window config, then fine-tune
Should
L4
Pooling head: [CLS] vs mean vs attention pooling over all tokens
Only [CLS] is used
Attention pooling helps when evidence is scattered
Cheap
Should
L5
Harness version (from scratch, LRA-Text): window only vs window + global [CLS] vs window + global tokens at every full stop
Clean ablation with no pretraining confound
Global [CLS] matters most when the window spans few words, as with bytes
Minutes per run
Must
Practical notes: cap Hyperpartisan at 2,048 tokens (its 95th percentile is 1,975 wordpieces). Its test split is 65 documents, so one document moves accuracy by 1.5 points; report mean ± std over 5 seeds or the comparison means nothing.
At 1M tokens today (frontier slide). Longformer's pattern became the standard way to make long context affordable: Gemma 3 runs five 1,024-token sliding-window layers per fully global layer and cites Longformer for it, and ModernBERT does the same for encoders. Its global tokens have a close cousin in "attention sinks": StreamingLLM-style cache policies keep the first few tokens plus a recent window. Frontier experiment: F2 in Track C.
Linformer (Wang et al., 2020)
Linformer was never tested on long documents: every downstream result is at n = 512 or 1,024 on SST-2, IMDB, QNLI and QQP. On the one long classification benchmark where others ran it, LRA-Text (byte-level IMDB, 4K), it scores 56.1% vs 65.4% for full attention. That gap is your opening, and the likely cause is testable.
What to present (about 25 minutes):
• The claim. Spectrum analysis of RoBERTa attention at n = 512 shows a long-tailed singular-value spectrum, more skewed in higher layers.
• The theory, critically. Theorem 1 says attention is ε-approximable with rank Θ(log n), via Johnson–Lindenstrauss with a random R. Theorem 2 says k = Θ(d log d / ε²) suffices, independent of n. Plug in the paper's constant k = 5 ln n / (ε² − ε³) with n = 4,096 and ε = 0.1: k ≈ 4,620, larger than n itself. These are existence results; the learned E, F are what actually do the work.
• Mechanism. E_i, F_i project keys and values along the sequence axis to k rows, so attention is n×k. Sharing options: headwise, key-value, layerwise (one matrix for the whole model).
• Results. At n = 512, k = 256 with layerwise sharing, the GLUE+IMDB average is 92.30 vs RoBERTa's 92.25. Pretraining perplexity at fixed k = 256 stays flat from n = 512 to 4,096. Speed at n = 4,096, k = 128: 3.4× faster inference, 14× less memory.
• Limits to name. E is tied to absolute positions and to a fixed n; it cannot do causal masking; it is blind to content. The paper lists pooling and strided convolution as alternative projections but never tests them.
The working hypothesis for the 56.1%: LRA-Text lengths are 1,296 ± 893 bytes, padded to 4,096. A short review only touches the first few hundred columns of E, so each length sees a different, mostly unused projection. Transformer-LS makes the same diagnosis without testing a direct fix.
Proposed experiments:
ID
Experiment
What is new vs the paper
Hypothesis
Cost
Priority
N1
Length-adaptive projection: replace fixed E, F with (a) adaptive mean-pooling of the real, unpadded length into k bins, (b) learned E resampled by linear interpolation to the real length, (c) the k lowest DCT basis vectors computed on the real length
The paper names pooling but never runs it; none of these are length-aware
Closes most of the 56 → 65 gap on LRA-Text, because the failure is length variance, not low rank
One function change in the harness; 4 variants × 3 seeds
Must
N2
Spectrum at long range: effective rank (singular values needed for 90% energy) of the trained full-attention model's attention at n = 1K, 2K, 4K, per layer
Paper measured only n = 512 on RoBERTa
Effective rank grows with n on long documents, which bounds how small k can be
Analysis of a model you train anyway
Must
N3
Shift robustness: insert m random bytes at the start at test time, m from 0 to 512; plot accuracy vs m for Linformer, N1(a), dynamic projection, full attention
Robustness to insertion is argued by Transformer-LS but never measured directly
Linformer degrades with m; the pooled and dynamic versions stay flat
Inference only
Should
N4
k × n sweep: k in {32, 64, 128, 256}, n in {1K, 2K, 4K} on a classification task
Theorem 2's n-independence was checked only through pretraining perplexity
Accuracy at fixed k drops as n grows, against the theorem's spirit
12 short runs
Should
Closest prior work for N1(c): DCT-Former (Scribano et al., 2022) compresses attention with the DCT. Cite it; your contribution is the length-aware drop-in comparison inside Linformer on variable-length documents.
For a signal-processing audience, N1 reads naturally: learned E is a position-locked filter bank; pooling is a box-filter decimator; DCT keeps the k lowest-frequency components of the key/value sequence.
At 1M tokens today (frontier slide). A fixed projection along the sequence did not reach frontier models. The cheap-attention line survives in two other forms: linear attention interleaved with softmax layers (MiniMax-01's lightning attention, evaluated on RULER up to 1M tokens), and learned compression of token blocks inside sparse attention (DeepSeek's NSA, next section). Frontier experiment: F5 tests Linformer's low-rank claim on Llama-3.1-8B at 128K.
Long-Short Transformer (Zhu et al., NeurIPS 2021)
Present Transformer-LS as the synthesis of the two previous talks: Longformer's window plus Linformer's projection, each repaired. It is also the paper with the most suspicious detail in this set. Its best LRA-Text configuration uses a window of w = 1 and a projection rank of r = 1, which questions whether that benchmark needs long-range attention at all.
What to present (about 25 minutes):
• Short branch. Segment-wise sliding window: split the sequence into segments of length w; each token attends to its own segment plus w/2 tokens on each side (2w keys). Faster in plain PyTorch than per-token windows.
• Long branch. Dynamic projection P = softmax(K·W_P), an n×r matrix normalised over tokens; projected keys and values are PᵀK and PᵀV. It is content-dependent, so it survives insertions and length changes that break Linformer's fixed E.
• Aggregation. Each query takes one softmax over the union of its 2w local keys and the r global keys, rather than splitting heads between patterns.
• DualLN, the subtle part. A weighted average of zero-mean vectors has lower variance, so projected keys and values have smaller norms. At initialisation that biases attention toward the local branch and starves the global branch of gradient. Separate LayerNorms for each branch fix the scale. Evidence: lower validation loss on enwik8 and text8, and ImageNet 81.3% → 81.9% for CvT*-LS-13.
• Causal version. Projections are computed per segment of length l, and a token attends only to segments that contain no future token; the window fills the gap.
• Results. LRA with tiny models (2 layers, d = 64, 2 heads, mean of 4 seeds): Text 68.4 ± 0.8 vs full attention 65.4 ± 0.3 and Linformer 56.1 ± 1.5; dynamic projection alone reaches 66.3. enwik8 test BPC 0.97 with 110M parameters, matching Compressive Transformer at 227M.
Proposed experiments:
ID
Experiment
What is new vs the paper
Hypothesis
Cost
Priority
S1
DualLN on classification: with vs without, plus a new measurement, the share of attention mass going to the global branch per layer over training
DualLN was ablated only on char-LM and ImageNet, and only by final loss
Without DualLN the global share stays near zero on LRA-Text and accuracy drops toward window-only
2 variants × 3 seeds + logging
Must
S2
Does the benchmark need long range? Grid w in {1, 8, 32} × r in {1, 8, 32}; then shuffle 64-byte chunks of each test document and re-measure
The paper reports w = r = 1 as best without discussing it
Accuracy barely moves under chunk shuffling, so LRA-Text behaves like a bag of local features; this motivates the evidence-position probe in the synthesis section
9 short runs + inference
Must
S3
RoPE in the short branch only vs both branches vs none
The long branch carries no position information after projection
RoPE helps the local branch; adding it before projection adds nothing
3 variants
Should
S4
Multi-resolution long branch: replace the single global projection with per-segment compression (r slots per 512-token segment, mean-pool or strided conv)
The global P discards where in the document evidence came from
Helps when a document has two separate evidence blocks (Paired Book Summary style)
Moderate code change
Should
S5
Rank schedule: r decreasing with depth (64 → 16) vs constant
Linformer's spectrum suggests higher layers are lower-rank; nobody applied it here
Same accuracy at lower cost
2 variants
Stretch
Code: the official repo's lra folder is PyTorch, built on the Nyströmformer harness, and already includes softmax, Linformer, Nyström, Performer and Reformer attention. Use it as the shared harness (next section).
At 1M tokens today (frontier slide). The closest modern relative is DeepSeek's Native Sparse Attention (ACL 2025 best paper). Each query attends through three branches — compressed block summaries, top-k selected blocks at full resolution, and a sliding window — and the sparsity is trained in from pretraining rather than added afterwards. Present it as long-short attention plus a selection step, with the Compressive Transformer's block compression inside. NSA is out of reach for this budget, so use it as the section's closing slide and as the template for the hybrid.
Compressive Transformer (Rae et al., ICLR 2020)
The Compressive Transformer was built for language modelling, speech and RL; it was never used for classification. Adapting it is itself your novel contribution. The interesting question is training signal: a language model gets a loss at every token, while a classifier gets one label at the very end, and the paper stops gradients from flowing into memory.
What to present (about 25 minutes):
• Mechanism. Transformer-XL processes a sequence in windows of nₛ tokens and keeps a FIFO memory of nₘ past activations per layer. When activations are evicted, the Compressive Transformer compresses them by rate c into a second FIFO of n_cm slots instead of discarding them. Attention runs over [compressed memory; memory; current window].
• Range vs cost. Temporal range rises from ℓ·nₘ to ℓ·(nₘ + c·n_cm) while attention cost stays O(nₛ² + nₛ(nₘ + n_cm)). With nₘ = n_cm = n/2 and c = 3, range doubles at equal cost.
• Compression functions and losses. Enwik8 BPC (lower is better): conv trained end-to-end by BPTT 0.996, max pool 0.986, conv + auto-encoding loss 0.984, mean pool 0.982, most-used 0.980, dilated conv + attention-reconstruction 0.977, conv + attention-reconstruction 0.973. The attention-reconstruction loss is deliberately lossy: keep what current queries attend to, drop the rest. Its gradients are stopped from entering the main network.
• Results. Enwik8 0.97 BPC; WikiText-103 17.1 perplexity, with the largest gains on rare words (about 20% for words seen under 1,000 times); PG-19, the 28,752-book benchmark the paper introduced, 33.6 vs 36.3 test perplexity for Transformer-XL.
• Evidence the memory is used. Average attention rises at the boundary from the oldest regular memory to compressed memory, so the model keeps salient information there.
• Stated limits. Extra complexity, and "unlikely to help" on tasks without long-range reasoning. Listed future work: adaptive compression rates by layer.
How to adapt it for classification: split the document into windows of 512 tokens, process them left to right with bidirectional attention inside each window, then let a [CLS] token in the last window attend over [compressed memory; memory; last window]. Closest prior work: ERNIE-Doc (Ding et al., ACL 2021), a recurrence Transformer that reads each document twice and was evaluated on IMDB and Hyperpartisan, without compression.
Proposed experiments:
ID
Experiment
What is new vs the paper
Hypothesis
Cost
Priority
C1
Classification adaptation at equal attention budget: window only (no memory) vs Transformer-XL memory vs Compressive (c = 4)
The paper never classifies
Compressive wins only when evidence sits early in the document
Moderate code; reuse an existing PyTorch implementation
Must
C2
Gradient path: stop-gradient memory (paper) vs full backprop through all 8 windows of a 4K document (gradient checkpointing)
For LM, stop-gradient barely hurts; for one end-of-document label it may decide everything
Full backprop beats stop-gradient by a wide margin on classification
2 variants × 3 seeds
Must
C3
Compression objective: attention-reconstruction (paper's best) vs task loss only vs both
Attention-reconstruction keeps what local queries need, not what the final label needs
Task loss, or both, beats reconstruction alone on classification, reversing the LM ranking
3 variants
Should
C4
Two-pass read: fill compressed memory in a first pass over the whole document, then read again with that memory (ERNIE-Doc's retrospective feed, with compression)
Combination not tested in either paper
Early windows can now see a summary of later text; gains on the evidence-at-end variant
About 2× compute
Should
C5
Compression rate per layer: c = 2 in lower layers, c = 8 in upper layers vs constant c = 4
The authors' own future work
Same accuracy with fewer slots, or better range at equal slots
2 variants
Stretch
Also replicate the paper's attention-bucket figure for the final [CLS]: how much attention mass goes to compressed memory, memory and the last window. On the synthetic evidence task (synthesis section), you know where the evidence was, so you can check whether [CLS] attends to the right compressed slot.
Reference implementation: lucidrains/compressive-transformer-pytorch (MIT, built for language modelling) gives memory handling and the reconstruction loss; you add the windowing and the [CLS] head.
At 1M tokens today (frontier slide). The idea lives in two places. At inference, KV-cache compression decides which past keys and values to keep, drop or merge; NVIDIA's kvpress collects dozens of these methods and notes that 1M tokens of Llama-3.1-70B cache need up to 330 GB in fp16. In training, recurrent-memory models (RMT, ARMT) carry compressed state across segments and reach millions of tokens on BABILong. Frontier experiments: F3 (compress vs evict) and F4 (train a small memory model and test it at 1M+).
RoFormer (Su et al., 2021)
RoFormer is the odd one out: it does not make attention cheaper, it changes how position enters attention, and that axis composes with every other paper here. Its own long-text evidence is thin, and its famous "long-term decay" story has since been challenged. That leaves room for a sharp critique and clean experiments.
What to present (about 20 minutes):
• Derivation. Require ⟨f_q(x_m, m), f_k(x_n, n)⟩ to depend only on x_m, x_n and m − n. In 2-D the solution is multiplication by e^{imθ}. In d dimensions, split into d/2 planes, each rotated at frequency θ_i = 10000^{−2(i−1)/d}. It is multiplicative, norm-preserving, and leaves values untouched. Implementation: x·cos(mθ) + rotate_half(x)·sin(mθ).
• Signal-processing view (own this slide). Each plane is a carrier at frequency θ_i; q·k sees only the phase difference (m − n)θ_i. Wavelengths run from about 6 tokens to tens of thousands. With head dimension 64 and base 10,000, 16 of the 32 planes never complete one cycle within 512 tokens, and 9 still do not within 4,096. Those slow planes behave almost like position-free channels.
• Long-term decay, critically. The paper bounds the score's magnitude with an Abel-summation upper bound that shrinks with distance; that is a bound, not the actual score. Barbero et al. (ICLR 2025) argue decay is not why RoPE works: Gemma 7B uses the highest frequencies to build positional heads and the lowest ones as semantic channels, and their p-RoPE drops the lowest frequencies.
• Linear attention. RoPE is applied to φ(q), φ(k) in the numerator only, so weights are no longer guaranteed non-negative or normalised; the paper admits this.
• Results, read carefully. WMT14 En-De 27.3 → 27.5 BLEU. On GLUE, RoFormer is lower on SST-2 (90.7 vs 93.5), QNLI (88.0 vs 90.5) and MNLI (80.2 vs 84.6), and higher on MRPC, STS-B and QQP. The BERT row matches the BERT paper's test-server numbers while RoFormer reports validation scores, so the QQP gap (86.4 vs 71.2) is not like-for-like.
• The long-text claim. On CAIL2019-SCM (Chinese legal case matching), test accuracy is 68.10% for WoBERT-512, 68.29% for RoFormer-512 and 69.79% for RoFormer-1024. The gain comes from reading 1,024 tokens, which absolute-position WoBERT cannot do; at equal length the two are within 0.2 points.
• Legacy. ModernBERT (2024) is Longformer + RoFormer: every third layer is global attention with RoPE base 160,000, the rest are 128-token local windows with base 10,000, at a native 8,192 tokens.
Proposed experiments:
ID
Experiment
What is new vs the paper
Hypothesis
Cost
Priority
R1
Position-encoding bake-off on the harness: none (NoPE), learned absolute, sinusoidal, RoPE. Train at 1K tokens, test at 1K, 2K and 4K
The paper never tests length extrapolation for classification at equal length
RoPE degrades gracefully, absolute fails beyond 1K, NoPE does better than expected on bag-like LRA-Text
4 variants × 3 seeds
Must
R2
Frequency design for long documents: base 10K vs 100K vs 500K, and p-RoPE (lowest 25% of frequencies removed), in and out of the training length
Base and p-RoPE were studied for decoder LLMs, not for long-document encoders trained on classification
A larger base helps at 4K; p-RoPE helps when evidence must be matched across long distances
4 variants
Must
R3
Binned RoPE for low-rank attention: with Linformer's pooled projection (N1a), each pooled key covers a known span, so rotate it by its span centre
Low-rank attention loses relative position after projection; RoFormer only showed RoPE on kernel attention
Restores order sensitivity; helps ListOps more than Text
Small code change on top of N1
Should
R4
Which frequencies matter: at inference, zero the highest vs the lowest rotary frequencies of a trained RoPE classifier and measure the accuracy drop; plot real mean attention vs distance next to the paper's bound
Tests Barbero et al.'s claim at small scale, on classification
Removing high frequencies hurts more; real attention does not decay monotonically
Inference only
Should
R5
Fine-tuning track: ModernBERT-base vs Longformer-base on the same datasets and lengths
Shows where the RoPE + local/global design ended up
ModernBERT matches or beats Longformer at a fraction of the wall-clock time
2 models × 3 seeds
Should
At 1M tokens today (frontier slide). Rotary positions are the default in open long-context models, and most of the long-context story is re-tuning their frequencies. Gemma 3 raises the global layers' base from 10K to 1M and uses positional interpolation; Qwen uses YaRN and Dual Chunk Attention for 128K–1M; MiniMax-01 rotates only half the dimensions; Llama 4 Scout interleaves layers with no positional encoding between RoPE layers (iRoPE), adds inference-time attention temperature scaling, and claims 10M tokens. Frontier experiment: F1, training-free RoPE scaling on RULER.
Synthesis: one experiment across all five, and where the field went
Close the talk with one figure that puts all five ideas on the same question: can the model use evidence wherever it sits in the document? Then propose a hybrid built only from the components that won their own experiments.
The evidence-position task (about one day to build). LRA-Text cannot answer the question, because you do not know where its evidence is. Build a task where you do:
1. Take label-neutral filler text (Wikipedia or PG-19 paragraphs) at lengths L of 1K, 2K and 4K.
2. Invent two sets of about 20 short marker phrases, one per class, that never occur in the filler.
3. Variant A (retrieve): plant one marker at relative position p in {0.05, 0.25, 0.5, 0.75, 0.95}; the label is that marker's class.
4. Variant B (combine): plant two markers at least L/2 apart; the label is the XOR of their classes. Either marker alone predicts nothing, so a model that cannot combine distant evidence scores 50%.
5. Train with random positions; evaluate per position bucket. Plot a heatmap of accuracy, models × position, one panel per length.
State predictions on a slide before showing results: full attention truncated to 512 fails past the first 512 tokens; Linformer degrades with length; Longformer and Long-Short solve A; Compressive with stop-gradient fails on early markers and recovers with full backprop; B separates models that combine evidence from models that only find it.
The proposed hybrid. Long-short attention with DualLN, RoPE in the local branch, length-adaptive projection (N1) with binned RoPE (R3) in the global branch, and a global [CLS]. For documents beyond 4K, wrap it in the compressive memory from C1/C2 with full backprop. Compare at matched FLOPs against full attention and the five originals on LRA-Text and both task variants. If it does not win, present that plainly; a clean null result with a clear reason is worth more than a tuned win.
Model
Local window
Global tokens
Low-rank projection
Memory over time
Rotary position
Longformer
Yes
Yes ([CLS])
No
No
No
Linformer
No
No
Yes (fixed)
No
No
Long-Short
Yes
No
Yes (dynamic)
No
No
Compressive
Within window
No
No
Yes
No
RoFormer
No
No
No
No
Yes
ModernBERT (2024)
Yes, 128 tokens
Every third layer fully global
No
No
Yes
Proposed hybrid
Yes
Yes ([CLS])
Yes (length-adaptive)
Optional, beyond 4K
Yes
Where the field went (closing, about 5 minutes).
• Convergence. Long-context encoders kept local windows, periodic global attention and RoPE (ModernBERT). Fixed low-rank projection is rarely used in today's long encoders.
• Kernels changed the economics. FlashAttention computes exact attention without materialising the n×n matrix, so exact attention at 4K–8K is often fast enough. Park et al. measured Longformer at about 12× the time of 512-token BERT. Show your own wall-clock numbers here.
• Benchmarks. LRA trains from scratch; Amos, Berant and Gupta (ICLR 2024) showed that self-pretraining on the task's own data changes model rankings substantially. Stretch experiment: MLM-pretrain each Track A model on LRA-Text bytes for a few thousand steps, then re-rank.
• Baselines. Simple truncation and sentence selection remain hard to beat on real datasets (Park et al., 2022).
Execution plan, priorities and likely questions
Freeze the Must results on Oct 18 and spend the last four days on figures, slides and one full dry run. Work after the freeze goes only into Should items that are already running.
The harness is the critical path: Track A cannot start until it runs a full-attention LRA-Text baseline and someone has timed it. Track B needs no harness, so it starts on day 2.
Priority
Experiments
Rule
Must
L1, L5, N1, N2, S1, S2, C1, C2, R1, R2, evidence task variant A
Finished and frozen by Oct 18, at least 3 seeds
Should
L3, L4, N3, N4, S3, S4, C3, C4, R3, R4, R5, variant B, the hybrid
Run only once every Must item in your section is done
Stretch
L2, S5, C5, self-pretraining re-ranking, Inverted EURLEX
Mention as future work if not done
Track C priorities: Must F1, F3, F6; Should F2, F4, F5, F7. Run Track C in the same Oct 13–18 window as Track A; it needs no harness, only a 40–80 GB GPU, the benchmark repositories and, for F7, API credits.
Questions to rehearse, with the short answer:
• "Isn't Longformer just a CNN with a few global tokens?" Mostly, yes; the global tokens give every pair of tokens a two-hop path, and removing them costs 8.3 points on WikiHop.
• "If Theorem 2 says k is independent of n, why does Linformer fail on LRA-Text?" The theorem is an existence result with a random projection and constants that exceed n at ε = 0.1; the failure is length variance, which N1 tests.
• "Is LRA-Text even a long-range task?" Show S2 (w = r = 1 is best; chunk shuffling) and the TF-IDF baseline.
• "Why not use full attention with FlashAttention?" At 4K that is often the right answer; show your wall-clock numbers and where the crossover would be.
• "How does RoPE extend past the training length?" Methods like position interpolation rescale the frequencies; R1 and R2 are the encoder-side test of the same idea.
• "Why does the Compressive Transformer stop gradients into memory?" Backprop through many windows is expensive and language modelling has a loss at every token; C2 shows what that costs a classifier.
• "Are your differences significant?" Show seeds and standard deviations; on Hyperpartisan one test document equals 1.5 points.
"Frontier models read 1M tokens; why train 4K models at all?" Nobody can pretrain a 1M-token model on a course budget, so the five mechanisms can only be compared fairly at small scale. Track C then shows which descendants survive at 128K–1M, on the benchmarks frontier labs report, and the RULER chart shows that a claimed window is not a usable one.
References and code
The five assigned papers:
• Longformer: The Long-Document Transformer — Beltagy, Peters, Cohan, 2020
• Linformer: Self-Attention with Linear Complexity — Wang et al., 2020
• Long-Short Transformer: Efficient Transformers for Language and Vision — Zhu et al., NeurIPS 2021
• Compressive Transformers for Long-Range Sequence Modelling — Rae et al., ICLR 2020
• RoFormer: Enhanced Transformer with Rotary Position Embedding — Su et al., 2021
Supporting work cited in this plan:
• Efficient Classification of Long Documents Using Transformers — Park, Vyas, Shah, ACL 2022 (simple baselines, Paired Book Summary, Inverted EURLEX)
• ERNIE-Doc: A Retrospective Long-Document Modeling Transformer — Ding et al., ACL 2021 (recurrence for classification)
• DCT-Former: Efficient Self-Attention with Discrete Cosine Transform — Scribano et al., 2022 (prior work for N1c)
• Round and Round We Go! What Makes Rotary Positional Encodings Useful? — Barbero et al., ICLR 2025 (against the decay story; p-RoPE)
• ModernBERT: Smarter, Better, Faster, Longer — Warner et al., 2024 (local/global + RoPE encoder)
• Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors — Amos, Berant, Gupta, ICLR 2024 (not opened for this plan; read before citing numbers)
Code and checkpoints:
• NVIDIA/transformer-ls — the lra folder is the Track A harness
• google-research/long-range-arena — LRA data download
• amazon-science/efficient-longdoc-classification — Hyperpartisan, Book Summary, TextRank baselines
• allenai/longformer and the allenai/longformer-base-4096 checkpoint
• answerdotai/ModernBERT-base checkpoint on Hugging Face
• lucidrains/compressive-transformer-pytorch — memory and reconstruction-loss logic
• cscribano/DCT-Former-Public — DCT attention reference
Frontier track (added):
• RULER — Hsieh et al., 2024; code, results table and effective-length rule
• LongBench v2 — Bai et al., 2024; dataset and leaderboard
• HELMET — Yen et al., ICLR 2025; includes many-shot classification
• OpenAI MRCR — dataset card with bins and grading script
• BABILong — Kuratov et al., NeurIPS 2024 Datasets and Benchmarks
• Associative Recurrent Memory Transformer — 2024; BABILong results to 50M tokens
• NVIDIA/kvpress — KV-cache compression methods and evaluation
• Native Sparse Attention — Yuan et al., ACL 2025
• Gemma 3 Technical Report — Gemma Team, 2025
• Qwen2.5-7B-Instruct-1M model card — Dual Chunk Attention, VRAM needs
• MiniMax-01: Scaling Foundation Models with Lightning Attention — 2025
• Llama 4 announcement — Meta, April 2025 (iRoPE; read via secondary coverage for this plan)
• Long Context Benchmarks: All Three Hit 1M — Now What? — March 2026 survey; AI-compiled, so use it only to find the vendor model cards it cites
