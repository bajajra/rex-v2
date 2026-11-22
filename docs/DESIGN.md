# rex-v2 design: a sub-200M-body looped MoE decoder in the style of Inkling 2 Mini

Status: design specification, revision 2 (2026-11-22). No implementation exists yet. The implementer builds exactly what is written here; anything not specified is a defect in this document, not a free choice.

Revision 2 replaces the earlier 32k-vocabulary / residual-stream / single-gate draft. The user's requirements that drive it: full Inkling vocabulary; the 200M cap applies to non-embedding parameters; attention residuals as the default; the same loop mechanism as Inkling 2 Mini; training up to 256k context; learnings from CSA2 (DeepSeek-V4.1 report).

Provenance tags used throughout:

| Tag | Meaning |
|---|---|
| **[I]** | Copied from Inkling 2 Mini as publicly described (config.json, HF blog, Goodfire write-up) |
| **[I?]** | Inkling has the feature but its form is undisclosed; the form here is my fill-in |
| **[K]** | From the Attention Residuals paper (Kimi Team), the source of Inkling's residual design |
| **[D]** | From the DeepSeek-V4.1-Flash technical report (user-supplied PDF) |
| **[S]** | Standard practice with published evidence (cited) |
| **[J]** | My judgement, no direct evidence |

---

## 1. Summary

rex-v2 is Inkling 2 Mini's text decoder topology at one quarter of the width: 24 stored layers in four blocks of six, the middle two blocks run three times with shared weights (48 layer applications), attention residuals instead of a residual stream, per-layer cross-pass readers with low-rank keep/gain gates on the Q, K, V and MoE inputs, five sliding-window layers per full-attention layer, and a routed SwiGLU MoE in every layer.

| | |
|---|---|
| Non-embedding ("body") parameters | **197,855,808** (cap: < 200,000,000) |
| Embedding parameters (201,024 x 576, tied in/out) | **115,789,824** |
| Total, tied (default) | 313,645,632 |
| Total if untied | 429,435,456 (untying costs +115,789,824) |
| Body parameters a token touches once (stored-active) | 78,416,448 |
| Body parameters weighted by applications (effective-active, weight matrices only) | 156,653,568 |
| Layer applications per token | 48 (40 sliding-window, 8 full-attention) |

### Position on "MoE spends parameters to save compute, looping spends compute to save parameters"

The concern is right, and the evidence at this scale leans against MoE under a fixed parameter cap:

- Abnar et al. find that for a fixed total parameter count, "for smaller models (up to about 500·10^6 parameters), the optimal sparsity remains at 0 (i.e., dense) for most compute budgets" (https://arxiv.org/abs/2501.12370, App. D.2). That holds at fixed training compute, not only at equal data.
- Against that, MoEUT (shared layers + fine-grained MoE) slightly beats parameter-matched dense non-shared transformers at 44M-1B parameters (C4 perplexity 13.24 vs 13.40 at 244M) at roughly matched compute (https://arxiv.org/abs/2405.16039). MobileLLM finds that sub-billion models prefer deep-thin shapes and that repeating blocks with shared weights adds accuracy at zero parameter cost (https://arxiv.org/abs/2402.14905).

My expectation, to be tested rather than assumed: a dense looped model at matched body parameters (baseline B2, §11) beats rex-v2 at equal tokens, because it spends about 2.4x the matmul compute per token with every parameter active. At equal FLOPs the outcome is open. The design reacts to this in two ways: sparsity is kept low (28 experts, top-8, 3.5x) rather than Inkling's 16x, and the baselines needed to falsify the MoE are specified with exact counts. Kill criterion: if rex-v2 does not beat B1 (same model, dense FFN of the active width, 78.0M body) by a clear margin at equal tokens, the 119M of inactive expert parameters are dead weight.

### Costs the user's choices carry (stated so nobody is surprised)

- The 201,024-token vocabulary makes the tied embedding 115.8M parameters (59% of the body's size) and the output projection 232M FLOPs per token forward, which is 37% of forward compute at short context. On English-only data most of those rows are never targets.
- Per-layer cross-pass gates roughly double activation memory in passes 2 and 3 (§6.4). Without gradient checkpointing the model stores about 4 MB of activations per token.
- At 262,144 tokens, full-attention layers are 83% of forward compute (§6.3). The 3-pass loop makes 8 full-attention applications out of 4 stored full-attention layers.

---

## 2. Final configuration

| Group | Setting | Value | Tag |
|---|---|---|---|
| Vocabulary | tokenizer | Inkling 2 Mini `tokenizer.json` (Apache-2.0, ungated) | [I] |
| | `vocab_size` (table rows) / `unpadded_vocab_size` | 201,024 / 200,058 | [I] |
| | embeddings | tied input/output | [J] (undisclosed in Inkling) |
| Width/depth | `d_model` | 576 (= 2304 / 4) | [J] |
| | stored layers / block size / blocks | 24 / 6 / 4 | [I] |
| | looped (core) layers | 6..17 (blocks B2, B3) | [I] |
| | passes `T` | 3, fixed | [I] |
| Residual | type | attention residuals over block sums, normalised values | [I][K] |
| Loop | cross-pass readers | per core layer, separate for Q, K, V, MoE | [I] |
| | gate rank | 16 (= 64 / 4) | [I] scaled |
| Attention | `head_dim` | 128 | [I], decided on merits §7 |
| | query heads / KV heads | 6 / 2 (inner 768 = 4/3 d; KV 256 = 4/9 d, Inkling's ratios) | [I] scaled |
| | full-attention layers | 5, 11, 17, 23 (last of each block) | [I] |
| | sliding window | 1,024 tokens in the other 20 layers | [I] |
| | position | RoPE (base 10,000) in sliding-window layers; none in full-attention layers | [J] |
| | QK treatment | QK-norm with learned per-head scale (init 1.44) | [S] |
| | long-context logit scaling | full-attention layers: x (1 + 0.1 ln(max(n, 8192)/8192)) | [I?] |
| | output gate | per head, 2·sigmoid, identity at init | [I?] |
| MoE | experts / top-k / expert hidden | 28 / 8 / 144 (= d/4) | [J] |
| | shared experts | 0 | [I] |
| | gate weights | softmax over selected logits, times `route_scale` = 8 | [I] |
| | router bias | on | [I] |
| | balancing | Switch aux loss 0.01 with smoothed load + z-loss 0.001 | [S]/[J] |
| Norm | RMSNorm, eps | 1e-6 | [I] |
| Output | final logit softcap | 30·tanh(x/30) | [I] |
| Context | `max_len` | 262,144, reached by curriculum (§6.5) | user |

Ready-to-use config (YAML; every key is consumed by the spec below):

```yaml
name: rex-v2-base
model:
  vocab_size: 201024
  unpadded_vocab_size: 200058
  bos_id: 200028          # <|begin_of_text|>
  eot_id: 199999          # <|endoftext|>
  tie_embeddings: true
  d_model: 576
  n_layers: 24
  block_size: 6           # layers per AttnRes block
  core_start: 6           # first looped layer (inclusive)
  core_end: 18            # last looped layer (exclusive)
  n_passes: 3
  residual: attnres       # attnres | stream (stream only for baseline B0, requires n_passes: 1)
  gate_rank: 16
  rms_eps: 1.0e-6
  attn:
    head_dim: 128
    n_q_heads: 6
    n_kv_heads: 2
    window: 1024
    global_layers: [5, 11, 17, 23]
    local_pos: rope
    rope_base: 10000.0
    global_pos: none      # none | rope (ablation; rope uses global_rope_base)
    global_rope_base: 1.0e6
    qk_mode: norm         # norm | l2clip (ablation; clip threshold 2*sqrt(head_dim))
    qk_scale_init: 1.44
    log_scale_alpha: 0.1
    log_scale_n_floor: 8192
    out_gate: true
    backend: flash        # flash | reference
  ffn:
    type: moe             # moe | dense
    n_experts: 28
    top_k: 8
    expert_hidden: 144
    route_scale: 8.0
    router_bias: true
    dense_hidden: 1152    # used only when type: dense
    dispatch: sorted      # sorted | dense_ref (tests only)
  final_logit_softcap: 30.0
loss:
  lb_coef: 0.01
  lb_load_ema: 0.9
  z_coef: 0.001
  balance: aux            # aux | bias (ablation)
  bias_update_speed: 0.001
  loss_chunk_tokens: 2048
train:
  total_tokens: 6.0e9
  tokens_per_step: 262144
  mb_tokens: 65536        # micro-batch token budget; lower it to fit memory
  grad_ckpt: layer        # none | layer
  tok_chunk: null         # e.g. 32768: nested checkpoint of token-local parts for very long sequences
  optimizer: adamw        # adamw | muon (ablation)
  lr: 4.0e-4
  lr_min_ratio: 0.1
  warmup_steps: 1000
  betas: [0.9, 0.95]
  adam_eps: 1.0e-8
  weight_decay: 0.1
  grad_clip: 1.0
  precision: bf16
  pass_sampling: fixed    # fixed | uniform (ablation: T ~ U{1..n_passes} per micro-batch)
  curriculum:             # fractions of total_tokens; see §6.5
    - {until: 0.82, max_len: 8192,   long_doc_frac: 0.15}
    - {until: 0.94, max_len: 65536,  long_doc_frac: 0.50}
    - {until: 1.00, max_len: 262144, long_doc_frac: 0.70}
data:
  main: HuggingFaceFW/fineweb-edu (sample-10BT)
  long_docs: deepmind/pg19
```

---

## 3. Exact parameter counts

Counted with a throwaway script (kept outside the repo). The repo must contain `scripts/count_params.py` that reproduces these from the config with the formulas in the right-hand column; acceptance test T1 compares both against the instantiated model.

Symbols: d=576, V=201,024, L=24, Hq=6, Hkv=2, dh=128, E=28, h=144, r=16, core layers C=12, full-attention layers G=4.

| Component | Formula | Parameters |
|---|---|---|
| Token embedding (tied with output projection) — **excluded from cap** | V·d | 115,789,824 |
| Embedding RMSNorm gain | d | 576 |
| Attention W_Q | L·d·Hq·dh | 10,616,832 |
| Attention W_K | L·d·Hkv·dh | 3,538,944 |
| Attention W_V | L·d·Hkv·dh | 3,538,944 |
| Attention W_O | L·Hq·dh·d | 10,616,832 |
| QK scale (per head) | L·Hq | 144 |
| Output gate (weight + bias) | L·(Hq·d + Hq) | 83,088 |
| Sublayer RMSNorm gains (attention, FFN) | 2·L·d | 27,648 |
| Depth readers, attention sublayers | (L − 1 + 2·G)·d = 31·d | 17,856 |
| Depth readers, MoE sublayers | L·d | 13,824 |
| Depth reader, output | d | 576 |
| Final RMSNorm gain | d | 576 |
| Router weight | L·E·d | 387,072 |
| Router bias | L·E | 672 |
| Expert W_in (gate and up, stacked) | L·E·2h·d | 111,476,736 |
| Expert W_out | L·E·d·h | 55,738,368 |
| Cross-pass reader queries u | C·4·d | 27,648 |
| Cross-pass gate P | C·4·r·2d | 884,736 |
| Cross-pass gate F | C·4·d·r | 442,368 |
| Cross-pass gate G | C·4·d·r | 442,368 |
| **Body total** | | **197,855,808** |

Per stored layer: attention matrices 1,179,648; MoE 6,983,452 (router 16,156 + experts 6,967,296); each core layer adds 149,760 of cross-pass parameters. Headroom under the cap: 2,144,192 (1.07%). E=29 would give 203,841,624 and violates the cap; E=27 gives 191,869,992.

Depth reader count is 56 (31 attention + 24 MoE + 1 output), which equals the count Goodfire reports for Inkling. That report does not explain 31; the reading used here, 23 + 4·2 (no reader for layer 0's attention, which has a single source, and separate Q/K/V readers in the four full-attention layers), is my inference **[I?]**.

Active-parameter accounting: inactive experts are L·(E−k)·3dh = 119,439,360, so stored-active body = 78,416,448. Effective-active = 156,653,568, defined as weight matrices only, each counted once per application: per application the attention matrices (1,179,648), the output-gate weight (3,456), the router weight (16,128) and the 8 selected experts (1,990,656), times 48 applications, plus `P`, `F`, `G` of the four gates (147,456) for each of the 24 gated applications. All vectors are excluded: router bias, output-gate bias, gains, QK scales, depth and cross-pass queries. (Revision 2 printed 156,654,912, which wrongly included the router bias.)

---

## 4. What is taken from Inkling 2 Mini and what is not

| Inkling feature | Here | Reason |
|---|---|---|
| 24 layers, 4 blocks of 6, core = blocks 2-3, 3 passes, shared weights | kept exactly | [I] |
| Attention residuals over block sums, normalised values, embedding as block 0 | kept | [I][K] |
| Cross-pass readers for Q, K, V, MoE with rank-r keep/gain gates; every pass restarts from the prelude | kept, rank scaled 64 -> 16 | [I] |
| 5 sliding-window (1,024) layers per full-attention layer | kept | [I] |
| head_dim 128, Q inner width 4/3 d, KV width 4/9 d (GQA 3:1) | kept | [I], §7 |
| MoE in every layer, no shared expert, selected scores normalised x8, router bias | kept | [I] |
| 128 experts, top-8, expert hidden d/2 | 28 experts, top-8, hidden d/4 | cap; §5.7 |
| RMSNorm eps 1e-6, final softcap 30, vocabulary | kept | [I] |
| Learned relative-position logits (d_rel 16, extent 1,024) | replaced by RoPE in windowed layers, nothing in full layers | per-pair logit bias needs a custom kernel; same property kept (position information never extends past the window) [J] |
| Q/K L2 clipping at 2·sqrt(head_dim) | QK-norm default, clip as ablation switch | QK-norm has small-scale evidence (https://arxiv.org/abs/2409.02060 §4.2.5); clip has none public [S] |
| `log_scaling_alpha` 0.1, `log_scaling_n_floor` 8192 | numbers kept, form guessed | [I?], §5.6 |
| Learned sink | dropped | a prepended sink key falls outside flash-attn's window alignment; the output gate covers the same need [J] |
| Output gate | kept, form guessed | [I?] |
| Kernel-4 short convolutions, dynamic K/V convolutions | dropped | form undisclosed; DeepSeek-V4.1 dropped the short causal convolution from its Engram module because its gains "do not justify the added complexity" [D]; five windowed layers precede every full layer |
| MLP gate/up softcaps (4.0 / 25.0), value norm | dropped | form undisclosed, no small-scale evidence, extra activation memory [J] |
| Vision tower, DSpark drafter, FP4/FP8 weights | out of scope | text-only |

Undisclosed in every public source, therefore decided here: embedding tying, norm placement details, all initialisation, load balancing, optimiser and schedule, how the loop is trained.

---

## 5. Architecture specification

### 5.1 Notation and data layout

Every tensor is flat over tokens. A micro-batch is a packed buffer of whole sequences (documents or crops):

- `ids[N]` int64, `cu[n_seq+1]` int32 cumulative sequence offsets (`cu[0]=0`, `cu[-1]=N`), `pos[N]` = position of each token inside its sequence.
- All operations except attention are token-local. Attention never crosses a sequence boundary ("sample-level attention masking" [D]).
- `nrm(x) = x / sqrt(mean(x², dim=-1) + eps)`, parameter-free, computed in float32.
- `RMSNorm_g(x) = g ⊙ nrm(x)`, `g ∈ R^d` initialised to 1.
- Blocks are numbered B1..B4 (layers 0-5, 6-11, 12-17, 18-23). `V_b` is the completed sum of block b; `R` is the running sum of the block in progress.

### 5.2 Embedding

`e = Emb[ids]` (d). `v0 = RMSNorm_emb(e)` **[I]**. With `tie_embeddings`, the output projection uses the same matrix `Emb`.

### 5.3 Depth-wise mix (attention residual)

```
DepthMix(q, sources=[s_1..s_n]):          # q ∈ R^d learned; each s_j is [N, d]
    ŝ_j = nrm(s_j)
    a_j = (ŝ_j · q) / sqrt(d)             # [N]
    α   = softmax over j of a             # float32
    return Σ_j α_j ⊙ ŝ_j                  # values are normalised (attn_res_normalize_values) [I]
```

Source lists **[I][K]**:

| Reader location | Sources |
|---|---|
| Sublayer in B1 | v0, R |
| Sublayer in B2, pass t | v0, V1, R |
| Sublayer in B3, pass t | v0, V1, V2[t], R |
| Sublayer in B4 | v0, V1, V2[T], V3[T], R |
| Output reader | v0, V1, V2[T], V3[T], V4 |

`R` is omitted from the list while it is empty, i.e. for the attention sublayer of the first layer of a block (this matches the Kimi reference code; Inkling's handling is not stated **[I?]**). For an MoE sublayer `R` already includes that layer's attention output. Running sums start empty at every block and at every pass.

All queries `q` are initialised to zero, giving a uniform average over sources at step 0. The Kimi paper states this is required: "all pseudo-query vectors must be initialized to zero ... prevents training volatility" (https://arxiv.org/abs/2603.15031 §5).

Consequence used as an acceptance test: multiplying every output projection inside one block by a positive constant leaves the model's logits unchanged. The invariance is exact only as `rms_eps` -> 0; with eps > 0 a source of mean square m is perturbed by a relative amount of order eps/m, and top-k routing can turn that into a discrete change. At default init the block-sum RMS is about 0.1-0.3 in the base config (eps/m ≤ 1e-4) and about 0.025 in the tiny test config (eps/m ≈ 1.6e-3). Neither is an init-scale problem, because values and keys are normalised and only eps/m matters; `rms_eps` stays 1e-6 [I] and no init is rescaled for small configs.

### 5.4 Layer schedule and forward pass

```
forward(ids, cu, pos, T=n_passes):
    v0 = RMSNorm_emb(Emb[ids])
    done = [v0]

    R = None
    for l in 0..5:   R = Layer_l(done, R, prev=None)              # B1 (prelude)
    V1 = R; done = [v0, V1]

    prev = None
    for t in 1..T:                                                # core, shared weights
        R = None
        for l in 6..11:  R = Layer_l(done, R, prev)               # B2
        V2 = R
        R = None
        for l in 12..17: R = Layer_l(done + [V2], R, prev)        # B3
        V3 = R
        prev = (nrm(V2), nrm(V3))                                 # read by pass t+1

    done = [v0, V1, V2, V3]                                       # last pass only
    R = None
    for l in 18..23: R = Layer_l(done, R, prev=None)              # B4 (coda)
    V4 = R

    x = DepthMix(q_out, [v0, V1, V2, V3, V4])
    hfinal = RMSNorm_f(x)
    return hfinal                                                 # logits in §5.8
```

```
Layer_l(done, R, prev):
    S = done + ([R] if R is not None else [])
    # --- attention sublayer ---
    if l is a full-attention layer:  xQ, xK, xV = DepthMix(q^Q_l, S), DepthMix(q^K_l, S), DepthMix(q^V_l, S)
    else:                            xQ = xK = xV = DepthMix(q^A_l, S)
    if prev is not None:             xQ, xK, xV = GateMix^Q_l(xQ, prev), GateMix^K_l(xK, prev), GateMix^V_l(xV, prev)
    o_attn = Attention_l(RMSNorm_attn,l(xQ), RMSNorm_attn,l(xK), RMSNorm_attn,l(xV))
    R = o_attn if R is None else R + o_attn
    # --- MoE sublayer ---
    x = DepthMix(q^M_l, done + [R])
    if prev is not None:             x = GateMix^M_l(x, prev)
    o_moe = FFN_l(RMSNorm_ffn,l(x))
    return R + o_moe
```

Config validation requires `n_layers = 4·block_size`, `core_start = block_size`, `core_end = 3·block_size`; the schedule above is written for the base values. `prev` is non-`None` only for core layers in passes 2 and 3. One gain vector `RMSNorm_attn,l` is shared by the three attention input streams **[J]**. Layer 0 has no `q^A` (single source, the mix is the identity).

Every pass starts from the same inputs (v0, V1) with empty running sums; information from the previous pass arrives only through `GateMix` **[I]** ("Every pass through the looped core starts from the same inputs ... Information moves between passes through cross-pass messages", https://huggingface.co/blog/inkling2mini).

### 5.5 Cross-pass message and gate

For core layer l and input s ∈ {Q, K, V, M}, with own parameters `u ∈ R^d`, `P ∈ R^{r×2d}`, `F, G ∈ R^{d×r}` **[I]** (equations from https://www.goodfire.com/research/interpreting-inkling-2-mini, appendix):

```
GateMix(x, prev=(v̂2, v̂3)):
    b_j = (v̂_j · u) / sqrt(d),  j ∈ {2, 3}
    β   = softmax over j of b
    m   = β_2 ⊙ v̂2 + β_3 ⊙ v̂3
    z   = P · concat(nrm(x), nrm(m))          # [N, r]
    κ   = 1 − tanh(F z)                        # keep,  in (0, 2)
    γ   = tanh(G z)                            # gain,  in (−1, 1)
    return κ ⊙ x + γ ⊙ m
```

No biases, no explicit pass index: Inkling shows none, and Goodfire's finding that the model marks the last pass in an activation direction indicates it does not need one **[I]**.

Initialisation (undisclosed in Inkling; **[J]**): `u = 0`, `F = 0`, `G = 0`, `P ~ N(0, 0.02²)`. Then κ = 1 and γ = 0 exactly, so at initialisation every pass computes the same block sums and the T-pass model equals the 1-pass model. F and G receive gradient from step 0; P and u receive gradient from step 1.

Loop training decisions:

- Full backpropagation through all three passes. With T = 3 there is nothing to truncate; Geiping et al. truncate to the last 8 of ~32 iterations (https://arxiv.org/abs/2502.05171).
- Fixed T = 3 **[I]** (`is_varloop: false` in the released model). `pass_sampling: uniform` is an ablation mirroring Inkling's separate variable-loop research checkpoint.
- Stability across passes comes from the residual design, not from extra machinery: every sublayer input is a convex mix of unit-RMS vectors, then gated with bounded κ, γ, then RMS-normalised, and running sums restart per pass. Shared layers therefore see the same input scale in every pass. This removes the failure MoEUT describes for pre-norm shared layers, where the residual norm grows with depth and a shared layer cannot rescale its output per depth (https://arxiv.org/abs/2405.16039).
- The failure to watch for is the one Geiping et al. report as "Bad Run 2": identical loss at 1 and many recurrences, the model ignoring the incoming state. §10 monitors it.

### 5.6 Attention

Inputs `nQ, nK, nV` ([N, d], already normalised). Parameters: `W_Q ∈ R^{768×d}`, `W_K, W_V ∈ R^{256×d}`, `W_O ∈ R^{d×768}`, `s ∈ R^6`, `W_g ∈ R^{6×d}`, `b_g ∈ R^6`. No other biases.

```
q = (W_Q nQ).view(N, 6, 128);  k = (W_K nK).view(N, 2, 128);  v = (W_V nV).view(N, 2, 128)
q = nrm(q); k = nrm(k)                         # per head over the 128 dims  (qk_mode: norm)  [S]
if layer is sliding-window (or global_pos == rope): q, k = RoPE(q, pos), RoPE(k, pos)
q = q * s[h]                                   # learned per-head scale, init 1.44
if layer is full-attention:  q = q * (1 + alpha * ln(max(pos + 1, n_floor) / n_floor))
a = CausalAttention(q, k, v, cu, window)       # scale 1/sqrt(128); query head h uses KV head h // 3
a = a * (2 * sigmoid(W_g nQ + b_g))[:, :, None]   # output gate per head
o_attn = W_O a.view(N, 768)
```

- Visibility: token i attends to token j iff both are in the same sequence and `pos_i − window < pos_j ≤ pos_i`. Full-attention layers have `window = ∞`.
- Effective logit: `s_h · λ_i · sqrt(128) · cos(q_i, k_j)`. Initial scale 1.44 follows nanochat's 1.2 x 1.2 (https://github.com/karpathy/nanochat, `nanochat/gpt.py`).
- `qk_mode: l2clip` (ablation, **[I]**): instead of `nrm`, `q ← q·min(1, c/‖q‖)`, same for k, `c = 2·sqrt(128) = 22.627`; `s` is then fixed to 1.
- RoPE: for a head vector split in halves `(x1, x2)`, `θ_i = base^(−i/64)`, i = 0..63, angle `φ = pos·θ_i`; output `(x1 cos φ − x2 sin φ, x1 sin φ + x2 cos φ)`. Tables are computed in float64 for positions up to 262,143 and then cast; computing angles in float32 loses the high frequencies at large positions.
- Position in full-attention layers is deliberately absent **[J]**. Reasoning: most training tokens are seen at short lengths; with rotary on full layers the low frequencies would be learned as non-rotating content dimensions and then start rotating at lengths first seen late in training. With no positional term, nothing in a full-attention layer depends on distance, so behaviour at 262k differs from 8k only in the number of keys. Order information reaches those layers through the five preceding RoPE layers and the causal mask (https://arxiv.org/abs/2305.19466). Inkling bounds its positional signal the same way (`rel_extent` 1024 = window). `global_pos: rope` is an ablation.
- Length scaling **[I?]**: Inkling's config has `log_scaling_alpha: 0.1`, `log_scaling_n_floor: 8192` with no public definition. The form above is my reading, motivated by Scalable-Softmax (attention flattens as the number of keys grows; scale logits with log n; https://arxiv.org/abs/2501.19399). It is exactly 1 for the first 8,192 positions, 1.21 at 65,536 and 1.35 at 262,144.
- Output gate **[I?]**: lets a head emit nothing. In windowed layers the first token is outside the window, so the usual trick of parking attention on it is unavailable (https://arxiv.org/abs/2309.17453). `W_g = 0, b_g = 0` gives gate 1 at initialisation.

Backends (both mandatory):

- `reference`: per sequence, explicit float32 `softmax(QKᵀ/sqrt(dh) + mask)` with a boolean [n, n] mask. Used on CPU, in tests, and for attention statistics.
- `flash`: `flash_attn.flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len, causal=True, window_size=(window−1, 0))` for windowed layers and `window_size=(−1, −1)` for full layers. It accepts the packed layout directly, GQA included. §7 lists the checked constraints.

### 5.7 Feed-forward: routed MoE

Input `n` ([N, d], normalised). Parameters: `W_r ∈ R^{E×d}`, `b_r ∈ R^E`, `W_in ∈ R^{E×2h×d}` (rows 0..h−1 gate, h..2h−1 up), `W_out ∈ R^{E×d×h}`. No expert biases.

```
logits = n.float() @ W_rᵀ + b_r                  # [N, E], float32
top_v, top_i = topk(logits, k=8)                 # no noise
w = route_scale * softmax(top_v, dim=-1)         # [N, 8], sums to 8   [I]
Expert_e(n) = W_out[e] · ( silu(g) ⊙ u ),  (g, u) = split(W_in[e] · n)
o_moe = Σ_{j=1..8} Expert_{top_i[j]}(n) with the hidden activation scaled by w[j]
```

Scaling the h-dimensional hidden activation by `w[j]` before `W_out` is identical to scaling the output and stores 144 instead of 576 values per assignment.

Shape decisions:

- Top-8 **[I]** with hidden d/4 rather than d/2: at this cap, hidden d/2 with top-8 leaves only 14 experts (1.75x sparsity). Halving expert size keeps Inkling's routing fan-out and doubles the number of experts. Finer experts are supported by OLMoE (8 -> 32 experts worth about 10%, 32 -> 64 another 1-2%; https://arxiv.org/abs/2409.02060 §4.1.2), Krajewski et al. (https://arxiv.org/abs/2402.07871), and MoEUT, which uses d_expert = 128 with K up to 16 at 44M-244M (https://arxiv.org/abs/2405.16039). Active hidden width is 8·144 = 2d.
- No shared expert **[I]**; OLMoE found a shared expert slightly worse (§4.1.3).
- MoE in every layer including the first and last **[I]**. `ffn.type: dense` exists for the baselines.
- No routing noise or jitter **[S]** (OLMoE, Abnar et al. both use none).

Auxiliary losses, per stored layer, pooling tokens from **all applications of that layer** in the micro-batch (a core layer pools its three passes, so experts may specialise by pass without penalty **[J]**):

```
f_mb[i] = (assignments to expert i) / (N_apps · k)             # no gradient
f̃ ← ρ·f̃ + (1 − ρ)·f_mb        (buffer, init 1/E, ρ = lb_load_ema)
P̄[i]   = mean over tokens and applications of softmax(logits)[i]
L_LB    = E · Σ_i f̃[i] · P̄[i]                                  # equals 1.0 when uniform
L_Z     = mean over tokens and applications of (logsumexp(logits))²
loss    = L_CE + lb_coef · Σ_layers L_LB + z_coef · Σ_layers L_Z
```

- Form and coefficients are Switch/OLMoE **[S]** (https://arxiv.org/abs/2101.03961; OLMoE §4.1.6-7: 0.01 and 0.001; without the balance loss OLMoE's first layer collapsed onto one expert).
- The smoothed load `f̃` is **[J]**, motivated by Qiu et al.: when a micro-batch is close to one sequence, the standard loss forces uniform routing within that sequence and suppresses specialisation (https://arxiv.org/abs/2501.11873). At 262k a micro-batch is one document. ρ = 0 recovers the standard loss. The buffer `f̃` is written exactly once per training-mode forward of a micro-batch (never by a checkpoint recomputation); in eval mode the loss is computed from the same formula but the buffer is left untouched, so evaluation never changes training state.
- `balance: bias` (ablation, DeepSeek-V3/V4.1 **[D]**; https://arxiv.org/abs/2412.19437 §2.1.2): a non-learned buffer `c ∈ R^E` per layer is added to `logits` for the top-k selection only; after each optimiser step `c_i += bias_update_speed · sign(1/E − f_step[i])`; `lb_coef` is set to 1e-4.

Dead experts: an expert is dead if its load over an evaluation window is below 0.1/E. No revival mechanism is built. With 8 of 28 selected, death is unlikely; if any expert stays dead for 2,000 steps after warm-up, the run is restarted with `lb_coef: 0.03`, and if that fails with `balance: bias`.

Dispatch (`sorted`, default, dropless):

```
flat_e = top_i.reshape(-1); tok = arange(N).repeat_interleave(k)
order = argsort(flat_e, stable=True); counts = bincount(flat_e, minlength=E)
xs = n[tok[order]]; ws = w.reshape(-1)[order]
for e with counts[e] > 0, on its contiguous slice:
    g, u = split(xs_slice @ W_in[e]ᵀ);  hid = silu(g) * u * ws_slice[:, None];  ys_slice = hid @ W_out[e]ᵀ
o_moe = zeros(N, d).index_add_(0, tok[order], concat(ys))
```

`dense_ref` evaluates every expert on every token and masks; it costs E/k = 3.5x and exists only as the reference for tests. The sorted path has data-dependent shapes; `torch.compile` must not be required for correctness.

Dense FFN (baselines): `W_in ∈ R^{2H×d}`, `W_out ∈ R^{d×H}`, `o = W_out(silu(g) ⊙ u)`.

### 5.8 Output and loss

```
logits = hfinal @ Embᵀ                      # only columns 0..unpadded_vocab_size−1
logits = 30 * tanh(logits / 30)             # float32
L_CE   = mean cross-entropy over all positions that have a next token in the same sequence
```

Rows 200,058..201,023 of `Emb` are never read as inputs or outputs.

The head must be evaluated in chunks: 65,536 tokens x 200,058 logits is 52 GB in float32. Procedure: detach `hfinal` into a leaf; for each chunk of `loss_chunk_tokens` compute logits and the summed loss, call `backward` to accumulate gradients into `Emb` and into the leaf; then backpropagate the leaf's gradient into the body. This is exact.

### 5.9 Initialisation

| Parameter | Init | Tag |
|---|---|---|
| `Emb` | N(0, 0.02²) | [S] |
| All attention and expert matrices, including `W_O`, `W_out`; dense-FFN `W_in`, `W_out` in the baselines | truncated normal, std 0.02, cut at ±0.06 | [S] (OLMoE §4.2.2) |
| Router `W_r` / `b_r` | N(0, 0.02²) / 0 | [S] |
| RMSNorm gains | 1 | [S] |
| Depth queries `q` | 0 | [K] |
| Cross-pass `u`, `F`, `G` | 0 | [J] |
| Cross-pass `P` | N(0, 0.02²) | [J] |
| QK scale `s` | 1.44 | [S] |
| Output gate `W_g`, `b_g` | 0 | [J] |

Output projections must not be zero-initialised: a zero block sum has no direction and `nrm` would amplify the first tiny update to unit scale.

Expected loss at initialisation: about ln(200,058) + 0.12 = 12.32.

### 5.10 Inference state

Each of the 48 layer applications owns its KV cache **[I]** ("each visit keeps its own KV ... state"). Windowed applications keep the last 1,024 K/V; full-attention applications keep all. Cross-pass readers and depth readers use only the current token's block sums, so they add no cache. Full-attention cache: 8 applications x 512 values x 2 bytes = 8,192 bytes per token, 2.15 GB at 262,144.

---

## 6. Long context: CSA2, cost, memory, curriculum

### 6.1 What CSA2 is (from the report, §2.3, §3.1.2, §4.2)

DeepSeek-V4.1-Flash gives every layer two attention branches under one softmax: a sliding window (128 tokens) over that layer's own K/V, and a sparse global branch that reads the top 512 entries of a "main KV" cache selected by a light indexer. CSA2 compresses that global cache along three axes at once: entry size (a compact KV latent), sequence (every m tokens become one entry, m = 2 in the encoder, 1 in the decoder, with the earlier overlap and absolute position embedding removed), and layers. The layer axis is the new part. Each layer is statically assigned a mode: **Full** computes its own main KV, indexer keys and top-k indices; **Reindex** reuses the main KV and indexer keys of the latest Full layer but rescores with its own indexer query; **Reuse** reuses both the main KV and the latest indices and runs no indexer. Every mode still computes its own query and its own window K/V. Groups are one Full or Reindex layer followed by 3-5 Reuse layers; the first two layers have no global branch. A hierarchical indexer (added in post-training) lets later indexers score only a 16,384-entry candidate pool chosen by the first. Training used sparse attention from scratch at 64K with no dense warm-up, extended to 1M at 34T of 45T tokens, with sample-level attention masking. The stated goal is KV-cache bytes (890 per token) and prefill cost at serving time.

### 6.2 Adopt, adapt, reject

| CSA2 / V4.1 element | Decision | Why |
|---|---|---|
| Train the final attention form from scratch; extend to maximum length late in training | **Adopt** | §6.5 mirrors it: longest stage starts at 94% of tokens, inside the LR decay as in V4.1 |
| Sample-level attention masking for packed sequences | **Adopt** | realised by the packed layout and the varlen kernel, no padding |
| Global K/V built from a different depth than the query (Full/Reuse, and CED projecting decoder KV from the encoder's last state) | **Adapt** | full-attention layers get separate depth readers for Q, K and V, so the model learns which block sums its long-lived K/V are built from; costs 8 vectors |
| Window-only early layers; small windows suffice once a global path exists | **Adapt** | layers 0-4 are window-only already; window stays 1,024 [I] with 256 as an ablation |
| Bounded, RMS-normalised K before caching | already true | QK-norm |
| Top-k sparse global attention with an indexer; Reindex/Reuse index sharing; hierarchical indexer | **Reject** | they remove serving cost, not training cost on one GPU: gathering 512 selected entries for 262,144 queries is 137 GB at our KV width, while dense full attention at inner width 768 is affordable (§6.3). Selection errors are also a robustness risk the report itself flags (§6) |
| Cross-layer or cross-pass reuse of one K/V cache | **Reject** | would save 1 GB of inference cache at 262k and contradicts the required per-pass K and V readers |
| Sequence compression (m tokens per entry) | **Reject for v1** | needs a staircase causal mask that neither flash-attn nor SDPA provides; it is the first thing to revisit if stage-3 cost must drop |
| FP4 KV cache, CED, SWA bounded replay | **Reject** | serving-side |
| Muon for matrices, AdamW for the rest, weight decay 0.1 | ablation | §9 |

### 6.3 Compute per token

Forward FLOPs per token for a sequence of length N (2 x effective-active matmul parameters; attention 4·768 per query-key pair; windowed layers see min(pos+1, 1024) keys; head 2·V·d):

| N | Matmul | 40 windowed apps | 8 full apps | Output head | Forward total | Train (x3) | Train with checkpointing (x4) |
|---|---|---|---|---|---|---|---|
| 1,024 | 313 M | 63 M | 13 M | 232 M | 0.62 G | 1.86 G | 2.48 G |
| 8,192 | 313 M | 118 M | 101 M | 232 M | 0.76 G | 2.29 G | 3.05 G |
| 65,536 | 313 M | 125 M | 805 M | 232 M | 1.48 G | 4.43 G | 5.90 G |
| 262,144 | 313 M | 126 M | 3,221 M | 232 M | 3.89 G | 11.68 G | 15.57 G |

- A windowed application costs 3.1 M FLOPs per token at any length beyond the window. A full application costs 1.5 k·N FLOPs per token: 0.4 G at 262,144.
- The loop turns 4 stored full-attention layers into 8 applications and 20 windowed layers into 40, so it doubles every attention term; without the loop (T = 1) the 262,144 row is 2.06 G forward.
- One 262,144-token sequence costs about 4.1e15 FLOPs per training step with checkpointing.

Wall-clock, under assumed delivered rates that the first smoke run must replace with measurements (0.7e14 FLOP/s for a 24 GB consumer card, 2.5e14 for an H100-class card; MoE dispatch overhead is the main unknown): 58 s or 16 s per 262,144-token sequence, i.e. 4.5k or 16k tokens/s, against roughly 26k or 93k tokens/s at short lengths with checkpointing.

### 6.4 Memory

Estimated activations kept for backward, per token per layer application (elements; about 2.5 bytes each in mixed bf16/float32):

| Application type | Elements | Dominant terms |
|---|---|---|
| Ungated (prelude, coda, pass 1) | ~20 k | MoE gather 8·d and hidden; attention q/k/v |
| Gated (core, passes 2-3) | ~46 k | four GateMix readers at ~11·d each |

- Without checkpointing: about 4 MB per token, 32 GB for one 8,192-token sequence. This is not viable as the default.
- `grad_ckpt: layer` **(default)**: each layer application is one `torch.utils.checkpoint` segment whose inputs are `R`, the completed sources, and `prev`. Stored state is one d-vector per token per application (49·d·2 bytes = 56 KB per token) plus one application's activations during its backward (about 115 KB per token). Compute becomes about 4/3 of the uncheckpointed cost.
- `tok_chunk`: inside a segment, the token-local parts (DepthMix, GateMix, MoE) are run per chunk of tokens under a nested checkpoint, leaving only attention activations at full length.
- Implementing GateMix as one custom autograd function that saves only `x`, `m`, `z` would cut the gated figure by about a third; it is an optimisation, not a requirement.

| Sequence length | Boundaries | One application | With `tok_chunk: 32768` | Plus model, optimiser, head chunk | Fits |
|---|---|---|---|---|---|
| 65,536 | 3.7 GB | 7.6 GB | — | ~8 GB | 24 GB |
| 131,072 | 7.4 GB | 15.1 GB | 3.8 + 2.3 GB | ~8 GB | 24 GB, tight |
| 262,144 | 14.8 GB | 30.3 GB | 3.8 + 4.5 GB | ~8 GB | 48 GB and up (about 31 GB) |

These are estimates (±30%); acceptance test T16 replaces them with measurements. On a 24 GB card the last curriculum stage must be capped at 131,072.

### 6.5 Length curriculum

One optimiser schedule; stages switch by tokens consumed. A step is always 262,144 tokens, accumulated over micro-batches of at most `mb_tokens`.

| Stage | Token share | `max_len` | Long-document share of tokens | Approx. share of compute |
|---|---|---|---|---|
| 1 | 0-82% | 8,192 | 15% | 64% |
| 2 | 82-94% | 65,536 | 50% | 15% |
| 3 | 94-100% | 262,144 | 70% | 21% |

- Sequences are whole documents; a document longer than `max_len` contributes one random contiguous crop. Every sequence starts with `bos_id`; `eot_id` is appended only when the crop contains the document's end.
- Short documents keep appearing in every stage so that short-context quality does not drift.
- Nothing in the model needs a length-specific warm-up: windowed layers never see a distance above 1,023, and full layers carry no positional term. The stages exist to train long-range use and to exercise the length scaling factor, which only departs from 1 beyond 8,192.
- In stage 3 a step holds one or two documents, so gradient noise is high; the learning rate is at its floor by then. If stage-3 loss is unstable, raise `tokens_per_step` to 524,288 for that stage.

Data is the real limit on 256k training, not the model: web documents average about 1,000 tokens, so only book-length sources exercise the long path (§8).

---

## 7. Head dimension and attention kernels

Question: do FlashAttention-2 and other fast kernels work at head_dim 64? **Yes; 64 and 128 are both fully supported.** Checked sources:

- flash-attention README: "All head dimensions up to 256"; fp16 and bf16 only (https://github.com/Dao-AILab/flash-attention).
- `csrc/flash_attn/flash_api.cpp`: `TORCH_CHECK(head_size <= 256, ...)`, `TORCH_CHECK(head_size % 8 == 0, ...)`, `TORCH_CHECK(num_heads % num_heads_k == 0, ...)` in every entry point including the varlen one; window sizes ≥ sequence length are treated as unlimited.
- `flash_fwd_kernel.h`: with a left window the kernel starts at `n_block_min = max(0, (... − window_size_left) / kBlockN)`, so windowed attention costs O(N·W), not O(N²).
- `flash_attn_interface.py` docstring: with `causal=True` and unequal lengths the mask is aligned to the bottom-right corner; this is why a prepended sink key cannot be used with a window.
- PyTorch `scaled_dot_product_attention`: the flash path requires q, k, v to share the last dimension and be ≤ 256 (`aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`, `check_head_dim_size_flash`); the FlashAttention-4 path accepts ≤ 256 on device capability 9.x and ≤ 128, 192/128, or 256 on 10.x. SDPA has no window argument and no packed-sequence form, and an explicit `attn_mask` rules out its flash path, which is why `flash-attn` is the production backend here.
- FlashAttention-3 (Hopper only) and FlashAttention-4 are mentioned in the README; their window and varlen support was not verified, so they are not relied on.

Decision: **head_dim 128, 6 query heads, 2 KV heads, attention inner width 768 > d** **[I]**, on these merits:

- With QK-norm the largest logit is `s·sqrt(dh)` while a random key's logit has standard deviation `s`. Picking one key out of N needs roughly `s·sqrt(dh) − s²/2 > ln N`; at N = 262,144 (ln N = 12.5) and s = 2 the margin is 1.5 for dh = 64 and 8.1 for dh = 128. Larger heads retrieve more reliably at long range. This is my derivation **[J]**; the low-rank-bottleneck argument for fixing head size independently of d is published (https://arxiv.org/abs/2002.07028).
- Decoupling heads x head_dim from d (768 vs 576) is what Inkling does (3,072 vs 2,304) and costs 1.18 M parameters per layer, 14% of the body.
- GQA at 3:1 is kept because the K/V of full-attention layers are now a real cost: 2.15 GB of cache at 262k with 2 KV heads, 6.4 GB with 6.

The cost is fewer distinct heads (6 per layer). Ablation 8 compares 12 x 64 with 4 KV heads: the four projection matrices are identical in size, and the per-head QK scales and output gates add 83,232 parameters (body 197,939,040, still under the cap).

---

## 8. Tokenizer and data

Tokenizer **[I]**: `tokenizer.json` from https://huggingface.co/thinkingmachines/Inkling-2-Mini (Apache-2.0, not gated), loaded with `tokenizers`. Ids 0..200,057 are valid; specials sit at 199,998..200,057. Only `<|begin_of_text|>` (200,028) and `<|endoftext|>` (199,999) are used in pre-training. The tokenizer is pinned by file hash in the data-prep manifest.

Default data (the user has not specified any; English assumed):

- Main corpus: FineWeb-Edu `sample-10BT`, 9.67M English documents, ODC-By (https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Nominally 10B tokens (the dataset page does not say under which tokenizer); the count under the Inkling tokenizer must be measured during prep.
- Long-document corpus: PG-19, 28,602 training books, Apache-2.0 (https://huggingface.co/datasets/deepmind/pg19; https://arxiv.org/abs/1911.05507). Books average roughly 90k tokens (my conversion from the paper's word counts; measure during prep), so most stage-3 sequences will be 50k-200k tokens and only the longest books fill 262,144. The dataset repo uses a loading script; read its parquet conversion instead. Held-out: PG-19 validation and test books, plus the last FineWeb-Edu shard.
- Prep output: uint32 token shards plus a document-offset index, so the loader can draw whole documents and crops by length without re-tokenising.

Tokens needed (default `total_tokens: 6e9`):

- Lower bound from the usual ~20 tokens per parameter for dense models (https://arxiv.org/abs/2203.15556) applied to the body: 4B.
- Expert weights are 119M inactive plus 48M active of the 198M; each expert sees 8/28 of the token-applications of its layer. Giving experts the same number of updates per parameter as a dense layer pushes the figure towards 6-10B.
- 6B is about 4.6B FineWeb-Edu tokens plus 1.4B book tokens (under one epoch of each). With checkpointing and the assumed rates of §6.3 it is about 2.0e19 FLOPs: roughly 22 hours on an H100-class card or 3.3 days on a 24 GB card. Tiers: smoke 50M, ablation 600M, full 6B.
- Losses are reported in bits per byte as well as nats per token, because the 200k tokenizer makes per-token loss incomparable with published small-model numbers.

---

## 9. Training recipe

| Item | Setting | Tag |
|---|---|---|
| Optimiser | AdamW, β = (0.9, 0.95), eps 1e-8 | [S] |
| Peak LR | 4e-4 (sweep 2e-4 / 4e-4 / 8e-4 in the smoke tier) | [S] range: OLMoE 4e-4, Moonlight 4.2e-4, DeepSeek-V4.1 2.6e-4, MoEUT 2.5e-4 |
| Schedule | linear warm-up 1,000 steps, cosine to 0.1x peak at the last step | [S] |
| Step size | 262,144 tokens (22,888 steps for 6B) | [J] |
| Weight decay 0.1 | all matrices: `Emb`, attention, experts, router weight, output-gate weight | [S] |
| Weight decay 0 | RMSNorm gains, biases, QK scales, depth queries, cross-pass `u`, `P`, `F`, `G` | [J]: decay on F and G pulls the loop shut |
| Gradient clipping | global norm 1.0 | [S] |
| Precision | bf16 autocast; float32 for `nrm`, DepthMix softmax, router and auxiliary losses, softcap and cross-entropy | [S] |
| Dropout | none | [S] |
| Checkpointing | `grad_ckpt: layer` | §6.4 |
| Loop | full backpropagation through 3 passes, fixed T | §5.5 |

The shared core weights accumulate gradient from three passes; no per-pass rescaling is applied (Adam normalises it).

Muon (ablation 6, not the default because its interaction with zero-initialised readers and gates is untested): for attention and expert matrices only, per expert slice; Nesterov momentum 0.95; five Newton-Schulz steps with (3.4445, −4.7750, 2.0315); update = 0.2·sqrt(max(rows, cols))·NS(·) so the AdamW learning rate and weight decay 0.1 carry over; everything else stays on AdamW (https://arxiv.org/abs/2502.16982, eqs. 1-4; about 2x compute efficiency over AdamW claimed, and the paper's own model is an MoE; DeepSeek-V4.1 uses the same split with RMS 0.18 [D]).

Throughput notes:

- A forward pass runs 48 x 28 expert slices. Micro-batches should be large (default 65,536 tokens) so each slice is a substantial matmul; this is the main reason checkpointing is on by default.
- The head is 37% of short-context compute; keep its chunks at 2,048 tokens or more.
- Sorting and `bincount` force one host sync per MoE application (48 per forward). Acceptable.

---

## 10. Monitoring and evaluation

Logged every 100 steps unless noted; "eval" items run every 1,000 steps on fixed held-out batches.

MoE is working if:

| Metric | Definition | Healthy |
|---|---|---|
| Load entropy | H(f)/ln E per stored layer, and per pass for core layers | > 0.95 |
| Max/min load ratio | per layer | < 4 |
| Dead experts | load < 0.1/E over the eval window | 0 |
| Router confidence | mean of the largest of the 8 gate weights / 8 | rises from 0.125, stays < 0.6 |
| Pass overlap (eval) | mean Jaccard overlap of a token's expert sets between passes, per core layer | below 1; equal to 1 means passes route identically |
| `L_LB`, `L_Z` | per layer | `L_LB` near 1.0 |
| Value of experts | rex-v2 vs B1 loss at equal tokens | clear gap |

The loop is working if:

| Metric | Definition | Healthy |
|---|---|---|
| Loss at forced T = 1, 2, 3 (eval) | same weights, fewer passes | loss(1) clearly above loss(3) after early training; equal values mean the loop is ignored |
| Message ablation (eval) | loss with γ forced to 0 in all GateMix | above normal loss |
| Gate statistics | RMS of γ and mean of κ per reader type (Q, K, V, M), per core layer, per pass | γ RMS leaves 0 within the first few hundred steps |
| β | mean weight on V3 vs V2 per reader | Inkling reads mostly V3; informational |
| Pass change | cosine and relative norm of V3[t] − V3[t−1] | neither 0 nor growing |
| Token correlation | mean pairwise cosine between tokens of `nrm(V3[t])` | not approaching 1 (collapse signature in https://arxiv.org/abs/2502.05171 §4.3) |

Attention residuals are working if:

| Metric | Definition |
|---|---|
| α maps (eval) | mean depth-mix weights per reader; should depart from uniform and favour recent sources with persistent weight on v0 (Kimi Fig. 8; Goodfire Fig. 4) |
| Block-sum RMS | RMS of V1..V4 and of running sums; unbounded growth is harmless to the forward pass but signals a scale race; a value below 0.01 (mean square within 100x of eps) means eps is distorting the mix |

Long context is working if:

| Metric | Definition |
|---|---|
| Loss by position bucket (eval, PG-19) | 0-1k, 1k-8k, 8k-64k, 64k-262k; must not rise with position |
| Context truncation (eval) | loss on the same target tokens given the last 1k / 8k / 64k / all context; the difference is the value of long context |
| Repeat probe (eval) | a held-out passage of 512 tokens repeated after a gap of 2k / 16k / 128k unrelated tokens; loss on the repeat should be far below loss on first occurrence |
| Attention distance (eval, reference backend, short sequences) | share of full-layer attention mass beyond 1,024 tokens |

Optimisation and system: training and validation loss in nats and bits per byte, gradient norm and clip fraction, update-to-weight ratio per parameter group, share of logits beyond ±25 (softcap saturation), QK scales, output-gate means, tokens/s, peak memory, sequence-length histogram per stage.

---

## 11. Dense baselines

All share the tokenizer, data, curriculum and recipe. Counts are exact body parameters.

| ID | Purpose | Definition | Body | Embedding | Forward FLOPs/token at N=8,192 |
|---|---|---|---|---|---|
| rex-v2 | — | §2 | 197,855,808 | 115,789,824 | 0.76 G |
| A1 | loop off | rex-v2 with `n_passes: 1` (no cross-pass parameters) | 196,058,688 | 115,789,824 | 0.49 G |
| B1 | dense, matched active parameters and compute | rex-v2 with `ffn.type: dense`, `dense_hidden: 1152` | 78,028,704 | 115,789,824 | 0.76 G |
| B2 | dense, matched body, looped | d=896, 9 query / 3 KV heads x 128, dense hidden 1,856, gate rank 24, T=3 | 190,255,280 | 180,117,504 | 1.45 G |
| B3 | dense, matched body, no loop | B2 with `n_passes: 1` | 186,083,504 | 180,117,504 | 0.90 G |
| B0 | plain transformer anchor | B3 with `residual: stream` | 186,033,328 | 180,117,504 | 0.90 G |

`residual: stream` is the ordinary pre-norm stream: `x = RMSNorm_emb(Emb[ids])`; `x += Attention(RMSNorm(x))`; `x += FFN(RMSNorm(x))`; `hfinal = RMSNorm_f(x)`; attention and FFN modules unchanged.

B0's initial loss is higher than the others' (12.73, reproduced; the base model's 12.30 and the 11.5 and 12.34 below are as reported by the stage-1 implementation). With a plain stream and tied embeddings the final state at initialisation is still dominated by the token's own embedding, so the logit of the input token itself averages about 11.5 and takes roughly 30% of the probability mass; with that one logit removed the loss is 12.34. Attention residuals dilute the embedding to one source in five, which is why the other configs do not show it. This is a transient of the first few dozen steps. B0's initialisation is deliberately left unchanged so that it remains a standard-init anchor.

Every comparison is reported twice: at equal training tokens and at equal training FLOPs (evaluate checkpoints at matched cumulative FLOPs). B2 is the configuration behind the claim that a dense looped model wins at equal data; B1 tests whether the experts earn their parameters.

---

## 12. Ablation plan (ordered)

Each run uses the 600M-token tier (2,289 steps) with stage-1 settings and `warmup_steps: 200` unless stated; run 0 uses `smoke.yaml` (§13). An MoE's advantage can appear late because experts train on a fraction of tokens, so any conclusion that decides the final architecture is re-checked at 6B.

| # | Question | Runs |
|---|---|---|
| 0 | Does it train; which LR | 50M tokens at LR 2e-4 / 4e-4 / 8e-4; all §14 tests green first |
| 1 | Do the experts earn their parameters | rex-v2 vs B1 |
| 2 | Does the loop earn its compute | rex-v2 vs A1; also forced-T evaluation of rex-v2 |
| 3 | Dense at matched body | B2 and B3 vs rex-v2, equal tokens and equal FLOPs |
| 4 | Attention residuals | B0 vs B3 |
| 5 | Cross-pass machinery | four readers (default) vs one reader and gate shared by Q, K, V, M per layer vs γ fixed at 0 |
| 6 | Optimiser | Muon vs AdamW |
| 7 | MoE shape and balancing | 14 experts / top-4 / hidden 288 (body 197,661,936); `balance: bias`; `lb_load_ema: 0` |
| 8 | Attention | `qk_mode: l2clip`; 12 x 64 query heads with 4 KV heads (+83,232 parameters); window 256; `n_kv_heads: 6` |
| 9 | Long context (needs stages 2-3, 1.2B tokens) | curriculum vs stage 1 only, both evaluated at 65k and 262k; `global_pos: rope`; `log_scale_alpha: 0` |
| 10 | Variable passes | `pass_sampling: uniform`, evaluated at T = 1, 2, 3 |

---

## 13. Repository layout

```
rex-v2/
  docs/DESIGN.md
  configs/            rex-v2-base.yaml, a1.yaml, b0.yaml, b1.yaml, b2.yaml, b3.yaml, smoke.yaml
                      # smoke.yaml = base model; train.total_tokens: 5.0e7 (190 steps); train.warmup_steps: 20;
                      # train.curriculum: [{until: 1.00, max_len: 8192, long_doc_frac: 0.15}]
  rex/
    config.py         dataclasses, validation (e.g. stream requires n_passes == 1)
    model/
      norm.py         nrm, RMSNorm
      attnres.py      DepthMix
      loop.py         GateMix
      rope.py
      attention.py    projections, QK treatment, length scaling, output gate
      attn_backends.py  reference and flash (varlen) kernels
      moe.py          router, sorted dispatch, dense_ref, auxiliary statistics
      ffn.py          dense SwiGLU
      layer.py        Layer_l
      model.py        forward schedule, checkpoint segments, inference caches
    train/
      losses.py       chunked head, auxiliary losses, load buffers
      optim.py        parameter groups, AdamW, Muon
      schedule.py     LR and curriculum
      trainer.py      micro-batch accumulation, logging, checkpoints
    data/
      tokenizer.py    pinned Inkling tokenizer
      prep.py         shard + document index
      loader.py       token-budget batches of whole documents and crops
    eval/
      metrics.py      §10 metrics
      probes.py       forced-T, message ablation, truncation, repeat probe
  scripts/            train.py, eval.py, prep_data.py, count_params.py, bench.py
  tests/              one file per group in §14
```

---

## 14. Acceptance tests

Tolerances: 1e-5 relative in float64 on the reference backend unless stated. "Tiny" = d 64, 8 layers in blocks of 2 (core 2..5), full-attention layers [1, 3, 5, 7], 2 query / 1 KV heads x 32, 6 experts top-2 hidden 16, rank 4, window 8, vocabulary 512 with 500 unpadded ids.

| # | Test | Pass condition |
|---|---|---|
| T1 | Parameter count | base config: body = 197,855,808 and embedding = 115,789,824, each row of §3 individually; A1, B0-B3 match §11; `count_params.py` agrees with the instantiated model |
| T2 | Initial loss | on random ids, cross-entropy `L_CE` within [12.1, 12.6] for the base config |
| T3 | Loop identity at init | with default init, logits for T = 1, 2, 3 are equal, and V2[t], V3[t] are equal for all t |
| T4 | Loop is live | after setting F, G to random values, logits for T = 1 and T = 3 differ; with γ forced to 0 and κ to 1 they are equal again |
| T5 | DepthMix | with q = 0 the weights are exactly 1/n; weights sum to 1; output is unchanged when any source is multiplied by a positive scalar |
| T6 | Block-scale invariance | tiny model at default init, float64, with `rms_eps: 1e-12` for this test only: multiplying all `W_O` and expert `W_out` of one block by 3.0 leaves logits unchanged within 1e-8, for a prelude, a core and the coda block. No weight rescaling in the test. The base config at default eps is not asserted (see §5.3) |
| T7 | Source lists | a hook records the sources of every reader in every pass; they equal the table in §5.3; reader count is 56 for the base config |
| T8 | Causality | changing token j changes no logit at positions < j, for every pass |
| T9 | Window | in a single windowed attention module, the output at i is unchanged by tokens at positions ≤ i − window |
| T10 | Sequence isolation | in a packed batch, changing any token of sequence A changes nothing in sequence B; packed result equals running each sequence alone |
| T11 | Kernel equivalence (GPU) | flash and reference backends agree within 2e-2 absolute in bf16 on windowed and full layers, with GQA, on packed sequences of unequal length including one longer than the window |
| T12 | RoPE | attention logits depend only on position differences (shifting all positions by 100,000 leaves a windowed layer's output unchanged within 1e-3 in float32); tables built in float64 |
| T13 | Router | exactly 8 distinct experts per token; gate weights sum to 8.0; no gradient reaches unselected experts |
| T14 | Dispatch | `sorted` equals `dense_ref` in outputs and gradients |
| T15 | Auxiliary losses | with ρ = 0: `L_LB` = 1.0 for uniform load and uniform probabilities, and E/k = 3.5 when every token puts all probability mass evenly on the same 8 experts; statistics for a core layer pool all 3 passes |
| T16 | Checkpointing | gradients with `grad_ckpt: layer` (and with `tok_chunk`) equal those with `none`; record measured peak memory per token at 8k, 65k and the largest length the test GPU holds, and update §6.4 |
| T17 | Chunked head | loss and all gradients equal the unchunked computation; padded rows of `Emb` have zero gradient |
| T18 | Parameter groups | every parameter is in exactly one group; the no-decay list matches §9 |
| T19 | Gradient reach | after 2 optimiser steps on the tiny model every parameter except padded embedding rows has a non-zero gradient (at step 0 only `P` and `u` may be zero) |
| T20 | Overfit | tiny model reaches cross-entropy `L_CE` < 0.05 on one fixed batch of 4 sequences x 64 tokens within 500 steps, optimising the full loss with auxiliary losses on. The threshold is on `L_CE` alone: the auxiliary term has a floor of about `lb_coef` x (number of MoE layers), 0.08 in the tiny model and 0.24 in the base model |
| T21 | Incremental decoding | token-by-token generation with per-application caches reproduces the full-sequence logits |
| T22 | Baselines | B0 with `residual: stream` runs T8 and T20, and T2 with two bands: `L_CE` within [12.1, 13.0] as is, and within [12.1, 12.6] when each position's logit for its own input token id is excluded from the softmax (§11 explains the difference); config validation rejects stream with `n_passes` > 1 |
| T23 | Long sequence (GPU) | one forward and backward at the stage-3 `max_len` completes within memory on the target card, and the flash backend accepts that length |
| T24 | Curriculum loader | every sequence starts with `bos_id`, never exceeds the stage `max_len`, `eot_id` only at true document ends, token budget per step within one sequence of 262,144 |

---

## 15. Open questions for the user

1. Long-document data. The default pairs FineWeb-Edu with PG-19 books. If the target domain is code, logs or transcripts, the long-context stages should use that instead; the choice changes what 256k training can teach.
2. GPU memory class. 262,144-token training needs about 31 GB (±30%) by the estimate in §6.4, i.e. a 48 GB card; on a 24 GB card the last stage is capped at 131,072.
3. Tied embeddings are assumed. Untying adds 115,789,824 parameters outside the cap and no body compute.

---

## 16. References

- Inkling 2 Mini config: https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json
- Inkling 2 Mini model card and blog: https://huggingface.co/thinkingmachines/Inkling-2-Mini , https://huggingface.co/blog/inkling2mini
- Goodfire, Interpreting Inkling 2 Mini: https://www.goodfire.com/research/interpreting-inkling-2-mini
- Kimi Team, Attention Residuals: https://arxiv.org/abs/2603.15031
- DeepSeek-AI, DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression (user-supplied PDF; §2.3 CSA2, §2.5, §3.1.2, §4.2)
- Abnar et al., Parameters vs FLOPs: https://arxiv.org/abs/2501.12370
- Csordás et al., MoEUT: https://arxiv.org/abs/2405.16039
- Liu et al., MobileLLM: https://arxiv.org/abs/2402.14905
- Geiping et al., recurrent depth: https://arxiv.org/abs/2502.05171
- Muennighoff et al., OLMoE: https://arxiv.org/abs/2409.02060
- Fedus et al., Switch Transformers: https://arxiv.org/abs/2101.03961
- Krajewski et al., fine-grained MoE scaling: https://arxiv.org/abs/2402.07871
- Qiu et al., global-batch load balancing: https://arxiv.org/abs/2501.11873
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- Liu et al., Muon is Scalable: https://arxiv.org/abs/2502.16982
- Nakanishi, Scalable-Softmax: https://arxiv.org/abs/2501.19399
- Kazemnejad et al., positional encoding and length generalisation: https://arxiv.org/abs/2305.19466
- Bhojanapalli et al., low-rank bottleneck in multi-head attention: https://arxiv.org/abs/2002.07028
- Xiao et al., attention sinks: https://arxiv.org/abs/2309.17453
- Hoffmann et al., compute-optimal training: https://arxiv.org/abs/2203.15556
- Rae et al., PG-19: https://arxiv.org/abs/1911.05507
- flash-attention: https://github.com/Dao-AILab/flash-attention (README; `csrc/flash_attn/flash_api.cpp`; `csrc/flash_attn/src/flash_fwd_kernel.h`; `flash_attn/flash_attn_interface.py`)
- PyTorch SDPA: https://github.com/pytorch/pytorch (`torch/nn/functional.py` docstring; `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`)
- nanochat: https://github.com/karpathy/nanochat
