# The 14 Dimensions → 14 Overprovisionings → Techniques Map

## The 7 per-token dimensions (86% of signal)

| PC | Variance | What it measures | Physical meaning | Overprovisioning it reveals | Fix |
|---|---|---|---|---|---|
| 1 | 29.8% | layer_7, layer_1, layer_5 | **Layer dynamics** — how fast the hidden state evolves across depth | Depth uniformity: 30 layers for every token | Adaptive depth / early exit |
| 2 | 21.0% | layer_9, hnorm_0, cluster_0 | **State location** — where in the manifold the hidden state sits | Width uniformity: 2560 dims regardless of position in manifold | Adaptive width / bottleneck |
| 3 | 14.7% | content_conf, logit_gap, sup_1 | **Distribution sharpness** — how peaked the prediction is | Vocab projection: 128K logits when only ~10 tokens are plausible | Sparse vocab / top-K projection |
| 4 | 7.8% | treuse_2, top10_cov, sup_1 | **Lexical predictability** — whether we've seen this pattern before | KV cache: full cache when patterns repeat | Cache dedup / pattern compression |
| 5 | 4.9% | cluster_1, layer_5, vel_0 | **Cluster stability** — whether the state is in a familiar region | Attention breadth: full N² when state is in a known cluster | Sparse attention / H2O |
| 6 | 4.4% | cluster_1, sup_0, mom_0 | **Superposition clarity** — how tightly grouped candidates are | Activation precision: full precision when candidates are trivial | Adaptive precision |
| 7 | 3.6% | mom_0, treuse_2, logit_gap | **Distribution shape** — skewness of the probability cloud | Softmax scope: 128K softmax when shape is trivial | Truncated / sparse softmax |

## The 7 cross-token dimensions (14% of signal)

| PC | Variance | What it measures | Physical meaning | Overprovisioning | Fix |
|---|---|---|---|---|---|
| 8 | 3.3% | sup_0 (embedding spread) | **Candidate diversity** — semantic spread of plausible tokens | Embedding table: 128K×2560 when only clusters are active | Embedding factorization |
| 9 | 3.0% | agreement_count | **Head consensus** — whether prediction heads agree | Head redundancy: 4 heads when they agree | Adaptive head count |
| 10 | 2.2% | treuse, nbr | **Temporal locality** — how similar to recent states | Temporal redundancy: full compute when state barely changed | Delta computation / skip-if-similar |
| 11 | 1.9% | nbr, vel (opposite signs) | **Stability tension** — spatial proximity vs temporal change | Per-position uniformity: same compute in stable vs turbulent | Position-adaptive compute |
| 12 | 1.4% | logit_gap, mom, vel | **Decision clarity** — how clear-cut the choice is | Verification overhead: verify when decision is obvious | Skip verification / gate |
| 13 | 0.7% | rg_2, hnorm, layer_9 | **Scale coherence** — same info at different scales | Scale uniformity: same arch at all resolutions | Multi-resolution / progressive |
| 14 | 0.7% | layer_5, cluster_1, rg_2 | **Depth-width coupling** — early convergence + cluster | Independent width/depth: should co-adapt but don't | Coupled width-depth allocation |

---

## Every known technique, mapped to dimensions

### FREE (code changes only, no retraining, no overhead)

| Technique | Dims addressed | CPU | GPU | Expected gain | How |
|---|---|---|---|---|---|
| **Skip weak MLP layers** | 1 (depth) | ✓ | ✓ | 15-30% | Comment out MLPs where spin-glass frustration ≈ random |
| **Sparse vocab top-K** | 3, 7 (sharpness, softmax) | ✓ | ✓ | 5-15% | Only compute logits for top-1K common tokens when conf is high |
| **KV cache eviction (H2O)** | 4, 5 (lexical, attention) | ✓ | ✓ | 10-20% memory | Evict low-attention KV entries, keep heavy-hitters |
| **Sliding window attention** | 5 (attention breadth) | ✓ | ✓ | 10-30% | Attend to last N positions, not full context |
| **Skip verification (confidence gate)** | 12 (decision clarity) | ✓ | ✓ | 5-10% | If head-0 softmax peak > τ, accept without backbone verify |

### CHEAP (hours of work, maybe light training on cached data)

| Technique | Dims addressed | CPU | GPU | Expected gain | How |
|---|---|---|---|---|---|
| **Bottleneck heads (Hydra rank-64)** | 2, 8 (width, embedding) | ✓ | ✓ | 90% head memory, 1.1× speed | Already trained. Deploy rank-64 instead of full-rank |
| **Unified gate (our 17-feature MLP)** | 1,3,12 (depth, vocab, verify) | ✓ | ✓ | 1.1-1.2× | Already trained. Routes easy/hard per token |
| **Delta hidden skip** | 10 (temporal locality) | ✓ | ✓ | 5-10% | If ||h_t - h_{t-1}|| < ε, reuse previous prediction |
| **Head pruning by agreement** | 9 (head consensus) | ✓ | ✓ | 25% head cost | When heads agree, run only head-0 next step |

### MEDIUM (days of work, retraining required)

| Technique | Dims addressed | CPU | GPU | Expected gain | How |
|---|---|---|---|---|---|
| **CALM early-exit** | 1 (depth) | okay | ✓✓ | 1.3-2× GPU | Train per-layer classifiers, exit when confident |
| **Matryoshka nested dimension** | 2, 6 (width, precision) | ✓ | ✓ | adaptive width | Train at multiple ranks, deploy at lowest-sufficient |
| **Layer-specific width (fat trunk thin branches)** | 1, 2, 14 (depth, width, coupling) | needs retrain | needs retrain | 1.5-2× | Narrow late layers to 64-256 dims |
| **Speculative decoding (EAGLE)** | 12 (verify) | ✗ | ✓✓ | 2-3× GPU | Separate draft model, batch verify |
| **Progressive token generation** | 13 (scale) | ✓ | ✓ | uncertain | Generate at coarse resolution first, refine |

### EXPENSIVE (weeks/months, significant compute)

| Technique | Dims addressed | CPU | GPU | Expected gain | How |
|---|---|---|---|---|---|
| **Manifold-shaped architecture** | ALL 14 | needs training | needs training | 3-10× | Architecture width/depth follows measured geometry |
| **Mixture of Experts (MoE)** | 1, 2, 5 (depth, width, attention) | ✓ | ✓✓ | 2-4× | Route tokens to specialized sub-networks |
| **Retrieval-augmented generation** | 4, 5 (lexical, attention) | ✓ | ✓ | variable | Replace attention over long context with retrieval |
| **Full architectural redesign (SSM/Mamba)** | 5, 10, 11 (attention, temporal, stability) | ✓✓ | ✓ | 2-5× | Replace attention with linear-time state space |

---

## MULTI-DIMENSIONAL techniques (address 2+ dims simultaneously)

These are the high-value targets:

| Technique | Dims | Why it's multi-dimensional |
|---|---|---|
| **Skip weak MLP layers** | 1 + 2 (depth + width) | Removing a layer removes BOTH its depth cost AND its width computation |
| **Unified gate** | 1 + 3 + 12 (depth + vocab + verify) | One signal decides depth, vocab scope, and verification — all from same 7-dim manifold read |
| **Fat trunk / thin branches** | 1 + 2 + 14 (depth + width + coupling) | Co-adapts width and depth, addresses their coupling directly |
| **Matryoshka + gate** | 2 + 3 + 6 (width + vocab + precision) | Adaptive rank determines both embedding width and effective vocab precision |
| **MoE + early exit** | 1 + 2 + 5 (depth + width + attention) | Expert selection is width-adaptive; early exit is depth-adaptive |
| **Bottleneck heads + sparse vocab** | 2 + 3 + 8 (width + vocab + embedding) | Bottleneck narrows the prediction space, sparse vocab narrows the output space |

---

## The Hydra bottleneck insight (what you flagged)

Rank-64 on 2560 = 97% accuracy at 2.5% parameters.

Why this is big: it's NOT just compression. It's a **measurement of the decision manifold's rank at the output layer.** Only 64 directions in the 2560-dim space carry next-token information. The other 2496 are bulk.

And it was FASTER (less memory bandwidth) despite being an ADDITIONAL computation. Compression = speedup. This inverts the normal engineering assumption that "more processing = slower."

The pattern to replicate: anywhere the manifold is thin, compressing to it doesn't just save memory — it saves TIME because you move less data. Apply this to:
- Attention (project K,V to thin manifold before attending)
- MLP layers (bottleneck the FFN intermediate dimension)
- The vocab projection (project h to manifold, THEN to vocab)

Each of these is the Hydra insight applied to a different overprovisioning.

---

## The 66% holographic bulk

We measured: participation ratio of raw hidden states = 85 out of 2560 ambient dims = 3.3%. Meaning 96.7% of the hidden state's LINEAR capacity is unused. The nonlinear manifold is even thinner (7-14 dims out of 2560).

"Redundant" doesn't mean "removable" — the bulk is needed for the COMPUTATION that produces the boundary information. But it means the OUTPUT of each layer can be COMPRESSED before passing to the next layer. The layer doesn't need to communicate in 2560 dims if only 64 carry information.

This is exactly what the thin-branches architecture does: compress the inter-layer communication to match the information content.

---

## Physics inspirations still untapped

**From electron cloud physics:**
- **Born-Oppenheimer separation**: Separate fast variables (within-token dynamics) from slow variables (cross-token context). Compute fast variables only when slow variables change. Analog: if the context hasn't changed much (low velocity), reuse the previous token's computation.
- **Perturbation theory**: Instead of computing the full wavefunction each step, compute the CHANGE from the previous step. For stable regions: δh = small perturbation, not full forward pass.

**From holographic principle:**
- **Holographic screen**: The boundary doesn't just contain the information — it contains it in a SPECIFIC encoding. The lm_head is the wrong decoder for the boundary; a learned 64-dim decoder (Hydra) is the right one.
- **Bulk reconstruction**: You CAN reconstruct the bulk from the boundary (AdS/CFT tells us so). This means: if you have the 14-dim boundary state, you can reconstruct the full 2560-dim state IF NEEDED, without running the backbone. The reconstruction is cheaper than the forward pass.

**From quantum computing:**
- **Amplitude amplification**: Don't search the full 128K vocab uniformly. Amplify the likely tokens, suppress the unlikely ones. This is sparse softmax with physics justification.
- **Quantum error correction**: The 14-dim manifold is the "logical" subspace. The other 2546 dims are "syndrome" qubits. You can detect errors (degenerate output) by checking syndrome without full decoding.
