# ORBIT Architecture Specification

## Core idea

Replace every layer's weight matrix W with a factorized form:

```
W = A · diag(s) · B
```

Where:
- **B** ∈ R^(r_max × input_dim): projects input to bottleneck space
- **s** ∈ R^(r_max): learnable scaling factors ("orbital gates")
- **A** ∈ R^(output_dim × r_max): projects bottleneck back to output space

Apply L1 penalty on `s` during training. The optimizer naturally drives bulk dimensions to zero. The surviving dimensions ARE the orbital — discovered by training, not hand-specified.

## Why this works (physics justification)

### Measured orbital confinement curve (BitNet 2B)

```
Easy tokens:  PR  6 → 36 → 16  (expands, explores, collapses — orbital forms)
Hard tokens:  PR  7 → 23 → 41  (expands, explores, keeps expanding — superposition)
```

The confinement tells us:
- **Early layers need ~6-12 active dims** (initial representation is simple)
- **Mid layers need ~36-72 active dims** (exploration phase — many competing completions)
- **Late layers need ~16-32 active dims for easy tokens** (orbital formed, decision made)
- **Late layers need ~41-82 active dims for hard tokens** (still deciding)

The L1 penalty on `s` should naturally discover this trajectory — early layers' `s` vectors will have few surviving entries, mid layers more, late layers fewer (for easy tokens) or more (for hard tokens).

### Phase-transition model (from user insight)

The orbital doesn't form by smooth convergence. It forms by **phase transition with rotation**:
1. Layer 15: model commits to a direction (hidden state in a structured subspace)
2. Layers 16-25: state ROTATES within the shrinking subspace to the correct answer
3. Layer 29: orbital fully confined, rotation complete

Velocity INCREASES through depth (rotation is fast). Dimensionality COLLAPSES (subspace shrinks). These aren't contradictory — they're the signature of a confined rotator.

The bottleneck preserves the **subspace**, not the exact position. The rotation happens inside the bottleneck, at the rank the bottleneck allows.

## Implementation: three training phases

### Phase 1: Warm-up (no L1, standard training)
- Replace all linear layers with A·diag(s)·B at r_max = 256
- Initialize s = 1 (full rank)
- Train normally for E/3 epochs
- Let the orbitals find their natural orientation
- **Checkpoint the orientations — this is the warm-up state**

### Phase 2: Annealing (ramp up L1)
- Gradually increase λ_L1 from 0 to λ_max over E/3 epochs
- s values for bulk dimensions decay toward zero
- Monitor per-layer effective rank (count of |s_i| > ε)
- **Expected outcome**: rank trajectory matches the measured confinement curve

### Phase 3: Surgery (prune dead dimensions)
- For each layer, permanently delete dimensions where |s_i| < ε
- Multiply surviving A_pruned · diag(s_surviving) · B_pruned
- Result: each layer has its OWN natural rank
- **Expected outcome**: early layers ~12-dim, mid layers ~36-dim, late layers ~16-dim (easy tokens)

## The input-adaptive version (Dynamic ORBIT)

Static ORBIT: one fixed rank per layer (discovered by L1).
Dynamic ORBIT: rank varies per token per layer.

For Dynamic ORBIT, replace the static `s` with a **gate-predicted** `s`:

```
s_token = gate(h_early) → [r_max] soft mask
```

The gate (our 7-dim manifold reader) outputs a per-dimension mask for each token. Easy tokens get narrow masks (few active dims). Hard tokens get wide masks.

Training: Gumbel-Softmax on the gate output for differentiability. The gate learns to predict which dimensions each token needs — which is equivalent to predicting the orbital trajectory.

## Relationship to existing work

| Technique | What it does | How ORBIT differs |
|---|---|---|
| LoRA | Low-rank adaptation of weights (for fine-tuning) | ORBIT discovers the rank FROM TRAINING, doesn't assume it |
| Matryoshka | Nested-rank embeddings | ORBIT applies to backbone layers, not just embeddings |
| Mixture of Experts | Route tokens to specialized sub-networks | ORBIT routes to rank (bandwidth), not to expert (content) |
| CALM early-exit | Skip layers for easy tokens | ORBIT narrows layers, doesn't skip them |
| Pruning | Remove weights post-training | ORBIT integrates pruning INTO training via L1 |

## The validation experiment

After training, measure the per-layer effective rank:
- If it matches the measured confinement curve (6→36→16 for easy) → physics and training agree
- If it's different → the L1 discovered a different structure, worth investigating
- If it's uniformly high → compression doesn't work for this model, L1 too weak

The confinement curve is the **ground truth** from our measurements. It tells us what the trained ORBIT model should look like.

## What we can test NOW (before full training)

1. **Factorize one layer**: take layer 29's weights, decompose via SVD, apply L1 on the singular values, measure prediction accuracy vs rank. This simulates the surgery phase for one layer.

2. **Gate-predicted rank**: use our trained gate to predict per-token rank, measure what rank is sufficient for 95% accuracy at each layer.

3. **Fine-tune with bottleneck**: freeze early layers, add A·diag(s)·B to late layers, fine-tune on cached hidden states with L1. See if the confinement curve emerges.

## Compute estimates (post-surgery)

Conservative: each layer settles at ~4× its PR rank
- Early: rank 24-48
- Mid: rank 92-144
- Late (easy): rank 64-128
- Late (hard): rank 164-256

At these ranks, per-layer compute is (rank/2560)² × full_layer_cost for the attention and rank/2560 × full for MLP. Weighted average: ~5-15× reduction. Per token.

Aggressive: each layer settles at ~2× its PR rank
- 60-125× reduction (matching our earlier theoretical model)

The truth is somewhere in between. The experiment determines where.
