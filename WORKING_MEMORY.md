# ORBIT Working Memory — DO NOT REPEAT THESE EXPERIMENTS

## PROVEN (do not re-test)
- Hydra rank-64 on BitNet layer 29 output: 97% accuracy (held-out, 98K training tokens)
- Manifold is 7-dim per-seq, model-agnostic (TwoNN, BitNet + Llama)
- Orbital confinement: easy PR 6→36→16, hard 7→23→41
- Gate works at 95% fidelity (5-seed replicated)
- Weight SVD gives 33× wall-clock speedup (but 0% quality — WRONG BASIS)
- GPU works: 89 GB VRAM, ROCm 7.13 nightly, Radeon 8060S gfx1151

## FAILED — DO NOT RETRY
- PCA inter-layer compression: 99.1% was OVERFIT (train=test). Real generalization: 17-18%
- PCA doesn't generalize regardless of calibration size (2000 tokens → still 17%)
- Weight SVD factorization: produces garbage (0% match) — weight directions ≠ data directions
- Activation-aware SVD on weights: still 0% match — same KV cache mismatch
- Small-data KL distillation (225 tokens, 30 epochs): 7.3% — massively undertrained
- Single-layer learned bottleneck with 2000 tokens: 34% — needs much more data
- PCA hooks during generation: KV cache mismatch corrupts output

## ROOT CAUSE OF FAILURES
1. PCA finds variance directions, NOT prediction directions. Curved manifold.
2. Weight factorization breaks because weights encode knowledge that's NOT low-rank.
3. Inter-layer compression compounds errors: 1% error/layer × 24 layers = catastrophic.
4. KV cache: compressed hidden states produce different K,V than uncompressed → mismatch in generation.
5. All training attempts used INSUFFICIENT DATA (<2K tokens for 896-dim projection).

## WHAT MUST BE DONE (Gemini's approach, PROPERLY)
1. Load Qwen 0.5B as BOTH teacher (frozen) and student (with bottlenecks)
2. Insert differentiable A·diag(s)·B bottleneck INSIDE each layer (before Q,K,V computation)
3. Generate 10K+ training tokens from teacher
4. Train student via KL(teacher || student) for 200+ epochs
5. Add L1 on s vectors after 50 warmup epochs
6. The end-to-end backprop teaches layers to COORDINATE — layer 15 leaves info layer 16 needs
7. Must use INSIDE-layer bottleneck (not hooks) to avoid KV mismatch

## KEY INSIGHT FROM USER
"Why aren't we using the curvature data? The manifold is curved."
- PCA is flat. Hydra is curved. That's why Hydra gets 97% and PCA gets 17%.
- The bottleneck must be LEARNED (nonlinear) not projected (linear).
- We already have curvature from Hydra on BitNet. Use it as initialization/validation.
- Don't keep re-proving what Hydra already proved.

## LATEST RESULTS (Gemini approach, first attempt)
- 8K tokens, 200 epochs, rank-128 inside-layer bottleneck on Qwen 0.5B
- Peak generalization: 51.3% at epoch 81 (before L1)
- KL dropped 17→0.5 (training loss improved but generalization plateaued)
- Generation: 3% match (still garbage output)
- CONCLUSION: 8K tokens insufficient for 5.5M params. Need 100K+ (like Hydra's 98K).
- The approach IS working — 51% > 17% (PCA) > 7% (earlier KL attempt)
- Running 100K tokens next.

## WHAT NOT TO DO
- Don't re-test PCA generalization (proven: fails)
- Don't re-test weight SVD (proven: wrong basis)
- Don't train on <5K tokens (proven: insufficient)
- Don't use output hooks for generation-mode compression (proven: KV mismatch)
- Don't test batch-mode chain reaction as if it proves generation works (proven: overfit)
- Don't train on <50K tokens for 5M+ params (proven: overfits at 8K)

## RESULT: 33K tokens, 300 epochs (Gemini approach)
- Peak generalization: 67.6% at epoch 176 (batch-mode on unseen prompts)
- Trend: 8K→51%, 33K→68%. More data consistently helps.
- Generation still diverges (~3% exact match)
- KL: 13.4 → 0.19 (training converges, generalization plateaus)
- All 128 dims survived L1 (too weak or needed)
- CONCLUSION: approach works, needs 100K-300K tokens for 90%+ target

## OUTPUT-ONLY BOTTLENECK (no chain reaction)
Qwen 0.5B (H=896, 27K train tokens, 100 epochs):
  rank-32: 47.3%, rank-64: 51.2%, rank-128: 57.6%, rank-256: 55.7% (overfits)
  Peak at rank-128. Drops at rank-256 = overfitting (too many params for 27K tokens).

Qwen 1.5B (H=1536, 2.2K train tokens, 100 epochs):
  rank-64: 32.9%, rank-512: 36.3%
  Massively undertrained. 2.2K tokens for 1.6M param bottleneck.

BitNet 2B Hydra (H=2560, 98K train tokens, 1000 steps):
  rank-64: 97%  ← THE GOLD STANDARD

## THE PATTERN
Every result improves with more training data:
  PCA (0 training): 17%
  KL 8K tokens: 51%
  KL 33K tokens: 68%
  Hydra 98K tokens: 97%

The approach WORKS. It's a data scaling problem, not an architecture problem.
Estimated need: 100K-300K diverse tokens for 90%+ on Qwen.

## WHAT TO DO NEXT
1. Generate 100K+ training tokens from Qwen (takes ~30 min on GPU)
2. Train output-only bottleneck with proper data scale
3. If output reaches 90%+, add inter-layer bottlenecks incrementally
4. OR: go back to BitNet where 97% is already proven, build full ORBIT there

## DEFINITIVE OUTPUT-ONLY TEST (C4 real text, 27K tokens, 200 epochs)
Qwen 0.5B (H=896):
  rank-32: 33.8%, rank-64: 37.1%, rank-128: 40.6%, rank-256: 42.6%
  
CONCLUSION: Qwen 0.5B output is NOT compressible like BitNet.
- BitNet Hydra rank-64 = 97% (on 98K model-specific tokens)
- Qwen 0.5B rank-256 = 42.6% (on 27K C4 tokens)

Likely causes:
1. H=896 is too small — less redundancy than H=2560
2. BitNet ternary weights → cleaner manifold (spin-glass ground state)
3. C4 real text is harder than model-generated text

NEXT: Test on larger model (H=1536 or H=2048+) OR go back to BitNet.
The physics (7-dim manifold, orbital confinement) is real. The 
compression just needs a model with enough redundancy to exploit.
