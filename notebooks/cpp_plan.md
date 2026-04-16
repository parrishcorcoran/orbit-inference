# unified-gate C++ Integration Plan (v0.1 — 2026-04-15)

The Python reference reproduces the gate frontier within ±0.001. This doc lays out the C++ integration target for the **systems paper** follow-up. Wall-clock speedup is the goal; Python feature extraction at ~1 sec/seq is not deployable.

---

## Deliverable

A C++ binary `llama-medusa-gated` (fork of bitnet.cpp's `llama-medusa`) that:

1. Runs Medusa speculative decoding as today
2. At each step, extracts the 20 gate features from in-memory tensors
3. Scores with `gate_k20.pt` (26 KB MLP)
4. If `score ≥ τ` (calibrated threshold for target λ): **accept the Medusa draft without running the verifier**
5. Otherwise: run verifier as today

Measured claims the systems paper needs:
- Tokens/sec improvement vs baseline `llama-medusa` at matched fidelity
- Tokens/joule improvement (power-meter reading)
- Fidelity holding at the target λ on long generations (1024+ tokens)

---

## Architecture

Three new modules under `bitnet.cpp/3rdparty/llama.cpp/examples/medusa-gated/`:

### 1. `gate_artifact.h` / `gate_artifact.cpp`
Loads `gate_k20.pt` at startup. The artifact is a Python pickle — two options:
- **A (pragmatic)**: Export weights via a Python script to a flat binary (`gate_k20.bin`: feature_indices[20], mu[20], sd[20], W0[64x20], b0[64], W1[64x64], b1[64], W2[1x64], b2[1], τ[4]). C++ reads 26 KB of floats. Fastest.
- **B (cleaner)**: Use libtorch's TorchScript. Load directly. Heavier runtime dependency.

Recommend **A**. Python script `export_gate_to_bin.py` lives in the unified-gate repo.

### 2. `features.h` / `features.cpp`
Implements the 20 feature extractors over `ggml_tensor`s and the llama.cpp token vocabulary. Per-step cost is the concern — targets:

| Feature group | per-step cost | notes |
|---|---|---|
| `content_conf`, `logit_gap`, `top10_cov` | free | reuse head-0 softmax values |
| `agreement_count` | few int compares | reuse head-{0..3} argmaxes |
| `cluster_mindist`, `cluster_entropy` | O(K·H) = 32·2560 = 82k mul | precompute centers at load |
| `layer_5/7/9` (Ryu-Takayanagi) | 3× cos sim O(H) | **needs mid-layer hidden tap** |
| `nbr_0` (neighborhood H2O) | O(W·H), W=recent window | circular buffer of last-W hidden states |
| `phase_svd`, `rg_div`, `moments`, `norm`, `velocity` | O(H) each | trivial |
| `treuse_2` (token-reuse rank) | O(W) over recent tokens | circular token buffer |
| `sup_0/1` (superposition) | O(K·H) = 32·2560 | top-K token-embedding spread |

Estimated total per-step cost: **~300k float ops**, 300 μs on modern CPU. Compared to ~30 ms for a BitNet forward pass, gate overhead is ~1%.

### 3. `gate_mlp.h` / `gate_mlp.cpp`
Applies the 64×64 MLP. Sixty-four ReLU × 20 + 64×64 + 64 = ~5k mul-adds. Negligible.

### 4. Main loop surgery
In `llama-medusa` main loop at the point where Medusa drafts are verified against the backbone: gate's `score ≥ τ_λ` decides skip/verify. Simplest integration point: skip the verifier call when gated-accept.

---

## Mid-layer hidden state extraction

The three `layer_5/7/9` features require hidden-state taps at intermediate layers. Today's `llama-medusa` only exposes the final hidden state. Two approaches:

**A. cb_eval hook**: `llama.cpp` has a `cb_eval` callback that fires during graph execution. The `hidden-dump` binary already uses this to capture `result_norm`. Extend to capture `l_out-5`, `l_out-15` in addition.

**B. Forward modification**: directly modify `llama_decode_internal` to cache layer-5/15 outputs. Faster but invasive.

Recommend **A** for the integration pass; profile and upgrade to B only if cb_eval overhead shows in profiles.

---

## Milestones

1. **Week 1**: Export `gate_k20.pt` → `gate_k20.bin` flat format. Write `gate_artifact.cpp` loader. Unit test: loads, inference matches Python on held-out seq 36 within float precision.

2. **Week 2**: Implement features (the 14 fast ones). Skip Ryu-Takayanagi for now (needs cb_eval work). Run against cached hidden-dump'd data. Verify: gate score matches Python within ε on seq 36.

3. **Week 3**: cb_eval hooks for mid-layer hidden taps. Add `layer_5/7/9` features. Full feature parity with Python.

4. **Week 4**: Wire into `llama-medusa` main loop. Measure:
   - Tokens/sec vs ungated baseline
   - Fidelity on long generation held-out
   - Tokens/joule with power meter
   - Target: **>1.2× tokens/sec** at λ=0.95 fidelity. (10% skip * small per-step cost - feature overhead)

5. **Week 5**: Regression testing, profile, merge to `unified-gate` repo as `cpp/` subdirectory.

---

## Risks and mitigations

- **Fidelity drift under sequence length**: gate was trained on 2048-token seqs. At 8k generation, feature distributions may shift. Mitigation: run measurement experiment on 8k held-out early.
- **Power-meter accuracy**: CPU power measurement is noisy. Mitigation: run long (>10 min) generations for stable averaging.
- **Cache invalidation**: token-reuse and neighborhood features depend on recent state. If batching changes or beam search is enabled, those features may break. Mitigation: for v1, restrict to single-stream greedy decoding.
- **cb_eval overhead**: may add 1-2 ms per step. Mitigation: profile early; fall back to direct graph modification if needed.

---

## Out of scope for v1

- Batch decoding
- Beam search
- Streaming API (gate is per-step; needs adapter for streaming)
- GPU path (CPU-only for systems paper)
- Non-BitNet models (deferred to v2 cross-model paper)

---

## Open questions for Parrish

- Target platform for systems paper numbers: i9-13900K? Specific Xeon? Record-for-record reproducibility matters.
- Power-meter setup: software (RAPL) or hardware (Kill-A-Watt)? Both are acceptable, pick one and stick with it.
- Fidelity target for the paper: stick with λ=0.95 as the headline or also report λ=0.90 for the "more aggressive" setting?

---

## Files that will need modification

In `bitnet.cpp`:
- `3rdparty/llama.cpp/examples/medusa-gated/CMakeLists.txt` (new)
- `3rdparty/llama.cpp/examples/medusa-gated/*.{h,cpp}` (new, ~600 LOC total)
- `3rdparty/llama.cpp/common/arg.cpp` — add `--gate-path` flag
- CMake top-level to include new example

In `unified-gate`:
- `scripts/export_gate_to_bin.py` (new)
- `cpp/` subdirectory (mirror of C++ code)
- README update with C++ usage

Out-of-scope for the Python repo (keep it focused).

---

## Test plan

- **Unit**: gate inference matches Python bit-exactly on held-out seq 36
- **Integration**: long-generation fidelity ≥λ on 10 seed prompts
- **Perf**: tokens/sec regression suite (baseline `llama-medusa` vs gated)
- **Power**: 10-min sustained generation with RAPL sampling every 100ms

---

## Ready-to-start checklist

When a session picks this up:
1. `cd /home/cpinchington/bitnet.cpp && git status` — make sure tree is clean
2. Check `gate_k20.pt` is still at `/home/cpinchington/unified-gate/gate_k20.pt`
3. Start with milestone 1 (gate_k20.bin export) — ship that small win first, don't try to land all 5 milestones at once
4. Open a WIP commit after milestone 1, let Parrish review before proceeding
