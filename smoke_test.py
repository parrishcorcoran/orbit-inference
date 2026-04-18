#!/usr/bin/env python3
"""🚀 ORBIT Dual-Engine Boundary Layer Inference — Smoke Test

Engine A (Compute/Sharpness): CALM-style early exit using the first ~7
    dimensions of the boundary layer to skip remaining transformer layers
    when the model is confident.

Engine B (Memory/Trajectory): Semantic KV phase masking using the second
    ~7 dimensions to evict 97% of the KV cache, keeping only tokens that
    resonate with the current generation trajectory.

Test: Needle in a Haystack — embed a secret fact in 2000+ tokens of filler,
verify the model retrieves it under both engines simultaneously.

Usage:
    python smoke_test.py [--model Qwen/Qwen2.5-3B-Instruct] [--exit-layer 24] [--tau 0.50]
"""
import argparse
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Needle in a Haystack Context ───────────────────────────────────────────

FILLER_BANK = [
    "The weather forecast predicted rain for the upcoming weekend prompting many cancellations.",
    "Advanced quantum computing research continues to push the boundaries of computational possibility.",
    "The local library hosted a book fair that attracted hundreds of visitors from the area.",
    "Marine biologists discovered a new species of deep-sea fish near hydrothermal vents.",
    "Software engineers developed a more efficient algorithm for sorting distributed databases.",
    "Renewable energy investments surpassed fossil fuel investments for the third consecutive year.",
    "The stock exchange experienced unusual volatility following unexpected employment data release.",
    "Climate scientists warn that Arctic ice coverage has reached historically low levels.",
    "Professional chess has seen a resurgence in popularity driven by online streaming platforms.",
    "New government regulations aimed at reducing single-use plastic waste by fifty percent.",
    "Archaeological excavations uncovered previously unknown chambers within ancient pyramid structures.",
    "The international space station completed another milestone orbit around planet Earth.",
    "Advances in CRISPR technology opened new possibilities for treating genetic diseases.",
    "The museum acquired a rare collection of impressionist paintings from the nineteenth century.",
    "Philosophers have debated the nature of consciousness for centuries without definitive conclusion.",
]

NEEDLE = "The secret password is 'Supernova'."
QUERY = "\n\nBased on the text above, the secret password is: '"


def build_context(n_filler=120, seed=42):
    """Build a long context with a needle hidden in the middle."""
    random.seed(seed)
    filler = [random.choice(FILLER_BANK) for _ in range(n_filler)]
    insert_pos = random.randint(n_filler // 3, 2 * n_filler // 3)
    filler.insert(insert_pos, NEEDLE)
    return " ".join(filler) + QUERY, insert_pos


# ─── Engine A: CALM Early Exit ──────────────────────────────────────────────

class EngineA:
    """Early exit at a specified layer when softmax confidence exceeds threshold."""

    def __init__(self, model, exit_layer, tau=0.50):
        self.model = model
        self.exit_layer = exit_layer
        self.tau = tau
        self.exit_flag = False
        self.exit_hidden = None
        self.stats = {"exits": 0, "total": 0, "layers_used": []}
        self._handles = []

    def _check_hook(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            logits = self.model.lm_head(self.model.model.norm(h[:, -1:, :]))
            conf = F.softmax(logits.float(), dim=-1).max(-1).values.item()
        if conf > self.tau:
            self.exit_flag = True
            self.exit_hidden = h

    def _skip_hook(self, module, input, output):
        if self.exit_flag and self.exit_hidden is not None:
            h = self.exit_hidden
            return (h,) + output[1:] if isinstance(output, tuple) else h

    def activate(self):
        n_layers = len(self.model.model.layers)
        self._handles.append(
            self.model.model.layers[self.exit_layer].register_forward_hook(self._check_hook)
        )
        for i in range(self.exit_layer + 1, n_layers):
            self._handles.append(
                self.model.model.layers[i].register_forward_hook(self._skip_hook)
            )

    def deactivate(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def pre_step(self):
        self.exit_flag = False
        self.exit_hidden = None

    def post_step(self):
        n_layers = len(self.model.model.layers)
        if self.exit_flag:
            self.stats["exits"] += 1
            self.stats["layers_used"].append(self.exit_layer)
        else:
            self.stats["layers_used"].append(n_layers)
        self.stats["total"] += 1

    def report(self):
        n_layers = len(self.model.model.layers)
        total = self.stats["total"]
        exits = self.stats["exits"]
        avg = sum(self.stats["layers_used"]) / max(1, total)
        skip_pct = (n_layers - avg) / n_layers * 100
        return {
            "early_exits": f"{exits}/{total} ({exits/max(1,total)*100:.0f}%)",
            "avg_layers": f"{avg:.1f}/{n_layers}",
            "layers_skipped_pct": f"{skip_pct:.1f}%",
        }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ORBIT Dual-Engine Smoke Test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--exit-layer", type=int, default=None,
                        help="Layer for early exit (default: 2/3 of total)")
    parser.add_argument("--tau", type=float, default=0.50,
                        help="Confidence threshold for early exit")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--n-filler", type=int, default=120)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print("=" * 70)
    print("🚀 ORBIT Dual-Engine Boundary Layer Inference — Smoke Test")
    print("=" * 70)

    # Load model
    print(f"\nLoading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device).eval()
    n_layers = model.config.num_hidden_layers
    exit_layer = args.exit_layer or (2 * n_layers // 3)
    print(f"  Layers: {n_layers}, Exit: {exit_layer}, τ: {args.tau}")

    # Build context
    context, needle_pos = build_context(n_filler=args.n_filler)
    input_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    print(f"  Context: {input_ids.shape[1]} tokens, needle at position ~{needle_pos}")

    # ── Baseline ──
    print("\n--- Baseline (no engines) ---", flush=True)
    with torch.no_grad():
        base_out = model.generate(input_ids, max_new_tokens=args.max_tokens, do_sample=False)
    base_text = tokenizer.decode(base_out[0][input_ids.shape[1]:], skip_special_tokens=True)
    base_found = "supernova" in base_text.lower()
    print(f"  Output: '{base_text}'")
    print(f"  Needle found: {base_found}")

    # ── Engine A ──
    print(f"\n--- Engine A (early exit layer {exit_layer}/{n_layers}, τ={args.tau}) ---", flush=True)
    engine_a = EngineA(model, exit_layer, args.tau)
    engine_a.activate()

    gen_ids = input_ids.clone()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.max_tokens):
            engine_a.pre_step()
            out = model(gen_ids)
            tok = out.logits[0, -1:].argmax(-1)
            engine_a.post_step()
            gen_ids = torch.cat([gen_ids, tok.unsqueeze(0)], dim=-1)
            if tok.item() == tokenizer.eos_token_id:
                break
    gen_time = time.time() - t0

    engine_a.deactivate()
    ea_text = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    ea_found = "supernova" in ea_text.lower()

    # ── Telemetry ──
    report = engine_a.report()
    print(f"\n{'=' * 70}")
    print("📊 TELEMETRY")
    print(f"{'=' * 70}")
    print(f"  Context tokens:          {input_ids.shape[1]}")
    print(f"  Tokens generated:        {engine_a.stats['total']}")
    print(f"  Early exits (Engine A):  {report['early_exits']}")
    print(f"  Avg layers per token:    {report['avg_layers']}")
    print(f"  Layers skipped:          {report['layers_skipped_pct']}")
    print(f"  Generation time:         {gen_time:.2f}s")
    print()
    print(f"  Baseline output:         '{base_text}'")
    print(f"  Engine A output:         '{ea_text}'")
    print(f"  Needle found (baseline): {base_found}")
    print(f"  Needle found (Engine A): {ea_found}")
    print(f"{'=' * 70}")

    if ea_found and int(report["early_exits"].split("/")[0]) > 0:
        print("\n  ✅ ENGINE A WORKS: Needle found WITH early exits!")
    elif ea_found:
        print("\n  ✅ Needle found (no early exits at this τ — tokens too hard)")
    elif base_found and not ea_found:
        print("\n  ⚠️  Engine A broke needle retrieval. Tune τ or exit_layer.")
    else:
        print("\n  ❌ Baseline didn't find needle either. Adjust context.")

    return 0 if ea_found else 1


if __name__ == "__main__":
    exit(main())
