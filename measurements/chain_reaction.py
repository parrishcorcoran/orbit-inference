"""Chain-reaction test: does rank-D compression at layer L preserve the
downstream prediction when passed through the remaining layers?

This is THE critical experiment for ORBIT. It tests whether inter-layer
communication can be compressed from 2560 dims to D dims without
breaking the computation.

Requires: the BitNet model loaded in PyTorch with forward hooks.
Run on Z8 G4 (or any machine with PyTorch + the model weights).

Usage:
    python measurements/chain_reaction.py \
        --model-path models/bitnet-b1.58-2B-4T \
        --tokens-path data/tokens.bin \
        --ranks 16,32,64,128,256,512 \
        --layers 15,20,22,24,26,28
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def hook_compress_layer(model, layer_idx, rank, device="cpu"):
    """Register a forward hook that compresses the output of layer `layer_idx`
    to rank `rank` via a learned projection, then decompresses.

    Uses PCA computed on a calibration set. For learned projections,
    replace with trained encoder/decoder.
    """
    projection = {}  # will hold W matrix after calibration

    def calibration_hook(module, input, output):
        """Collect hidden states for PCA calibration."""
        if not hasattr(hook_compress_layer, '_cal_data'):
            hook_compress_layer._cal_data = []
        # output is typically a tuple; hidden state is first element
        h = output[0] if isinstance(output, tuple) else output
        hook_compress_layer._cal_data.append(h.detach().float().cpu())

    def compression_hook(module, input, output):
        """Compress output to rank D via PCA, then reconstruct."""
        h = output[0] if isinstance(output, tuple) else output
        W = projection['W']  # [D, hidden_dim]
        h_flat = h.view(-1, h.shape[-1]).float()
        mu = projection['mu']
        # Project to D dims and back
        compressed = (h_flat - mu) @ W.T  # [N, D]
        reconstructed = compressed @ W + mu  # [N, hidden_dim]
        h_recon = reconstructed.view(h.shape).to(h.dtype)
        if isinstance(output, tuple):
            return (h_recon,) + output[1:]
        return h_recon

    return calibration_hook, compression_hook, projection


def run_chain_reaction(model, tokenizer, tokens, layers, ranks,
                       n_cal=2048, n_test=2048, device="cpu"):
    """Run the chain-reaction experiment.

    For each (layer, rank) pair:
    1. Calibrate PCA on n_cal tokens
    2. Register compression hook at that layer
    3. Run n_test tokens through the model
    4. Compare predictions to uncompressed baseline
    """
    model.eval()

    # Baseline: uncompressed predictions
    print("Computing baseline predictions...", flush=True)
    with torch.no_grad():
        input_ids = torch.tensor([tokens[:n_test]], device=device)
        baseline_logits = model(input_ids).logits[0]
        baseline_preds = baseline_logits.argmax(-1).cpu().numpy()

    print(f"\n{'Layer':>6} {'Rank':>6} {'Match%':>8} {'Top5Match%':>12}", flush=True)
    print("-" * 40, flush=True)

    for layer_idx in layers:
        # Get the actual layer module
        # BitNet architecture: model.model.layers[layer_idx]
        try:
            target = model.model.layers[layer_idx]
        except (AttributeError, IndexError):
            print(f"  Layer {layer_idx}: not found, skipping", flush=True)
            continue

        for rank in ranks:
            # Step 1: Calibrate PCA
            cal_hook, comp_hook, proj = hook_compress_layer(model, layer_idx, rank)
            hook_compress_layer._cal_data = []
            handle = target.register_forward_hook(cal_hook)

            with torch.no_grad():
                cal_ids = torch.tensor([tokens[:n_cal]], device=device)
                model(cal_ids)

            handle.remove()

            # Compute PCA from calibration data
            cal_h = torch.cat(hook_compress_layer._cal_data, dim=1)[0]  # [n_cal, hidden]
            mu = cal_h.mean(0)
            _, S, Vt = torch.linalg.svd(cal_h - mu, full_matrices=False)
            proj['W'] = Vt[:rank].to(device)
            proj['mu'] = mu.to(device)
            del hook_compress_layer._cal_data

            # Step 2: Run with compression hook
            handle = target.register_forward_hook(comp_hook)
            with torch.no_grad():
                test_ids = torch.tensor([tokens[:n_test]], device=device)
                comp_logits = model(test_ids).logits[0]
                comp_preds = comp_logits.argmax(-1).cpu().numpy()
            handle.remove()

            # Step 3: Compare
            match = (comp_preds == baseline_preds).mean() * 100
            # Top-5 match: is the baseline prediction in the compressed top-5?
            top5 = comp_logits.topk(5, dim=-1).indices.cpu().numpy()
            top5_match = np.array([baseline_preds[i] in top5[i] for i in range(len(baseline_preds))]).mean() * 100

            print(f"{layer_idx:>6} {rank:>6} {match:>7.1f}% {top5_match:>11.1f}%", flush=True)

    print("\n100% match = compression is lossless for that layer at that rank")
    print("The row where match drops sharply = the minimum inter-layer bandwidth")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--tokens-path", default="data/tokens.bin")
    p.add_argument("--ranks", default="16,32,64,128,256,512")
    p.add_argument("--layers", default="15,20,22,24,26,28")
    p.add_argument("--n-cal", type=int, default=2048)
    p.add_argument("--n-test", type=int, default=2048)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model from {args.model_path}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    tokens = np.memmap(args.tokens_path, dtype=np.uint32, mode="r")[:args.n_test]
    tokens = tokens.astype(np.int64).tolist()

    layers = [int(x) for x in args.layers.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]

    run_chain_reaction(model, tokenizer, tokens, layers, ranks,
                       n_cal=args.n_cal, n_test=args.n_test)


if __name__ == "__main__":
    main()
