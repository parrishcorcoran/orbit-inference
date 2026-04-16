# Running ORBIT on HP Z8 G4 (196 GB RAM)

## Why Z8 G4

The critical experiment (inter-layer compression with live computation) requires:
- Loading BitNet 2B model (~1.2 GB GGUF)
- Caching hidden states at multiple layers (~250 MB per layer × 30 layers = ~7.5 GB)
- Training bottleneck projections with full 128K vocab cross-entropy (~4 GB per batch of logits)
- Running modified forward passes with compressed hidden states

Total peak memory: ~15-20 GB. Your Z8 G4 with 196 GB has plenty of headroom.

## Setup

```bash
# 1. Clone repos
git clone https://github.com/parrishcorcoran/orbit-inference
git clone https://github.com/parrishcorcoran/MedusaBitNet
git clone https://github.com/parrishcorcoran/unified-gate

# 2. Python environment
cd orbit-inference
python -m venv .venv
source .venv/bin/activate
pip install torch numpy transformers huggingface_hub

# 3. Install unified-gate package
pip install -e ../unified-gate

# 4. Download model (if not already present)
# The official Microsoft BitNet GGUF:
mkdir -p ../MedusaBitNet/models/bitnet-b1.58-2B-4T
cd ../MedusaBitNet/models/bitnet-b1.58-2B-4T
# Download from HuggingFace: microsoft/bitnet-b1.58-2B-4T-gguf
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf ggml-model-i2_s.gguf \
    --local-dir .
cd ../../../orbit-inference

# 5. Build bitnet.cpp (for hidden-dump and medusa binaries)
cd ../MedusaBitNet
# Follow bitnet.cpp build instructions:
# cmake -B build && cmake --build build -j$(nproc)
# Binaries needed: build/bin/llama-hidden-dump, build/bin/llama-medusa-gated

# 6. Cache hidden states (if not already done)
# This takes ~30 min for 48 sequences × 5 layers
cd ../MedusaBitNet
for layer in 5 15 28 29; do
    echo "Caching layer $layer..."
    # See cache_hidden_gguf_v2.py for the caching script
done
```

## Run the critical experiment

```bash
cd orbit-inference

# The inter-layer compression test
# Tests: compress h at layer L to rank D, run through remaining layers,
# measure if the prediction survives.
python measurements/inter_layer_compression.py \
    --medusabitnet-root ../MedusaBitNet \
    --layers 20,22,24,26,28 \
    --ranks 16,32,64,128,256,512

# Expected output: a table showing prediction accuracy at each layer × rank
# If accuracy is high at low ranks → ORBIT works
# If accuracy drops sharply → inter-layer bandwidth is wider than expected
```

## Run the orbital confinement measurement

```bash
# Verify the orbital confinement curve
python measurements/orbital_confinement.py \
    --medusabitnet-root ../MedusaBitNet

# Expected: easy tokens PR peaks then drops, hard tokens PR keeps rising
```

## Run the full measurement suite

```bash
# All measurements in sequence
python -m pytest measurements/ -v
```

## Notes

- The Z8 G4's Xeon CPUs are slower per-core than consumer chips but have massive memory bandwidth. The 128K vocab matmul that bottlenecked the Fedora workstation should run much faster with 196 GB of DDR4 bandwidth.
- All experiments are CPU-only. No GPU required (BitNet uses ternary LUT inference).
- If you have the Medusa GGUF with retrained heads (`ggml-model-i2_s-medusa-official-v2.gguf`), copy it to `MedusaBitNet/models/bitnet-b1.58-2B-4T/`.
