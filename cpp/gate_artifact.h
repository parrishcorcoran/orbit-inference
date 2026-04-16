#pragma once
// gate_artifact.h — loads gate_k20.bin (flat binary) and runs the skip MLP.
//
// Usage:
//   GateArtifact gate;
//   gate.load("gate_k20.bin");
//   float features[70];  // fill from decode state
//   float score = gate.score(features);
//   if (score >= gate.threshold(0.95f)) { /* skip verifier */ }

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

struct GateArtifact {
    uint32_t K        = 0;   // number of selected features (20)
    uint32_t hidden   = 0;   // MLP hidden dim (64)
    uint32_t n_thresh = 0;

    std::vector<uint32_t> feature_indices;
    std::vector<float>    mu;
    std::vector<float>    sd;

    // MLP weights: net = Linear(K, H) -> ReLU -> Linear(H, H) -> ReLU -> Linear(H, 1)
    std::vector<float> W0;   // [hidden, K]
    std::vector<float> b0;   // [hidden]
    std::vector<float> W1;   // [hidden, hidden]
    std::vector<float> b1;   // [hidden]
    std::vector<float> W2;   // [hidden]  (stored as 1×hidden, row-major)
    float              b2 = 0.0f;

    // Calibrated thresholds: lambda -> tau
    struct ThreshEntry { float lambda; float tau; };
    std::vector<ThreshEntry> thresholds;

    bool load(const std::string & path) {
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "gate: cannot open %s\n", path.c_str());
            return false;
        }

        auto read_u32 = [&]() -> uint32_t {
            uint32_t v; fread(&v, 4, 1, f); return v;
        };
        auto read_f32_vec = [&](size_t n) -> std::vector<float> {
            std::vector<float> v(n);
            fread(v.data(), 4, n, f);
            return v;
        };

        K        = read_u32();
        hidden   = read_u32();
        n_thresh = read_u32();

        feature_indices.resize(K);
        fread(feature_indices.data(), 4, K, f);

        mu = read_f32_vec(K);
        sd = read_f32_vec(K);
        W0 = read_f32_vec(hidden * K);
        b0 = read_f32_vec(hidden);
        W1 = read_f32_vec(hidden * hidden);
        b1 = read_f32_vec(hidden);
        W2 = read_f32_vec(hidden);
        fread(&b2, 4, 1, f);

        thresholds.resize(n_thresh);
        for (uint32_t i = 0; i < n_thresh; i++) {
            fread(&thresholds[i].lambda, 4, 1, f);
            fread(&thresholds[i].tau,    4, 1, f);
        }

        fclose(f);
        fprintf(stderr, "gate: loaded %s  K=%u hidden=%u thresholds=%u\n",
                path.c_str(), K, hidden, n_thresh);
        return true;
    }

    // Score a single token's features. features_full is the 70-dim feature vector.
    float score(const float * features_full) const {
        // Select + normalize
        std::vector<float> x(K);
        for (uint32_t i = 0; i < K; i++) {
            x[i] = (features_full[feature_indices[i]] - mu[i]) / (sd[i] + 1e-9f);
        }

        // Layer 0: h = ReLU(W0 @ x + b0)
        std::vector<float> h(hidden);
        for (uint32_t j = 0; j < hidden; j++) {
            float sum = b0[j];
            for (uint32_t i = 0; i < K; i++) {
                sum += W0[j * K + i] * x[i];
            }
            h[j] = sum > 0.0f ? sum : 0.0f;
        }

        // Layer 1: h = ReLU(W1 @ h + b1)
        std::vector<float> h2(hidden);
        for (uint32_t j = 0; j < hidden; j++) {
            float sum = b1[j];
            for (uint32_t i = 0; i < hidden; i++) {
                sum += W1[j * hidden + i] * h[i];
            }
            h2[j] = sum > 0.0f ? sum : 0.0f;
        }

        // Layer 2: logit = W2 @ h2 + b2
        float logit = b2;
        for (uint32_t i = 0; i < hidden; i++) {
            logit += W2[i] * h2[i];
        }

        // Sigmoid
        return 1.0f / (1.0f + expf(-logit));
    }

    float threshold(float lambda) const {
        float best_tau = 1.0f;
        float best_dist = 1e9f;
        for (const auto & t : thresholds) {
            float d = fabsf(t.lambda - lambda);
            if (d < best_dist) {
                best_dist = d;
                best_tau = t.tau;
            }
        }
        return best_tau;
    }

    bool should_skip(const float * features_full, float lambda = 0.95f) const {
        return score(features_full) >= threshold(lambda);
    }
};
