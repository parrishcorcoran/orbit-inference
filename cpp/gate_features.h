#pragma once
// gate_features.h — extract the 20 gate features from decode-step tensors.
//
// Features that need the 70-slot feature vector filled:
//   Slot  Name              Source
//   ----  ----              ------
//    0    content_conf      head-0 softmax max
//    2    logit_gap         head-0 top1 - top2 logit
//    5    top10_cov         head-0 cumulative top-10 softmax
//   14    agreement_count   head-0 argmax matches head-{1,2,3} lagged (bool-OR)
//   22    vel_0             ||h_t - h_{t-1}||
//   24    cluster_mindist   min distance to K-means center
//   25    cluster_entropy   soft-cluster entropy
//   26    mom_0             softmax 3rd moment (skewness)
//   28    hnorm_0           log(1 + ||h_t||)
//   30    fe_0              log(1 + 0.01 * cluster_mindist)
//   31    fe_1              fe_0 - 0.1 * cluster_entropy
//   32    nbr_0             min distance to recent hidden states
//   45    treuse_2          token-reuse rank within recent window
//   54    rg_2              distance to mean of recent 9 hidden states
//   55    sup_0             top-K token-embedding spread
//   56    sup_1             top-K effective rank (exp entropy)
//   57-69 layer_*           Ryu-Takayanagi (needs mid-layer taps, filled externally)
//
// The gate's feature_indices select a subset of these 70 slots.
// Feature computation cost: ~300k float ops per step (~300 μs on CPU).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>

struct GateFeatureState {
    // Configuration
    int hidden_dim = 2560;
    int vocab_size = 128256;
    int n_heads    = 4;
    int n_clusters = 32;
    int nbr_window = 20;
    int rg_window  = 9;
    int topk       = 32;

    // Precomputed cluster centers [n_clusters, hidden_dim]
    std::vector<float> cluster_centers;

    // Circular buffer of recent hidden states for neighborhood + velocity
    int buf_size = 64;
    std::vector<std::vector<float>> hidden_buf;
    int buf_pos = 0;
    int buf_count = 0;

    // Circular buffer of recent tokens for token-reuse
    std::vector<uint32_t> token_buf;
    int tok_pos = 0;
    int tok_count = 0;

    // Previous hidden state for velocity
    std::vector<float> h_prev;
    bool has_prev = false;

    // Lagged head predictions for agreement (ring buffer, depth 4)
    std::vector<uint32_t> head_pred_buf; // [4 * n_heads]
    int pred_pos = 0;
    int pred_count = 0;

    void init() {
        hidden_buf.resize(buf_size, std::vector<float>(hidden_dim, 0.0f));
        h_prev.resize(hidden_dim, 0.0f);
        token_buf.resize(buf_size, 0);
        head_pred_buf.resize(4 * n_heads, 0);
    }

    void load_cluster_centers(const float * data, int K, int H) {
        n_clusters = K;
        hidden_dim = H;
        cluster_centers.assign(data, data + K * H);
    }

    // Push current hidden state into the circular buffer
    void push_hidden(const float * h) {
        memcpy(hidden_buf[buf_pos % buf_size].data(), h, hidden_dim * sizeof(float));
        buf_pos++;
        if (buf_count < buf_size) buf_count++;
        // Update velocity state
        memcpy(h_prev.data(), h, hidden_dim * sizeof(float));
        has_prev = true;
    }

    void push_token(uint32_t tok) {
        token_buf[tok_pos % buf_size] = tok;
        tok_pos++;
        if (tok_count < buf_size) tok_count++;
    }

    void push_head_preds(const uint32_t * preds) {
        // preds is [n_heads] argmax per head
        memcpy(&head_pred_buf[(pred_pos % 4) * n_heads], preds, n_heads * sizeof(uint32_t));
        pred_pos++;
        if (pred_count < 4) pred_count++;
    }

    // ---- Feature extraction (fills slots in features[70]) ----

    void extract_all(
        const float * h_cur,          // [hidden_dim] current hidden state (result_norm)
        const float * head_logits,    // [n_heads, vocab_size] head logits
        const float * lm_head,        // [vocab_size, hidden_dim] for superposition
        uint32_t cur_token,
        float features[70]            // output: 70-slot feature vector
    ) {
        memset(features, 0, 70 * sizeof(float));

        // ---- Head-0 softmax stats (slots 0, 2, 5, 26) ----
        const float * logits0 = head_logits;  // first head
        // Softmax of head-0
        float max_logit = *std::max_element(logits0, logits0 + vocab_size);
        std::vector<float> probs0(vocab_size);
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            probs0[i] = expf(logits0[i] - max_logit);
            sum_exp += probs0[i];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < vocab_size; i++) probs0[i] *= inv_sum;

        // content_conf (slot 0)
        float conf = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            if (probs0[i] > conf) conf = probs0[i];
        }
        features[0] = conf;

        // logit_gap (slot 2): top1 - top2 logit
        float top1 = -1e30f, top2 = -1e30f;
        for (int i = 0; i < vocab_size; i++) {
            if (logits0[i] > top1) { top2 = top1; top1 = logits0[i]; }
            else if (logits0[i] > top2) { top2 = logits0[i]; }
        }
        features[2] = top1 - top2;

        // top10_cov (slot 5): sum of top-10 probs
        {
            std::vector<float> ps(probs0.begin(), probs0.end());
            std::partial_sort(ps.begin(), ps.begin() + 10, ps.end(), std::greater<float>());
            float s = 0.0f;
            for (int i = 0; i < 10; i++) s += ps[i];
            features[5] = s;
        }

        // mom_0 (slot 26): softmax skewness
        {
            double mean_tok = 0.0;
            for (int i = 0; i < vocab_size; i++) mean_tok += probs0[i] * (double)i;
            double var = 0.0, m3 = 0.0;
            for (int i = 0; i < vocab_size; i++) {
                double d = (double)i - mean_tok;
                var += probs0[i] * d * d;
                m3  += probs0[i] * d * d * d;
            }
            double sd = sqrt(var + 1e-12);
            features[26] = (float)(m3 / (sd * sd * sd));
        }

        // ---- agreement_count (slot 14) ----
        // Current head argmaxes
        uint32_t preds[4];
        for (int k = 0; k < std::min(n_heads, 4); k++) {
            const float * lk = head_logits + k * vocab_size;
            uint32_t best = 0;
            float best_v = lk[0];
            for (int i = 1; i < vocab_size; i++) {
                if (lk[i] > best_v) { best_v = lk[i]; best = i; }
            }
            preds[k] = best;
        }
        push_head_preds(preds);
        // agreement: any of head-{1,2,3} at lag-k match head-0 at current
        // bool-OR semantics (matches numpy 2.x training behavior)
        if (pred_count >= 2) {
            bool any_agree = false;
            uint32_t h0 = preds[0];
            for (int k = 1; k < std::min(4, pred_count); k++) {
                int lag_idx = ((pred_pos - 1 - k) % 4 + 4) % 4;
                uint32_t hk = head_pred_buf[lag_idx * n_heads + k];
                if (hk == h0) any_agree = true;
            }
            features[14] = any_agree ? 1.0f : 0.0f;
        }

        // ---- velocity (slot 22) ----
        if (has_prev) {
            float vel = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                float d = h_cur[i] - h_prev[i];
                vel += d * d;
            }
            features[22] = sqrtf(vel);
        }

        // ---- hidden norm (slot 28) ----
        {
            float norm = 0.0f;
            for (int i = 0; i < hidden_dim; i++) norm += h_cur[i] * h_cur[i];
            features[28] = logf(1.0f + sqrtf(norm));
        }

        // ---- cluster (slots 24, 25) + FE (30, 31) ----
        if (!cluster_centers.empty()) {
            float min_d = 1e30f;
            std::vector<float> neg_dists(n_clusters);
            for (int c = 0; c < n_clusters; c++) {
                float d2 = 0.0f;
                const float * center = cluster_centers.data() + c * hidden_dim;
                for (int i = 0; i < hidden_dim; i++) {
                    float diff = h_cur[i] - center[i];
                    d2 += diff * diff;
                }
                float d = sqrtf(d2);
                if (d < min_d) min_d = d;
                neg_dists[c] = -d * 0.01f;
            }
            features[24] = min_d;

            // Soft-cluster entropy
            float max_nd = *std::max_element(neg_dists.begin(), neg_dists.end());
            float sum_e = 0.0f;
            for (int c = 0; c < n_clusters; c++) {
                neg_dists[c] = expf(neg_dists[c] - max_nd);
                sum_e += neg_dists[c];
            }
            float ent = 0.0f;
            for (int c = 0; c < n_clusters; c++) {
                float p = neg_dists[c] / sum_e;
                if (p > 1e-12f) ent -= p * logf(p);
            }
            features[25] = ent;

            // Free energy (slot 30, 31)
            features[30] = logf(1.0f + min_d * 0.01f);
            features[31] = features[30] - ent * 0.1f;
        }

        // ---- neighborhood (slot 32) ----
        if (buf_count > 1) {
            float min_nbr = 1e30f;
            int lookback = std::min(nbr_window, buf_count - 1);
            for (int j = 1; j <= lookback; j++) {
                int idx = ((buf_pos - 1 - j) % buf_size + buf_size) % buf_size;
                float d2 = 0.0f;
                for (int i = 0; i < hidden_dim; i++) {
                    float diff = h_cur[i] - hidden_buf[idx][i];
                    d2 += diff * diff;
                }
                float d = sqrtf(d2);
                if (d < min_nbr) min_nbr = d;
            }
            features[32] = min_nbr;
        }

        // ---- token reuse rank (slot 45) ----
        {
            int rank = 0;
            int lookback = std::min(30, tok_count);
            for (int j = 0; j < lookback; j++) {
                int idx = ((tok_pos - 1 - j) % buf_size + buf_size) % buf_size;
                if (token_buf[idx] == cur_token) { rank = j + 1; break; }
            }
            features[45] = (float)rank;
        }

        // ---- RG div9 (slot 54) ----
        if (buf_count >= rg_window) {
            float mean[2560] = {0};
            for (int j = 0; j < rg_window; j++) {
                int idx = ((buf_pos - 1 - j) % buf_size + buf_size) % buf_size;
                for (int i = 0; i < hidden_dim; i++) mean[i] += hidden_buf[idx][i];
            }
            float d2 = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                mean[i] /= rg_window;
                float diff = h_cur[i] - mean[i];
                d2 += diff * diff;
            }
            features[54] = sqrtf(d2);
        }

        // ---- superposition (slots 55, 56) ----
        if (lm_head != nullptr) {
            // Top-K token indices from head-0 probs
            std::vector<int> top_idx(topk);
            std::vector<float> top_p(topk, 0.0f);
            for (int i = 0; i < vocab_size; i++) {
                for (int k = 0; k < topk; k++) {
                    if (probs0[i] > top_p[k]) {
                        for (int kk = topk - 1; kk > k; kk--) {
                            top_p[kk] = top_p[kk-1];
                            top_idx[kk] = top_idx[kk-1];
                        }
                        top_p[k] = probs0[i];
                        top_idx[k] = i;
                        break;
                    }
                }
            }

            // sup_0: spread of top-K token embeddings
            // centroid, then mean distance to centroid
            std::vector<float> centroid(hidden_dim, 0.0f);
            for (int k = 0; k < topk; k++) {
                const float * emb = lm_head + top_idx[k] * hidden_dim;
                for (int i = 0; i < hidden_dim; i++) centroid[i] += emb[i];
            }
            for (int i = 0; i < hidden_dim; i++) centroid[i] /= topk;
            float spread = 0.0f;
            for (int k = 0; k < topk; k++) {
                const float * emb = lm_head + top_idx[k] * hidden_dim;
                float d2 = 0.0f;
                for (int i = 0; i < hidden_dim; i++) {
                    float diff = emb[i] - centroid[i];
                    d2 += diff * diff;
                }
                spread += sqrtf(d2);
            }
            features[55] = spread / topk;

            // sup_1: effective rank = exp(entropy of top-K probs)
            float p_sum = 0.0f;
            for (int k = 0; k < topk; k++) p_sum += top_p[k];
            float ent = 0.0f;
            for (int k = 0; k < topk; k++) {
                float p = top_p[k] / (p_sum + 1e-12f);
                if (p > 1e-12f) ent -= p * logf(p);
            }
            features[56] = expf(ent);
        }

        // Push state for next step
        push_hidden(h_cur);
        push_token(cur_token);
    }
};
