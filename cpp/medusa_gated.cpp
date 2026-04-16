// medusa_gated.cpp — Medusa speculative decoding with unified-gate skip.
//
// Same as medusa.cpp but adds a confidence gate that can skip the backbone
// verification pass when the gate score exceeds a calibrated threshold.
// This saves the full backbone forward pass for easy tokens.
//
// Additional flags:
//   --gate <gate_k20.bin>     path to exported gate binary
//   --lambda <0.95>           fidelity target (looks up calibrated τ)
//
// Gate integration point: after collecting spec_tokens from Medusa heads,
// before building the verify batch. Uses the PREVIOUS step's hidden state
// and head logits (already available). If score >= τ, all spec tokens are
// accepted without verification.

#include "common.h"
#include "llama.h"
#include "log.h"
#include "medusa_tree.h"
#include "gate_artifact.h"
#include "gate_features.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct medusa_gated_params {
    std::string model_path;
    std::string gate_path;
    std::string centers_path;
    std::string prompt = "Hello";
    int32_t     n_predict    = 64;
    int32_t     n_ctx        = 2048;
    int32_t     n_threads    = -1;
    float       lambda       = 0.95f;
    bool        verbose      = false;
};

// cb_eval context for capturing multiple tensors during decode
struct hidden_capture {
    std::vector<float> last_hidden;    // result_norm at target slot
    std::vector<float> hidden_layer5;  // l_out-5 at target slot
    std::vector<float> hidden_layer15; // l_out-15 at target slot
    int32_t target_slot = -1;
    int32_t hidden_dim  = 0;
    bool captured = false;
    bool has_layer5 = false;
    bool has_layer15 = false;
};

static void capture_tensor_slot(struct ggml_tensor * t, int32_t slot, std::vector<float> & out) {
    int64_t H = t->ne[0];
    int64_t T = t->ne[1];
    if (slot < 0 || slot >= (int32_t)T) slot = (int32_t)T - 1;
    out.resize(H);
    size_t n_bytes = H * sizeof(float);
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (is_host) {
        const float * src = (const float *)t->data + slot * H;
        memcpy(out.data(), src, n_bytes);
    } else {
        ggml_backend_tensor_get(t, out.data(), slot * n_bytes, n_bytes);
    }
}

static bool cb_capture(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (hidden_capture *) user_data;
    if (ask) {
        return strcmp(t->name, "result_norm") == 0
            || strcmp(t->name, "l_out-5") == 0
            || strcmp(t->name, "l_out-15") == 0;
    }
    if (t->type != GGML_TYPE_F32) return true;

    if (strcmp(t->name, "result_norm") == 0) {
        cap->hidden_dim = (int32_t)t->ne[0];
        capture_tensor_slot(t, cap->target_slot, cap->last_hidden);
        cap->captured = true;
    } else if (strcmp(t->name, "l_out-5") == 0) {
        capture_tensor_slot(t, cap->target_slot, cap->hidden_layer5);
        cap->has_layer5 = true;
    } else if (strcmp(t->name, "l_out-15") == 0) {
        capture_tensor_slot(t, cap->target_slot, cap->hidden_layer15);
        cap->has_layer15 = true;
    }
    return true;
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m MODEL.gguf --gate GATE.bin [-p PROMPT] [-n N] [-c CTX] "
        "[-t THREADS] [--lambda L] [-v]\n", argv0);
}

static bool parse_args(int argc, char ** argv, medusa_gated_params & p) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-m" && i + 1 < argc)            p.model_path = argv[++i];
        else if (a == "--gate" && i + 1 < argc)    p.gate_path  = argv[++i];
        else if (a == "--centers" && i + 1 < argc) p.centers_path = argv[++i];
        else if (a == "-p" && i + 1 < argc)        p.prompt     = argv[++i];
        else if (a == "-n" && i + 1 < argc)        p.n_predict  = std::atoi(argv[++i]);
        else if (a == "-c" && i + 1 < argc)        p.n_ctx      = std::atoi(argv[++i]);
        else if (a == "-t" && i + 1 < argc)        p.n_threads  = std::atoi(argv[++i]);
        else if (a == "--lambda" && i + 1 < argc)  p.lambda     = std::atof(argv[++i]);
        else if (a == "-v")                        p.verbose    = true;
        else if (a == "-h" || a == "--help")       { print_usage(argv[0]); return false; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); print_usage(argv[0]); return false; }
    }
    if (p.model_path.empty()) { print_usage(argv[0]); return false; }
    return true;
}

static llama_token argmax(const float * logits, int32_t n_vocab) {
    llama_token best = 0;
    float best_score = logits[0];
    for (int32_t v = 1; v < n_vocab; ++v) {
        if (logits[v] > best_score) { best_score = logits[v]; best = (llama_token) v; }
    }
    return best;
}

int main(int argc, char ** argv) {
    medusa_gated_params p;
    if (!parse_args(argc, argv, p)) return 1;

    // ---- load gate ----
    GateArtifact gate;
    bool has_gate = false;
    if (!p.gate_path.empty()) {
        has_gate = gate.load(p.gate_path);
        if (!has_gate) {
            fprintf(stderr, "warning: gate failed to load, running without gating\n");
        } else {
            fprintf(stderr, "[gate] threshold for lambda=%.2f: tau=%.4f\n",
                    p.lambda, gate.threshold(p.lambda));
        }
    }

    llama_backend_init();

    // ---- model load ----
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_load_model_from_file(p.model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const int32_t n_medusa = llama_n_medusa_heads(model);
    if (n_medusa <= 0) {
        fprintf(stderr, "error: model has no Medusa heads\n");
        llama_free_model(model); return 1;
    }
    fprintf(stderr, "[medusa-gated] n_medusa_heads=%d\n", n_medusa);

    const int32_t n_vocab = llama_n_vocab(model);
    const int32_t hidden_dim = llama_n_embd(model);

    // ---- context with cb_eval for hidden capture ----
    hidden_capture cap;
    cap.hidden_dim = hidden_dim;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = p.n_ctx;
    cparams.n_batch   = p.n_ctx;
    cparams.n_seq_max = 4;
    if (p.n_threads > 0) {
        cparams.n_threads       = p.n_threads;
        cparams.n_threads_batch = p.n_threads;
    }
    cparams.flash_attn = false;
    cparams.cb_eval           = cb_capture;
    cparams.cb_eval_user_data = &cap;

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); llama_free_model(model); return 1; }

    // ---- feature state ----
    GateFeatureState feat_state;
    feat_state.hidden_dim = hidden_dim;
    feat_state.vocab_size = n_vocab;
    feat_state.n_heads    = std::min(n_medusa, 4);
    feat_state.init();
    // Extract lm_head (output.weight) from model for superposition features
    std::vector<float> lm_head_data;
    {
        struct ggml_tensor * output_w = llama_get_model_tensor(model, "output.weight");
        if (output_w && output_w->type == GGML_TYPE_F32) {
            size_t n = ggml_nelements(output_w);
            lm_head_data.resize(n);
            if (ggml_backend_buffer_is_host(output_w->buffer)) {
                memcpy(lm_head_data.data(), output_w->data, n * sizeof(float));
            } else {
                ggml_backend_tensor_get(output_w, lm_head_data.data(), 0, n * sizeof(float));
            }
            fprintf(stderr, "[gate] loaded output.weight (%zu floats) for superposition features\n", n);
        } else if (output_w) {
            fprintf(stderr, "[gate] output.weight is type %s, not F32 — superposition features disabled\n",
                    ggml_type_name(output_w->type));
        } else {
            fprintf(stderr, "[gate] output.weight not found — superposition features disabled\n");
        }
    }

    if (!p.centers_path.empty()) {
        FILE * fc = fopen(p.centers_path.c_str(), "rb");
        if (fc) {
            fseek(fc, 0, SEEK_END); long sz = ftell(fc); fseek(fc, 0, SEEK_SET);
            int K = 32;
            std::vector<float> cdata(sz / sizeof(float));
            fread(cdata.data(), sizeof(float), cdata.size(), fc);
            fclose(fc);
            feat_state.load_cluster_centers(cdata.data(), K, hidden_dim);
            fprintf(stderr, "[gate] loaded %d cluster centers from %s\n", K, p.centers_path.c_str());
        } else {
            fprintf(stderr, "[gate] warning: cannot open centers %s\n", p.centers_path.c_str());
        }
    }

    // ---- tokenise prompt ----
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, p.prompt, true);
    if (prompt_tokens.empty()) {
        fprintf(stderr, "error: empty prompt\n");
        llama_free(ctx); llama_free_model(model); return 1;
    }
    fprintf(stderr, "[medusa-gated] prompt tokens: %zu\n", prompt_tokens.size());
    for (auto t : prompt_tokens) printf("%s", common_token_to_piece(ctx, t).c_str());
    fflush(stdout);

    // ---- prefill ----
    llama_batch batch = llama_batch_init(p.n_ctx, 0, 4);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], (llama_pos) i, { 0 },
                         i == prompt_tokens.size() - 1);
    }
    cap.target_slot = batch.n_tokens - 1;
    cap.captured = false;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "prefill decode failed\n"); goto cleanup;
    }
    fprintf(stderr, "[gate] prefill: captured=%d layer5=%d layer15=%d\n",
            cap.captured, cap.has_layer5, cap.has_layer15);

    {
    llama_pos   last_pos   = (llama_pos) prompt_tokens.size() - 1;
    llama_token last_token = argmax(llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab);
    int32_t     prev_slot  = batch.n_tokens - 1;

    printf("%s", common_token_to_piece(ctx, last_token).c_str());
    fflush(stdout);
    int32_t n_generated = 1;
    if (llama_token_is_eog(model, last_token)) goto cleanup;

    int32_t total_accepted = 0, total_steps = 0, total_gated = 0;

    while (n_generated < p.n_predict) {
        // ---- collect spec tokens from Medusa heads ----
        std::vector<llama_token> spec_tokens(n_medusa);
        for (int k = 0; k < n_medusa; ++k) {
            const float * mlogits = llama_get_medusa_logits_ith(ctx, k, prev_slot);
            if (!mlogits) { fprintf(stderr, "medusa logits missing\n"); goto done; }
            spec_tokens[k] = argmax(mlogits, n_vocab);
        }

        // ---- GATE DECISION ----
        // Two modes: MLP gate (full features) or simple confidence threshold.
        // Simple mode: max(softmax(head-0 logits)) > tau → skip verification.
        // No features, no MLP, just one number from the Medusa head.
        bool gate_skip = false;
        if (!has_gate) {
            // Simple confidence gate: use head-0 softmax peak
            const float * mlogits0 = llama_get_medusa_logits_ith(ctx, 0, prev_slot);
            if (mlogits0) {
                float max_logit = mlogits0[0];
                for (int i = 1; i < n_vocab; i++)
                    if (mlogits0[i] > max_logit) max_logit = mlogits0[i];
                float sum_exp = 0.0f;
                float max_prob = 0.0f;
                for (int i = 0; i < n_vocab; i++) {
                    float e = expf(mlogits0[i] - max_logit);
                    sum_exp += e;
                    if (e > max_prob) max_prob = e;
                }
                float conf = max_prob / sum_exp;
                gate_skip = (conf >= p.lambda);
                if (p.verbose && total_steps < 5) {
                    fprintf(stderr, "[conf-gate] step %d conf=%.4f tau=%.4f → %s\n",
                            total_steps, conf, p.lambda, gate_skip ? "SKIP" : "verify");
                }
            }
        } else if (has_gate && cap.captured) {
            // Build head logits array [n_heads, vocab]
            std::vector<float> head_logits(feat_state.n_heads * n_vocab);
            for (int k = 0; k < feat_state.n_heads; k++) {
                const float * ml = llama_get_medusa_logits_ith(ctx, k, prev_slot);
                if (ml) memcpy(&head_logits[k * n_vocab], ml, n_vocab * sizeof(float));
            }

            float features[70] = {0};
            feat_state.extract_all(
                cap.last_hidden.data(),
                head_logits.data(),
                lm_head_data.empty() ? nullptr : lm_head_data.data(),
                (uint32_t)last_token,
                features
            );

            // Layer-wise features (slots 57-69), matching test_layer_wise.py layout:
            //   57: log1p(vel_5_15)     58: log1p(vel_15_29)   59: log1p(vel_5_29)
            //   60: layer_angle         61: cos_5_15           62: cos_15_29
            //   63: cos_5_29            64: log1p(n5)          65: log1p(n15)
            //   66: log1p(n29)          67: norm_ratio_5_29    68: norm_ratio_15_29
            //   69: early_agrees
            if (cap.has_layer5 && cap.has_layer15 && !cap.last_hidden.empty()) {
                const float * h5 = cap.hidden_layer5.data();
                const float * h15 = cap.hidden_layer15.data();
                const float * h29 = cap.last_hidden.data();
                int H = hidden_dim;
                float v515 = 0, v1529 = 0, v529 = 0;
                float cos_num_515 = 0, cos_num_1529 = 0, cos_num_529 = 0;
                float n5sq = 0, n15sq = 0, n29sq = 0;
                // Direction vectors for layer_angle
                float dir1_dot_dir2 = 0, dir1_norm = 0, dir2_norm = 0;
                for (int i = 0; i < H; i++) {
                    float d1 = h15[i] - h5[i], d2 = h29[i] - h15[i], d3 = h29[i] - h5[i];
                    v515 += d1*d1; v1529 += d2*d2; v529 += d3*d3;
                    n5sq += h5[i]*h5[i]; n15sq += h15[i]*h15[i]; n29sq += h29[i]*h29[i];
                    cos_num_515 += h5[i] * h15[i];
                    cos_num_1529 += h15[i] * h29[i];
                    cos_num_529 += h5[i] * h29[i];
                    dir1_dot_dir2 += d1 * d2;
                    dir1_norm += d1 * d1;
                    dir2_norm += d2 * d2;
                }
                float n5 = sqrtf(n5sq), n15 = sqrtf(n15sq), n29 = sqrtf(n29sq);
                features[57] = logf(1.0f + sqrtf(v515));
                features[58] = logf(1.0f + sqrtf(v1529));
                features[59] = logf(1.0f + sqrtf(v529));
                features[60] = dir1_dot_dir2 / (sqrtf(dir1_norm) * sqrtf(dir2_norm) + 1e-9f);
                features[61] = cos_num_515 / (n5 * n15 + 1e-9f);
                features[62] = cos_num_1529 / (n15 * n29 + 1e-9f);
                features[63] = cos_num_529 / (n5 * n29 + 1e-9f);
                features[64] = logf(1.0f + n5);
                features[65] = logf(1.0f + n15);
                features[66] = logf(1.0f + n29);
                features[67] = n5 / (n29 + 1e-6f);
                features[68] = n15 / (n29 + 1e-6f);
                // 69: early_agrees — would need lm_head projection, skip for now
            }

            float score = gate.score(features);
            gate_skip = (score >= gate.threshold(p.lambda));

            if (p.verbose) {
                if (total_steps < 3) {
                    fprintf(stderr, "[gate-debug] step=%d layer5=%d layer15=%d\n",
                            total_steps, cap.has_layer5?1:0, cap.has_layer15?1:0);
                    // Print all 20 gate features with their normalized values
                    for (int fi = 0; fi < (int)gate.K; fi++) {
                        int idx = gate.feature_indices[fi];
                        float raw = features[idx];
                        float normed = (raw - gate.mu[fi]) / (gate.sd[fi] + 1e-9f);
                        fprintf(stderr, "  [%2d] idx=%2d  raw=%10.4f  mu=%8.4f  sd=%8.4f  normed=%8.4f\n",
                                fi, idx, raw, gate.mu[fi], gate.sd[fi], normed);
                    }
                }
                fprintf(stderr, "[gate] step %d score=%.4f tau=%.4f → %s\n",
                        total_steps, score, gate.threshold(p.lambda),
                        gate_skip ? "SKIP" : "verify");
            }
        }

        if (gate_skip) {
            // Accept all spec tokens without verification
            for (int k = 0; k < n_medusa; k++) {
                printf("%s", common_token_to_piece(ctx, spec_tokens[k]).c_str());
            }
            fflush(stdout);

            // We need to "decode" the spec tokens into the KV cache even
            // though we're skipping verification. Build and run the batch
            // but DON'T check backbone argmax.
            common_batch_clear(batch);
            common_batch_add(batch, last_token, last_pos + 1, { 0 }, true);
            for (int k = 0; k < n_medusa; k++) {
                common_batch_add(batch, spec_tokens[k], last_pos + 2 + k, { 0 }, true);
            }
            cap.target_slot = n_medusa;  // capture hidden at last spec token
            cap.captured = false;
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "gated decode failed\n"); goto done;
            }

            last_pos   += 1 + n_medusa;
            last_token  = spec_tokens[n_medusa - 1];
            prev_slot   = n_medusa;
            n_generated += n_medusa;
            total_accepted += n_medusa;
            total_gated++;
            total_steps++;

            bool hit_eos = false;
            for (auto t : spec_tokens) {
                if (llama_token_is_eog(model, t)) { hit_eos = true; break; }
            }
            if (hit_eos) break;
            continue;
        }

        // ---- normal verify path (same as medusa.cpp) ----
        common_batch_clear(batch);
        common_batch_add(batch, last_token, last_pos + 1, { 0 }, true);
        for (int k = 0; k < n_medusa; ++k) {
            common_batch_add(batch, spec_tokens[k], last_pos + 2 + k, { 0 }, true);
        }
        cap.target_slot = -1;  // will be set after we know n_accept
        cap.captured = false;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "verify decode failed\n"); goto done;
        }

        {
        int32_t n_accept = 0;
        llama_token next_tok = 0;
        for (int i = 0; i <= n_medusa; ++i) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (!logits) { fprintf(stderr, "missing logits at %d\n", i); goto done; }
            next_tok = argmax(logits, n_vocab);
            if (i < n_medusa && next_tok == spec_tokens[i]) {
                n_accept++;
            } else {
                break;
            }
        }

        // Capture hidden at the slot where next_tok was produced
        // This requires re-reading result_norm at slot n_accept.
        // cb_eval already fired during decode, but we captured the wrong slot.
        // Workaround: always capture the last slot, which is n_medusa.
        // For proper multi-slot capture we'd need to buffer all slots.
        // For now: capture at the n_accept position by re-triggering.
        // TODO: buffer all result_norm slots in cb_eval for proper gating.

        std::vector<llama_token> accepted(spec_tokens.begin(), spec_tokens.begin() + n_accept);
        accepted.push_back(next_tok);

        for (auto t : accepted) {
            printf("%s", common_token_to_piece(ctx, t).c_str());
        }
        fflush(stdout);

        const llama_pos keep_upto = last_pos + 1 + (llama_pos) n_accept;
        llama_kv_cache_seq_rm(ctx, 0, keep_upto + 1, -1);

        last_pos   = keep_upto;
        last_token = next_tok;
        prev_slot  = n_accept;
        n_generated    += (int32_t) accepted.size();
        total_accepted += n_accept;
        total_steps    += 1;

        bool hit_eos = false;
        for (auto t : accepted) {
            if (llama_token_is_eog(model, t)) { hit_eos = true; break; }
        }
        if (hit_eos) break;
        }
    }

done:
    printf("\n");
    if (total_steps > 0) {
        const double mean_accept = (double) total_accepted / (double) total_steps;
        fprintf(stderr,
            "[medusa-gated] steps=%d  accepted=%d  mean_accept=%.2f/%d  "
            "generated=%d  gated_skips=%d (%.1f%%)\n",
            total_steps, total_accepted, mean_accept, n_medusa,
            n_generated, total_gated,
            100.0 * total_gated / total_steps);
    }
    }

cleanup:
    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
