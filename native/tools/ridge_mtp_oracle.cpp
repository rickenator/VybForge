// ridge_mtp_oracle — run llama.cpp's OWN MTP draft head and measure its acceptance.
//
// Why this exists: P4.13 compares our blk.64 implementation against the main model's greedy picks, and
// the bar for that comparison has been unset because the oracle's own MTP acceptance was not measurable
// through the CLI. `params.speculative.types = draft-mtp` exists and the server wires it up, but at this
// commit the field has no CLI surface in this build (`--speculative.types` is rejected, and the help's
// speculative section is draft-MODEL config only). The API, however, is public: set a second context's
// ctx_type = LLAMA_CONTEXT_TYPE_MTP with ctx_other pointing at the target context, and feed it a batch
// carrying BOTH the next token and the target's hidden (the MTP path keys off batch.embd).
//
// Method, teacher-forced exactly like P4.13 so the numbers are comparable: decode the fixture's prompt
// ids in the target context, take its per-token hidden (post final norm — the fixture's hidden_stage),
// then for each position t feed the MTP context (token = the true next token, embd = hidden row t) and
// compare the argmax of its logits with the fixture's recorded pick at k = t+2, under the fixture's own
// rule (equality where the top-2 margin clears margin_bar, membership of {top1,top2} below it).
//
// Build:  g++ -O2 -o native/build/ridge_mtp_oracle native/tools/ridge_mtp_oracle.cpp \
//             -I ~/Projects/llama.cpp/include -L ~/Projects/llama.cpp/build/bin \
//             -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,$HOME/Projects/llama.cpp/build/bin
// Run:    native/build/ridge_mtp_oracle <model.gguf> <fixture.fix>

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

struct Pick { int k = 0; int t1 = 0; double lp1 = 0; int t2 = 0; double lp2 = 0; std::string rule; };

static std::vector<long> parse_ids(const std::string & s) {
    std::vector<long> v; std::stringstream ss(s); long x;
    while (ss >> x) v.push_back(x);
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <fixture.fix>\n", argv[0]); return 2; }

    // ---- read the fixture: everything the run needs comes from it, including the ids ----
    std::ifstream f(argv[2]);
    if (!f) { fprintf(stderr, "cannot open fixture %s\n", argv[2]); return 2; }
    std::vector<long> prompt_ids, gen_ids;
    std::vector<Pick> picks;
    double margin_bar = 0.5;
    std::string hidden_file;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string tag;
        ls >> tag;
        if (tag == "prompt_ids" || tag == "gen_ids") {
            std::string rest = line.substr(line.find(tag) + tag.size());
            std::vector<long> ids = parse_ids(rest);
            if (tag == "prompt_ids") prompt_ids = ids; else gen_ids = ids;
        } else if (tag == "margin_bar") {
            ls >> margin_bar;
        } else if (tag == "hidden_file") {
            ls >> hidden_file;
        } else if (tag == "pos_top1") {
            Pick p;
            ls >> p.k >> p.t1 >> p.lp1 >> p.t2 >> p.lp2 >> p.rule;
            picks.push_back(p);
        }
    }
    if (prompt_ids.empty() || picks.empty()) { fprintf(stderr, "fixture lacks prompt_ids or pos_top1\n"); return 2; }

    // the true next token after position t: the prompt if it exists there, else the oracle's own
    // greedy continuation (gen_ids) — teacher forcing needs the TRUE token, not our draft.
    auto next_token_after = [&](size_t t) -> long {
        if (t + 1 < prompt_ids.size()) return prompt_ids[t + 1];
        size_t gi = t + 1 - prompt_ids.size();
        return gi < gen_ids.size() ? gen_ids[gi] : -1;
    };
    (void) hidden_file;   // the hidden is taken from the live target context, not the fixture file

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("ORACLE_MTP model=%s n_embd=%d n_vocab=%d prompt=%zu picks=%zu bar=%.3f\n",
           argv[1], n_embd, n_vocab, prompt_ids.size(), picks.size(), margin_bar);

    // ---- target context: embeddings on, no pooling, so the per-token hidden is exposed ----
    llama_context_params cpa = llama_context_default_params();
    cpa.n_ctx     = 512;
    cpa.n_batch   = 512;
    cpa.n_ubatch  = 512;
    cpa.n_seq_max = 1;
    cpa.embeddings  = true;
    cpa.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctxA = llama_init_from_model(model, cpa);
    if (!ctxA) { fprintf(stderr, "target context init failed\n"); return 1; }

    // ---- the MTP draft context: same model, ctx_type = MTP, linked to the target ----
    llama_context_params cpb = llama_context_default_params();
    cpb.n_ctx     = 512;
    cpb.n_batch   = 1;
    cpb.n_ubatch  = 1;
    cpb.n_seq_max = 1;
    cpb.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    cpb.ctx_other = ctxA;
    llama_context * ctxB = llama_init_from_model(model, cpb);
    if (!ctxB) { fprintf(stderr, "MTP context init failed (no nextn tensors?)\n"); return 1; }
    printf("ORACLE_MTP contexts ready (target + MTP)\n");

    // ---- decode the prompt in the target ----
    llama_batch ba = llama_batch_init((int) prompt_ids.size(), 0, 1);
    ba.n_tokens = (int) prompt_ids.size();
    for (size_t i = 0; i < prompt_ids.size(); i++) {
        ba.token[i]    = (llama_token) prompt_ids[i];
        ba.pos[i]      = (llama_pos) i;
        ba.n_seq_id[i] = 1;
        ba.seq_id[i][0] = 0;
        ba.logits[i]   = 0;
    }
    if (llama_decode(ctxA, ba) != 0) { fprintf(stderr, "target decode failed\n"); return 1; }
    const float * hid = llama_get_embeddings(ctxA);
    if (!hid) { fprintf(stderr, "target produced no embeddings (is embeddings=true honoured?)\n"); return 1; }
    printf("ORACLE_MTP target hidden: %zu rows x %d\n", prompt_ids.size(), n_embd);

    // ---- per position: feed the MTP context (true next token + hidden row t), take its argmax ----
    int checked = 0, agree = 0;
    for (size_t t = 0; t + 2 <= (size_t) picks.back().k + 1 && t < prompt_ids.size(); t++) {
        long tok = next_token_after(t);
        if (tok < 0) break;
        int k = (int) t + 2;
        // the fixture's pick for this k, if it recorded one
        const Pick * p = nullptr;
        for (const auto & q : picks) if (q.k == k) { p = &q; break; }
        if (!p) continue;

        llama_batch bb = llama_batch_init(1, n_embd, 1);
        bb.n_tokens = 1;
        bb.token[0]    = (llama_token) tok;
        bb.pos[0]      = (llama_pos) (t + 1);
        bb.n_seq_id[0] = 1;
        bb.seq_id[0][0] = 0;
        bb.logits[0]   = 1;
        memcpy(bb.embd, hid + t * (size_t) n_embd, sizeof(float) * (size_t) n_embd);

        if (llama_decode(ctxB, bb) != 0) { fprintf(stderr, "MTP decode failed at t=%zu\n", t); llama_batch_free(bb); return 1; }
        const float * lg = llama_get_logits_ith(ctxB, 0);
        if (!lg) { fprintf(stderr, "no logits from MTP at t=%zu\n", t); llama_batch_free(bb); return 1; }
        llama_batch_free(bb);

        int best = 0; float bv = lg[0];
        for (int v = 1; v < n_vocab; v++) if (lg[v] > bv) { bv = lg[v]; best = v; }

        double margin = p->lp1 - p->lp2;
        bool hit = (p->rule == "exact") ? (best == p->t1) : (best == p->t1 || best == p->t2);
        checked++; if (hit) agree++;
        printf("ORACLE_MTP   k=%3d tok=%-7ld oracle=%-7d margin=%7.3f %-11s -> argmax=%-7d %s\n",
               k, tok, p->t1, margin, p->rule.c_str(), best, hit ? "ok" : "MISS");
    }

    printf("ORACLE_MTP_ACCEPTANCE agree=%d/%d (%.1f%%)  [the oracle's OWN MTP head, same teacher forcing]\n",
           agree, checked, checked ? 100.0 * agree / checked : 0.0);

    llama_batch_free(ba);
    llama_free(ctxB); llama_free(ctxA);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
