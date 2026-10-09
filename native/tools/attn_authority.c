// attn_authority.c — a Ridge attention block's stages, computed with ggml's own ops (unit 10.3a).
//
// The reconnaissance flagged one substantial risk in Ridge's attention path: `attn_q.weight` is a
// JOINT Q+gate projection (`5120 x 12288` = one matrix carrying, per head, a 256-dim q block followed
// by a 256-dim gate block), and everything downstream depends on splitting it the way llama.cpp does.
// This harness does that split with `ggml_view_3d` and the rest with the real ops — `ggml_mul_mat`,
// `ggml_rms_norm`, `ggml_rope_multi` — and dumps the stages, so the engine has an authority to match
// instead of a reading of the source to believe.
//
// Scope: the whole block front-to-back — the Q/gate split, the two per-head RMS norms, rope, causal
// attention, the output gate `attn * sigmoid(gate)` and `wo`. Every stage is where a wrong reading is
// silent: a swapped q/gate half, a norm over the whole projection, rope across the whole head, a
// non-causal attention, or a raw gate multiply would all still produce plausible numbers.
//
// Usage: attn_authority in.bin out.bin n_head head_dim n_kv S D n_rot eps kq_scale attn_mode s0 s1 s2 [threads]
//   attn_mode = explicit (hand-rolled, n_kv must equal n_head) | flash (ggml_flash_attn_ext; GQA-capable)
//   in.bin (f32, ggml order — ne0 fastest, so numpy reads each as its last axis):
//     W_qg   (n_head*2*head_dim, D)   the joint Q+gate projection
//     W_k    (n_kv*head_dim,    D)
//     W_v    (n_kv*head_dim,    D)
//     W_o    (n_head*head_dim,  D)
//     norm_q (head_dim)
//     norm_k (head_dim)
//     hidden (S, D)
//   out.bin: stages concatenated as f32, in this order (counts derivable from the args):
//     qg (S, n_head*2*hd), q_pre (S, n_head, hd), gate_pre (S, n_head, hd),
//     q_norm (S, n_head, hd), q_rope (S, n_head, hd), k_norm (S, n_kv, hd), k_rope (S, n_kv, hd),
//     scores (S_q, S, n_head) [ne = (S_kv, S_q, n_head)], probs (same shape),
//     attn (S, n_head, hd), gated (S, n_head, hd), out (S, D)
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void * xmalloc(size_t n) {
    void * p = malloc(n);
    if (!p) { fprintf(stderr, "ATTN_ERR oom\n"); exit(1); }
    return p;
}

int main(int argc, char ** argv) {
    if (argc < 16 || argc > 17) {
        fprintf(stderr, "usage: %s in.bin out.bin n_head head_dim n_kv S D n_rot eps kq_scale attn_mode s0 s1 s2 [threads]\n", argv[0]);
        return 2;
    }
    const int64_t nh   = atoll(argv[3]);
    const int64_t hd   = atoll(argv[4]);
    const int64_t nkv  = atoll(argv[5]);
    const int64_t S    = atoll(argv[6]);
    const int64_t D    = atoll(argv[7]);
    const int     nd   = atoi(argv[8]);
    const float   eps  = (float) atof(argv[9]);
    const float   kqs  = (float) atof(argv[10]);
    const char *  amode = argv[11];
    const int     flash = (strcmp(amode, "flash") == 0);
    int sections[4] = { atoi(argv[12]), atoi(argv[13]), atoi(argv[14]), 0 };
    (void) sections;

    const int64_t nqg = nh * 2 * hd;
    const int64_t nk  = nkv * hd;

    // every f32 the fixture holds, in the order they are laid out in in.bin
    size_t n_float = (size_t) (nqg * D + nk * D + nk * D + nh * hd * D + hd + hd + S * D);
    float * fbuf = (float *) xmalloc(n_float * sizeof(float));
    {
        FILE * f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "ATTN_ERR open %s\n", argv[1]); return 2; }
        size_t got = fread(fbuf, sizeof(float), n_float, f);
        fclose(f);
        if (got != n_float) { fprintf(stderr, "ATTN_ERR short read %zu != %zu\n", got, n_float); return 3; }
    }
    float * Wqg = fbuf;
    float * Wk  = Wqg + nqg * D;
    float * Wv  = Wk + nk * D;
    float * Wo  = Wv + nk * D;
    float * nmq = Wo + nh * hd * D;
    float * nmk = nmq + hd;
    float * hid = nmk + hd;

    struct ggml_init_params ip = { .mem_size = (size_t) 512 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ATTN_ERR ggml_init\n"); return 4; }

    struct ggml_tensor * t_wqg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, nqg);
    struct ggml_tensor * t_wk  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, nk);
    struct ggml_tensor * t_nmq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    struct ggml_tensor * t_nmk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    struct ggml_tensor * t_h   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, S);
    memcpy(t_wqg->data, Wqg, (size_t) nqg * D * sizeof(float));
    memcpy(t_wk ->data, Wk,  (size_t) nk  * D * sizeof(float));
    memcpy(t_nmq->data, nmq, (size_t) hd  * sizeof(float));
    memcpy(t_nmk->data, nmk, (size_t) hd  * sizeof(float));
    memcpy(t_h  ->data, hid, (size_t) S   * D * sizeof(float));

    // ── the joint projection, then the split that the whole path depends on ──
    struct ggml_tensor * qg  = ggml_mul_mat(ctx, t_wqg, t_h);                  // (nqg, S)
    struct ggml_tensor * qg3 = ggml_reshape_3d(ctx, qg, 2 * hd, nh, S);        // per-head: q then gate
    // q = the first hd of each 2*hd block, gate = the second (llama.cpp's own view arithmetic)
    struct ggml_tensor * q  = ggml_view_3d(ctx, qg3, hd, nh, S, qg3->nb[1], qg3->nb[2], 0);
    struct ggml_tensor * gt = ggml_view_3d(ctx, qg3, hd, nh, S, qg3->nb[1], qg3->nb[2], (size_t) hd * sizeof(float));

    // ── per-head RMS norm (ne0 = hd), then rope over the first n_dims ──
    struct ggml_tensor * qn = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), t_nmq);
    // K needs the same per-head shape as Q: without the reshape the norm would run over the
    // whole n_kv*head_dim width and rope would see a 2-D tensor (ggml asserts a->ne[2] == pos length)
    struct ggml_tensor * kp = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, t_wk, t_h), hd, nkv, S);
    struct ggml_tensor * kn = ggml_mul(ctx, ggml_rms_norm(ctx, kp, eps), t_nmk);

    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    for (int64_t i = 0; i < S; i++) { ((int32_t *) pos->data)[i] = (int32_t) i; }
    struct ggml_tensor * qr = ggml_rope_multi(ctx, qn, pos, NULL, nd, (int[4]) { 11, 11, 10, 0 },
                                             GGML_ROPE_TYPE_NEOX, 40960, 1e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    struct ggml_tensor * kr = ggml_rope_multi(ctx, kn, pos, NULL, nd, (int[4]) { 11, 11, 10, 0 },
                                             GGML_ROPE_TYPE_NEOX, 40960, 1e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    struct ggml_tensor * t_wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, nk);
    struct ggml_tensor * t_wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nh * hd, D);
    memcpy(t_wv->data, Wv, (size_t) nk * D * sizeof(float));
    memcpy(t_wo->data, Wo, (size_t) nh * hd * D * sizeof(float));

    // ── attention (causal), the OUTPUT GATE, and wo ──
    // scores need K transposed for the V product afterwards, exactly as llama.cpp's non-flash path
    // does: KQ = mul_mat(k, q) -> (n_kv, n_head, S); V is transposed to (n_kv, head_dim, S) so that
    // mul_mat(V_t, probs) yields (head_dim, n_head, S).
    // The shape dance llama.cpp's non-flash path performs. ggml's 3-D mul_mat collapses ne0 and treats
    // ne2 as the batch, so attention over TOKENS needs the token axis in ne1: permute k/q/v from
    // (hd, heads, S) to (hd, S, heads). Then KQ = mul_mat(k, q) is (S_kv, S_q, n_head) — the key axis is
    // ne0, which is what soft_max reduces and what the causal mask indexes. (This fixture therefore uses
    // n_kv == n_head; GQA with fewer kv heads needs the repeat llama.cpp applies to K/V, which is a
    // separate step and is not claimed here.)
    struct ggml_tensor * vp  = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, t_wv, t_h), hd, nkv, S);
    // ggml_flash_attn_ext asserts its mask is F16, so this one is built as F16
    struct ggml_tensor * mask2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S, S);
    for (int64_t qi = 0; qi < S; qi++) {
        for (int64_t ki = 0; ki < S; ki++) {
            ((ggml_fp16_t *) mask2->data)[ki + qi * S] = ggml_fp32_to_fp16((ki > qi) ? -INFINITY : 0.0f);
        }
    }

    struct ggml_tensor * kqs_t;
    struct ggml_tensor * pr;
    struct ggml_tensor * at;
    if (nkv == nh) {
        // The hand-rolled path, which ggml's batched mul_mat can only express when the batch (head) dims
        // match. It is the cross-check on the flash op below, and it is why this fixture ran with
        // n_kv == n_head until now. Uses the view for at_ex, but the same 2-D mask as flash.
        struct ggml_tensor * qp  = ggml_permute(ctx, qr, 0, 2, 1, 3);        // (hd, S, n_head)
        struct ggml_tensor * kpp = ggml_permute(ctx, kr, 0, 2, 1, 3);       // (hd, S, n_kv)
        struct ggml_tensor * vpp = ggml_cont(ctx, ggml_transpose(ctx, ggml_permute(ctx, vp, 0, 2, 1, 3)));
        struct ggml_tensor * kq  = ggml_mul_mat(ctx, kpp, qp);              // (S_kv, S_q, n_head)
        struct ggml_tensor * kqm = ggml_diag_mask_inf(ctx, kq, 0);
        (void) kqm;
        kqs_t = ggml_scale(ctx, kq, kqs);
        // the mask is applied INSIDE the op, so `scores` is deliberately the unmasked scaled scores
        struct ggml_tensor * kq3 = ggml_reshape_3d(ctx, ggml_scale(ctx, kq, 1.0f), S, S, nh);
        struct ggml_tensor * mask3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, S, 1);
        for (int64_t qi = 0; qi < S; qi++) {
            for (int64_t ki = 0; ki < S; ki++) {
                ((float *) mask3->data)[ki + qi * S] = (ki > qi) ? -INFINITY : 0.0f;
            }
        }
        pr = ggml_soft_max_ext(ctx, kq3, mask3, kqs, 0.0f);
        struct ggml_tensor * atp = ggml_mul_mat(ctx, vpp, pr);
        struct ggml_tensor * at_ex = ggml_cont(ctx, ggml_permute(ctx, atp, 0, 2, 1, 3));
        // NOT cast to F16: llama.cpp does that at its call site, but doing the same here (with
        // ggml_flash_attn_ext_set_prec F32) turned the output into nans, so the F16 cast is not the
        // explanation for the mismatch below and is left out.
        struct ggml_tensor * kt = ggml_cont(ctx, ggml_permute(ctx, kr, 0, 2, 1, 3));
        struct ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vp, 0, 2, 1, 3));
        struct ggml_tensor * atf = ggml_flash_attn_ext(ctx, qr, kt, vt, mask2, kqs, 0.0f, 0.0f);
        // flash returns (hd, n_tokens, n_head) — measured, not assumed — so permute to the (hd, n_head,
        // n_tokens) that the split half of this harness and the gate multiply use
        at = flash ? ggml_cont(ctx, ggml_permute(ctx, atf, 0, 2, 1, 3)) : at_ex;
    } else {
        // GQA (n_kv < n_head): the hand-rolled products cannot be expressed, so the flash op — the one
        // llama.cpp uses for this architecture — is the authority, and the two diagnostic slots are
        // zeros. Keep the slot COUNT and ORDER identical in both modes: the reader walks positionally.
        struct ggml_tensor * kt = ggml_cont(ctx, ggml_permute(ctx, kr, 0, 2, 1, 3));
        struct ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, vp, 0, 2, 1, 3));
        struct ggml_tensor * atf = ggml_flash_attn_ext(ctx, qr, kt, vt, mask2, kqs, 0.0f, 0.0f);
        at = ggml_cont(ctx, ggml_permute(ctx, atf, 0, 2, 1, 3));
        kqs_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, S, nh);
        pr    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, S, nh);
    }
    struct ggml_tensor * gs  = ggml_sigmoid(ctx, gt);                   // the gate half of the joint projection
    struct ggml_tensor * gtd = ggml_mul(ctx, at, gs);
    struct ggml_tensor * gc2 = ggml_cont_2d(ctx, gtd, nh * hd, S);
    struct ggml_tensor * wo_out = ggml_mul_mat(ctx, t_wo, gc2);            // (D, S)

    // The splits are VIEWS: their ->data points into the parent with the parent's strides, so a linear
    // dump of a view reports the parent's layout. Copy them contiguous — that is what makes the two
    // split stages mean anything on their own.
    struct ggml_tensor * qc  = ggml_cont(ctx, q);
    struct ggml_tensor * gtc = ggml_cont(ctx, gt);

    // Mark every read-back tensor as a graph OUTPUT *before* the graph is allocated. Called after the
    // compute (or after ggml_new_graph) it is too late: the allocator has already handed a stage's buffer
    // to someone else, and the dump then reports that other tensor's values — which is exactly how
    // `attn` came to hold the scores (and why a change elsewhere in the graph moved it: the aliasing did).
    ggml_set_input(mask2);
    ggml_set_output(qg);  ggml_set_output(qc);  ggml_set_output(gtc);
    ggml_set_output(qn);  ggml_set_output(qr);  ggml_set_output(kn);  ggml_set_output(kr);
    ggml_set_output(kqs_t); ggml_set_output(pr);
    ggml_set_output(at);  ggml_set_output(gtd); ggml_set_output(wo_out);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, qg);
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, gt);
    ggml_build_forward_expand(gf, qc);
    ggml_build_forward_expand(gf, gtc);
    ggml_build_forward_expand(gf, at);
    ggml_build_forward_expand(gf, gtd);
    ggml_build_forward_expand(gf, wo_out);
    ggml_build_forward_expand(gf, qn);
    ggml_build_forward_expand(gf, qr);
    ggml_build_forward_expand(gf, kn);
    ggml_build_forward_expand(gf, kr);
    // An authority must be deterministic. Default to ONE thread (a threaded graph compute is the prime
    // suspect for a verdict that flipped between runs), and let the caller ask for more to test that.
    const int n_threads = (argc == 17) ? atoi(argv[16]) : 1;
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // Every tensor this harness READS BACK is marked as an output before the graph is built (and the
    // mask as an input), so the allocator keeps those buffers alive to the end. NOTE: this was added
    // while chasing what looked like buffer aliasing, and it was NOT the cause of that mismatch — the
    // cause was a parse-order bug in attn_verify.py (see the DUMPED comment there). Kept because it is
    // the correct way to read a stage back, and because a harness that dumps what it reads should say so.
    ggml_set_input(mask2);
    const struct { const char * nm; struct ggml_tensor * t; } stages[] = {
        { "qg", qg }, { "q_pre", qc }, { "gate_pre", gtc }, { "q_norm", qn },
        { "q_rope", qr }, { "k_norm", kn }, { "k_rope", kr },
        { "scores", kqs_t }, { "probs", pr },
        { "attn", at }, { "gated", gtd }, { "out", wo_out },
    };
    FILE * out = fopen(argv[2], "wb");
    if (!out) { fprintf(stderr, "ATTN_ERR create %s\n", argv[2]); return 2; }
    for (size_t i = 0; i < sizeof(stages) / sizeof(stages[0]); i++) {
        struct ggml_tensor * t = stages[i].t;
        const int64_t n = ggml_nelements(t);
        printf("ATTN_STAGE %s n0=%lld n1=%lld n2=%lld n=%lld\n", stages[i].nm,
               (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) n);
        if (fwrite(t->data, sizeof(float), (size_t) n, out) != (size_t) n) {
            fprintf(stderr, "ATTN_ERR short write\n"); return 3;
        }
    }
    fclose(out);
    printf("ATTN_OK n_head=%lld head_dim=%lld n_kv=%lld S=%lld D=%lld n_dims=%d eps=%g threads=%d mode=%s\n",
           (long long) nh, (long long) hd, (long long) nkv, (long long) S, (long long) D, nd, (double) eps,
           n_threads, amode);
    ggml_free(ctx);
    free(fbuf);
    return 0;
}
