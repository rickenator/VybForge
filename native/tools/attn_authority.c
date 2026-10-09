// attn_authority.c — a Ridge attention block's stages, computed with ggml's own ops (unit 10.3a).
//
// The reconnaissance flagged one substantial risk in Ridge's attention path: `attn_q.weight` is a
// JOINT Q+gate projection (`5120 x 12288` = one matrix carrying, per head, a 256-dim q block followed
// by a 256-dim gate block), and everything downstream depends on splitting it the way llama.cpp does.
// This harness does that split with `ggml_view_3d` and the rest with the real ops — `ggml_mul_mat`,
// `ggml_rms_norm`, `ggml_rope_multi` — and dumps the stages, so the engine has an authority to match
// instead of a reading of the source to believe.
//
// Scope: the Q/gate split, the two per-head RMS norms, and rope. Attention, the output gate and `wo`
// are NOT here yet (they are the next increment); this harness exists because the split and the norms
// are where a wrong reading is silent — a swapped q/gate half or a norm taken over the whole 6144-wide
// projection would still produce plausible numbers.
//
// Usage: attn_authority in.bin out.bin n_head head_dim n_kv S D n_rot eps sections... 
//   in.bin (f32, ggml order — ne0 fastest, so numpy reads each as its last axis):
//     W_qg   (n_head*2*head_dim, D)   the joint Q+gate projection
//     W_k    (n_kv*head_dim,    D)
//     norm_q (head_dim)
//     norm_k (head_dim)
//     hidden (S, D)
//   out.bin: stages concatenated as f32, in this order (counts derivable from the args):
//     qg (S, n_head*2*hd), q_pre (S, n_head, hd), gate_pre (S, n_head, hd),
//     q_norm (S, n_head, hd), q_rope (S, n_head, hd), k_norm (S, n_kv, hd), k_rope (S, n_kv, hd)
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void * xmalloc(size_t n) {
    void * p = malloc(n);
    if (!p) { fprintf(stderr, "ATTN_ERR oom\n"); exit(1); }
    return p;
}

int main(int argc, char ** argv) {
    if (argc != 14) {
        fprintf(stderr, "usage: %s in.bin out.bin n_head head_dim n_kv S D n_rot eps s0 s1 s2\n", argv[0]);
        return 2;
    }
    const int64_t nh   = atoll(argv[3]);
    const int64_t hd   = atoll(argv[4]);
    const int64_t nkv  = atoll(argv[5]);
    const int64_t S    = atoll(argv[6]);
    const int64_t D    = atoll(argv[7]);
    const int     nd   = atoi(argv[8]);
    const float   eps  = (float) atof(argv[9]);
    int sections[4] = { atoi(argv[10]), atoi(argv[11]), atoi(argv[12]), 0 };
    (void) sections;

    const int64_t nqg = nh * 2 * hd;
    const int64_t nk  = nkv * hd;

    // every f32 the fixture holds, in the order they are laid out in in.bin
    size_t n_float = (size_t) (nqg * D + nk * D + hd + hd + S * D);
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
    float * nmq = Wk + nk * D;
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

    // The splits are VIEWS: their ->data points into the parent with the parent's strides, so a linear
    // dump of a view reports the parent's layout. Copy them contiguous — that is what makes the two
    // split stages mean anything on their own.
    struct ggml_tensor * qc  = ggml_cont(ctx, q);
    struct ggml_tensor * gtc = ggml_cont(ctx, gt);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, qg);
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, gt);
    ggml_build_forward_expand(gf, qc);
    ggml_build_forward_expand(gf, gtc);
    ggml_build_forward_expand(gf, qn);
    ggml_build_forward_expand(gf, qr);
    ggml_build_forward_expand(gf, kn);
    ggml_build_forward_expand(gf, kr);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const struct { const char * nm; struct ggml_tensor * t; } stages[] = {
        { "qg", qg }, { "q_pre", qc }, { "gate_pre", gtc }, { "q_norm", qn },
        { "q_rope", qr }, { "k_norm", kn }, { "k_rope", kr },
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
    printf("ATTN_OK n_head=%lld head_dim=%lld n_kv=%lld S=%lld D=%lld n_dims=%d eps=%g\n",
           (long long) nh, (long long) hd, (long long) nkv, (long long) S, (long long) D, nd, (double) eps);
    ggml_free(ctx);
    free(fbuf);
    return 0;
}
