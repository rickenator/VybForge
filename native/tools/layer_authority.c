// layer_authority.c — build ONE qwen35 linear-attention block out of ggml's own ops and run it, the
// fifth unit of the phase-4 layer wiring (VybForge #10, item 1).
//
// The four earlier units verified the pieces in isolation against the same libggml llama.cpp runs
// (the recurrence, the short convolution, the two norms + epilogue, the projections and gates). This
// one verifies the WIRING: that our numpy reference assembles those pieces in the order, shapes and
// residual structure src/models/qwen35.cpp uses. Per-stage dumps are written so a divergence names
// the stage that is wrong instead of "the layer".
//
// Wiring transcribed from qwen35.cpp (build_layer_attn_linear + the block loop) for a recurrent
// layer, with the block's input `x` = inpL:
//
//   xn   = RMSNorm(x, attn_norm)                      (build_norm, LLM_NORM_RMS)
//   qkv  = mul_mat(wqkv,      xn)                     [qkv_dim, T, B]
//   z    = mul_mat(wqkv_gate, xn)                     [value_dim, T, B]
//   beta = sigmoid(mul_mat(ssm_beta,  xn))            [1, H_v, T, B]
//   gate = softplus(mul_mat(ssm_alpha, xn) + ssm_dt) * ssm_a   [1, H_v, T, B]
//   conv = silu(ssm_conv(concat(window, qkv), ssm_conv1d))
//   q/k/v = views of conv at channel offsets 0, key_dim, 2*key_dim
//   q,k  = l2_norm(q), l2_norm(k)
//   out  = gated_delta_net(q, k, v, gate, beta, state)          (K=1: out || new state)
//   y    = mul_mat(ssm_out, reshape(RMSNorm(out, ssm_norm) * SiLU(z), [value_dim, T, B]))
//   x_out = x + y                                    (the block's attn residual)
//
// The window (conv state) and the delta-net state come from the input file, so this authority can run
// any STEP of a sequence: zero for the first token, or the state carried out of the previous step.
// T is still 1 — the op's multi-token kernel is not characterised — but a sequence is a chain of
// single-token steps, which is exactly what decode does.
//
// Usage: layer_authority in.bin outdir n_embd head_k_dim n_k_heads n_v_heads d_conv T B eps n_threads
//   in.bin, in this order, each in ggml ORDER (the memory of a tensor whose ne is given, which is a
//   numpy array of the reversed shape):
//     x             (T, n_embd)
//     attn_norm     (n_embd)
//     wqkv          (qkv_dim, n_embd)        qkv_dim = 2*head_k_dim*n_k_heads + head_v_dim*n_v_heads
//     wqkv_gate     (value_dim, n_embd)
//     ssm_beta      (H_v, n_embd)
//     ssm_alpha     (H_v, n_embd)
//     ssm_dt        (H_v)
//     ssm_a         (H_v)
//     ssm_conv1d    (qkv_dim, d_conv)
//     ssm_norm      (head_v_dim)
//     ssm_out       (n_embd, value_dim)
//   outdir receives one <name>.bin per stage, same ggml-order convention.
#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void * xmalloc(size_t n) {
    void * p = malloc(n);
    if (!p) { fprintf(stderr, "LA_ERR oom %zu\n", n); exit(1); }
    return p;
}
static void write_exact(const char * path, const void * src, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "LA_ERR create %s\n", path); exit(2); }
    if (fwrite(src, 1, n, f) != n) { fprintf(stderr, "LA_ERR short write %s\n", path); exit(3); }
    fclose(f);
}

static char g_outdir[4096];

// Stages are collected while the graph is built and written after the compute, so the writing never
// depends on allocation order. A view is materialised with ggml_cont when it is not contiguous —
// ggml_cont preserves ne, so its memory is exactly what a numpy array of the reversed shape holds.
#define MAX_STAGES 32
static struct { const char * name; struct ggml_tensor * t; } g_stages[MAX_STAGES];
static int g_nstages = 0;

static void stage(struct ggml_context * ctx, struct ggml_cgraph * gf,
                  struct ggml_tensor * t, const char * name) {
    if (!ggml_is_contiguous(t)) {
        t = ggml_cont(ctx, t);                  // pending node: must be expanded into the graph
        ggml_build_forward_expand(gf, t);
    }
    if (g_nstages == MAX_STAGES) { fprintf(stderr, "LA_ERR too many stages\n"); exit(2); }
    g_stages[g_nstages].name = name;
    g_stages[g_nstages].t = t;
    g_nstages++;
}

static void write_stages(void) {
    for (int i = 0; i < g_nstages; i++) {
        char path[4200];
        snprintf(path, sizeof(path), "%s/%s.bin", g_outdir, g_stages[i].name);
        struct ggml_tensor * t = g_stages[i].t;
        write_exact(path, t->data, (size_t) ggml_nbytes(t));
        fprintf(stderr, "LA_STAGE %-12s ne=(%lld,%lld,%lld,%lld) bytes=%zu\n",
                g_stages[i].name, (long long) t->ne[0], (long long) t->ne[1],
                (long long) t->ne[2], (long long) t->ne[3], (size_t) ggml_nbytes(t));
    }
}

// RMSNorm along ne0 with a per-channel weight (build_norm, LLM_NORM_RMS).
static struct ggml_tensor * rms_norm_w(struct ggml_context * ctx, struct ggml_tensor * x,
                                       struct ggml_tensor * w, float eps) {
    struct ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, w);
}

int main(int argc, char ** argv) {
    if (argc != 12) {
        fprintf(stderr,
                "usage: %s in.bin outdir n_embd head_k_dim n_k_heads n_v_heads d_conv T B eps n_threads\n",
                argv[0]);
        return 2;
    }
    snprintf(g_outdir, sizeof(g_outdir), "%s", argv[2]);
    const int64_t n_embd    = atoll(argv[3]);
    const int64_t head_dim  = atoll(argv[4]);
    const int64_t H_k       = atoll(argv[5]);
    const int64_t H_v       = atoll(argv[6]);
    const int64_t d_conv    = atoll(argv[7]);
    const int64_t T         = atoll(argv[8]);
    const int64_t B         = atoll(argv[9]);
    const float   eps       = strtof(argv[10], NULL);
    const int     nthr      = atoi(argv[11]);

    const int64_t key_dim   = head_dim * H_k;
    const int64_t value_dim = head_dim * H_v;
    const int64_t qkv_dim   = 2 * key_dim + value_dim;

    if (T != 1) {
        fprintf(stderr, "LA_ERR T must be 1: the multi-token kernel is not characterised\n");
        return 2;
    }

    // ---- inputs ------------------------------------------------------------------------------
    const size_t n_x   = (size_t) T * n_embd;
    const size_t n_xn  = (size_t) n_embd;
    const size_t n_wq  = (size_t) qkv_dim * n_embd;
    const size_t n_wg  = (size_t) value_dim * n_embd;
    const size_t n_wa  = (size_t) H_v * n_embd;
    const size_t n_hv  = (size_t) H_v;
    const size_t n_cv  = (size_t) qkv_dim * d_conv;
    const size_t n_hd  = (size_t) head_dim;
    const size_t n_wo  = (size_t) n_embd * value_dim;
    const size_t n_win = (size_t) qkv_dim * (d_conv - 1);
    const size_t n_st  = (size_t) head_dim * head_dim * H_v * B;
    const size_t total = n_x + n_xn + n_wq + n_wg + 2 * n_wa + 2 * n_hv + n_cv + n_hd + n_wo
                       + n_win + n_st;

    struct ggml_init_params ip = { .mem_size = (size_t) 1024 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "LA_ERR ggml_init\n"); return 4; }

    // The window/state blocks are OPTIONAL: a file without them is a first-token step (both zero),
    // which keeps the older fixtures valid (unit 5 writes no window/state).
    const size_t n_tail = n_win + n_st;
    const size_t n_base = total - n_tail;
    float * buf = (float *) xmalloc(total * sizeof(float));
    FILE * f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "LA_ERR open %s\n", argv[1]); return 2; }
    if (fread(buf, sizeof(float), n_base, f) != n_base) { fprintf(stderr, "LA_ERR short read\n"); return 3; }
    memset(buf + n_base, 0, n_tail * sizeof(float));
    size_t got = fread(buf + n_base, sizeof(float), n_tail, f);   // short read => zeros, i.e. a fresh step
    (void) got;
    fclose(f);

    size_t o = 0;
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T);
    memcpy(x->data, buf + o, n_x * sizeof(float)); o += n_x;
    struct ggml_tensor * attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    memcpy(attn_norm->data, buf + o, n_xn * sizeof(float)); o += n_xn;
    struct ggml_tensor * wqkv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, qkv_dim);
    memcpy(wqkv->data, buf + o, n_wq * sizeof(float)); o += n_wq;
    struct ggml_tensor * wgate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, value_dim);
    memcpy(wgate->data, buf + o, n_wg * sizeof(float)); o += n_wg;
    struct ggml_tensor * wbeta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, H_v);
    memcpy(wbeta->data, buf + o, n_wa * sizeof(float)); o += n_wa;
    struct ggml_tensor * walpha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, H_v);
    memcpy(walpha->data, buf + o, n_wa * sizeof(float)); o += n_wa;
    struct ggml_tensor * ssm_dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v);
    memcpy(ssm_dt->data, buf + o, n_hv * sizeof(float)); o += n_hv;
    struct ggml_tensor * ssm_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v);
    memcpy(ssm_a->data, buf + o, n_hv * sizeof(float)); o += n_hv;
    struct ggml_tensor * conv1d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, qkv_dim);
    memcpy(conv1d->data, buf + o, n_cv * sizeof(float)); o += n_cv;
    struct ggml_tensor * ssm_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
    memcpy(ssm_norm->data, buf + o, n_hd * sizeof(float)); o += n_hd;
    struct ggml_tensor * ssm_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, value_dim, n_embd);
    memcpy(ssm_out->data, buf + o, n_wo * sizeof(float)); o += n_wo;

    // ---- the layer, in qwen35.cpp's order ------------------------------------------------------
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    struct ggml_tensor * xn = rms_norm_w(ctx, x, attn_norm, eps);
    stage(ctx, gf, xn, "xn");

    struct ggml_tensor * qkv = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wqkv, xn), qkv_dim, T, B);
    struct ggml_tensor * z   = ggml_mul_mat(ctx, wgate, xn);
    stage(ctx, gf, qkv, "qkv");
    stage(ctx, gf, z, "z");

    struct ggml_tensor * beta = ggml_sigmoid(ctx, ggml_reshape_4d(
            ctx, ggml_mul_mat(ctx, wbeta, xn), 1, H_v, T, B));
    stage(ctx, gf, beta, "beta");

    struct ggml_tensor * alpha = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, walpha, xn), H_v, T, B);
    alpha = ggml_add(ctx, alpha, ssm_dt);
    alpha = ggml_softplus(ctx, alpha);
    alpha = ggml_mul(ctx, alpha, ssm_a);
    struct ggml_tensor * gate = ggml_reshape_4d(ctx, alpha, 1, H_v, T, B);
    stage(ctx, gf, gate, "gate");

    // conv input, exactly as build_conv_state does it: the (d_conv-1)-frame window (zero for a fresh
    // sequence) concatenated along ne0 with the tokens. qkv is (qkv_dim, T, B), so it is transposed
    // to (T, qkv_dim, B) first and the concat gives (ncs, qkv_dim, B) — which is what ggml_ssm_conv
    // takes. Built as ops, so it is computed inside the graph rather than read before qkv exists.
    struct ggml_tensor * conv_win = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_conv - 1, qkv_dim, B);
    memcpy(conv_win->data, buf + o, n_win * sizeof(float)); o += n_win;
    struct ggml_tensor * qkv_t = ggml_transpose(ctx, qkv);
    struct ggml_tensor * conv_in = ggml_concat(ctx, conv_win, qkv_t, 0);
    stage(ctx, gf, conv_in, "conv_in");

    struct ggml_tensor * conv = ggml_ssm_conv(ctx, conv_in, conv1d);
    struct ggml_tensor * conv_silu = ggml_silu(ctx, conv);
    stage(ctx, gf, conv_silu, "conv_silu");

    // the q|k|v split: the model's ggml_view_4d over the conv output's channel axis.
    const size_t nb1_qkv = (size_t) qkv_dim * sizeof(float);
    struct ggml_tensor * q_conv = ggml_view_4d(ctx, conv_silu, head_dim, H_k, T, B,
            (size_t) head_dim * sizeof(float), nb1_qkv, nb1_qkv * T, 0);
    struct ggml_tensor * k_conv = ggml_view_4d(ctx, conv_silu, head_dim, H_k, T, B,
            (size_t) head_dim * sizeof(float), nb1_qkv, nb1_qkv * T,
            (size_t) key_dim * sizeof(float));
    struct ggml_tensor * v_conv = ggml_view_4d(ctx, conv_silu, head_dim, H_v, T, B,
            (size_t) head_dim * sizeof(float), nb1_qkv, nb1_qkv * T,
            (size_t) 2 * key_dim * sizeof(float));
    stage(ctx, gf, v_conv, "v_conv");

    struct ggml_tensor * q_n = ggml_l2_norm(ctx, q_conv, eps);
    struct ggml_tensor * k_n = ggml_l2_norm(ctx, k_conv, eps);
    stage(ctx, gf, q_n, "q_norm");
    stage(ctx, gf, k_n, "k_norm");

    // the delta-net state, zero for the first token of a sequence.
    struct ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, head_dim, H_v, B);
    memcpy(state->data, buf + o, n_st * sizeof(float)); o += n_st;

    struct ggml_tensor * gdn = ggml_gated_delta_net(ctx, q_n, k_n, v_conv, gate, beta, state, /*K=*/1);
    if (!gdn) { fprintf(stderr, "LA_ERR gated_delta_net refused\n"); return 5; }
    // K=1: the result buffer is the output [S_v,H_v,T,B] followed by the new state [S_v,S_v,H_v,B].
    struct ggml_tensor * gdn_out = ggml_view_4d(ctx, gdn, head_dim, H_v, T, B,
            (size_t) head_dim * sizeof(float), (size_t) head_dim * H_v * sizeof(float),
            (size_t) head_dim * H_v * T * sizeof(float), 0);
    stage(ctx, gf, gdn_out, "gdn_out");
    // K=1: out [S_v,H_v,T,B] is immediately followed by the new state [S_v,S_v,H_v,B] in one buffer.
    struct ggml_tensor * state_out = ggml_view_4d(ctx, gdn, head_dim, head_dim, H_v, B,
            (size_t) head_dim * sizeof(float), (size_t) head_dim * head_dim * sizeof(float),
            (size_t) head_dim * head_dim * H_v * sizeof(float),
            (size_t) head_dim * H_v * T * B * sizeof(float));
    stage(ctx, gf, state_out, "state_out");

    struct ggml_tensor * z2d = ggml_reshape_4d(ctx, z, head_dim, H_v, T, B);
    struct ggml_tensor * epi = ggml_mul(ctx, rms_norm_w(ctx, gdn_out, ssm_norm, eps), ggml_silu(ctx, z2d));
    stage(ctx, gf, epi, "epi");

    struct ggml_tensor * final_out = ggml_reshape_3d(ctx, epi, value_dim, T, B);
    struct ggml_tensor * y = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, ssm_out, final_out), n_embd, T * B);
    stage(ctx, gf, y, "y");

    struct ggml_tensor * layer_out = ggml_add(ctx, y, x);
    stage(ctx, gf, layer_out, "layer_out");

    // one graph, one pass; every stage above is an ancestor of layer_out or already expanded.
    ggml_build_forward_expand(gf, layer_out);
    ggml_graph_compute_with_ctx(ctx, gf, nthr);

    write_stages();

    printf("LA_OK n_embd=%lld head_dim=%lld Hk=%lld Hv=%lld d_conv=%lld T=%lld B=%lld eps=%g\n",
           (long long) n_embd, (long long) head_dim, (long long) H_k, (long long) H_v,
           (long long) d_conv, (long long) T, (long long) B, eps);
    ggml_free(ctx);
    free(buf);
    return 0;
}
