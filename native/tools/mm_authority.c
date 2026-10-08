// mm_authority.c — run ggml's OWN mul_mat and the linear-attention input gating, the fourth unit of
// the phase-4 layer wiring (VybForge #10, item 1).
//
// Why: the five projections of a qwen35 linear-attention layer (wqkv, wqkv_gate, ssm_beta, ssm_alpha,
// ssm_out) are "ordinary mul_mats", but no harness here calls ggml_mul_mat yet, and the layer's input
// path is not only a matmul — from src/models/qwen35.cpp (build_layer_attn_linear):
//
//   beta  = sigmoid(reshape(mul_mat(ssm_beta,  cur), 1, H_v, T, B))
//   alpha = mul_mat(ssm_alpha, cur)
//   alpha = softplus(alpha + ssm_dt) * ssm_a        // ssm_a is the GGUF's blk.%d.ssm_a, already -exp(A_log)
//
// so the conventions to pin are: which operand is the weight (ggml_mul_mat(w, x) = w^T x), the
// per-head bias broadcast (ssm_dt/ssm_a have ne0 = H_v and add/multiply across the head axis), and
// the two activation forms. They are NOT interchangeable with their obvious alternatives:
//
//   sigmoid (ggml/src/ggml-cpu/vec.h:936)   1.f/(1.f + expf(-x))
//   softplus (unary-ops.cpp op_softplus)    (x > 20.0f) ? x : logf(1.0f + expf(x))
//          -> a threshold at 20, and logf(1+expf(x)) rather than log1p(expf(x))
//
// Usage: mm_authority in.bin out.bin mm|beta|alpha n_col n_out n_tokens n_threads
//   in.bin  = w (n_col * n_out floats) — in GGML ORDER: the memory of a tensor of ne (n_col, n_out),
//             which is a numpy (n_out, n_col) array; getting this backwards silently transposes every
//             matmul (the trap the third unit's doc paragraph warns about, hit for real while writing
//             this harness and caught by the very "mis-read buffer" check the verifier prints).
//             x (n_col * n_tokens floats, same convention: numpy (n_tokens, n_col))
//             then, for mode=alpha only: dt (n_out floats), a (n_out floats)
//   out.bin = mm:    (n_out * n_tokens) floats, ggml ne = (n_out, n_tokens)
//             beta:  sigmoid of the above
//             alpha: (softplus(mm + dt) * a), same shape
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void * xmalloc(size_t n) {
    void * p = malloc(n);
    if (!p) { fprintf(stderr, "MM_ERR oom\n"); exit(1); }
    return p;
}
static void read_exact(const char * path, void * dst, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "MM_ERR open %s\n", path); exit(2); }
    size_t r = fread(dst, 1, n, f);
    fclose(f);
    if (r != n) { fprintf(stderr, "MM_ERR short read %zu != %zu\n", r, n); exit(3); }
}
static void write_exact(const char * path, const void * src, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "MM_ERR create %s\n", path); exit(2); }
    if (fwrite(src, 1, n, f) != n) { fprintf(stderr, "MM_ERR short write\n"); exit(3); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s in.bin out.bin mm|beta|alpha n_col n_out n_tokens n_threads\n", argv[0]);
        return 2;
    }
    const char * mode = argv[3];
    const int64_t n_col = atoll(argv[4]);
    const int64_t n_out = atoll(argv[5]);
    const int64_t n_tok = atoll(argv[6]);
    const int    nthr  = atoi(argv[7]);
    const int    is_alpha = strcmp(mode, "alpha") == 0;

    const size_t n_w = (size_t) n_col * n_out;
    const size_t n_x = (size_t) n_col * n_tok;
    const size_t n_b = is_alpha ? 2 * (size_t) n_out : 0;
    float * buf = (float *) xmalloc((n_w + n_x + n_b) * sizeof(float));
    read_exact(argv[1], buf, (n_w + n_x + n_b) * sizeof(float));

    struct ggml_init_params ip = { .mem_size = (size_t) 256 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "MM_ERR ggml_init\n"); return 4; }

    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_col, n_out);
    memcpy(w->data, buf, n_w * sizeof(float));
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_col, n_tok);
    memcpy(x->data, buf + n_w, n_x * sizeof(float));

    struct ggml_tensor * r = ggml_mul_mat(ctx, w, x);   // (n_out, n_tokens)
    if (strcmp(mode, "beta") == 0) {
        r = ggml_sigmoid(ctx, r);
    } else if (is_alpha) {
        struct ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_out);
        memcpy(dt->data, buf + n_w + n_x, n_out * sizeof(float));
        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_out);
        memcpy(a->data, buf + n_w + n_x + n_out, n_out * sizeof(float));
        r = ggml_add(ctx, r, dt);                       // per-head bias, broadcast over ne1/ne2
        r = ggml_softplus(ctx, r);
        r = ggml_mul(ctx, r, a);                        // per-head scale
    } else if (strcmp(mode, "mm") != 0) {
        fprintf(stderr, "MM_ERR unknown mode %s\n", mode);
        return 2;
    }
    if (!r) { fprintf(stderr, "MM_ERR op refused\n"); return 5; }
    ggml_set_output(r);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, nthr);

    write_exact(argv[2], r->data, (size_t) n_out * n_tok * sizeof(float));
    printf("MM_OK mode=%s n_col=%lld n_out=%lld n_tokens=%lld out_ne=(%lld,%lld,%lld,%lld)\n",
           mode, (long long) n_col, (long long) n_out, (long long) n_tok,
           (long long) r->ne[0], (long long) r->ne[1], (long long) r->ne[2], (long long) r->ne[3]);
    ggml_free(ctx);
    free(buf);
    return 0;
}
