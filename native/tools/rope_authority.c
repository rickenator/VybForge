// rope_authority.c — run ggml's OWN MRoPE op for a qwen35 attention layer (phase 4, unit 10).
//
// Why: a Ridge attention layer rotates only n_dims = 64 of each 256-dim head, through
// `ggml_rope_multi` with `rope.dimension_sections = [11,11,10,0]`. The engine's `qwen3rope` kernel
// rotates ALL head dims with pairs (i, i + HD/2), so it cannot be pointed at this case — but exactly
// WHICH layout the op uses is not something to reconstruct from source (the pairing, the offset and
// the frequency progression interact; `rotate_pairs`'s `ic = i0/scale` plus its `n_offset` argument
// has an off-by-one waiting in it). This harness is the authority instead: it links the same libggml
// llama.cpp runs and asks the real op.
//
// Usage: rope_authority in.bin out.bin head_dim n_head n_tokens n_dims mode s0 s1 s2 s3
//   in.bin  = q (head_dim * n_head * n_tokens F32, GGML ORDER: ne = (head_dim, n_head, n_tokens),
//              i.e. numpy (n_tokens, n_head, head_dim))
//             then pos (n_tokens * 4 INT32) — the four MRoPE positions per token; for a TEXT model
//             all four are the same token position, which is the case under test.
//   out.bin = the rotated q (same shape/order)
//   mode    = imrope | mrope | neox
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void * xmalloc(size_t n) {
    void * p = malloc(n);
    if (!p) { fprintf(stderr, "ROPE_ERR oom\n"); exit(1); }
    return p;
}
static void read_exact(const char * path, void * dst, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ROPE_ERR open %s\n", path); exit(2); }
    size_t r = fread(dst, 1, n, f);
    fclose(f);
    if (r != n) { fprintf(stderr, "ROPE_ERR short read %zu != %zu\n", r, n); exit(3); }
}
static void write_exact(const char * path, const void * src, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ROPE_ERR create %s\n", path); exit(2); }
    if (fwrite(src, 1, n, f) != n) { fprintf(stderr, "ROPE_ERR short write\n"); exit(3); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 13) {
        fprintf(stderr, "usage: %s in.bin out.bin head_dim n_head n_tokens n_dims mode s0 s1 s2 s3 n_pos\n", argv[0]);
        return 2;
    }
    const int64_t hd   = atoll(argv[3]);
    const int64_t nh   = atoll(argv[4]);
    const int64_t ntok = atoll(argv[5]);
    const int     nd   = atoi(argv[6]);
    const char *  mode = argv[7];
    int sections[4] = { atoi(argv[8]), atoi(argv[9]), atoi(argv[10]), atoi(argv[11]) };
    // NEOX wants ONE position per token; MRoPE/IMROPE want four planes (t, h, w, e). The op
    // asserts on the length, so the caller states it rather than the harness guessing.
    const int64_t n_pos = atoll(argv[12]);

    const size_t n_q = (size_t) hd * nh * ntok;
    const size_t n_p = (size_t) n_pos;
    float *   qbuf = (float *) xmalloc(n_q * sizeof(float));
    int32_t * pos  = (int32_t *) xmalloc(n_p * sizeof(int32_t));
    read_exact(argv[1], qbuf, n_q * sizeof(float));
    {
        FILE * f = fopen(argv[1], "rb");
        if (!f) { fprintf(stderr, "ROPE_ERR open %s\n", argv[1]); return 2; }
        if (fseek(f, (long) (n_q * sizeof(float)), SEEK_SET) != 0) { return 2; }
        if (fread(pos, sizeof(int32_t), n_p, f) != n_p) { fprintf(stderr, "ROPE_ERR short pos\n"); return 3; }
        fclose(f);
    }

    int mode_id = 0;
    if      (strcmp(mode, "imrope") == 0) mode_id = GGML_ROPE_TYPE_IMROPE;
    else if (strcmp(mode, "mrope")  == 0) mode_id = GGML_ROPE_TYPE_MROPE;
    else if (strcmp(mode, "neox")   == 0) mode_id = GGML_ROPE_TYPE_NEOX;
    else { fprintf(stderr, "ROPE_ERR unknown mode %s\n", mode); return 2; }

    struct ggml_init_params ip = { .mem_size = (size_t) 256 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ROPE_ERR ggml_init\n"); return 4; }

    struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, nh, ntok);
    memcpy(q->data, qbuf, n_q * sizeof(float));
    struct ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos);
    memcpy(p->data, pos, n_p * sizeof(int32_t));

    // The parameters a plain (non-YaRN, non-LongRoPE) text run uses: freq_scale 1, ext_factor 0,
    // attn_factor 1, beta_fast/slow at their defaults.
    struct ggml_tensor * r = ggml_rope_multi(ctx, q, p, NULL, nd, sections, mode_id,
                                             /*n_ctx_orig*/ 40960, /*freq_base*/ 1e7f, /*freq_scale*/ 1.0f,
                                             /*ext_factor*/ 0.0f, /*attn_factor*/ 1.0f,
                                             /*beta_fast*/ 32.0f, /*beta_slow*/ 1.0f);
    if (!r) { fprintf(stderr, "ROPE_ERR op refused\n"); return 5; }
    ggml_set_output(r);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    write_exact(argv[2], r->data, n_q * sizeof(float));
    printf("ROPE_OK mode=%s hd=%lld nh=%lld ntok=%lld n_dims=%d sections=%d,%d,%d,%d out_ne=(%lld,%lld,%lld,%lld)\n",
           mode, (long long) hd, (long long) nh, (long long) ntok, nd,
           sections[0], sections[1], sections[2], sections[3],
           (long long) r->ne[0], (long long) r->ne[1], (long long) r->ne[2], (long long) r->ne[3]);
    ggml_free(ctx);
    free(qbuf); free(pos);
    return 0;
}
