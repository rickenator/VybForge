// conv_authority.c — run ggml's OWN short-convolution op (GGML_OP_SSM_CONV), the causal depthwise
// conv a qwen35 recurrent layer applies to q||k||v over all 10240 channels.
//
// Why: the convolution is the other novel op in the layer, and it carries state between tokens (the
// last d_conv-1 frames). As with the delta rule, the authority is the real op from the libggml this
// llama.cpp was built with, not a copy of the maths.
//
// Shapes, from ggml.c:5559-5572 —
//   sx: 3D (d_conv-1+n_t, d_inner, n_seqs)      c: matrix (d_conv, d_inner)
//   d_conv = c->ne[0], n_t = sx->ne[0] - d_conv + 1, result = (d_inner, n_t, n_seqs)
//
// Usage: conv_authority in.bin out.bin d_conv d_inner n_t n_threads
//   in.bin = s (ncs * d_inner floats, ggml order) then c (d_conv * d_inner floats)
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void * xmalloc(size_t n) {
    void * p = malloc(n);
    if (!p) { fprintf(stderr, "CONV_ERR oom\n"); exit(1); }
    return p;
}
static void read_exact(const char * path, void * dst, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "CONV_ERR open %s\n", path); exit(2); }
    size_t r = fread(dst, 1, n, f);
    fclose(f);
    if (r != n) { fprintf(stderr, "CONV_ERR short read %zu != %zu\n", r, n); exit(3); }
}
static void write_exact(const char * path, const void * src, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "CONV_ERR create %s\n", path); exit(2); }
    if (fwrite(src, 1, n, f) != n) { fprintf(stderr, "CONV_ERR short write\n"); exit(3); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s in.bin out.bin d_conv d_inner n_t n_threads\n", argv[0]);
        return 2;
    }
    const int64_t d_conv  = atoll(argv[3]);
    const int64_t d_inner = atoll(argv[4]);
    const int64_t n_t     = atoll(argv[5]);
    const int     nthr    = atoi(argv[6]);
    const int64_t ncs     = d_conv - 1 + n_t;

    const size_t n_s = (size_t) ncs * d_inner;
    const size_t n_c = (size_t) d_conv * d_inner;
    float * buf = (float *) xmalloc((n_s + n_c) * sizeof(float));
    read_exact(argv[1], buf, (n_s + n_c) * sizeof(float));

    struct ggml_init_params ip = { .mem_size = (size_t) 64 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "CONV_ERR ggml_init\n"); return 4; }

    struct ggml_tensor * sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncs, d_inner, 1);
    memcpy(sx->data, buf, n_s * sizeof(float));
    struct ggml_tensor * cw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner);
    memcpy(cw->data, buf + n_s, n_c * sizeof(float));

    struct ggml_tensor * r = ggml_ssm_conv(ctx, sx, cw);
    if (!r) { fprintf(stderr, "CONV_ERR op refused\n"); return 5; }
    ggml_set_output(r);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, nthr);

    write_exact(argv[2], r->data, (size_t) d_inner * n_t * sizeof(float));
    printf("CONV_OK d_conv=%lld d_inner=%lld n_t=%lld ncs=%lld\n",
           (long long) d_conv, (long long) d_inner, (long long) n_t, (long long) ncs);
    ggml_free(ctx);
    free(buf);
    return 0;
}
