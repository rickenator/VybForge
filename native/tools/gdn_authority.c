// gdn_authority.c — run ggml's OWN Gated DeltaNet op, the way llama.cpp's qwen35 builds it.
//
// Why a C harness at all: the recurrent layer's novel math lives inside GGML_OP_GATED_DELTA_NET,
// implemented in ggml-cpu/ops.cpp. llama.cpp's CPU default resolves fused_gdn_ar/fused_gdn_ch to
// true (llama-context.cpp:232-233, re-resolved against backend support at :561-562), so BOTH decode
// and prefill go through build_delta_net_fused -> ggml_gated_delta_net(..., K=1). This harness links
// the libggml that llama.cpp was built with and calls that same op, so it executes the identical
// implementation rather than a copy of it.
//
// Layout, taken from build_delta_net_fused (src/models/delta-net-base.cpp:401-420): with K=1 the
// op's single result buffer is the attention output [S_v, H_v, T, B] immediately followed by the
// final state [S_v, S_v, H_v, B].
//
// Usage: gdn_authority in.bin out.bin state_out.bin S_v H_k H_v T B n_threads
//   in.bin = q,k,v,gate,beta,state concatenated as float32, in that order, each in ggml's layout
//            (q,k: [S,H_k,T,B]; v: [S,H_v,T,B]; gate,beta: [1,H_v,T,B]; state: [S,S,H_v,B])
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void * xmalloc(size_t n) {
    void * p = malloc(n);
    if (!p) { fprintf(stderr, "GDA_ERR oom %zu\n", n); exit(1); }
    return p;
}

static void read_exact(const char * path, void * dst, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "GDA_ERR open %s\n", path); exit(2); }
    size_t r = fread(dst, 1, n, f);
    fclose(f);
    if (r != n) { fprintf(stderr, "GDA_ERR short read %s (%zu != %zu)\n", path, r, n); exit(3); }
}

static void write_exact(const char * path, const void * src, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "GDA_ERR create %s\n", path); exit(2); }
    if (fwrite(src, 1, n, f) != n) { fprintf(stderr, "GDA_ERR short write %s\n", path); exit(3); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 10) {
        fprintf(stderr, "usage: %s in.bin out.bin state_out.bin S_v H_k H_v T B n_threads\n", argv[0]);
        return 2;
    }
    const int64_t S      = atoll(argv[4]);
    const int64_t H_k    = atoll(argv[5]);
    const int64_t H_v    = atoll(argv[6]);
    const int64_t T      = atoll(argv[7]);
    const int64_t B      = atoll(argv[8]);
    const int     nthr   = atoi(argv[9]);

    if (S <= 0 || H_k <= 0 || H_v <= 0 || T <= 0 || B <= 0 || H_v % H_k != 0) {
        fprintf(stderr, "GDA_ERR bad dims\n");
        return 2;
    }

    const size_t n_q = (size_t) S * H_k * T * B;
    const size_t n_k = n_q;
    const size_t n_v = (size_t) S * H_v * T * B;
    const size_t n_g = (size_t) H_v * T * B;
    const size_t n_b = n_g;
    const size_t n_s = (size_t) S * S * H_v * B;
    const size_t total = n_q + n_k + n_v + n_g + n_b + n_s;

    float * buf = (float *) xmalloc(total * sizeof(float));
    read_exact(argv[1], buf, total * sizeof(float));

    struct ggml_init_params ip = { .mem_size = (size_t) 256 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "GDA_ERR ggml_init\n"); return 4; }

    size_t off = 0;
    struct ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, B);
    memcpy(q->data, buf + off, n_q * sizeof(float)); off += n_q;
    struct ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, B);
    memcpy(k->data, buf + off, n_k * sizeof(float)); off += n_k;
    struct ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_v, T, B);
    memcpy(v->data, buf + off, n_v * sizeof(float)); off += n_v;
    struct ggml_tensor * g  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, T, B);
    memcpy(g->data, buf + off, n_g * sizeof(float)); off += n_g;
    struct ggml_tensor * be = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, T, B);
    memcpy(be->data, buf + off, n_b * sizeof(float)); off += n_b;
    struct ggml_tensor * st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H_v, B);
    memcpy(st->data, buf + off, n_s * sizeof(float)); off += n_s;

    // TEMPORARY DEBUG: what the harness put into the k tensor vs what is actually in it. If these
    // two disagree, the copy (not the op) is at fault; if they agree and the op still prints
    // different numbers for k_d[0..3], the op's pointer arithmetic is not what the source reads as.
    if (getenv("GGML_GDN_DEBUG")) {
        const float * kint = buf + n_q;             // k is the second block in the file
        const float * kten = (const float *) k->data;
        fprintf(stderr, "GDN_HARNESS intended k[0..3]=%.9g %.9g %.9g %.9g\n",
                (double) kint[0], (double) kint[1], (double) kint[2], (double) kint[3]);
        fprintf(stderr, "GDN_HARNESS tensor   k[0..3]=%.9g %.9g %.9g %.9g   data=%p\n",
                (double) kten[0], (double) kten[1], (double) kten[2], (double) kten[3], (void *) kten);
    }

    struct ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, be, st, /*K=*/1);
    if (!result) { fprintf(stderr, "GDA_ERR op refused\n"); return 5; }
    ggml_set_output(result);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, nthr);

    write_exact(argv[2], result->data, n_v * sizeof(float));
    write_exact(argv[3], (const char *) result->data + n_v * sizeof(float), n_s * sizeof(float));

    printf("GDA_OK S=%lld Hk=%lld Hv=%lld T=%lld B=%lld out=%zu state=%zu\n",
           (long long) S, (long long) H_k, (long long) H_v, (long long) T, (long long) B, n_v, n_s);

    ggml_free(ctx);
    free(buf);
    return 0;
}
