// norm_authority.c — run ggml's OWN normalisation ops and the gated RMS epilogue, the third unit of
// the phase-4 layer wiring (VybForge #10, item 1).
//
// Why: the l2 normalisation on q/k and the gated RMS epilogue on the recurrence output are the last
// two pieces of a recurrent layer before the projections. They are ordinary ops, but "ordinary" is a
// summary, not evidence, and the two variants of the l2 norm differ by an ulp plus a convention:
//
//   l2  (ggml/src/ggml-cpu/ops.cpp, GGML_OP_L2_NORM):  scale = 1.0f/fmaxf(sqrtf(sum), eps)
//       eps is a FLOOR applied AFTER the sqrt.
//   rms (ops.cpp ggml_compute_forward_rms_norm_f32):   scale = 1.0f/sqrtf(mean + eps)
//       eps is INSIDE the sqrt.
//
// Both accumulate in double, both square in f32 first, both take the reciprocal in f32. The model's
// gated epilogue is exactly `RMSNorm(output, ssm_norm) * SiLU(z)` (qwen35.cpp build_norm_gated), so
// mode `epilogue` builds that composition out of the same ops the model's graph uses.
//
// Usage: norm_authority in.bin out.bin mode eps n_col n_rows n_threads
//   mode = l2 | rms | epilogue
//   in.bin  = x     (n_col * n_rows floats, ggml order: ne0 = n_col contiguous, per row)
//             then, for mode=epilogue only:
//             w    (n_col floats)
//             gate (n_col * n_rows floats, same layout as x)
//   Shapes: the tensor is 2D (ne0 = n_col, ne1 = n_rows); both ops normalise along ne0 only, and
//   every trailing dim of the model's tensor is just another row here.
#include "ggml.h"
#include "ggml-cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void * xmalloc(size_t n) {
    void * p = malloc(n);
    if (!p) { fprintf(stderr, "NORM_ERR oom\n"); exit(1); }
    return p;
}
static void read_exact(const char * path, void * dst, size_t n) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "NORM_ERR open %s\n", path); exit(2); }
    size_t r = fread(dst, 1, n, f);
    fclose(f);
    if (r != n) { fprintf(stderr, "NORM_ERR short read %zu != %zu\n", r, n); exit(3); }
}
static void write_exact(const char * path, const void * src, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "NORM_ERR create %s\n", path); exit(2); }
    if (fwrite(src, 1, n, f) != n) { fprintf(stderr, "NORM_ERR short write\n"); exit(3); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: %s in.bin out.bin l2|rms|epilogue eps n_col n_rows n_threads\n", argv[0]);
        return 2;
    }
    const char * mode = argv[3];
    const float  eps  = strtof(argv[4], NULL);
    const int64_t n_col = atoll(argv[5]);
    const int64_t n_rows = atoll(argv[6]);
    const int     nthr = atoi(argv[7]);
    const int     is_epilogue = strcmp(mode, "epilogue") == 0;

    const size_t n_x = (size_t) n_col * n_rows;
    const size_t n_w = is_epilogue ? (size_t) n_col : 0;
    const size_t n_g = is_epilogue ? n_x : 0;
    float * buf = (float *) xmalloc((n_x + n_w + n_g) * sizeof(float));
    read_exact(argv[1], buf, (n_x + n_w + n_g) * sizeof(float));

    struct ggml_init_params ip = { .mem_size = (size_t) 64 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "NORM_ERR ggml_init\n"); return 4; }

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_col, n_rows);
    memcpy(x->data, buf, n_x * sizeof(float));

    struct ggml_tensor * r = NULL;
    if (strcmp(mode, "l2") == 0) {
        r = ggml_l2_norm(ctx, x, eps);
    } else if (strcmp(mode, "rms") == 0) {
        r = ggml_rms_norm(ctx, x, eps);
    } else if (is_epilogue) {
        // exactly build_norm_gated: RMSNorm(input, weights) * SiLU(gate)
        struct ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_col);
        memcpy(w->data, buf + n_x, n_w * sizeof(float));
        struct ggml_tensor * gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_col, n_rows);
        memcpy(gate->data, buf + n_x + n_w, n_g * sizeof(float));

        struct ggml_tensor * normalized = ggml_rms_norm(ctx, x, eps);
        normalized = ggml_mul(ctx, normalized, w);
        struct ggml_tensor * gated_silu = ggml_silu(ctx, gate);
        r = ggml_mul(ctx, normalized, gated_silu);
    } else {
        fprintf(stderr, "NORM_ERR unknown mode %s\n", mode);
        return 2;
    }
    if (!r) { fprintf(stderr, "NORM_ERR op refused\n"); return 5; }
    ggml_set_output(r);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, nthr);

    write_exact(argv[2], r->data, n_x * sizeof(float));
    printf("NORM_OK mode=%s eps=%g n_col=%lld n_rows=%lld\n",
           mode, eps, (long long) n_col, (long long) n_rows);
    ggml_free(ctx);
    free(buf);
    return 0;
}
