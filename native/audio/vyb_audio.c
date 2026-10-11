// vyb_audio.c — minimal, header-free audio I/O for the Vyb voice loop (L1).
//
// Why this shape: the audio dev headers (libasound2-dev / libpipewire-0.3-dev)
// are not installed on godzilla, and privilege escalation from an agent session
// is not available. libasound.so.2 IS present at runtime, and pipewire-alsa
// routes it to PipeWire (so `default` sees the BT earbuds and the USB audio
// device alike). So we dlopen ALSA and call it through function pointers we
// declare ourselves: no headers, no link-time dependency, nothing privileged.
//
// The ABI facts hand-written here are the stable ALSA ones:
//   snd_pcm_stream_t : PLAYBACK=0, CAPTURE=1
//   snd_pcm_format_t : FLOAT_LE=14
//   snd_pcm_access_t : RW_INTERLEAVED=3
//   snd_pcm_uframes_t = unsigned long
// snd_strerror gives readable text for every error number, so a wrong
// assumption shows up as a message, never as a silently wrong buffer.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef struct snd_pcm snd_pcm_t;
typedef unsigned long va_frames;

#define VA_CHUNK 1024u

typedef int (*fn_open)(snd_pcm_t **, const char *, int, int);
typedef int (*fn_close)(snd_pcm_t *);
typedef int (*fn_set_params)(snd_pcm_t *, int, int, unsigned, unsigned, int, unsigned);
typedef long (*fn_readi)(snd_pcm_t *, void *, va_frames);
typedef long (*fn_writei)(snd_pcm_t *, const void *, va_frames);
typedef int (*fn_prepare)(snd_pcm_t *);
typedef int (*fn_drain)(snd_pcm_t *);
typedef int (*fn_recover)(snd_pcm_t *, int, int);
typedef const char *(*fn_strerror)(int);

static void *g_lib;
static fn_open p_open;
static fn_close p_close;
static fn_set_params p_set_params;
static fn_readi p_readi;
static fn_writei p_writei;
static fn_prepare p_prepare;
static fn_drain p_drain;
static fn_recover p_recover;
static fn_strerror p_strerror;

static char g_err[512] = "ok";

const char *va_lasterr(void) { return g_err; }

int va_init(void)
{
    if (g_lib) return 0;
    g_lib = dlopen("libasound.so.2", RTLD_NOW | RTLD_LOCAL);
    if (!g_lib) {
        snprintf(g_err, sizeof g_err, "dlopen libasound.so.2: %s", dlerror());
        return -1;
    }
#define BIND(name)                                                     \
    do {                                                               \
        *(void **)(&p_##name) = dlsym(g_lib, "snd_pcm_" #name);         \
        if (!p_##name) {                                                \
            snprintf(g_err, sizeof g_err, "missing snd_pcm_" #name);    \
            return -2;                                                  \
        }                                                               \
    } while (0)
    BIND(open); BIND(close); BIND(set_params); BIND(readi); BIND(writei);
    BIND(prepare); BIND(drain); BIND(recover);
#undef BIND
    *(void **)(&p_strerror) = dlsym(g_lib, "snd_strerror");
    snprintf(g_err, sizeof g_err, "ok");
    return 0;
}

const char *va_error(int rc)
{
    static char buf[256];
    if (!p_strerror) {
        snprintf(buf, sizeof buf, "alsa error %d (no strerror)", rc);
        return buf;
    }
    const char *s = p_strerror(rc);
    return s ? s : "unknown";
}

// device: ALSA PCM name, e.g. "default" (-> PipeWire) or "plughw:1,0".
// Returns an opaque handle, or NULL (va_lasterr() says why).
void *va_open(const char *device, unsigned rate, unsigned channels, int capture)
{
    if (va_init() != 0) return NULL;
    snd_pcm_t *h = NULL;
    int r = p_open(&h, device, capture ? 1 : 0, 0);
    if (r < 0) {
        snprintf(g_err, sizeof g_err, "snd_pcm_open(%s): %s", device, va_error(r));
        return NULL;
    }
    // format FLOAT_LE(14), access RW_INTERLEAVED(3), soft resample ON,
    // 200 ms latency budget.
    r = p_set_params(h, 14, 3, channels, rate, 1, 200000);
    if (r < 0) {
        snprintf(g_err, sizeof g_err, "snd_pcm_set_params(%u Hz, %u ch): %s",
                 rate, channels, va_error(r));
        p_close(h); return NULL;
    }
    r = p_prepare(h);
    if (r < 0) {
        snprintf(g_err, sizeof g_err, "snd_pcm_prepare: %s", va_error(r));
        p_close(h); return NULL;
    }
    return h;
}

// Staging buffer for the f32 <-> f64 boundary. Vyb's Float is float64 (CDouble)
// and the compiler has no CFloat->Float cast, so the dtype widening happens here,
// one chunk at a time. Single-threaded use per device is the contract: the voice
// loop gives each open device exactly one reader/writer thread.
static float g_stage[VA_CHUNK];

// Frames actually read (>=0), or -1 on an unrecoverable error.
int va_read(void *handle, double *buf, unsigned frames)
{
    snd_pcm_t *h = (snd_pcm_t *)handle;
    unsigned done = 0;
    while (done < frames) {
        unsigned want = frames - done;
        if (want > VA_CHUNK) want = VA_CHUNK;
        long got = p_readi(h, g_stage, (va_frames)want);
        if (got < 0) {
            int r = p_recover(h, (int)got, 1 /*silent*/);
            if (r < 0) {
                snprintf(g_err, sizeof g_err, "snd_pcm_readi: %s", va_error((int)got));
                return done ? (int)done : -1;
            }
            p_prepare(h);
            break;
        }
        for (long k = 0; k < got; k++) buf[done + k] = (double)g_stage[k];
        done += (unsigned)got;
        if (got < (long)want) break;
    }
    return (int)done;
}

// Frames actually written (>=0), or -1 on an unrecoverable error.
int va_write(void *handle, const double *buf, unsigned frames)
{
    snd_pcm_t *h = (snd_pcm_t *)handle;
    unsigned done = 0;
    while (done < frames) {
        unsigned want = frames - done;
        if (want > VA_CHUNK) want = VA_CHUNK;
        for (unsigned k = 0; k < want; k++) {
            double v = buf[done + k];
            if (v > 1.0) v = 1.0;
            if (v < -1.0) v = -1.0;
            g_stage[k] = (float)v;
        }
        long put = p_writei(h, g_stage, (va_frames)want);
        if (put < 0) {
            int r = p_recover(h, (int)put, 1);
            if (r < 0) {
                snprintf(g_err, sizeof g_err, "snd_pcm_writei: %s", va_error((int)put));
                return done ? (int)done : -1;
            }
            break;
        }
        done += (unsigned)put;
        if (put < (long)want) break;
    }
    return (int)done;
}

void va_close(void *handle, int drain)
{
    snd_pcm_t *h = (snd_pcm_t *)handle;
    if (!h) return;
    if (drain && p_drain) p_drain(h);
    p_close(h);
}