// vyb_audio.h — curated FFI surface for the Vyb voice loop's audio I/O (L1).
//
// Mirrors vyb_audio.c. Kept deliberately small: the module exposes start/stop of
// a device plus block reads/writes of float32 frames, and nothing else. The
// library is a shim we compile ourselves (see build.sh) — it dlopens ALSA at
// runtime, so this posting needs no ALSA dev headers and no link dependency.
#ifndef VYB_AUDIO_H
#define VYB_AUDIO_H

/* Load libasound and resolve the symbols we use. 0 on success. */
int va_init(void);

/* Text of the failure from the last va_* call that failed. Never NULL.
   Declared in the Vyb binding as loc<CChar>; read it with strlen +
   String::from_bytes (the CString view is not a readable raw pointer). */
const char *va_lasterr(void);

/* ALSA error number -> text (borrowed pointer into libasound). */
const char *va_error(int rc);

/* Open `device` ("default" routes through PipeWire, so it follows the system
   default source/sink — BT earbuds included). capture=1 opens an input.
   Returns an opaque handle, or NULL (va_lasterr() says why).

   No int* out-params anywhere in this surface, on purpose: a scalar out-param
   (loc<CInt>) inside a `freedom` block in an imported module miscompiled into a
   bad pointer (segfault inside snd_pcm_open), while the same code at file scope
   worked. Errors travel through va_lasterr() instead. */
void *va_open(const char *device, unsigned rate, unsigned channels, int capture);

/* Read up to `frames` mono frames into `buf` (float64: Vyb's Float is CDouble
   and there is no CFloat->Float cast, so the f32 device data is widened here).
   Returns the count read (a short read is normal), or -1 (va_lasterr()). */
int va_read(void *h, double *buf, unsigned frames);

/* Write `frames` mono frames from `buf` (float64; clamps to [-1,1] and narrows).
   Returns the count written, or -1. */
int va_write(void *h, const double *buf, unsigned frames);

/* Close the device. drain=1 flushes queued playback first. */
void va_close(void *h, int drain);

#endif /* VYB_AUDIO_H */
