# Vyb executable

`bin/vyb-configurator` is a Linux Vyb executable. It talks only to the local
Ollama listener at `127.0.0.1:11434`; it has no Hermes dependency and never
invokes VybOS tooling. It is a portable CPU fallback, not the recommended
interviewer runtime: use GPU-backed Ollama or the hosted backends in
`./run.sh` for a useful session.

Run it with:

```bash
./run-vyb.sh
```

It defaults to the CPU-friendlier `qwen3:4b`. Select the 8B model when useful:

```bash
VYBAICONF_MODEL=qwen3:8b ./run-vyb.sh
```

Human prompts go to stderr. Each model result is emitted on stdout as one
Ollama JSON response envelope; the nested `message.content` is instructed to
be a VybOS configurator JSON object. The executable does not build, apply, or
modify configurations.
