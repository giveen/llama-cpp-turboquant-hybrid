# llama-cpp-turboquant-hybrid

Hybrid speculative decoding for llama.cpp: composes **MTP** (Multi-Token Prediction) and **DFlash** (Block Diffusion) draft paths into a single decoder pass.

Current state: **working end-to-end on this fork**. Target loads, hybrid init composes DFlash + MTP impls, and the server returns completions with draft stats (`draft_n`, `draft_n_accepted`).

## What this is

This branch adds a new speculative type:

- `draft-hybrid-mtp-dflash` — runs DFlash first, then extends with MTP, and presents the combined draft block to the target verifier.
- Provenance tracking per token so acceptance updates each backend’s stats independently.
- New CLI knobs:
  - `--hybrid-df-depth N` — max DFlash tree depth used in the composed draft
  - `--hybrid-mtp-length N` — how many MTP extension tokens to append

## Repo layout

- `common/speculative.cpp` — hybrid composer impl + speculative init changes
- `common/arg.cpp` / `common/common.h` — CLI wiring and params
- `src/llama-context.cpp` — debug guard adjustment for multi-token draft batches
- `docs/HYBRID-MTP-DFLASH-DESIGN.md` — design notes and sequence diagram
- `test-models/` — local test artifacts, ignored by git

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build -j --target llama-server
```

## Run

```bash
./build/bin/llama-server \
  -m test-models/target/Qwen3.5-4B-Q4_K_M.gguf \
  -md test-models/dflash-gguf/Qwen3.5-4B-DFlash.Q8_0.gguf \
  --spec-type draft-hybrid-mtp-dflash \
  --spec-draft-n-max 8 \
  --hybrid-df-depth 4 \
  --hybrid-mtp-length 3 \
  -c 32768 -ngl 999 --jinja --port 18080 --no-webui -np 1 -bs
```

Notes:
- This repo does **not** ship model files. Download a DFlash draft GGUF that matches your target family separately.
- The `-bs` / backend-sampling path is required for the current multi-token draft batch shape in this fork.

## Known limitations

- The backend sampling guard in `src/llama-context.cpp` is still stricter than upstream; the hybrid path needs `-bs` or an equivalent batch-output path.
- DFlash draft heads are model-family-specific. A Qwen3.5 DFlash head will not work with a Gemma target.
- Benchmarks so far are partial. Default and MTP runs are stable. DFlash server path was unstable in this environment. Hybrid is proven functional but needs cleaner comparison runs.

## Next steps

- Tighten the backend sampling guard to the configured draft max instead of a fixed cap.
- Add a minimal standalone harness under `examples/` so hybrid can be validated without the server.
- Rebase onto latest upstream llama.cpp and minimize local patches.
