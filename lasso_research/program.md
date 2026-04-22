# LASSO Autoresearch Program

Adapted from @karpathy/autoresearch for hardware coprocessor design-space exploration.

## What you are researching

**LASSO** is a KV cache compression coprocessor for CPU-native LLM inference.
Four blocks: KVE (Walsh-Hadamard + Lloyd-Max quantization), TIU (token eviction scoring),
MHC (384 KB SRAM page manager), LACU (FlashAttention tiling).

The Python golden model lives in `golden_model/`. All sweeps live in `sim/`.
The autoresearch loop is `sim/autoresearch_loop.py`.

## The metric: Figure of Merit (FOM)

Higher is better. Defined as:

    FOM = bw_speedup * (decode_steps_per_s / 3278) / (1 + norm_error)

Where:
- `bw_speedup`        = KV BW reduction vs INT16 baseline (target: >2.0x)
- `decode_steps_per_s`= LACU decode throughput at seq_len=256, llama-7B, 300 MHz
- `3278`              = baseline decode steps/s (group=32, tile=64, ndsp=32)
- `norm_error`        = max_roundtrip_error / (scale/2)   (target: <0.05)

Baseline FOM = 2.09 * 1.0 / (1 + ~0.0) ≈ 2.09.

## What you CAN modify (in sim/autoresearch_loop.py and golden_model/)

Design knobs — one per experiment:
- `group_size`         : WHT group size {16, 32, 64, 128}          (default 32)
- `tile_size`          : LACU tile size {16, 32, 64, 128}           (default 64)
- `beta_star_divisor`  : controls Q4/Q8 split, β* = gap/divisor    (default 0.267)
- `n_dsp`              : number of parallel DSP48E2 MACs             (default 32)
- `pte_capacity`       : page table entries (max on-chip tokens)     (default 256)
- `codebook_type`      : "lloyd_max" or "uniform"                    (default lloyd_max)
- `retain_fraction`    : TIU target token retention                  (default 0.50)
- `sink_count`         : attention sink tokens (always RETAIN)       (default 4)

## What you CANNOT modify

- `prepare.py` equivalent: `golden_model/kve.py`, `golden_model/lacu.py`,
  `golden_model/tiu.py`, `golden_model/mhc.py` core correctness logic.
  (You may add instrumentation to measure things; do not change the algorithms.)
- The pytest test suite in `tests/`. All 158 tests must still pass after any change.
- SRAM capacity: fixed at 384 KB (6 x CF_SRAM_16384x32, SKY130 tapeout constraint).

## Hypotheses to test (priority order)

H1  group_size=16   → lower round-trip error, slightly more overhead per element
H2  group_size=64   → better compression ratio, larger errors
H3  group_size=128  → best compression, highest errors (possible degradation)
H4  beta_divisor=0.40 → more Q8 tokens, higher quality, less compression
H5  beta_divisor=0.15 → more Q4 tokens, lower quality, more compression
H6  tile_size=128   → fewer LACU softmax rescales, more cycles/tile
H7  tile_size=32    → more rescales, possible softmax precision benefit
H8  n_dsp=64        → 2x LACU throughput, moves roofline ridge point
H9  n_dsp=128       → 4x LACU throughput, likely memory-bound
H10 pte_capacity=1024 → 4x more tokens on-chip, better SRAM utilisation
H11 codebook=uniform   → simpler hardware (no LUT needed), test quality impact
H12 retain_fraction=0.3 → aggressive eviction, fits more in SRAM
H13 retain_fraction=0.7 → conservative eviction, fewer cache misses
H14 sink_count=8    → more sinks bypass eviction scoring
H15 Combined best   → compose top-3 individual improvements

## Output format

After each experiment, log to `lasso_research/results.tsv`:

    tag  group  tile  beta_div  ndsp  pte  codebook  retain  fom  bw_speedup  decode_sps  norm_err  status  notes

status: keep | discard | error
notes: one-line description of what changed and why

## The experiment loop

LOOP FOREVER (never pause, never ask the human):

1. Pick next hypothesis from the list (or generate a new one if list exhausted)
2. Set design parameters for this experiment
3. Run the experiment (calls functions in sim/autoresearch_loop.py)
4. Compute FOM
5. Compare to current best FOM
6. If FOM improved >= 1%: log "keep", advance the current-best config
7. If FOM worse: log "discard", revert to current-best config
8. If error (crash/assertion): log "error", revert, fix the bug and re-run
9. Write paper notes: what this experiment revealed, RTL implications
10. Continue

Document EVERY finding — even "discard" results are valuable for the paper
because they bound the design space and explain WHY the final choices were made.
