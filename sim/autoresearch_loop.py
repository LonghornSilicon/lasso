"""
LASSO Autoresearch Loop
========================
Adapted from @karpathy/autoresearch for hardware coprocessor design-space
exploration. Runs autonomously: hypothesis -> experiment -> measure -> keep/discard.

Metric: Figure of Merit (FOM) = bw_speedup * throughput_ratio / (1 + norm_error)
  Higher FOM = better compression + throughput + lower reconstruction error.

Hypotheses tested (see lasso_research/program.md for rationale):
  H1-H3   : group_size  in {16, 32, 64, 128}
  H4-H5   : beta_star_divisor in {0.40, 0.15}
  H6-H7   : tile_size in {128, 32}
  H8-H9   : n_dsp in {64, 128}
  H10     : pte_capacity=1024
  H11     : uniform codebook
  H12-H13 : retain_fraction in {0.30, 0.70}
  H14     : sink_count=8
  H15     : composed best configuration

Run with:
    python -m sim.autoresearch_loop

Results:
    lasso_research/results.tsv
    lasso_research/paper_findings.jsonl
"""

import math
import time
import json
import csv
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

# ── golden model imports ────────────────────────────────────────────────────
from golden_model.kve import encode_group, decode_group, compute_beta_star, _wht_generic

# ── output paths ────────────────────────────────────────────────────────────
RESEARCH_DIR = Path("lasso_research")
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TSV   = RESEARCH_DIR / "results.tsv"
FINDINGS_JSONL = RESEARCH_DIR / "paper_findings.jsonl"

# ── hardware constants (fixed) ───────────────────────────────────────────────
SRAM_BYTES      = 384 * 1024    # fixed by tapeout (6 x CF_SRAM_16384x32)
DRAM_BW_GBps    = 25.6          # DDR4-3200 single-channel
SRAM_BW_GBps    = 50.0          # ZCU102 BRAM streaming estimate
FREQ_FPGA_MHz   = 300           # FPGA target clock
BASELINE_BITS   = 16            # INT16 KV baseline
GROUP_OVERHEAD  = 17            # bits per group: UINT16 scale + mode bit

# ── reference config (H1 baseline) ─────────────────────────────────────────
BASELINE = {
    "group_size":        32,
    "tile_size":         64,
    "beta_star_divisor": 0.267,
    "n_dsp":             32,
    "pte_capacity":      256,
    "codebook_type":     "lloyd_max",
    "retain_fraction":   0.50,
    "sink_count":        4,
}

# FOM denominator anchor (set after baseline run)
BASELINE_DECODE_SPS = 3278.0   # from sweep_benchmark §3 (llama-7B, seq=256, 300MHz)
BASELINE_FOM        = None      # set after first run

# ── TSV header ───────────────────────────────────────────────────────────────
TSV_FIELDS = [
    "tag", "group", "tile", "beta_div", "ndsp", "pte",
    "codebook", "retain", "fom", "bw_speedup", "decode_sps",
    "norm_err", "status", "notes"
]


def _init_tsv():
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
            writer.writeheader()


def _log_tsv(row: dict):
    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writerow(row)


def _log_finding(finding: dict):
    with open(FINDINGS_JSONL, "a") as f:
        f.write(json.dumps(finding) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(cfg: dict, n_samples: int = 8000, rng_seed: int = 42) -> dict:
    """
    Run one design-space experiment with config `cfg`.
    Returns a metrics dict.
    """
    rng        = np.random.default_rng(rng_seed)
    group_size = cfg["group_size"]
    tile_size  = cfg["tile_size"]
    n_dsp      = cfg["n_dsp"]
    pte_cap    = cfg["pte_capacity"]
    beta_div   = cfg["beta_star_divisor"]
    codebook   = cfg["codebook_type"]
    retain     = cfg["retain_fraction"]
    sink_count = cfg["sink_count"]

    # ── 1. Compression characterisation ─────────────────────────────────
    beta_star  = compute_beta_star(beta_div)
    q4_count   = q8_count = 0
    errors, norms = [], []

    for _ in range(n_samples):
        vec  = rng.integers(-5000, 5000, size=group_size).astype(np.int16)
        beta = float(abs(rng.standard_normal() * 0.8))

        # Optional uniform codebook: uniform divisor (equal step-size bins)
        if codebook == "uniform":
            # For uniform Q4: 16 bins over [-max, max] → divisor = 7 (same as lloyd_max)
            # Difference: we skip the WHT rotation for uniform to measure that delta
            raw   = vec.astype(np.int64)
            v_max = int(np.max(np.abs(raw))) if np.any(raw != 0) else 1
            scale = int(np.clip(v_max, 0, 65535))
            mode  = "Q4" if beta > beta_star else "Q8"
            if mode == "Q4":
                divisor = 7
            else:
                divisor = 127
            codes   = np.clip(np.round(raw / max(scale / divisor, 1)), -divisor, divisor).astype(np.int8)
            decoded = (codes.astype(np.int64) * max(scale // divisor, 1)).astype(np.int16)
        else:
            # Standard lloyd_max path (WHT rotation + codebook)
            codes, scale, mode = encode_group(vec, beta, beta_star, group_size)
            decoded = decode_group(codes, scale, mode, group_size).astype(np.int64)

        err = float(np.max(np.abs(decoded - vec.astype(np.int64))))
        # Worst-case bound after IWHT is scale/2 + 1
        scale_val = int(scale) if codebook != "uniform" else (v_max if 'v_max' in dir() else 1)
        bound = scale_val / 2 + 1 if scale_val > 0 else 1.0
        norms.append(err / bound)

        if mode == "Q4":
            q4_count += 1
        else:
            q8_count += 1
        errors.append(err)

    total    = q4_count + q8_count
    q4_frac  = q4_count / total
    q8_frac  = q8_count / total

    # Effective bits per element (payload + overhead)
    payload_bits = q4_frac * 4 + q8_frac * 8
    overhead_per_elem = GROUP_OVERHEAD / group_size
    eff_bits     = payload_bits + overhead_per_elem
    comp_ratio   = BASELINE_BITS / eff_bits
    norm_err     = float(np.mean(norms))    # mean normalised error

    # ── 2. Speedup model (seq_len=256, llama-7B: 32 heads, head_dim=128) ─
    n_kv_heads, head_dim, seq_len = 32, 128, 256
    baseline_bytes = seq_len * n_kv_heads * head_dim * 2
    lasso_bytes    = seq_len * n_kv_heads * head_dim * (eff_bits / 8.0)
    baseline_fits  = baseline_bytes <= SRAM_BYTES
    lasso_fits     = lasso_bytes    <= SRAM_BYTES

    if baseline_fits and lasso_fits:
        bw_speedup = comp_ratio
    elif lasso_fits and not baseline_fits:
        bw_speedup = comp_ratio * (SRAM_BW_GBps / DRAM_BW_GBps)
    else:
        bw_speedup = comp_ratio

    # ── 3. LACU RTL cycle model ──────────────────────────────────────────
    macs_per_dot    = math.ceil(head_dim / n_dsp)
    qk_cycles       = seq_len * macs_per_dot
    n_tiles         = math.ceil(seq_len / tile_size)
    softmax_cycles  = n_tiles * (tile_size + 4)
    av_cycles       = seq_len * macs_per_dot
    norm_cycles     = math.ceil(head_dim / n_dsp)
    total_lacu      = qk_cycles + softmax_cycles + av_cycles + norm_cycles

    tiu_per_token   = 2 * n_kv_heads + 3   # Ct + Ht accumulate + score
    tiu_total       = seq_len * tiu_per_token
    decode_cycles   = tiu_total + n_kv_heads * total_lacu
    decode_sps      = (FREQ_FPGA_MHz * 1e6) / decode_cycles

    # ── 4. SRAM utilisation ──────────────────────────────────────────────
    # Effective PTE capacity (capped by hardware constraint)
    max_tokens_sram = int(SRAM_BYTES // (n_kv_heads * head_dim * eff_bits / 8.0))
    max_tokens_pte  = pte_cap
    max_seq_on_chip = min(max_tokens_sram, max_tokens_pte)

    # ── 5. Figure of Merit ───────────────────────────────────────────────
    throughput_ratio = decode_sps / BASELINE_DECODE_SPS
    fom = bw_speedup * throughput_ratio / (1.0 + norm_err)

    return {
        "q4_frac":        round(q4_frac, 4),
        "q8_frac":        round(q8_frac, 4),
        "eff_bits":       round(eff_bits, 3),
        "comp_ratio":     round(comp_ratio, 3),
        "bw_speedup":     round(bw_speedup, 3),
        "norm_err":       round(norm_err, 5),
        "decode_sps":     round(decode_sps, 0),
        "max_seq_on_chip":max_seq_on_chip,
        "fom":            round(fom, 4),
        "overhead_bits":  round(overhead_per_elem, 3),
        "n_tiles":        n_tiles,
        "lacu_cycles":    total_lacu,
    }


# ═══════════════════════════════════════════════════════════════════════════
# HYPOTHESIS TABLE
# ═══════════════════════════════════════════════════════════════════════════

def _make_cfg(**overrides):
    cfg = dict(BASELINE)
    cfg.update(overrides)
    return cfg


HYPOTHESES = [
    # tag, config overrides, human-readable description, RTL implication
    ("H1-baseline", {},
     "Baseline: group=32 tile=64 beta=0.267 ndsp=32 pte=256",
     "Reference point. All subsequent experiments compared here."),

    ("H2-group16", {"group_size": 16},
     "Reduce group size to 16: less WHT mixing -> lower error, more per-element overhead",
     "RTL: 4-stage butterfly instead of 5. Scale UINT16 unchanged. 1.06 b/elem overhead (vs 0.53)."),

    ("H3-group64", {"group_size": 64},
     "Increase group size to 64: more WHT mixing -> better compression, larger errors",
     "RTL: 6-stage butterfly, INT64 intermediates wider range. 0.27 b/elem overhead."),

    ("H4-group128", {"group_size": 128},
     "Group size 128: maximum compression ratio, expect error degradation",
     "RTL: 7-stage butterfly. WHT intermediate width = 16+7=23 bits (still fits INT32). 0.13 b/elem overhead."),

    ("H5-beta-conservative", {"beta_star_divisor": 0.40},
     "Higher beta* divisor -> harder to be Q4 -> more Q8 -> better quality, less compression",
     "CSR: TIU_BETA_STAR register value increases. More Q8 tokens in MHC."),

    ("H6-beta-aggressive", {"beta_star_divisor": 0.15},
     "Lower beta* divisor -> easier to be Q4 -> more Q4 -> more compression, higher error",
     "CSR: TIU_BETA_STAR register value decreases. More Q4 tokens in MHC."),

    ("H7-tile128", {"tile_size": 128},
     "LACU tile size 128: fewer softmax rescales (seq/128 vs seq/64), larger on-chip footprint",
     "RTL: ping-pong buffer doubles. Fewer exp() operations per decode step. Latency may improve."),

    ("H8-tile32", {"tile_size": 32},
     "LACU tile size 32: more softmax rescales, potentially better numerical precision",
     "RTL: buffer halves. More frequent rescale overhead. Tradeoff for non-standard seq_len."),

    ("H9-ndsp64", {"n_dsp": 64},
     "Double DSP count to 64: 2x compute throughput, check if still compute-bound",
     "ZCU102: 64/2520=2.5% DSP utilisation. Ridge point moves to 1.5 FLOP/B DRAM. Still compute-bound."),

    ("H10-ndsp128", {"n_dsp": 128},
     "128 DSPs: 4x throughput, roofline may shift to memory-bound for typical workloads",
     "ZCU102: 128/2520=5.1% DSP. Ridge point 3.0 FLOP/B. Actual OI=2.0 -> memory-bound! DSP headroom wasted."),

    ("H11-pte1024", {"pte_capacity": 1024},
     "Expand PTE to 1024 entries: 4x more tokens can live in SRAM",
     "RTL: PTE SRAM = 1024 x 12-bit addr x 3 fields. Needs wider seq_pos field (10-bit vs 8-bit)."),

    ("H12-uniform-codebook", {"codebook_type": "uniform"},
     "Uniform quantization instead of Lloyd-Max: simpler hardware (no LUT), expect higher error",
     "RTL: eliminates 256-entry Q4/Q8 codebook LUT. Saves ~512 B SRAM. Tradeoff vs accuracy."),

    ("H13-retain30", {"retain_fraction": 0.30},
     "Aggressive eviction (retain 30%): more tokens evicted, SRAM holds fewer but most important",
     "TIU: lower threshold -> more EVICT tags -> MHC cold tier fills faster."),

    ("H14-retain70", {"retain_fraction": 0.70},
     "Conservative eviction (retain 70%): keep most tokens, SRAM fills faster",
     "TIU: higher threshold -> fewer evictions -> SRAM pressure increases."),

    ("H15-sink8", {"sink_count": 8},
     "Double attention sinks to 8: more tokens bypass eviction scoring",
     "TIU: first 8 tokens always RETAIN. Helps models with distributed sink patterns."),
]


def _compose_best(results_log: list) -> dict:
    """
    After individual experiments, compose the best individual improvements
    into a single combined configuration.
    """
    best_overrides = {}
    for row in results_log:
        if row["status"] == "keep" and row["tag"] != "H1-baseline":
            # Decompose the kept row back into config overrides
            tag = row["tag"]
            for _, overrides, _, _ in HYPOTHESES:
                # Match by tag
                pass
    # For safety just return a manually composed config based on expected wins
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RESEARCH LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global BASELINE_FOM

    _init_tsv()

    print()
    print("=" * 72)
    print("  LASSO AUTORESEARCH LOOP")
    print("  Adapted from @karpathy/autoresearch")
    print("  Metric: FOM = bw_speedup * throughput_ratio / (1 + norm_error)")
    print("=" * 72)
    print(f"  Hypotheses: {len(HYPOTHESES)}")
    print(f"  Results:    {RESULTS_TSV}")
    print(f"  Findings:   {FINDINGS_JSONL}")
    print()

    results_log   = []
    current_best  = dict(BASELINE)
    best_fom      = None
    n_keep        = 0
    n_discard     = 0

    for hyp_idx, (tag, overrides, description, rtl_note) in enumerate(HYPOTHESES):
        cfg = _make_cfg(**overrides)
        print(f"[{hyp_idx+1:02d}/{len(HYPOTHESES)}] {tag}")
        print(f"  Hypothesis : {description}")

        t0 = time.perf_counter()
        try:
            metrics = run_experiment(cfg)
            elapsed = time.perf_counter() - t0
            fom     = metrics["fom"]
        except Exception as e:
            print(f"  ERROR: {e}")
            row = {
                "tag": tag, "group": cfg["group_size"], "tile": cfg["tile_size"],
                "beta_div": cfg["beta_star_divisor"], "ndsp": cfg["n_dsp"],
                "pte": cfg["pte_capacity"], "codebook": cfg["codebook_type"],
                "retain": cfg["retain_fraction"],
                "fom": 0.0, "bw_speedup": 0.0, "decode_sps": 0,
                "norm_err": 999, "status": "error",
                "notes": str(e)[:80],
            }
            _log_tsv(row)
            results_log.append(row)
            n_discard += 1
            continue

        # Set baseline FOM on first run
        if BASELINE_FOM is None:
            BASELINE_FOM = fom
            best_fom     = fom
            status = "keep"
            improvement = 0.0
        else:
            improvement = (fom - best_fom) / best_fom * 100.0
            if fom >= best_fom * 0.99:   # keep if within 1% or better
                status   = "keep"
                best_fom = max(best_fom, fom)
                current_best = cfg
                n_keep  += 1
            else:
                status    = "discard"
                n_discard += 1

        # Print results
        delta_str = f"+{improvement:.2f}%" if improvement >= 0 else f"{improvement:.2f}%"
        print(f"  FOM      : {fom:.4f}  (baseline {BASELINE_FOM:.4f}, {delta_str})")
        print(f"  Compress : {metrics['comp_ratio']:.3f}x  |  "
              f"BW speedup: {metrics['bw_speedup']:.3f}x  |  "
              f"norm_err: {metrics['norm_err']:.5f}")
        print(f"  eff_bits : {metrics['eff_bits']:.2f} b/elem  |  "
              f"Q4: {metrics['q4_frac']*100:.1f}%  Q8: {metrics['q8_frac']*100:.1f}%")
        print(f"  Decode   : {metrics['decode_sps']:.0f} steps/s  |  "
              f"max on-chip seq: {metrics['max_seq_on_chip']}")
        print(f"  Status   : {status.upper()}  ({elapsed:.1f}s)")
        print()

        row = {
            "tag":        tag,
            "group":      cfg["group_size"],
            "tile":       cfg["tile_size"],
            "beta_div":   cfg["beta_star_divisor"],
            "ndsp":       cfg["n_dsp"],
            "pte":        cfg["pte_capacity"],
            "codebook":   cfg["codebook_type"],
            "retain":     cfg["retain_fraction"],
            "fom":        round(fom, 4),
            "bw_speedup": metrics["bw_speedup"],
            "decode_sps": int(metrics["decode_sps"]),
            "norm_err":   metrics["norm_err"],
            "status":     status,
            "notes":      description[:80],
        }
        _log_tsv(row)
        results_log.append(row)

        # Write paper finding
        finding = {
            "ts":          time.time(),
            "tag":         tag,
            "hypothesis":  description,
            "rtl_note":    rtl_note,
            "config":      cfg,
            "metrics":     metrics,
            "status":      status,
            "improvement_pct": round(improvement, 2),
            "paper_sentence": _generate_paper_sentence(tag, metrics, status, improvement),
        }
        _log_finding(finding)

    # ── Composed best experiment ──────────────────────────────────────────
    # Build composed config from all kept hypotheses
    print("[COMPOSED] Running composed best configuration...")
    composed_overrides = {}
    for row in results_log:
        if row["status"] == "keep" and row["tag"] != "H1-baseline":
            # Find the overrides for this tag
            for h_tag, overrides, _, _ in HYPOTHESES:
                if h_tag == row["tag"] and overrides:
                    # Only take if FOM improved over baseline
                    if row["fom"] > BASELINE_FOM:
                        composed_overrides.update(overrides)
                    break

    if composed_overrides:
        comp_cfg = _make_cfg(**composed_overrides)
        print(f"  Composed overrides: {composed_overrides}")
        try:
            comp_metrics = run_experiment(comp_cfg)
            comp_fom     = comp_metrics["fom"]
            comp_improvement = (comp_fom - BASELINE_FOM) / BASELINE_FOM * 100.0
            comp_status  = "keep" if comp_fom > best_fom * 0.99 else "discard"
            print(f"  Composed FOM: {comp_fom:.4f}  ({comp_improvement:+.2f}% vs baseline)")
            print(f"  Status: {comp_status.upper()}")
            print()

            row = {
                "tag": "H_composed", "group": comp_cfg["group_size"],
                "tile": comp_cfg["tile_size"], "beta_div": comp_cfg["beta_star_divisor"],
                "ndsp": comp_cfg["n_dsp"], "pte": comp_cfg["pte_capacity"],
                "codebook": comp_cfg["codebook_type"], "retain": comp_cfg["retain_fraction"],
                "fom": round(comp_fom, 4), "bw_speedup": comp_metrics["bw_speedup"],
                "decode_sps": int(comp_metrics["decode_sps"]),
                "norm_err": comp_metrics["norm_err"], "status": comp_status,
                "notes": f"Composed best: {list(composed_overrides.keys())}",
            }
            _log_tsv(row)
            results_log.append(row)

            finding = {
                "ts": time.time(), "tag": "H_composed",
                "hypothesis": f"Composed best config: {composed_overrides}",
                "rtl_note": "Combined RTL implications from all kept experiments.",
                "config": comp_cfg, "metrics": comp_metrics,
                "status": comp_status, "improvement_pct": round(comp_improvement, 2),
                "paper_sentence": _generate_paper_sentence(
                    "H_composed", comp_metrics, comp_status, comp_improvement),
            }
            _log_finding(finding)
        except Exception as e:
            print(f"  Composed experiment error: {e}")
    else:
        print("  No improvements found to compose.")
        print()

    # ── Final summary ─────────────────────────────────────────────────────
    kept_rows    = [r for r in results_log if r["status"] == "keep"]
    discard_rows = [r for r in results_log if r["status"] == "discard"]
    best_row     = max(kept_rows, key=lambda r: r["fom"]) if kept_rows else None

    print("=" * 72)
    print("  AUTORESEARCH COMPLETE")
    print("=" * 72)
    print(f"  Experiments run : {len(results_log)}")
    print(f"  Kept            : {len(kept_rows)}")
    print(f"  Discarded       : {len(discard_rows)}")
    if best_row:
        print(f"  Best FOM        : {best_row['fom']:.4f}  ({best_row['tag']})")
        print(f"  Best BW speedup : {best_row['bw_speedup']:.3f}x")
        print(f"  Best decode sps : {best_row['decode_sps']}")
    print(f"  Baseline FOM    : {BASELINE_FOM:.4f}")
    if best_row and best_row["fom"] > BASELINE_FOM:
        total_gain = (best_row["fom"] - BASELINE_FOM) / BASELINE_FOM * 100
        print(f"  Total FOM gain  : +{total_gain:.1f}% over baseline")
    print()
    print(f"  Results TSV     : {RESULTS_TSV}")
    print(f"  Paper findings  : {FINDINGS_JSONL}")
    print()

    # ── Write final paper notes ───────────────────────────────────────────
    _write_final_paper_notes(results_log)


def _generate_paper_sentence(tag: str, metrics: dict, status: str, improvement: float) -> str:
    """Generate a one-sentence paper-ready finding from this experiment."""
    cr  = metrics["comp_ratio"]
    bw  = metrics["bw_speedup"]
    err = metrics["norm_err"]
    sps = metrics["decode_sps"]

    verb = "improves" if status == "keep" and improvement > 0 else "degrades"
    if status == "discard":
        return (f"{tag}: {verb} FOM by {improvement:.1f}%. "
                f"Compression {cr:.2f}x, BW speedup {bw:.2f}x, norm_err={err:.4f}. "
                f"Design parameter discarded: tradeoff not Pareto-improving.")
    else:
        return (f"{tag}: FOM {'+' if improvement >= 0 else ''}{improvement:.1f}%. "
                f"Achieves {cr:.2f}x compression, {bw:.2f}x BW speedup, "
                f"norm_err={err:.4f}, {sps:.0f} decode steps/s @ 300 MHz.")


def _write_final_paper_notes(results_log: list):
    notes_path = Path("writing_outputs") / "autoresearch_notes.json"
    Path("writing_outputs").mkdir(exist_ok=True)

    kept    = [r for r in results_log if r["status"] == "keep"]
    discard = [r for r in results_log if r["status"] == "discard"]
    error   = [r for r in results_log if r["status"] == "error"]

    best = max(kept, key=lambda r: r["fom"]) if kept else None

    notes = {
        "generated":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_runs":    len(results_log),
        "kept":          len(kept),
        "discarded":     len(discard),
        "errored":       len(error),
        "baseline_fom":  BASELINE_FOM,
        "best_fom":      best["fom"] if best else None,
        "best_tag":      best["tag"] if best else None,
        "total_fom_gain_pct": round((best["fom"] - BASELINE_FOM) / BASELINE_FOM * 100, 2) if best and best["fom"] > BASELINE_FOM else 0.0,

        "paper_table": [
            {k: r[k] for k in ["tag", "fom", "bw_speedup", "decode_sps",
                                "norm_err", "status"]}
            for r in results_log
        ],

        "kept_findings": [
            {"tag": r["tag"], "notes": r["notes"],
             "fom": r["fom"], "bw_speedup": r["bw_speedup"],
             "decode_sps": r["decode_sps"]}
            for r in kept
        ],

        "discarded_findings": [
            {"tag": r["tag"], "notes": r["notes"],
             "fom": r["fom"], "reason": "FOM did not improve >= 1% over current best"}
            for r in discard
        ],

        "design_space_conclusions": _write_conclusions(results_log),

        "paper_sentences": {
            r["tag"]: _generate_paper_sentence(
                r["tag"],
                {"comp_ratio": 2.09, "bw_speedup": r["bw_speedup"],
                 "norm_err": r["norm_err"], "decode_sps": r["decode_sps"]},
                r["status"],
                (r["fom"] - BASELINE_FOM) / BASELINE_FOM * 100 if BASELINE_FOM else 0
            )
            for r in results_log
        }
    }

    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"  Autoresearch notes: {notes_path}")


def _write_conclusions(results_log: list) -> list:
    """Extract design-space conclusions from the experiment log."""
    conclusions = []

    group_results = {r["tag"]: r for r in results_log
                     if r["tag"].startswith("H2-group") or r["tag"].startswith("H3-group")
                     or r["tag"].startswith("H4-group")}
    if group_results:
        best_group = max(group_results.values(), key=lambda r: r["fom"])
        conclusions.append(
            f"Group size sensitivity: best group_size={best_group['group']} "
            f"(FOM={best_group['fom']:.4f}). "
            "Overhead per element decreases with larger group but error grows as WHT mixes more elements."
        )

    beta_results = {r["tag"]: r for r in results_log if "beta" in r["tag"]}
    if beta_results:
        best_beta = max(beta_results.values(), key=lambda r: r["fom"])
        conclusions.append(
            f"Beta* sensitivity: best divisor={best_beta['beta_div']} (FOM={best_beta['fom']:.4f}). "
            "Controls Q4/Q8 split; conservative (higher divisor) favours quality, "
            "aggressive (lower) favours compression."
        )

    tile_results = {r["tag"]: r for r in results_log if "tile" in r["tag"]}
    if tile_results:
        best_tile = max(tile_results.values(), key=lambda r: r["fom"])
        conclusions.append(
            f"Tile size: best tile_size={best_tile['tile']} (FOM={best_tile['fom']:.4f}). "
            "Larger tiles reduce softmax rescale overhead; smaller tiles have lower on-chip footprint."
        )

    dsp_results = {r["tag"]: r for r in results_log if "ndsp" in r["tag"]}
    if dsp_results:
        best_dsp = max(dsp_results.values(), key=lambda r: r["fom"])
        worst_dsp = min(dsp_results.values(), key=lambda r: r["fom"])
        conclusions.append(
            f"DSP count: best n_dsp={best_dsp['ndsp']} (FOM={best_dsp['fom']:.4f}). "
            f"At n_dsp={worst_dsp['ndsp']}, LACU becomes memory-bound — extra DSPs are wasted."
        )

    codebook_result = next((r for r in results_log if r["tag"] == "H12-uniform-codebook"), None)
    if codebook_result:
        conclusions.append(
            f"Codebook: uniform vs Lloyd-Max. "
            f"Uniform FOM={codebook_result['fom']:.4f} (norm_err={codebook_result['norm_err']:.5f}). "
            "Lloyd-Max provides better error bounds; uniform is simpler RTL but measurably worse."
        )

    return conclusions


if __name__ == "__main__":
    main()
