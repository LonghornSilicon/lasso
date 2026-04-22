"""
MHC (Memory Hierarchy Controller) simulation sweep.

Tests:
  1. Page table exhaustion sweep
  2. Tier fill progression sweep
  3. Bank address boundary sweep
  4. Concurrent access simulation
  5. Precision round-trip sweep
  6. SRAM capacity verification
"""

import numpy as np
from sim.logger import SimLogger

from golden_model.mhc import MHC, PageTable, SRAMModel, _PAGE_TABLE_SIZE

BLOCK = "MHC"


def _sweep_page_table_exhaustion(logger: SimLogger, rng: np.random.Generator):
    """Sweep 1: Page table exhaustion."""
    test = "page_table_exhaustion"

    mhc = MHC()

    # Fill all 256 entries
    for i in range(256):
        mhc.write_kv(i, packed_kv=0x12345678, scale=100, mode="Q4", tag="RETAIN")

    # Attempt 257th write (new seq_pos=256 — but seq_pos is 8-bit capped at 255)
    # Actually the page table uses seq_pos as key; to test exhaustion we need a NEW seq_pos.
    # All 256 seq_pos slots (0..255) are filled. Trying to write a new one should overflow.
    # Since seq_pos is 8-bit in PTE but PageTable uses it as dict key with no inherent range:
    # Write seq_pos=256 (outside 8-bit range but PageTable doesn't enforce 8-bit on key)
    try:
        mhc.write_kv(256, packed_kv=0xDEADBEEF, scale=50, mode="Q4", tag="RETAIN")
        logger.log(
            BLOCK, test, "FAIL",
            {"entries": 256, "attempt": "257th"},
            "No OverflowError raised when page table full — overflow not detected",
            severity="FAIL",
        )
    except OverflowError as e:
        logger.log(
            BLOCK, test, "OVERFLOW_DETECTED",
            {"entries": 256, "attempt": "257th"},
            f"Correctly raised OverflowError on 257th write: {e}",
            severity="PASS",
        )

    # Evict token 0, then attempt write again — slot should be reused
    # Note: evict marks as evicted but does NOT remove from page table dict,
    # so a new seq_pos=256 will still see 256 entries. We must check if
    # writing to an already-evicted seq_pos (0) succeeds (it overwrites).
    mhc.evict(0)
    # Try writing back to seq_pos=0 (already in table, just evicted — overwrite allowed)
    try:
        mhc.write_kv(0, packed_kv=0xCAFEBABE, scale=75, mode="Q8", tag="RETAIN")
        logger.log(
            BLOCK, test, "SLOT_REUSED",
            {"seq_pos": 0},
            "After eviction, write to seq_pos=0 succeeds (slot overwritten in page table)",
            severity="PASS",
        )
    except Exception as e:
        logger.log(
            BLOCK, test, "FAIL",
            {"seq_pos": 0},
            f"Write to evicted seq_pos=0 raised {type(e).__name__}: {e}",
            severity="FAIL",
        )


def _sweep_tier_fill_progression(logger: SimLogger, rng: np.random.Generator):
    """Sweep 2: Tier fill progression."""
    test = "tier_fill_progression"

    mhc = MHC(hot_thresh=128)  # first 128 tokens go hot
    hot_full_n = None

    for n in range(256):
        mhc.write_kv(n, packed_kv=n, scale=1, mode="Q4", tag="RETAIN")
        hot_pct = mhc.page_table.hot_fill_pct()
        cold_pct = mhc.page_table.cold_fill_pct()

        if hot_pct > 1.0:
            logger.log(
                BLOCK, test, "WARN",
                {"n": n, "hot_fill_pct": round(hot_pct, 4)},
                f"hot_fill_pct={hot_pct:.4f} > 100% at N={n}",
                severity="WARN",
            )
        if cold_pct > 1.0:
            logger.log(
                BLOCK, test, "WARN",
                {"n": n, "cold_fill_pct": round(cold_pct, 4)},
                f"cold_fill_pct={cold_pct:.4f} > 100% at N={n}",
                severity="WARN",
            )

        # Detect when hot tier first reaches full (hot_thresh tokens written)
        if hot_full_n is None and hot_pct >= (128 / 256):
            hot_full_n = n

    if hot_full_n is not None:
        logger.log(
            BLOCK, test, "EDGE",
            {"hot_full_at_n": hot_full_n, "hot_thresh": 128},
            f"Hot tier reached hot_thresh={128} (50% of page table) at N={hot_full_n}",
            severity="EDGE",
        )
    else:
        logger.log(
            BLOCK, test, "WARN",
            {"hot_thresh": 128},
            "Hot tier never appeared full in sweep of 256 tokens",
            severity="WARN",
        )


def _sweep_bank_address_boundary(logger: SimLogger, rng: np.random.Generator):
    """Sweep 3: Bank address boundary."""
    test = "bank_address_boundary"

    mhc = MHC()
    n_tokens = 256

    fail_count = 0
    for i in range(n_tokens):
        mhc.write_kv(i, packed_kv=i * 100, scale=i % 128, mode="Q4", tag="RETAIN")

    for i in range(n_tokens):
        pte = mhc.page_table.lookup(i)
        bank_sel_ok = 0 <= pte.bank_sel <= 5
        bank_row_ok = 0 <= pte.bank_row <= 16383

        if not bank_sel_ok or not bank_row_ok:
            fail_count += 1
            logger.log(
                BLOCK, test, "FAIL",
                {"seq_pos": i, "bank_sel": pte.bank_sel, "bank_row": pte.bank_row},
                f"Out-of-range address: bank_sel={pte.bank_sel} (valid 0-5), "
                f"bank_row={pte.bank_row} (valid 0-16383)",
                severity="FAIL",
            )

    if fail_count == 0:
        logger.log(
            BLOCK, test, "OK",
            {"n_tokens": n_tokens},
            f"All {n_tokens} tokens have valid bank_sel [0,5] and bank_row [0,16383]",
            severity="PASS",
        )


def _sweep_concurrent_access(logger: SimLogger, rng: np.random.Generator):
    """Sweep 4: Concurrent access simulation (interleaved write/read)."""
    test = "concurrent_access"

    mhc = MHC()
    seq_pos = 5

    # Write token 5 (first write)
    kv1 = 0xAAAABBBB
    scale1 = 100
    mhc.write_kv(seq_pos, packed_kv=kv1, scale=scale1, mode="Q4", tag="RETAIN")

    # Read token 5
    packed_kv_r1, scale_r1, mode_r1 = mhc.read_kv(seq_pos)

    # Write token 5 again (overwrite)
    kv2 = 0xCCCCDDDD
    scale2 = 200
    mhc.write_kv(seq_pos, packed_kv=kv2, scale=scale2, mode="Q8", tag="RETAIN")

    # Read token 5 again — should return second write's data
    packed_kv_r2, scale_r2, mode_r2 = mhc.read_kv(seq_pos)

    # The write stores: word0 = (packed_kv & 0xFFFF0000) | (scale & 0xFFFF)
    # word1 = packed_kv & 0xFFFFFFFF
    # read returns: scale = word0 & 0xFFFF, packed_kv = word1
    expected_packed_kv2 = kv2 & 0xFFFFFFFF
    expected_scale2 = scale2 & 0xFFFF
    expected_mode2 = "Q8"

    data_match = (packed_kv_r2 == expected_packed_kv2 and
                  scale_r2 == expected_scale2 and
                  mode_r2 == expected_mode2)

    if data_match:
        logger.log(
            BLOCK, test, "OK",
            {"seq_pos": seq_pos,
             "expected_packed_kv": hex(expected_packed_kv2),
             "got_packed_kv": hex(packed_kv_r2),
             "expected_scale": expected_scale2, "got_scale": scale_r2},
            "Second read returns second write's data correctly",
            severity="PASS",
        )
    else:
        logger.log(
            BLOCK, test, "FAIL",
            {"seq_pos": seq_pos,
             "expected_packed_kv": hex(expected_packed_kv2),
             "got_packed_kv": hex(packed_kv_r2),
             "expected_scale": expected_scale2, "got_scale": scale_r2,
             "expected_mode": expected_mode2, "got_mode": mode_r2},
            "Data mismatch: second read does not return second write's data",
            severity="FAIL",
        )


def _sweep_precision_round_trip(logger: SimLogger, rng: np.random.Generator):
    """Sweep 5: Precision round-trip sweep."""
    test = "precision_round_trip"

    for precision in ["Q4", "Q8", "bypass"]:
        mhc = MHC()
        seq_pos = 42
        mhc.write_kv(seq_pos, packed_kv=0x12345678, scale=256, mode=precision, tag="RETAIN")

        pte = mhc.page_table.lookup(seq_pos)
        if pte.precision == precision:
            logger.log(
                BLOCK, test, "OK",
                {"precision": precision, "seq_pos": seq_pos},
                f"PTE precision field matches written mode '{precision}'",
                severity="PASS",
            )
        else:
            logger.log(
                BLOCK, test, "FAIL",
                {"precision": precision, "pte_precision": pte.precision, "seq_pos": seq_pos},
                f"PTE precision mismatch: wrote '{precision}', read '{pte.precision}'",
                severity="FAIL",
            )


def _sweep_sram_capacity(logger: SimLogger, rng: np.random.Generator):
    """Sweep 6: SRAM capacity verification."""
    test = "sram_capacity"

    # Compute: 6 banks × 16384 rows × 4 bytes = 393,216 bytes = 384 KB
    n_banks = 6
    rows_per_bank = 16384
    bytes_per_word = 4
    total_bytes = n_banks * rows_per_bank * bytes_per_word
    total_kb = total_bytes / 1024

    sram = SRAMModel()
    assert sram.capacity_bytes() == total_bytes

    logger.log(
        BLOCK, test, "EDGE",
        {"n_banks": n_banks, "rows_per_bank": rows_per_bank,
         "bytes_per_word": bytes_per_word, "total_bytes": total_bytes,
         "total_kb": total_kb},
        f"SRAM: {n_banks} banks × {rows_per_bank} rows × {bytes_per_word} B = "
        f"{total_bytes} B = {total_kb:.0f} KB (confirmed 384 KB)",
        severity="EDGE",
    )

    # Q4 utilization at max PTE capacity (256 tokens × 32 bytes = 8 KB)
    q4_bytes = 256 * 32
    q4_util = q4_bytes / total_bytes * 100
    q8_bytes = 256 * 64
    q8_util = q8_bytes / total_bytes * 100

    logger.log(
        BLOCK, test, "WARN",
        {"pte_capacity": 256, "q4_sram_used_kb": q4_bytes // 1024,
         "q4_utilization_pct": round(q4_util, 2),
         "q8_sram_used_kb": q8_bytes // 1024,
         "q8_utilization_pct": round(q8_util, 2)},
        f"PTE capacity (256 entries) is the binding constraint, not SRAM capacity. "
        f"Q4: {q4_bytes} B used of {total_bytes} B ({q4_util:.1f}%). "
        f"Q8: {q8_bytes} B used ({q8_util:.1f}%).",
        severity="WARN",
    )


def run_mhc_sweeps(logger: SimLogger):
    """Run all MHC sweeps."""
    rng = np.random.default_rng(42)
    _sweep_page_table_exhaustion(logger, rng)
    _sweep_tier_fill_progression(logger, rng)
    _sweep_bank_address_boundary(logger, rng)
    _sweep_concurrent_access(logger, rng)
    _sweep_precision_round_trip(logger, rng)
    _sweep_sram_capacity(logger, rng)
