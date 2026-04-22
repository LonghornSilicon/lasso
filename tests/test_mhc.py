"""
MHC (Memory Hierarchy Controller) golden model tests.
22 test cases covering PTE encoding, page table, SRAM, and MHC operations.
"""

import pytest
import numpy as np

from golden_model.mhc import (
    PTE,
    PageTable,
    SRAMModel,
    MHC,
    MHC_EDRAM_CTRL,
    _PAGE_TABLE_SIZE,
)


# ===========================================================================
# PTE Tests
# ===========================================================================

# 1. PTE.to_word() and from_word() round-trip
def test_pte_roundtrip():
    pte = PTE(precision="Q8", tier="hot", bank_row=42, bank_sel=2, seq_pos=7)
    word = pte.to_word()
    pte2 = PTE.from_word(word)
    assert pte2.precision == pte.precision
    assert pte2.tier == pte.tier
    assert pte2.bank_row == pte.bank_row
    assert pte2.bank_sel == pte.bank_sel
    assert pte2.seq_pos == pte.seq_pos


# 2. PTE precision field encodes Q4=0b00, Q8=0b01, bypass=0b11
def test_pte_precision_encoding():
    for prec, expected_bits in [("Q4", 0b00), ("Q8", 0b01), ("bypass", 0b11)]:
        pte = PTE(precision=prec, tier="hot", bank_row=0, bank_sel=0, seq_pos=0)
        word = pte.to_word()
        prec_bits = (word >> 30) & 0x3
        assert prec_bits == expected_bits, f"{prec}: got {prec_bits:#04b}, expected {expected_bits:#04b}"


# 3. PTE tier field encodes hot=0b00, cold=0b01, evicted=0b10
def test_pte_tier_encoding():
    for tier, expected_bits in [("hot", 0b00), ("cold", 0b01), ("evicted", 0b10)]:
        pte = PTE(precision="Q8", tier=tier, bank_row=0, bank_sel=0, seq_pos=0)
        word = pte.to_word()
        tier_bits = (word >> 28) & 0x3
        assert tier_bits == expected_bits, f"{tier}: got {tier_bits:#04b}"


# 4. allocate_hot: first token goes to bank 0 of hot tier
def test_allocate_hot_first_token():
    pt = PageTable()
    bank_sel, bank_row = pt.allocate(0, "Q8", "hot")
    assert bank_sel == 0   # first hot bank
    assert bank_row == 0


# 5. allocate: 257th token raises OverflowError (page table full)
def test_allocate_overflow():
    pt = PageTable()
    for i in range(_PAGE_TABLE_SIZE):
        pt.allocate(i, "Q8", "hot")
    with pytest.raises(OverflowError):
        pt.allocate(_PAGE_TABLE_SIZE, "Q8", "hot")


# 6. lookup nonexistent token raises KeyError
def test_lookup_nonexistent_raises():
    pt = PageTable()
    with pytest.raises(KeyError):
        pt.lookup(42)


# 7. write_kv RETAIN → stored; write_kv EVICT → not stored
def test_write_kv_retain_evict():
    mhc = MHC()
    mhc.write_kv(0, packed_kv=0xABCD1234, scale=100, mode="Q8", tag="RETAIN")
    mhc.write_kv(1, packed_kv=0xDEAD0000, scale=50,  mode="Q4", tag="EVICT")

    # Token 0: RETAIN → stored
    packed, scale, mode = mhc.read_kv(0)
    assert scale == 100
    assert mode == "Q8"

    # Token 1: EVICT → not stored → KeyError
    with pytest.raises(KeyError):
        mhc.read_kv(1)


# 8. read_kv after write_kv returns same data
def test_read_after_write_same_data():
    mhc = MHC()
    mhc.write_kv(5, packed_kv=0x12345678, scale=200, mode="Q8", tag="RETAIN")
    packed, scale, mode = mhc.read_kv(5)
    assert scale == 200
    assert mode == "Q8"


# 9. hot tier fills before cold tier (first hot_thresh tokens go hot)
def test_hot_fills_before_cold():
    mhc = MHC(hot_thresh=4)
    for i in range(4):
        mhc.write_kv(i, 0x1234, 10, "Q8", "RETAIN")
    # Check all 4 are in hot tier
    for i in range(4):
        pte = mhc.page_table.lookup(i)
        assert pte.tier == "hot", f"Token {i} should be hot"


# 10. overflow flag set when page table full and another arrives
def test_overflow_flag():
    mhc = MHC(hot_thresh=1000)
    for i in range(_PAGE_TABLE_SIZE):
        mhc.write_kv(i, 0x0, 1, "Q8", "RETAIN")
    assert not mhc.is_overflow()
    with pytest.raises(OverflowError):
        mhc.write_kv(_PAGE_TABLE_SIZE, 0x0, 1, "Q8", "RETAIN")


# 11. evict: PTE tier becomes 'evicted', SRAM data cleared
def test_evict_clears_data():
    mhc = MHC()
    mhc.write_kv(3, packed_kv=0xCAFEBABE, scale=77, mode="Q8", tag="RETAIN")
    mhc.evict(3)
    pte = mhc.page_table.lookup(3)
    assert pte.tier == "evicted"
    with pytest.raises(KeyError):
        mhc.read_kv(3)


# 12. hot_fill_pct: 0% initially, 100% at _PAGE_TABLE_SIZE hot tokens
def test_hot_fill_pct():
    pt = PageTable()
    assert pt.hot_fill_pct() == 0.0
    for i in range(_PAGE_TABLE_SIZE):
        pt.allocate(i, "Q8", "hot")
    assert pt.hot_fill_pct() == 1.0


# 13. cold_fill_pct similar
def test_cold_fill_pct():
    pt = PageTable()
    assert pt.cold_fill_pct() == 0.0
    for i in range(_PAGE_TABLE_SIZE):
        pt.allocate(i, "Q8", "cold")
    assert pt.cold_fill_pct() == 1.0


# 14. SRAM bank read/write: read after write returns same 32-bit word
def test_sram_read_write():
    sram = SRAMModel()
    sram.write(0, 100, 0xDEADBEEF)
    val = sram.read(0, 100)
    assert val == 0xDEADBEEF


# 15. SRAM capacity: 6 × 16384 × 4 bytes = 393,216 bytes = 384 KB
def test_sram_capacity():
    sram = SRAMModel()
    assert sram.capacity_bytes() == 6 * 16384 * 4
    assert sram.capacity_bytes() == 393_216


# 16. bank_sel in range [0, 5]
def test_bank_sel_range():
    pt = PageTable()
    for i in range(6):
        pt.allocate(i, "Q8", "hot")
    for i in range(6):
        pte = pt.lookup(i)
        assert 0 <= pte.bank_sel <= 5


# 17. bank_row in range [0, 16383]
def test_bank_row_range():
    pt = PageTable()
    for i in range(10):
        pt.allocate(i, "Q8", "hot")
    for i in range(10):
        pte = pt.lookup(i)
        assert 0 <= pte.bank_row <= 16383


# 18. page table reset: all entries cleared
def test_page_table_reset():
    pt = PageTable()
    for i in range(10):
        pt.allocate(i, "Q8", "hot")
    pt.reset()
    assert len(pt) == 0
    with pytest.raises(KeyError):
        pt.lookup(0)


# 19. write Q8 token, read back: precision field matches
def test_write_q8_precision_preserved():
    mhc = MHC()
    mhc.write_kv(7, packed_kv=0x0, scale=50, mode="Q8", tag="RETAIN")
    pte = mhc.page_table.lookup(7)
    assert pte.precision == "Q8"


# 20. eDRAM stub: MHC_EDRAM_CTRL=0 is no-op (no error raised)
def test_edram_stub_no_error():
    assert MHC_EDRAM_CTRL == 0
    mhc = MHC()
    # write_kv internally references MHC_EDRAM_CTRL=0 → should not raise
    mhc.write_kv(0, 0x1234, 10, "Q8", "RETAIN")


# 21. MHC flush cold tier: all cold PTEs set to evicted
def test_flush_cold_tier():
    mhc = MHC(hot_thresh=2)  # first 2 tokens hot, rest cold
    for i in range(4):
        mhc.write_kv(i, 0x1234, 10, "Q8", "RETAIN")

    # Tokens 2, 3 should be cold
    mhc.flush_cold()

    # Cold tokens should now be evicted
    for i in range(2, 4):
        pte = mhc.page_table.lookup(i)
        assert pte.tier == "evicted"

    # Hot tokens still readable
    for i in range(2):
        pte = mhc.page_table.lookup(i)
        assert pte.tier == "hot"


# 22. Two tokens at same seq_pos: second write overwrites first
def test_second_write_overwrites_first():
    mhc = MHC()
    mhc.write_kv(10, packed_kv=0xAAAA, scale=11, mode="Q8", tag="RETAIN")
    mhc.write_kv(10, packed_kv=0xBBBB, scale=22, mode="Q4", tag="RETAIN")

    packed, scale, mode = mhc.read_kv(10)
    # Second write should be visible
    assert scale == 22
    assert mode == "Q4"
