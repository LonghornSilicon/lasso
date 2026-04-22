"""
Memory Hierarchy Controller (MHC) golden model for LASSO.

Architecture:
  - 256-entry page table (32-bit PTEs)
  - 6 SRAM banks of 16384 × 32-bit words = 384 KB total
    - Hot tier:  banks 0–3, 256 KB (up to 4 × 16384 rows)
    - Cold tier: banks 4–5, 128 KB (up to 2 × 16384 rows)
  - eDRAM stub: MHC_EDRAM_CTRL=0 is a no-op

PTE bit layout (32-bit word):
  [31:30] precision  Q4=0b00, Q8=0b01, bypass=0b11
  [29:28] tier       hot=0b00, cold=0b01, evicted=0b10
  [27:16] bank_row   (12 bits, 0..4095 within tile)
  [15: 8] bank_sel   (8 bits, but only [0,5] valid)
  [ 7: 0] seq_pos    (8 bits, 0..255)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

# Encoding maps
_PRECISION_ENC = {"Q4": 0b00, "Q8": 0b01, "bypass": 0b11}
_PRECISION_DEC = {v: k for k, v in _PRECISION_ENC.items()}
_TIER_ENC = {"hot": 0b00, "cold": 0b01, "evicted": 0b10}
_TIER_DEC = {v: k for k, v in _TIER_ENC.items()}

# Memory geometry
_NUM_BANKS = 6
_ROWS_PER_BANK = 16384
_HOT_BANKS = list(range(0, 4))   # banks 0–3
_COLD_BANKS = list(range(4, 6))  # banks 4–5
_MAX_HOT_SLOTS = len(_HOT_BANKS) * _ROWS_PER_BANK   # 65536
_MAX_COLD_SLOTS = len(_COLD_BANKS) * _ROWS_PER_BANK  # 32768
_PAGE_TABLE_SIZE = 256

# eDRAM control stub
MHC_EDRAM_CTRL = 0  # 0 = no-op


@dataclass
class PTE:
    """Page Table Entry for one KV token."""

    precision: str   # 'Q4', 'Q8', 'bypass'
    tier: str        # 'hot', 'cold', 'evicted'
    bank_row: int    # row within bank [0, 16383]
    bank_sel: int    # bank index [0, 5]
    seq_pos: int     # sequence position [0, 255]

    def to_word(self) -> int:
        """Encode PTE to 32-bit integer."""
        prec_bits = _PRECISION_ENC[self.precision]
        tier_bits = _TIER_ENC[self.tier]
        # bank_row stored in [27:16] — 12 bits (HW uses lower 12 bits of row)
        row_bits = self.bank_row & 0xFFF
        sel_bits = self.bank_sel & 0xFF
        pos_bits = self.seq_pos & 0xFF

        word = (
            (prec_bits << 30)
            | (tier_bits << 28)
            | (row_bits << 16)
            | (sel_bits << 8)
            | pos_bits
        )
        return word & 0xFFFFFFFF

    @classmethod
    def from_word(cls, word: int) -> "PTE":
        """Decode 32-bit integer to PTE."""
        word = word & 0xFFFFFFFF
        prec_bits = (word >> 30) & 0x3
        tier_bits = (word >> 28) & 0x3
        row_bits  = (word >> 16) & 0xFFF
        sel_bits  = (word >> 8) & 0xFF
        pos_bits  = word & 0xFF

        precision = _PRECISION_DEC.get(prec_bits, "Q8")
        tier = _TIER_DEC.get(tier_bits, "evicted")
        return cls(
            precision=precision,
            tier=tier,
            bank_row=row_bits,
            bank_sel=sel_bits,
            seq_pos=pos_bits,
        )


class PageTable:
    """
    Manages 256 PTE entries for KV token addressing.

    Allocates hot-tier slots first (banks 0–3), then cold-tier (banks 4–5).
    """

    def __init__(self):
        self._entries: Dict[int, PTE] = {}  # seq_pos → PTE
        self._hot_next_slot: int = 0    # monotone hot allocation counter
        self._cold_next_slot: int = 0   # monotone cold allocation counter

    def reset(self):
        """Clear all page table entries."""
        self._entries.clear()
        self._hot_next_slot = 0
        self._cold_next_slot = 0

    def allocate(self, seq_pos: int, precision: str, tier: str) -> Tuple[int, int]:
        """
        Allocate a slot for seq_pos.

        Parameters
        ----------
        seq_pos : int  (0–255)
        precision : str  ('Q4', 'Q8', 'bypass')
        tier : str  ('hot', 'cold')

        Returns
        -------
        (bank_sel, bank_row)

        Raises
        ------
        OverflowError if page table has 256 entries and a new one is requested
        (seq_pos not already present).
        """
        if seq_pos in self._entries:
            # Overwrite existing entry
            pte = self._entries[seq_pos]
            bank_sel = pte.bank_sel
            bank_row = pte.bank_row
        else:
            if len(self._entries) >= _PAGE_TABLE_SIZE:
                raise OverflowError(
                    f"Page table full ({_PAGE_TABLE_SIZE} entries). "
                    f"Cannot allocate seq_pos={seq_pos}."
                )
            if tier == "hot":
                slot = self._hot_next_slot
                self._hot_next_slot += 1
                bank_sel = _HOT_BANKS[slot % len(_HOT_BANKS)]
                bank_row = slot // len(_HOT_BANKS)
            else:
                slot = self._cold_next_slot
                self._cold_next_slot += 1
                bank_sel = _COLD_BANKS[slot % len(_COLD_BANKS)]
                bank_row = slot // len(_COLD_BANKS)

        pte = PTE(
            precision=precision,
            tier=tier,
            bank_row=bank_row,
            bank_sel=bank_sel,
            seq_pos=seq_pos,
        )
        self._entries[seq_pos] = pte
        return bank_sel, bank_row

    def lookup(self, seq_pos: int) -> PTE:
        """
        Lookup PTE for seq_pos.

        Raises
        ------
        KeyError if seq_pos not found.
        """
        if seq_pos not in self._entries:
            raise KeyError(f"seq_pos={seq_pos} not in page table")
        return self._entries[seq_pos]

    def evict(self, seq_pos: int):
        """
        Mark a token as evicted (tier = 'evicted').

        Raises
        ------
        KeyError if seq_pos not found.
        """
        pte = self.lookup(seq_pos)
        self._entries[seq_pos] = PTE(
            precision=pte.precision,
            tier="evicted",
            bank_row=pte.bank_row,
            bank_sel=pte.bank_sel,
            seq_pos=seq_pos,
        )

    def hot_fill_pct(self) -> float:
        """Fraction of hot slots used (0.0 – 1.0)."""
        hot_used = sum(
            1 for p in self._entries.values() if p.tier == "hot"
        )
        max_hot = _PAGE_TABLE_SIZE  # capped by page table, not SRAM
        return hot_used / max_hot

    def cold_fill_pct(self) -> float:
        """Fraction of cold slots used (0.0 – 1.0)."""
        cold_used = sum(
            1 for p in self._entries.values() if p.tier == "cold"
        )
        max_cold = _PAGE_TABLE_SIZE
        return cold_used / max_cold

    def flush_cold(self):
        """Evict all cold-tier tokens."""
        for seq_pos, pte in list(self._entries.items()):
            if pte.tier == "cold":
                self.evict(seq_pos)

    def __len__(self) -> int:
        return len(self._entries)


class SRAMModel:
    """
    Models 6 banks of 16384 × 32-bit SRAM words.

    Total capacity: 6 × 16384 × 4 = 393,216 bytes = 384 KB.
    """

    def __init__(self):
        # Each bank: array of 16384 32-bit words, initialized to 0
        self._banks = [
            np.zeros(_ROWS_PER_BANK, dtype=np.uint32)
            for _ in range(_NUM_BANKS)
        ]

    def _check(self, bank: int, row: int):
        if not (0 <= bank < _NUM_BANKS):
            raise ValueError(f"bank={bank} out of range [0, {_NUM_BANKS-1}]")
        if not (0 <= row < _ROWS_PER_BANK):
            raise ValueError(f"row={row} out of range [0, {_ROWS_PER_BANK-1}]")

    def read(self, bank: int, row: int) -> int:
        """Return 32-bit word at (bank, row)."""
        self._check(bank, row)
        return int(self._banks[bank][row])

    def write(self, bank: int, row: int, data: int):
        """Write 32-bit word to (bank, row)."""
        self._check(bank, row)
        self._banks[bank][row] = np.uint32(data & 0xFFFFFFFF)

    def clear(self, bank: int, row: int):
        """Zero out one word."""
        self._check(bank, row)
        self._banks[bank][row] = np.uint32(0)

    def capacity_bytes(self) -> int:
        """Total capacity in bytes."""
        return _NUM_BANKS * _ROWS_PER_BANK * 4  # 393216 = 384 KB


class MHC:
    """
    Memory Hierarchy Controller: combines PageTable + SRAMModel.

    Manages KV token storage with tiered SRAM, eviction, and overflow detection.

    The 32-bit SRAM word stores:
      [31:16] packed_kv low 16 bits (or scale if only 1 word per token)
      [15: 0] scale as INT16

    For multi-word tokens, additional words store packed_kv data.
    In this golden model we store up to 2 words per token:
      word0: [31:16] = packed_kv[0:16], [15:0] = scale (INT16)
      word1: packed_kv[16:32] (if present)

    Simplified: packed_kv is treated as a single 32-bit value for the golden model.
    """

    def __init__(self, hot_thresh: int = 128):
        """
        Parameters
        ----------
        hot_thresh : int
            Number of hot tokens before overflow flag is set (default 128,
            i.e. half the page table).
        """
        self._pt = PageTable()
        self._sram = SRAMModel()
        self._overflow = False
        self._hot_thresh = hot_thresh
        self._count = 0

    def reset(self):
        """Reset MHC: clear page table and SRAM."""
        self._pt.reset()
        self._sram = SRAMModel()
        self._overflow = False
        self._count = 0

    def write_kv(
        self,
        seq_pos: int,
        packed_kv: int,
        scale: int,
        mode: str,
        tag: str,
    ):
        """
        Write KV token to SRAM if tag is RETAIN.

        Parameters
        ----------
        seq_pos : int  Token position [0, 255]
        packed_kv : int  32-bit packed KV data
        scale : int  INT16 scale factor
        mode : str  'Q4', 'Q8', 'bypass'
        tag : str  'RETAIN' or 'EVICT'
        """
        if tag != "RETAIN":
            return  # EVICT: do not store

        # eDRAM stub: MHC_EDRAM_CTRL=0 → no-op (no external DRAM writes)
        _ = MHC_EDRAM_CTRL

        # Determine tier (hot if within hot threshold)
        tier = "hot" if self._count < self._hot_thresh else "cold"

        if seq_pos not in self._pt._entries:
            self._count += 1

        try:
            bank_sel, bank_row = self._pt.allocate(seq_pos, mode, tier)
        except OverflowError:
            self._overflow = True
            raise

        if self._count > _PAGE_TABLE_SIZE:
            self._overflow = True

        # Pack into two SRAM words:
        # word at row*2:   [31:16]=packed_kv[31:16], [15:0]=scale
        # word at row*2+1: [31:0]=packed_kv[31:0]
        # Use row*2 and row*2+1 (guaranteed within 16384 rows since page table ≤ 256)
        base_row = bank_row * 2
        word0 = ((packed_kv & 0xFFFF0000) | (scale & 0xFFFF)) & 0xFFFFFFFF
        word1 = packed_kv & 0xFFFFFFFF
        self._sram.write(bank_sel, base_row, word0)
        self._sram.write(bank_sel, base_row + 1, word1)

    def read_kv(self, seq_pos: int) -> Tuple[int, int, str]:
        """
        Read KV token from SRAM.

        Parameters
        ----------
        seq_pos : int

        Returns
        -------
        (packed_kv, scale, mode)

        Raises
        ------
        KeyError if seq_pos not stored or is evicted.
        """
        pte = self._pt.lookup(seq_pos)
        if pte.tier == "evicted":
            raise KeyError(f"seq_pos={seq_pos} has been evicted")

        base_row = pte.bank_row * 2
        word0 = self._sram.read(pte.bank_sel, base_row)
        word1 = self._sram.read(pte.bank_sel, base_row + 1)

        scale = word0 & 0xFFFF
        packed_kv = word1 & 0xFFFFFFFF
        mode = pte.precision
        return (packed_kv, scale, mode)

    def evict(self, seq_pos: int):
        """
        Evict a token: mark PTE as evicted, zero out SRAM data.
        """
        pte = self._pt.lookup(seq_pos)
        base_row = pte.bank_row * 2
        self._sram.clear(pte.bank_sel, base_row)
        self._sram.clear(pte.bank_sel, base_row + 1)
        self._pt.evict(seq_pos)

    def flush_cold(self):
        """Evict all cold-tier tokens."""
        # Collect cold tokens first (avoid modifying dict during iteration)
        cold_positions = [
            sp for sp, pte in self._pt._entries.items() if pte.tier == "cold"
        ]
        for sp in cold_positions:
            self.evict(sp)

    def is_overflow(self) -> bool:
        """Return True if page table has overflowed (> 256 tokens stored)."""
        return self._overflow

    @property
    def page_table(self) -> PageTable:
        return self._pt

    @property
    def sram(self) -> SRAMModel:
        return self._sram
