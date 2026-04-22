# LASSO Simulation Sweep Summary

Generated: 2026-04-22T08:19:14.902226Z

| Block | Test | Severity | Status | Params | Detail |
|-------|------|----------|--------|--------|--------|
| KVE | wht_overflow | WARN | WARN | magnitude=5000, actual_mag=5000, max_wht_output=160000 | WHT output 160000 exceeds INT16 range 32767 (expected — WHT expands range) |
| KVE | wht_overflow | WARN | WARN | magnitude=10000, actual_mag=10000, max_wht_output=320000 | WHT output 320000 exceeds INT16 range 32767 (expected — WHT expands range) |
| KVE | wht_overflow | WARN | WARN | magnitude=16000, actual_mag=16000, max_wht_output=512000 | WHT output 512000 exceeds INT16 range 32767 (expected — WHT expands range) |
| KVE | wht_overflow | WARN | WARN | magnitude=32767, actual_mag=32767, max_wht_output=1048544 | WHT output 1048544 exceeds INT16 range 32767 (expected — WHT expands range) |
| KVE | wht_overflow | WARN | WARN | magnitude=32768, actual_mag=32767, max_wht_output=1048544 | WHT output 1048544 exceeds INT16 range 32767 (expected — WHT expands range) (clipped from 32768 to 32767) |
| KVE | scale_saturation | FAIL | FAIL | group_size=32, input=all_INT16_MIN, mode=Q4, raw_scale=149796, stored_scale=32767 | KNOWN: abs(INT16_MIN) overflows INT16 scale register. Raw scale 149796 > 32767; clamped to 32767. RTL must clamp abs to 32767. |
| KVE | scale_saturation | FAIL | FAIL | group_size=64, input=all_INT16_MIN, mode=Q4, raw_scale=299593, stored_scale=32767 | KNOWN: abs(INT16_MIN) overflows INT16 scale register. Raw scale 299593 > 32767; clamped to 32767. RTL must clamp abs to 32767. |
| KVE | scale_saturation | FAIL | FAIL | group_size=128, input=all_INT16_MIN, mode=Q4, raw_scale=599186, stored_scale=32767 | KNOWN: abs(INT16_MIN) overflows INT16 scale register. Raw scale 599186 > 32767; clamped to 32767. RTL must clamp abs to 32767. |
| KVE | scale_saturation | FAIL | FAIL | group_size=128, input=all_INT16_MIN, mode=Q8, raw_scale=33026, stored_scale=32767 | KNOWN: abs(INT16_MIN) overflows INT16 scale register. Raw scale 33026 > 32767; clamped to 32767. RTL must clamp abs to 32767. |
| KVE | group_size_boundary | EDGE | RAISED_VALUEERROR | group_size=33 | Correctly raised ValueError for non-power-of-2 group_size=33: group_size must be a power of 2, got 33 |
| KVE | group_size_boundary | EDGE | RAISED_ERROR | group_size=0 | Correctly raised ValueError for group_size=0: math domain error |
| KVE | group_size_boundary | EDGE | WORKS | group_size=1, input=500, decoded=497 | group_size=1 encodes/decodes without error. decoded=497 |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.0, eviction_rate=0.0, n_samples=100 | eviction_rate=0.0000 at threshold=0.000 |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.05, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.050 |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.1, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.100 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.1, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.100 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.15, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.150 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.15, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.150 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.2, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.200 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.2, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.200 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.25, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.250 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.25, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.250 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.3, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.300 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.3, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.300 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.35, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.350 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.35, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.350 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.4, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.400 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.4, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.400 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.45, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.450 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.45, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.450 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.5, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.500 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.5, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.500 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.55, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.550 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.55, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.550 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.6, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.600 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.6, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.600 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.65, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.650 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.65, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.650 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.7, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.700 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.7, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.700 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.75, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.750 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.75, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.750 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.8, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.800 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.8, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.800 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.85, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.850 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.85, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.850 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.9, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.900 |
| TIU | eviction_rate_curve | WARN | WARN | threshold=0.9, eviction_rate=1.0 | 100% eviction at non-extreme threshold 0.900 — scoring may be too aggressive |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=0.95, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=0.950 |
| TIU | eviction_rate_curve | EDGE | EDGE | threshold=1.0, eviction_rate=1.0, n_samples=100 | eviction_rate=1.0000 at threshold=1.000 |
| TIU | weight_sensitivity | WARN | WARN | C_t=0.7, H_t=1.0, seq_len=10, threshold=0.5 | No flip point found across w_C in [0,1] — decision never changes |
| TIU | sink_count_boundary | EDGE | EDGE | sink_count=0, tokens_03_evictable=True | With sink_count=0 and threshold=1.1: tokens 0-3 can be evicted |
| TIU | sink_count_boundary | EDGE | EDGE | sink_count=100, seq_len=10, tok_idx=5, tag=RETAIN | sink_count=100 > seq_len=10 does not crash. tok_idx=5 tag=RETAIN |
| MHC | tier_fill_progression | EDGE | EDGE | hot_full_at_n=127, hot_thresh=128 | Hot tier reached hot_thresh=128 (50% of page table) at N=127 |
| MHC | sram_capacity | EDGE | EDGE | n_banks=6, rows_per_bank=16384, bytes_per_word=4, total_bytes=393216, total_kb=384.0 | SRAM: 6 banks × 16384 rows × 4 B = 393216 B = 384 KB (confirmed 384 KB) |
| MHC | sram_capacity | WARN | WARN | pte_capacity=256, q4_sram_used_kb=8, q4_utilization_pct=2.08, q8_sram_used_kb=16, q8_utilization_pct=4.17 | PTE capacity (256 entries) is the binding constraint, not SRAM capacity. Q4: 8192 B used of 393216 B (2.1%). Q8: 16384 B used (4.2%). |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=1, max_abs_error=0.0, rel_error=0.0 | seq_len=1: max_abs_error=0.00e+00, rel_error=0.00e+00 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=2, max_abs_error=2.22e-16, rel_error=9.77e-17 | seq_len=2: max_abs_error=2.22e-16, rel_error=9.77e-17 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=4, max_abs_error=2.22e-16, rel_error=1.32e-16 | seq_len=4: max_abs_error=2.22e-16, rel_error=1.32e-16 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=8, max_abs_error=2.22e-16, rel_error=1.82e-16 | seq_len=8: max_abs_error=2.22e-16, rel_error=1.82e-16 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=16, max_abs_error=2.22e-16, rel_error=1.74e-16 | seq_len=16: max_abs_error=2.22e-16, rel_error=1.74e-16 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=32, max_abs_error=3.33e-16, rel_error=3.22e-16 | seq_len=32: max_abs_error=3.33e-16, rel_error=3.22e-16 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=64, max_abs_error=1.11e-16, rel_error=2.71e-16 | seq_len=64: max_abs_error=1.11e-16, rel_error=2.71e-16 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=128, max_abs_error=1.39e-16, rel_error=3.6e-16 | seq_len=128: max_abs_error=1.39e-16, rel_error=3.60e-16 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=256, max_abs_error=1.11e-16, rel_error=4.34e-16 | seq_len=256: max_abs_error=1.11e-16, rel_error=4.34e-16 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=512, max_abs_error=1.39e-16, rel_error=8.32e-16 | seq_len=512: max_abs_error=1.39e-16, rel_error=8.32e-16 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=1024, max_abs_error=8.33e-17, rel_error=6.52e-16 | seq_len=1024: max_abs_error=8.33e-17, rel_error=6.52e-16 |
| LACU | seq_length_scaling | EDGE | EDGE | seq_len=2048, max_abs_error=9.71e-17, rel_error=9.83e-16 | seq_len=2048: max_abs_error=9.71e-17, rel_error=9.83e-16 |
| LACU | accumulator_overflow | OVERFLOW | OVERFLOW | head_dim=64, INT16_MAX=32767, max_dot_product=68715282496, INT32_MAX=2147483647, overflows=True | INT32 accumulator overflows for seq_len>=2048 with max-magnitude inputs; need INT64 or score scaling. 64 * 32767^2 = 6.87e+10 >> INT32_MAX=2.15e+09 |
| LACU | accumulator_overflow | FAIL | FAIL | overflow_seq_len=1, raw_dot_product=68715282496.0, INT32_MAX=2147483647 | KNOWN: Single QK dot product overflows INT32 at seq_len=1 (overflow is per-dot-product, independent of seq_len). dot(Q,K[i])=6.87e+10 > INT32_MAX=2.15e+09. RTL accumulator needs INT64 or input pre-scaling when inputs near INT16_MAX. |
| LACU | dsp_mapping_zcu | EDGE | EDGE | dsps_per_tile=32, zcu102_total_dsps=2520, zcu104_total_dsps=1728, lacu_dsps=32 | 32-wide dot product needs 32 DSP48E2 slices per tile computation |
| LACU | dsp_mapping_zcu | EDGE | EDGE | zcu102_dsps=2520, zcu104_dsps=1728, lacu_dsps=32, zcu102_utilization_pct=1.3, zcu104_utilization_pct=1.9 | ZCU102 has 2520 DSP48E2; LACU uses 32 (1.3%). ZCU104 has 1728; LACU uses 32 (1.9%). |
| LACU | dsp_mapping_zcu | EDGE | EDGE | macs_per_tile=2048, cycles_per_tile=64, freq_mhz=50, time_per_tile_us=1.28 | At 50 MHz: 32 MACs × 64 elements = 2048 multiply-adds per tile. One tile per 64 cycles = 1.28 μs per tile at 50 MHz |
| LACU | dsp_mapping_zcu | EDGE | EDGE | freq_mhz=300, time_per_tile_ns=213.3 | At 300 MHz (ZCU102 native): one tile per 213 ns |
