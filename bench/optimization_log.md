# Optimization Log

## 2026-03-10 17:41 CST

- Commit: `d8e4e9e`
- Change: switch the `N=512` dispatch in `src/SpMM_API.cu` from `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 2>` to `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 3>`.
- Rationale: `bench/ncu/benchMain_20260310_173412.csv` still showed persistent pipeline stalls on the active `N=512` kernel with `wait=1.911238`, `barrier=1.928103`, `mio=1.356246`, and tensor-pipe activity at `40.346497%`, so the next single-variable step was to deepen the `cp.async` pipeline without changing the tile.
- Result: cloud bench improved to `n2m4 0.13404 ms / 128.17 TFLOPs / ErrorRate 0.00`, versus `cusparseLt 0.14441 ms / 118.96 TFLOPs / ErrorRate 0.00`.

## 2026-03-10 17:50 CST

- Commit: `ef45a03`
- Change: switch the `N=512` dispatch in `src/SpMM_API.cu` from `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 3>` to `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 4>`.
- Rationale: `bench/ncu/benchMain_20260310_174542.csv` shows that the 3-stage version improved runtime while keeping the same register footprint (`165`) and same occupancy limits; tensor-pipe activity rose to `42.681516%`, barrier stall dropped to `1.619423`, and long scoreboard dropped to `0.732642`. The main remaining stall is still `wait=2.035925`, so the next single-variable test is one more stage of buffering.
- Result: invalid on the cloud A40 path. The `N=512` run reported `n2m4 0.00074 ms / ErrorRate 2097152.00`, and `bench/ncu/benchMain_20260310_175725.csv` contains no `SpMM_N2M4_Kernel`, which is consistent with the launch failing before the main kernel executed.

## 2026-03-10 18:40 CST

- Commit: `099c986`
- Change: revert the `N=512` dispatch in `src/SpMM_API.cu` back to `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 3>`, and add dynamic shared-memory limit checks in `SpMM_N2M4_Kernel_API`.
- Rationale: for `N2M4ConfigN128Balanced`, `stages=4` requests `102400` bytes of dynamic shared memory (`65536` for B, `32768` for A, `4096` for metadata), which exceeds the sm86/A40 opt-in limit of `101376` bytes. The previous code ignored `cudaFuncSetAttribute` failure, so an invalid configuration could slip through as a bogus near-zero runtime.
- Result: cloud bench recovered to `n2m4 0.13425 ms / 127.97 TFLOPs / ErrorRate 0.00`, versus `cusparseLt 0.14500 ms / 118.48 TFLOPs / ErrorRate 0.00`. The latest valid NCU `bench/ncu/benchMain_20260310_185126.csv` is effectively identical to the prior `stages=3` report, so pipeline-depth tuning is exhausted for this tile.

## 2026-03-10 19:05 CST

- Commit: `fb62106`
- Change: add a direct register-to-global output store path for the active `N=512` config `N2M4TilingConfig<16, 2, 4, 4, 4, 4>` in `src/SpMM_Kernel.cuh`, backed by a new `StoreToGlobalMemoryFromRegister_half` helper in `src/LoadAndStore.cuh`.
- Rationale: the current `stages=3` kernel is still paying a final `register -> shared -> global` round-trip plus a block-wide `__syncthreads()` even though each warp owns a disjoint output subtile. The latest NCU still shows `barrier=1.652997`, `mio=1.374762`, and `smsp__sass_inst_executed_op_shared_st.sum=32768`, so the next single-variable step is to remove that shared-memory writeback path only for the active `N=512` tile.
- Result: regressed on the cloud path. `bench/ncu/benchMain_20260310_193329.csv` shows the active `SpMM_N2M4_Kernel` runtime rising from `0.180666 ms` to `0.185764 ms` even though `smsp__sass_inst_executed_op_shared_st.sum` dropped from `32768` to `0`, because `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` doubled from `131072` to `262144`. The direct stores removed the shared-memory writeback, but they also broke the coalesced global-store pattern and lost throughput.

## 2026-03-10 20:00 CST

- Commit: `addb772`
- Change: switch the `N=512` dispatch in `src/SpMM_API.cu` from `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 3>` to `SpMM_N2M4_Kernel_API<N2M4ConfigN128Compact, 2>`.
- Rationale: after the direct-store regression, the best known valid baseline is still the shared-store `N2M4ConfigN128Balanced, 3` kernel. Its latest valid NCU (`bench/ncu/benchMain_20260310_185126.csv`) is limited by `wait=2.035946`, `math_pipe_throttle=1.944977`, `barrier=1.636837`, and `shared_mem_per_block=77.824 KB`, which locks it to one resident block per SM. The `128x64x64 / 128-thread / stage2` launch cuts dynamic shared memory to `34816B`, enabling two resident blocks per SM on sm86/A40 while keeping B/global-store traffic per output tile unchanged; the tradeoff is only extra A/metadata traffic along N, which is the cheaper side of the reuse loss.
- Result: regressed badly on the cloud path. `bench/ncu/benchMain_20260310_195636.csv` shows the active `SpMM_N2M4_Kernel` runtime rising to `0.209504 ms`, with `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` jumping from `6553600` to `8912896` and `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` rising from `0.705609` to `4.257484`. The extra residency did not compensate for the lost B/A reuse.

## 2026-03-10 20:15 CST

- Commit: `dcd5d90`
- Change: restore the `N=512` dispatch in `src/SpMM_API.cu` to `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 3>`, and replace the previous naive direct-output store with a coalesced warp-shuffle implementation for the active `N2M4ConfigN128Balanced` path in `src/LoadAndStore.cuh`.
- Rationale: the direct-store idea was only wrong because its write pattern doubled global-store sectors, not because bypassing shared memory was inherently bad. The best valid launch is still `N2M4ConfigN128Balanced, 3`, so the next single-variable step is to keep that launch fixed and retest only the output path with the same 128-bit row-wise store shape used by `StoreToGlobalMemoryFromShared`, but assembled directly from registers via warp shuffles. If this works, it should keep `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` near `131072` while eliminating the final shared-memory writeback and block-wide output barrier.
- Result: regressed again on the cloud path. `bench/ncu/benchMain_20260310_201800.csv` keeps `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` at `131072`, but the active `SpMM_N2M4_Kernel` runtime still rises to `0.202477 ms`, tensor-pipe activity drops from `42.572486%` to `38.583723%`, `mio_throttle` rises from `1.374505` to `1.787415`, and `lts__t_sectors.sum` rises from `6691113.866667` to `7323068.066667`. The coalesced direct store fixed store coalescing, but the extra shuffle/store instruction overhead still made the kernel slower.

## 2026-03-10 20:35 CST

- Commit: `140fdbe`
- Change: switch the `N=512` dispatch in `src/SpMM_API.cu` from `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 3>` to `SpMM_N2M4_Kernel_API<N2M4ConfigN128Wide, 3>`.
- Rationale: both direct-output store experiments are now dead ends, so the next safe step is to stop touching the output path and return to launch-policy search on the recovered 3-stage kernel. This keeps the same `128x128x64` tile and the same 3-stage `cp.async` depth that previously helped, but swaps the warp decomposition from `2x4` back to `4x2`. That changes only which side of the tile each warp carries more fragments for, and it automatically bypasses the direct-store specialization because that specialization only matches `N2M4ConfigN128Balanced`.
- Status: pending cloud validation.

## 2026-03-10 21:10 CST

- Commit: `b4ffbf6`
- Change: switch the `N=1024` dispatch in `src/SpMM_API.cu` from `SpMM_N2M4_Kernel_API<N2M4TilingConfig<16, 4, 8, 16, 2, 4>, 2>` to `SpMM_N2M4_Kernel_API<N2M4ConfigN128Wide, 3>`.
- Rationale: the old `N=1024` launch is not just slow, it is invalid. Its `TILE_N=1024 / BLOCK_THREADS=1024` shape requests `280576B` of dynamic shared memory, which exceeds the sm86/A40 opt-in limit of `101376B`, so the main kernel never launches and the near-zero `n2m4` runtime is bogus. The first valid recovery step is to reuse the already-proven `128x128x64 / 256-thread / stage3` tile so `N=1024` is split across 8 blocks in N while staying under the shared-memory limit.
- Status: pending cloud validation.
