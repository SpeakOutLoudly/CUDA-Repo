# Optimization Log

## 2026-03-10 17:41 CST

- Commit: `d8e4e9e`
- Change: switch the `N=512` dispatch in `src/SpMM_API.cu` from `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 2>` to `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 3>`.
- Rationale: `bench/ncu/benchMain_20260310_173412.csv` still showed persistent pipeline stalls on the active `N=512` kernel with `wait=1.911238`, `barrier=1.928103`, `mio=1.356246`, and tensor-pipe activity at `40.346497%`, so the next single-variable step was to deepen the `cp.async` pipeline without changing the tile.
- Result: cloud bench improved to `n2m4 0.13404 ms / 128.17 TFLOPs / ErrorRate 0.00`, versus `cusparseLt 0.14441 ms / 118.96 TFLOPs / ErrorRate 0.00`.

## 2026-03-10 17:50 CST

- Commit: pending
- Change: switch the `N=512` dispatch in `src/SpMM_API.cu` from `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 3>` to `SpMM_N2M4_Kernel_API<N2M4ConfigN128Balanced, 4>`.
- Rationale: `bench/ncu/benchMain_20260310_174542.csv` shows that the 3-stage version improved runtime while keeping the same register footprint (`165`) and same occupancy limits; tensor-pipe activity rose to `42.681516%`, barrier stall dropped to `1.619423`, and long scoreboard dropped to `0.732642`. The main remaining stall is still `wait=2.035925`, so the next single-variable test is one more stage of buffering.
- Status: pending cloud validation.
