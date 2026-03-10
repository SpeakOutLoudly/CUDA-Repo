#pragma once

// N:2M（2:4）结构化稀疏内核的 Tile 配置
// 目标：与 mma.sp.sync.aligned.m16n8k{16|32}（FP16->FP32 累加）保持一致的线程块切分

/* ============================================================
 * 1) 实现/布局相关的宏配置（不直接参与几何推导）
 *    - shared memory 的 padding（bank conflict 规避）
 *    - C fragment 在寄存器中的占用（按 mma 输出映射的经验/约定）
 * ============================================================ */
#define PADDING_SHARED_MEM_FOR_C 8  // 为消除 bank 冲突，在共享内存中为 C 矩阵的每一列额外增加的元素数，为了和 shared memory 对齐需要 padding 128bit
#define REG_PER_C_TENSOR_16_8 4     // 16*8 的 Float 结果均分到 32 条线程，每线程 4 个 32-bit accum


/* ============================================================
 * 2) 指令形状/硬件常量（K 作为模板参数）
 *    - MMA 的 (M,N,K) 形状、warp 大小
 * ============================================================ */
static constexpr int WARP_SIZE = 32;  // CUDA warp 包含多少条线程

// 指令形状：m16n8kx（K=16 或 32）
template<int K_>
struct SpMmaShape {
    static_assert(K_ == 16 || K_ == 32, "Only K=16 or K=32 is supported");
    static constexpr int MMA_M_SP = 16;  // 稀疏 MMA 指令的 M 维度
    static constexpr int MMA_N_SP = 8;   // 稀疏 MMA 指令的 N 维度
    static constexpr int MMA_K_SP = K_;  // 稀疏 MMA 指令的 K 维度
};


/* ============================================================
 * 3) 访存/流水相关常量（拷贝粒度、K 循环步长、Tile-K 覆盖）
 *    - 这部分里凡是依赖 K 的，都放进模板里派生
 * ============================================================ */

// 单位内存事务常量
static constexpr int HALF_PER_128bit   = 8;  // 128bit 可容纳 8 个 fp16
static constexpr int UINT32_PER_128bit = 4;  // 128bit 可容纳 4 个 uint32

// cp.async 异步加载相关（通常与 stage/tile 设计相关）
static constexpr int async_copy_unit_M = 16;
static constexpr int async_copy_unit_N = 8;
static constexpr int async_copy_unit_K = 64;  // 考虑到在进行列置换的时候是 64 列一组，所以这里最好固定为 64

// K 方向循环/Tile-K：依赖 MMA_K_SP（即 K）
template<int K_, int BLOCK_K_STEPS_ = 4>
struct SpKLoopConfig {
    using Shape = SpMmaShape<K_>;
    static constexpr int MMA_K_STEP    = Shape::MMA_K_SP;             // 16 or 32
    static constexpr int BLOCK_K_STEPS = BLOCK_K_STEPS_;              // 每个 Tile 内沿 K 方向的 mma.sp 次数
    static constexpr int TILE_K        = MMA_K_STEP * BLOCK_K_STEPS;  // 沿 K 维迭代计算的步长
};


/* ============================================================
 * 4) Block/Warp 级 tiling 配置（模板，派生 TILE_M / TILE_N / BLOCK_THREADS）
 *    - 输入：K、block 在 M/N 的 warp 切分、warp 在 N 的 tensor 数、K 循环次数、warp 在 M 的 tensor 数
 *    - 输出：block threads 数、输出 tile 的 M/N 覆盖、以及 TILE_K
 * ============================================================ */
    template<
        int _K,                    // 一般使用 32
        int _BLOCK_ROW_WARPS,
        int _BLOCK_COL_WARPS,
        int _WARP_COL_TENSORS,
        int _WARP_ROW_TENSORS = 1,  // warp 在 M 方向并行 tensor 数，默认为1
        int _BLOCK_K_STEPS = 2      // 可选：默认 2 次 mma
    >
    struct N2M4TilingConfig {
        // ---- 指令形状 ----
        using Shape = SpMmaShape<_K>;
        static constexpr int MMA_M_SP = Shape::MMA_M_SP;
        static constexpr int MMA_N_SP = Shape::MMA_N_SP;
        static constexpr int MMA_K_SP = Shape::MMA_K_SP;

        // ---- 输入配置 ----
        static constexpr int BLOCK_ROW_WARPS  = _BLOCK_ROW_WARPS;
        static constexpr int BLOCK_COL_WARPS  = _BLOCK_COL_WARPS;
        static constexpr int WARP_COL_TENSORS = _WARP_COL_TENSORS;
        static constexpr int WARP_ROW_TENSORS = _WARP_ROW_TENSORS;

        // ---- block 级派生：warp / thread 数 ----
        static constexpr int BLOCK_WARPS   = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
        static constexpr int BLOCK_THREADS = BLOCK_WARPS * WARP_SIZE;

        // ---- tile 级派生：输出 tile 的 M/N 覆盖 ----
        static constexpr int TILE_M = MMA_M_SP * (WARP_ROW_TENSORS * BLOCK_ROW_WARPS);
        static constexpr int TILE_N = MMA_N_SP * (WARP_COL_TENSORS * BLOCK_COL_WARPS);

        // ---- K 方向派生：一次推进 / 循环次数 / TILE_K ----
        using KLoop = SpKLoopConfig<_K, _BLOCK_K_STEPS>;
        static constexpr int MMA_K_STEP    = KLoop::MMA_K_STEP;
        static constexpr int BLOCK_K_STEPS = KLoop::BLOCK_K_STEPS;
        static constexpr int TILE_K        = KLoop::TILE_K;
    };


// ------------------------
// 你要的“两套”直接用别名给出来
// ------------------------
// template<int _BLOCK_ROW_WARPS, int _BLOCK_COL_WARPS, int _WARP_COL_TENSORS, int _BLOCK_K_STEPS = 4>
// using N2M4TilingConfigK16 =
//     N2M4TilingConfig<16, _BLOCK_ROW_WARPS, _BLOCK_COL_WARPS, _WARP_COL_TENSORS, _BLOCK_K_STEPS>;

// template<int _BLOCK_ROW_WARPS, int _BLOCK_COL_WARPS, int _WARP_COL_TENSORS, int _BLOCK_K_STEPS = 2>
// using N2M4TilingConfigK32 =
//     N2M4TilingConfig<32, _BLOCK_ROW_WARPS, _BLOCK_COL_WARPS, _WARP_COL_TENSORS, _BLOCK_K_STEPS>;

// 常用候选配置，便于 profiling 脚本和 launch policy 统一引用。
using N2M4ConfigN64Default = N2M4TilingConfig<16, 2, 2, 4, 2, 4>;
using N2M4ConfigN128Compact = N2M4TilingConfig<16, 2, 2, 4, 4, 4>;
using N2M4ConfigN128Wide = N2M4TilingConfig<16, 4, 2, 8, 2, 4>;
using N2M4ConfigN512WideN = N2M4TilingConfig<16, 2, 4, 4, 2, 4>;
