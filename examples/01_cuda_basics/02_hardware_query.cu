/**
 * [Module A] 02. GPU 硬件架构深度解析
 * 硬件拓扑侦探：挖掘 SM 架构、L2 Cache、Tensor Core 能力与带宽极限
 */

#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cstdio>

// --- 宏定义：生产级错误检查 ---
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- 辅助函数：根据架构推算每个 SM 的 CUDA Core 数量 ---
// 注意：CUDA API 不直接提供 "Total Cores"，必须根据 Compute Capability 推算
// 参考：NVIDIA CUDA Programming Guide -> Compute Capabilities
int get_cores_per_sm(int major, int minor) {
    // CUDA API 不直接给「总 CUDA Core」。未知 CC 返回 -1，由调用方印 Unknown。
    // Blackwell consumer (RTX 50 / GB20x)
    if (major == 12) {
        return 128;
    }
    // Hopper (H100)
    if (major == 9) {
        if (minor == 0) return 128; // GH100
    }
    // Ampere / Ada
    if (major == 8) {
        if (minor == 0) return 64;  // GA100 (A100)
        if (minor == 6) return 128; // GA102 (RTX 30)
        if (minor == 9) return 128; // Ada Lovelace (RTX 40) — sm_89, not Hopper
    }
    // Volta
    if (major == 7) {
        if (minor == 0) return 64;  // V100
    }
    // Pascal
    if (major == 6) {
        if (minor == 0) return 64;  // P100
        if (minor == 1) return 128; // GTX 1080
    }
    return -1; // Unknown (e.g. sm_100 datacenter Blackwell: don't guess)
}

// --- 辅助函数：格式化存储大小 ---
std::string format_bytes(size_t bytes) {
    char buf[64];
    if (bytes < 1024) sprintf(buf, "%zu B", bytes);
    else if (bytes < 1024 * 1024) sprintf(buf, "%.2f KB", bytes / 1024.0);
    else if (bytes < 1024 * 1024 * 1024) sprintf(buf, "%.2f MB", bytes / 1024.0 / 1024.0);
    else sprintf(buf, "%.2f GB", bytes / 1024.0 / 1024.0 / 1024.0);
    return std::string(buf);
}

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    printf("=================================================================\n");
    printf("   AI System Performance Lab - Hardware Topology Detective   \n");
    printf("=================================================================\n");
    printf("Detected %d CUDA Capable Device(s)\n\n", device_count);

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        int cc_major = prop.major;
        int cc_minor = prop.minor;
        int cores_per_sm = get_cores_per_sm(cc_major, cc_minor);

        printf("[Device %d]: %s\n", dev, prop.name);
        printf("-----------------------------------------------------------------\n");

        // --- 1. 计算能力与架构 ---
        printf("  [Architecture]\n");
        printf("    Compute Capability      : %d.%d ", cc_major, cc_minor);
        if (cc_major == 12) printf("(Blackwell consumer, e.g. RTX 50 / sm_120)\n");
        else if (cc_major == 10 || cc_major == 11) printf("(Blackwell datacenter class)\n");
        else if (cc_major == 9) printf("(Hopper)\n");
        else if (cc_major == 8 && cc_minor == 9) printf("(Ada Lovelace — not Hopper)\n");
        else if (cc_major == 8) printf("(Ampere)\n");
        else printf("(Volta/Pascal or older)\n");

        // --- 2. SM 宏观拓扑 ---
        printf("  [Compute Topology]\n");
        printf("    Multiprocessors (SMs)   : %d\n", prop.multiProcessorCount);
        if (cores_per_sm != -1) {
            printf("    CUDA Cores / SM         : %d\n", cores_per_sm);
            printf("    Total CUDA Cores        : %d\n", cores_per_sm * prop.multiProcessorCount);
        } else {
            printf("    CUDA Cores / SM         : Unknown (Architecture not indexed)\n");
        }
#if defined(CUDA_VERSION) && CUDA_VERSION < 12000
        // clockRate 在 CUDA 12+ 中已移除
        printf("    GPU Clock Rate          : %.0f MHz\n", prop.clockRate / 1000.0);
#else
        printf("    GPU Clock Rate          : N/A (removed in CUDA 12+)\n");
#endif

        // --- 3. 内存体系 (Memory Wall 分析关键) ---
        printf("  [Memory Hierarchy]\n");
        printf("    Global Memory (HBM/DDR) : %s\n", format_bytes(prop.totalGlobalMem).c_str());
        printf("    Memory Bus Width        : %d-bit\n", prop.memoryBusWidth);
#if defined(CUDA_VERSION) && CUDA_VERSION < 12000
        // memoryClockRate 在 CUDA 12+ 中已移除
        printf("    Memory Clock Rate       : %.0f MHz\n", prop.memoryClockRate / 1000.0);

        // 计算理论峰值带宽: (Clock * BusWidth * 2(DDR)) / 8 bits
        // 注意：memoryClockRate 单位是 kHz
        double bandwidth_gbps = (prop.memoryClockRate * (prop.memoryBusWidth / 8.0) * 2.0) / 1e6;
        printf("    Theoretical Bandwidth   : %.2f GB/s\n", bandwidth_gbps);
#else
        printf("    Memory Clock Rate       : N/A (removed in CUDA 12+)\n");
        // 在 CUDA 12+ 中，需要使用 nvml 或其他 API 获取内存时钟频率
        // 这里暂时显示 N/A
        printf("    Theoretical Bandwidth   : N/A (use nvml API for accurate value)\n");
#endif

        printf("    L2 Cache Size           : %s (chip-wide; not per-GPC)\n", format_bytes(prop.l2CacheSize).c_str());

        // --- 4. SM 内部资源 (Occupancy 分析关键) ---
        printf("  [SM Micro-Architecture]\n");
        printf("    Max Shared Mem / Block  : %s\n", format_bytes(prop.sharedMemPerBlock).c_str());
        printf("    Max Shared Mem (Opt-in) : %s (Dynamic)\n", format_bytes(prop.sharedMemPerBlockOptin).c_str());
        printf("    Max Registers / Block   : %d\n", prop.regsPerBlock);
        printf("    Max Threads / Block     : %d\n", prop.maxThreadsPerBlock);
        printf("    Max Threads / SM        : %d\n", prop.maxThreadsPerMultiProcessor);
        printf("    Warp Size               : %d\n", prop.warpSize);

        // --- 5. 现代特性支持侦测 ---
        printf("  [Modern Features Support]\n");

        // Unified Addressing (UVA) - 基础
        printf("    Unified Addressing      : %s\n", prop.unifiedAddressing ? "Yes" : "No");

        // Managed Memory (Page Migration)
        printf("    Managed Memory          : %s\n", prop.managedMemory ? "Yes" : "No");

        // TMA / Cluster: ISA floor is sm_90+. This is not a measured speedup.
        const bool tma_isa_floor = (cc_major >= 9);
        printf("    TMA ISA floor (sm_90+)  : %s\n", tma_isa_floor ? "Yes (measure in B-08)" : "No");
        printf("    Cluster ISA floor       : %s\n", tma_isa_floor ? "Yes (measure in B-08)" : "No");

        printf("\n");
    }

    return 0;
}