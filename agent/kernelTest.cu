#include <cstdio>
#include <cstdlib>

#include "benchmark_runner.h"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::printf("Wrong inputs. Correct usage: ./kernelTest M K N SPLIT_K\n");
        return 0;
    }

    const int M_GLOBAL = std::atoi(argv[1]);
    const int K_GLOBAL = std::atoi(argv[2]);
    const int N_GLOBAL = std::atoi(argv[3]);
    const int SPLIT_K = std::atoi(argv[4]);

    std::printf("M_GLOBAL: %d, K_GLOBAL: %d, N_GLOBAL: %d, SPLIT_K: %d\n",
                M_GLOBAL,
                K_GLOBAL,
                N_GLOBAL,
                SPLIT_K);
    return RunKernelTest(M_GLOBAL, K_GLOBAL, N_GLOBAL, SPLIT_K);
}
