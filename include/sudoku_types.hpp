#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#define GUARD 82
#define NO_SOL 83
#define N 9
#define N2 81
#define SQRTN 3

struct SudokuInstance {
    char* empty;
    uint64_t* rows1;
    uint64_t* rows2;
    uint64_t* cols1;
    uint64_t* cols2;
    uint64_t* boxes1;
    uint64_t* boxes2;
    uint32_t* number;
    uint32_t count;
};

cudaError_t SudokuCuda(const SudokuInstance& hostSi, const char* boardsPacked);