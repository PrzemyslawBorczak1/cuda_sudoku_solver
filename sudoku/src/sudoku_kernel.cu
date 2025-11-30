#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include "sudoku_types.hpp"
#include "sudoku_kernel.cuh"

__device__ void DFS(uint64_t rows1, uint64_t rows2,
    uint64_t cols1, uint64_t cols2,
    uint64_t boxes1, uint64_t boxes2,
    char* empty, char* stack)
{
    char depth = 0;
    stack[0] = -1;
    bool found = true;
    while (depth >= 0) {
        char ind = empty[depth];
        if (ind == GUARD) return;
        char row = ind / 9;
        char col = ind % 9;
        char box = (row / 3) * 3 + (col / 3);

        uint64_t* row_w;
        uint64_t* col_w;
        uint64_t* box_w;

        if (row < 7) row_w = &rows1; else { row -= 7; row_w = &rows2; }
        if (col < 7) col_w = &cols1; else { col -= 7; col_w = &cols2; }
        if (box < 7) box_w = &boxes1; else { box -= 7; box_w = &boxes2; }

        if (!found) {
            if (depth < 0) { stack[0] = NO_SOL; return; }
            uint64_t mask = (uint64_t)1 << stack[depth];
            *row_w &= ~(mask << (row * 9));
            *col_w &= ~(mask << (col * 9));
            *box_w &= ~(mask << (box * 9));
        }

        found = false;
        for (char num = stack[depth] + 1; num < 9; ++num) {
            uint64_t mask = (uint64_t)1 << num;
            if ((*row_w >> (row * 9) & mask) ||
                (*col_w >> (col * 9) & mask) ||
                (*box_w >> (box * 9) & mask)) continue;

            found = true;
            stack[depth] = num;
            *row_w |= mask << (row * 9);
            *col_w |= mask << (col * 9);
            *box_w |= mask << (box * 9);
            depth++;
            stack[depth] = -1;
            break;
        }
        if (!found) depth--;
    }
}

__global__ void sudokuKernel(SudokuInstance si, const char* boards)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (int)si.count) return;

    extern __shared__ char smem[];
    char* emptySh = smem + threadIdx.x * (2 * N2);
    char* stackSh = emptySh + N2;

    const char* emptyGlobal = si.empty + gid * N2;
    for (int i = 0; i < N2; ++i) {
        emptySh[i] = emptyGlobal[i];
        if (emptyGlobal[i] == GUARD) break;
    }

    DFS(si.rows1[gid], si.rows2[gid],
        si.cols1[gid], si.cols2[gid],
        si.boxes1[gid], si.boxes2[gid],
        emptySh, stackSh);

    // TODO: write back solution if needed
}

cudaError_t LaunchSudokuKernel(const SudokuInstance& dSi, const char* dBoards)
{
    int threads = 128;
    int blocks = (dSi.count + threads - 1) / threads;
    size_t sharedBytes = (size_t)threads * (2 * N2);
    sudokuKernel<<<blocks, threads, sharedBytes>>>(dSi, dBoards);
    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) return st;
    return cudaDeviceSynchronize();
}

cudaError_t SudokuCuda(const SudokuInstance& hostSi, const char* boardsPacked)
{
    cudaError_t st;
    SudokuInstance dSi{};
    dSi.count = hostSi.count;
    size_t count = hostSi.count;
    size_t masks   = count * sizeof(uint64_t);
    size_t empties = count * N2 * sizeof(char);
    size_t numbers = count * sizeof(uint32_t);

    auto alloc = [&](void** ptr, size_t sz) {
        st = cudaMalloc(ptr, sz);
        return st == cudaSuccess;
    };
    if (!alloc((void**)&dSi.rows1, masks) ||
        !alloc((void**)&dSi.rows2, masks) ||
        !alloc((void**)&dSi.cols1, masks) ||
        !alloc((void**)&dSi.cols2, masks) ||
        !alloc((void**)&dSi.boxes1, masks) ||
        !alloc((void**)&dSi.boxes2, masks) ||
        !alloc((void**)&dSi.empty, empties) ||
        !alloc((void**)&dSi.number, numbers)) goto Cleanup;

    auto cpy = [&](void* dst, const void* src, size_t sz) {
        st = cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice);
        return st == cudaSuccess;
    };
    if (!cpy(dSi.rows1, hostSi.rows1, masks) ||
        !cpy(dSi.rows2, hostSi.rows2, masks) ||
        !cpy(dSi.cols1, hostSi.cols1, masks) ||
        !cpy(dSi.cols2, hostSi.cols2, masks) ||
        !cpy(dSi.boxes1, hostSi.boxes1, masks) ||
        !cpy(dSi.boxes2, hostSi.boxes2, masks) ||
        !cpy(dSi.empty, hostSi.empty, empties) ||
        !cpy(dSi.number, hostSi.number, numbers)) goto Cleanup;

    st = LaunchSudokuKernel(dSi, boardsPacked);

Cleanup:
    cudaFree(dSi.rows1);
    cudaFree(dSi.rows2);
    cudaFree(dSi.cols1);
    cudaFree(dSi.cols2);
    cudaFree(dSi.boxes1);
    cudaFree(dSi.boxes2);
    cudaFree(dSi.empty);
    cudaFree(dSi.number);
    return st;
}