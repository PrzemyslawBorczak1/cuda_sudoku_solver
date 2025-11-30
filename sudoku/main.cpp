#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include "../include/sudoku_types.hpp"  // fixed relative path

// CPU helpers
SudokuInstance populate_boards(const char* bulk, int count);
char* read_to_buffer(int count, const char* path);
void clear_sudoku_instance(SudokuInstance& si);

static void usage() {
    std::printf("Usage: sudoku method count input_file\n");
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        usage();
        return 1;
    }
    const char* method = argv[1];
    int count = std::stoi(argv[2]);
    const char* inputPath = argv[3];

    if (count <= 0) {
        std::fprintf(stderr, "Invalid count\n");
        return 1;
    }
    if (std::strcmp(method, "gpu") != 0 && std::strcmp(method, "cpu") != 0) {
        std::fprintf(stderr, "Invalid method\n");
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    char* bulk = read_to_buffer(count, inputPath);
    SudokuInstance si = populate_boards(bulk, count);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::printf("Preparation %.3f ms\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    if (std::strcmp(method, "gpu") == 0) {
        cudaError_t st = SudokuCuda(si, nullptr);
        if (st != cudaSuccess) {
            std::fprintf(stderr, "SudokuCuda failed: %s\n", cudaGetErrorString(st));
        } else {
            std::printf("Kernel completed OK\n");
        }
    } else {
        std::printf("CPU mode not implemented yet.\n");
    }

    clear_sudoku_instance(si);
    std::free(bulk);
    return 0;
}