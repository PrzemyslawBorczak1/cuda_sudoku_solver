#pragma once
#include "sudoku_types.hpp"

// Kernel (one board per thread)
__global__ void sudokuKernel(SudokuInstance si, const char* boards);

// Launch helper
cudaError_t LaunchSudokuKernel(const SudokuInstance& dSi, const char* dBoards);