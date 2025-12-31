# CUDA Sudoku Solver (DFS)

This project implements a **Sudoku solver using Depth-First Search (DFS) with backtracking**, accelerated using **CUDA**.  
The solver uses GPU parallelism to speed up solving standard 9×9 Sudoku puzzles.

## Features

- Solves 9×9 Sudoku puzzles
- DFS and backtracking algorithm
- CUDA-based parallel implementation

## Technologies Used

- C++
- CUDA (NVCC)
- NVIDIA GPU

## Input Format

The Sudoku puzzle is provided as a text file.  
Empty cells are represented using `0`.

# Example
000400560010506090000097300009020040600005000000370000502000000063000000000960800

# Compling

## Cmake
```bash
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release .
cmake --build . --config Release
```


## make
```bash
make 
```


# Usage
```bash
sudoku gpu <count> <input_file> <output_file>
```


