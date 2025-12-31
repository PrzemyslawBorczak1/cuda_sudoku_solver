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

# Usage
sudoku method count input_file output_file
