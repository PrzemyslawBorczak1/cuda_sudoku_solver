
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define GUARD 82
#define NO_SOL 83

#define N 9
#define N2 81
#define SQRTN 3

//// do testow
//#include <iostream>
//#include <chrono>
//using namespace std;


__device__ void parse_and_run_dfs(char* board, char* sol, size_t* counter_sol);
__device__ void DFS(uint64_t rows1, uint64_t rows2,
    uint64_t cols1, uint64_t cols2,
    uint64_t boxes1, uint64_t boxes2,
    char* empty, char* stack);

__global__ void sudokuKernel(char* a, char* b, char* sol, size_t* counter_a, size_t* counter_b)
{
    int i = threadIdx.x; 
    // TODO do zmiany
    for(int i = 0; i < N2; i++)
    {
		//a[i] = a[i] - '1';
	}
   

    parse_and_run_dfs(a, sol, 0);
}


/// <summary>
/// Przyjmuje board jako char table 81 znakow ASCII ('0' dla pustego, '1'..'9') i wypelnia sol jako tablice 81 char (wewnetrzne cyfry 0..8 lub NO_SOL).
/// </summary>
__device__ void parse_and_run_dfs(char* board, char* sol, size_t* counter_sol) {
    uint64_t rows1 = 0;
    uint64_t rows2 = 0;
    uint64_t cols1 = 0;
    uint64_t cols2 = 0;
    uint64_t boxes1 = 0;
    uint64_t boxes2 = 0;

    //  inaczej zadklarowac?
    char empty[N2];// mniej jesli zalozyc mozna minimalna ilosc wypelnionych pol
    char empty_ptr = 0;

    char stack[N2];// mniej bo mozna odjac empty_ptr ale wtedy deklarowana dynamicznie

    char col = 0;
    char row = 0;
    char box = 0;
    for (char i = 0; i < N2; i++) {
        char num = board[i] - '1'; // TODO po BFS w zaleznosci od reprezentacji inne ustawianie

        if (num == -1) {
            empty[empty_ptr++] = i;
        }
        else {

            uint64_t bit_mask = (uint64_t)1 << (num);

            if (row < 7) {
                rows1 |= bit_mask << (row * 9);
            }
            else {
                rows2 |= bit_mask << ((row - 7) * 9);
            }

            if (col < 7) {
                cols1 |= bit_mask << (col * 9);
            }
            else {
                cols2 |= bit_mask << ((col - 7) * 9);
            }

            if (box < 7) {
                boxes1 |= bit_mask << (box * 9);
            }
            else {
                boxes2 |= bit_mask << ((box - 7) * 9);
            }
        }




        col += 1;
        if (col == N) {
            col = 0;
            row += 1;
            if (row % SQRTN == 0) {
                box = row;
            }
            else {
                box -= SQRTN - 1;
            }
        }
        else if (col % SQRTN == 0) {
            box += 1;
        }
    }

    empty[empty_ptr] = GUARD;
    // w ostateczenj wersji przepisane jako jedna funckja zeby nie kopiowac zmiennych rows1 rows2 etc
    DFS(rows1, rows2,
        cols1, cols2,
        boxes1, boxes2,
        empty, stack
    );

    // powinno byc inaczej
    if (stack[0] == NO_SOL)
    {
        // no sol
        sol[0] = NO_SOL;
    }

    for (int i = 0; i < N2; i++) {
        if (board[i] != 0) {
            sol[i] = board[i] - '1';
        }
    }

    int i = 0;
    char num = empty[i];
    while(num != GUARD) {
        sol[num] = stack[i];
        i++;
        num = empty[i];
    }

}

__device__ void DFS(uint64_t rows1, uint64_t rows2,
    uint64_t cols1, uint64_t cols2,
    uint64_t boxes1, uint64_t boxes2,
    char* empty, char* stack)
{
    char depth = 0;
    // every iteration starts with stack[] + 1 so the placeholder number needs to be -1
    stack[0] = -1;

    bool found = true;
    while (depth >= 0) {
        // at the end of indcies of empty indexes should be guard
        char ind = empty[depth];
        if (ind == GUARD) {
            // solved!!
            return;
        }

        // setup in wchich row, col, box we are
        char row = ind / 9;
        char col = ind % 9;
        char box = (row / 3) * 3 + (col / 3);

        uint64_t* row_w, * col_w, * box_w;

        // determine which variable has appropriate bits
        if (row < 7) {
            row_w = &rows1;
        }
        else {
            row -= 7;
            row_w = &rows2;
        }

        if (col < 7) {
            col_w = &cols1;
        }
        else {
            col -= 7;
            col_w = &cols2;
        }

        if (box < 7) {
            box_w = &boxes1;
        }
        else {
            box -= 7;
            box_w = &boxes2;
        }

        // deleteing mask of number that is being backtracked
        if (!found) {
            if (depth < 0) {
                // no sol
                stack[0] = NO_SOL;
                return;
            }

            uint64_t mask = (uint64_t)1 << (stack[depth]);

            *row_w &= ~(mask << (row * 9));
            *col_w &= ~(mask << (col * 9));
            *box_w &= ~(mask << (box * 9));

        }


        found = false;
        for (char num = stack[depth] + 1; num < 9; num++) {

            uint64_t mask = (uint64_t)1 << (num);

            if ((*row_w >> (row * 9) & mask) ||
                (*col_w >> (col * 9) & mask) ||
                (*box_w >> (box * 9) & mask))
                continue;

            found = true;
            stack[depth] = num;

            (*row_w) |= mask << (row * 9);
            (*col_w) |= mask << (col * 9);
            (*box_w) |= mask << (box * 9);


            depth++;
            stack[depth] = -1;
            break;

        }
        if (!found)
            depth--;

    }
}





cudaError_t SudokuCuda(const char* board, size_t size_buffer, char* solutions, size_t size_sol)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

   

	char* dev_a = 0;
	char* dev_b = 0;
	char* dev_solutions = 0;
	size_t* counter_a = 0;
    size_t* counter_b = 0;
	const int size_board = 81 * sizeof(char);

    cudaStatus = cudaMalloc((void**)&dev_a, size_buffer );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size_buffer);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_solutions, size_sol);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, board, size_board, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }



    cudaStatus = cudaMalloc((void**)&counter_a, sizeof(size_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&counter_b, sizeof(size_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    sudokuKernel << <1, 1 >> > (dev_a, dev_b, dev_solutions, counter_a, counter_b);


	//printf("size_t: %d\nsolutions: %d\nboard: %d\nsize: %d\n", sizeof(size_t), sizeof(solutions), sizeof(board), size_buffer);

 //   // test 
 //  sudokuKernel << <1, 1 >> > (dev_a, dev_b, dev_solutions, counter_a, counter_b);
 //   cudaDeviceSynchronize();


 //   // If you want host wall-clock including launch + sync:
 //   auto t0 = std::chrono::high_resolution_clock::now();
 //   sudokuKernel << <1, 1 >> > (dev_a, dev_b, dev_solutions, counter_a, counter_b);
 //   cudaDeviceSynchronize();
 //   auto t1 = std::chrono::high_resolution_clock::now();
 //   auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
 //   printf("Kernel time (host wall): %.3f milisekund\n", us / 1000.0);
	//// koniec testu




    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "sudokuKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sudokuKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(solutions, dev_solutions, size_sol, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_solutions);
    
    return cudaStatus;
}




int main()
{
    const char board[] = "530070000600195000098000060800060003400803001700020006060000280000419005000080079";
    // przy parsowaniu danych mozna obliczyc max ilosc wolnych pol wiec tez glebokosc DFS
	const size_t size = 1024 * 1024 * 1024; // 1 GB
	char* soulutions = new char[size / 1024];

    cudaError_t cudaStatus = SudokuCuda(board, size, soulutions, size/1024);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "SudokuCuda failed!");
        return 1;
    }

    for (int i = 0; i < 81; i++) {
        printf("%d ", soulutions[i] + 1);
        if ((i + 1) % 9 == 0) printf("\n");
    }
    printf("\n");


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}