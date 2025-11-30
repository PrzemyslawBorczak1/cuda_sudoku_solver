
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#include "stdio.h";



#define GUARD 82
#define NO_SOL 83

#define N 9
#define N2 81
#define SQRTN 3


#define THREADSPERBLOCK 1



//// do testow
//#include <iostream>
#include <chrono>
using namespace std;

struct SudokuInstance {
    char* empty;

    uint64_t* rows1;
    uint64_t* rows2;

    uint64_t* cols1;
    uint64_t* cols2;

    uint64_t* boxes1;
    uint64_t* boxes2;

    uint32_t* number; // moze byc zmiejszone na 16 jesli rozwazane co najwyzej 8k watkow

    uint32_t count;
};


__device__ void DFS(uint64_t rows1, uint64_t rows2,
    uint64_t cols1, uint64_t cols2,
    uint64_t boxes1, uint64_t boxes2,
    char* empty, char* stack);

__device__ void BFS(char* a, char* b, size_t* counter_a, size_t* counter_b);

__global__ void sudokuKernel(SudokuInstance si, char* board)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    if (gid >= (int)si.count) return;

    // Per-thread shared slice: [empty (N2 bytes) | stack (N2 bytes)]

    extern __shared__ char smem[];
    char* threadBase = smem + tid * (2 * N2);
    char* emptySh = threadBase;
    char* stackSh = threadBase + N2;


     char* emptyGlobal = si.empty + gid * N2;
    for (int i = 0; i < N2; ++i) {
        emptySh[i] = emptyGlobal[i];
		//printf("emptySh[%d] = %d\n", i, (int)emptySh[i]);
        if (emptyGlobal[i] == GUARD) break;
    }

	emptySh = emptyGlobal;


    DFS(
        si.rows1[gid], si.rows2[gid],
        si.cols1[gid], si.cols2[gid],
        si.boxes1[gid], si.boxes2[gid],
        emptySh, stackSh
    );


    if (stackSh[0] == NO_SOL)
        printf("%d No solution\n", gid);
    else {
        printf("%d Solution\n", gid);
        printf("%d empty: [%d,%d,%d]\n stack : [% d, % d, % d]\n", gid, emptySh[0], emptySh[1], emptySh[2], stackSh[0], stackSh[1], stackSh[2]);
    }

    return;
}

__device__ void sol_and_fill(uint64_t rows1, uint64_t rows2,
    uint64_t cols1, uint64_t cols2,
    uint64_t boxes1, uint64_t boxes2,
    char* empty, char* stack, char* board) {

    DFS(rows1, rows2,
        cols1, cols2,
        boxes1, boxes2,
        empty, stack);

	if (stack[0] == NO_SOL)
		printf("No solution\n");
    else {
		printf("Solution: ");
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






cudaError_t SudokuCuda(SudokuInstance si, char* board)
{


    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


	SudokuInstance d_si = {};

    size_t boardsCount = si.count;
    size_t maskCount64 = boardsCount * sizeof(uint64_t);
    size_t emptyCountChar = boardsCount * N2 * sizeof(char);
    size_t numberCount32 = boardsCount * sizeof(uint32_t);
    size_t boardsChars = boardsCount * N2 * sizeof(char); 



    if ((cudaStatus = cudaMalloc((void**)&d_si.rows1, maskCount64)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.rows2, maskCount64)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.cols1, maskCount64)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.cols2, maskCount64)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.boxes1, maskCount64)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.boxes2, maskCount64)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.empty, emptyCountChar)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.number, numberCount32)) != cudaSuccess) goto AllocError;
    char* d_boards = nullptr;
    if ((cudaStatus = cudaMalloc((void**)&d_boards, boardsCount * N2)) != cudaSuccess) goto AllocError;

    if ((cudaStatus = cudaMemcpy(d_si.rows1, si.rows1, maskCount64, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.rows2, si.rows2, maskCount64, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.cols1, si.cols1, maskCount64, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.cols2, si.cols2, maskCount64, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.boxes1, si.boxes1, maskCount64, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.boxes2, si.boxes2, maskCount64, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.empty, si.empty, emptyCountChar, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.number, si.number, numberCount32, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;

    d_si.count = si.count;

    int block =  d_si.count / THREADSPERBLOCK + 1;


    auto t0 = chrono::high_resolution_clock::now();


    sudokuKernel <<< block, THREADSPERBLOCK >>> (d_si, d_boards);

    auto t1 = chrono::high_resolution_clock::now();
    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    printf("alloc %.3f ms\n", us / 1000.0);


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
   /* cudaStatus = cudaMemcpy(solutions, dev_solutions, size_sol, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }*/

CopyError:
AllocError:
Error:
    // Cleanup
    cudaFree(d_si.rows1);
    cudaFree(d_si.rows2);
    cudaFree(d_si.cols1);
    cudaFree(d_si.cols2);
    cudaFree(d_si.boxes1);
    cudaFree(d_si.boxes2);
    cudaFree(d_si.empty);
    cudaFree(d_si.number);
    cudaFree(d_boards);

    return cudaStatus;
}


#include <string>
#include <cstdio> 
#include <chrono>

#define LINE_LEN 83

#define ERR(source) \
    (fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), perror(source), exit(EXIT_FAILURE))


#define BOARDS 80000 //80 tys




SudokuInstance create_instance() {

    SudokuInstance instance{};
    instance.empty = (char*)malloc(BOARDS * N2);

    instance.rows1 = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));
    instance.rows2 = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));

    instance.cols1 = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));
    instance.cols2 = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));

    instance.boxes1 = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));
    instance.boxes2 = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));

    instance.number = (uint32_t*)malloc(BOARDS * sizeof(uint32_t));

    return instance;
}

void clear_sudoku_instance(SudokuInstance* si) {
	free(si->empty);
	free(si->rows1);
	free(si->rows2);
	free(si->cols1);
	free(si->cols2);
	free(si->boxes1);
	free(si->boxes2);
	free(si->number);

}


bool test_set(uint64_t* part1, uint64_t* part2, int index, uint64_t mask) {

    if (index < 7) {
        if (*part1 & (mask << (index * 9))) {
            return false;
        }
        *part1 |= (mask << (index * 9));
    }
    else {
        index -= 7;
        if (*part2 & (mask << (index * 9))) {
            return false;
        }
        *part2 |= (mask << (index * 9));
    }
    return true;

}

int wrtie_board_to_instance(SudokuInstance si, int offset, char* board) {
    char empty_ptr = 0;
    si.rows1[offset] = si.rows2[offset] = 0;
    si.cols1[offset] = si.cols2[offset] = 0;
    si.boxes1[offset] = si.boxes2[offset] = 0;
    for (int i = 0; i < N2; ++i) {
        char ch = board[i];
        if (ch == '0') {
            si.empty[offset * N2 + empty_ptr++] = (char)i;
            continue;
        }

        int num = ch - '1';        // 0..8
        uint64_t mask = (uint64_t)1 << num;

        int row = i / 9;
        int col = i % 9;
        int box = (row / 3) * 3 + (col / 3);

        if (!test_set(si.rows1 + offset, si.rows2 + offset, row, mask))
            return 1;

        if (!test_set(si.cols1 + offset, si.cols2 + offset, col, mask))
            return 1;

        if (!test_set(si.boxes1 + offset, si.boxes2 + offset, box, mask))
            return 1;
    }
	si.empty[offset * N2 + empty_ptr] = GUARD;
    return 0;
}

// todo przetestowac jeszcze raz
void multiply_boards(SudokuInstance* sr, SudokuInstance* ds)
{
    SudokuInstance src = *sr;
    SudokuInstance dst = *ds;

    uint32_t out = 0;

    for (uint32_t i = 0; i < src.count; i++)
    {
        const size_t srcEmptyBase = (size_t)i * N2;
        char firstEmpty = src.empty[srcEmptyBase];
        // todo druga granica dla out
        if (firstEmpty == GUARD || out  >= BOARDS - i - N)
        { 
            if (out >= BOARDS) {
                fprintf(stderr, "multiply_boards: capacity exceeded (BOARDS)\n");
                break;
            }
            dst.rows1[out] = src.rows1[i];
            dst.rows2[out] = src.rows2[i];
            dst.cols1[out] = src.cols1[i];
            dst.cols2[out] = src.cols2[i];
            dst.boxes1[out] = src.boxes1[i];
            dst.boxes2[out] = src.boxes2[i];
            dst.empty[out * N2] = GUARD;
            dst.number[out] = src.number[i];
            ++out;
            continue;
        }

        int row = firstEmpty / N;
        int col = firstEmpty % N;
        int box = (row / SQRTN) * SQRTN + (col / SQRTN);

        uint64_t rowBits = (row < 7) ? (src.rows1[i] >> (row * 9)) : (src.rows2[i] >> ((row - 7) * 9));
        uint64_t colBits = (col < 7) ? (src.cols1[i] >> (col * 9)) : (src.cols2[i] >> ((col - 7) * 9));
        uint64_t boxBits = (box < 7) ? (src.boxes1[i] >> (box * 9)) : (src.boxes2[i] >> ((box - 7) * 9));

        for (int num = 0; num < 9; ++num)
        {
            uint64_t mask = (uint64_t)1ULL << num;

            if ((rowBits & mask) || (colBits & mask) || (boxBits & mask))
                continue; // digit not valid here

            if (out >= BOARDS) {
                fprintf(stderr, "multiply_boards: capacity exceeded while expanding\n");
                break;
            }

            // Copy base masks
            dst.rows1[out] = src.rows1[i];
            dst.rows2[out] = src.rows2[i];
            dst.cols1[out] = src.cols1[i];
            dst.cols2[out] = src.cols2[i];
            dst.boxes1[out] = src.boxes1[i];
            dst.boxes2[out] = src.boxes2[i];

            // Set the new digit bits
            if (row < 7)
                dst.rows1[out] |= mask << (row * 9);
            else
                dst.rows2[out] |= mask << ((row - 7) * 9);

            if (col < 7)
                dst.cols1[out] |= mask << (col * 9);
            else
                dst.cols2[out] |= mask << ((col - 7) * 9);

            if (box < 7)
                dst.boxes1[out] |= mask << (box * 9);
            else
                dst.boxes2[out] |= mask << ((box - 7) * 9);

            // Copy empty list excluding the first element
            size_t dstEmptyBase = (size_t)out * N2;
            int e = 0;
            for (;;)
            {
                char srcVal = src.empty[srcEmptyBase + 1 + e];
                dst.empty[dstEmptyBase + e] = srcVal;
                if (srcVal == GUARD)
                    break;
                ++e;
            }

            // Preserve board identification (or customize if needed)
            dst.number[out] = src.number[i];
            ++out;
        }
    }

    (*ds).count = out;
}



SudokuInstance populate_boards(char* boards, int count) {
	SudokuInstance generation_a = create_instance();

    SudokuInstance generation_b = create_instance();

    for (int i = 0; i < count; i++) {
        if(wrtie_board_to_instance(generation_a, i, boards + i * LINE_LEN) != 0) {
            fprintf(stderr, "Invalid board at line %d\n", i + 1);
            exit(EXIT_FAILURE);
		}
		generation_a.number[i] = i;
    }

    generation_a.count = count;


    const int MAX_ITER = 2; // adjust as needed
    int iter = 0;
    SudokuInstance* src = &generation_a;
    SudokuInstance* dst = &generation_b;
    uint32_t prevCount = src->count;
    while (iter < MAX_ITER && prevCount > 0) {
        multiply_boards(src, dst);
        double factor = prevCount ? (double)dst->count / (double)prevCount : 0.0;
        printf("iter %d: %u -> %u (x%.3f)\n", iter, prevCount, dst->count, factor);

        
        // Prepare next iteration
        prevCount = dst->count;
        SudokuInstance* tmp = src;
        src = dst;
        dst = tmp;
        iter++;
    }

	clear_sudoku_instance(dst);

	return *src;

}




void usage() {
    printf("Usage:\n");
    printf("  sudoku method count input_file output_file\n");
    printf("Where:\n");
    printf("  method       : cpu | gpu\n");
    printf("  count        : positive integer (<= number of lines in input_file)\n");
    printf("  input_file   : path to input text file with 81-digit boards per line (0 for empty)\n");
    printf("  output_file  : path to output text file (will be created/overwritten)\n");
}


char* read_to_buff(int count, char* path) {


    FILE* source = fopen(path, "rb");
    if (source == NULL) {
        fprintf(stderr, "Could not open input file: '%s'\n", path);
        usage();
        exit(EXIT_FAILURE);
    }

    size_t expected = (size_t)count * LINE_LEN * sizeof(char);

    char* buff = (char*)malloc(expected);
    if (!buff) {
        fprintf(stderr, "malloc bulk buffer");

        exit(EXIT_FAILURE);
    }

    size_t got = fread(buff, 1, expected, source);
    if (got != expected) {
        if (feof(source))
            fprintf(stderr, "Too short file got %d lines but needs %d lines\n", got / LINE_LEN, expected / LINE_LEN);
        usage();
        exit(EXIT_FAILURE);
    }

    fclose(source);

    return buff;
}
// todo debuugiing
void print_uint64_bits(uint64_t value)
{
    char buf[64 + 32 + 8 + 2]; // bits + spaces + 7 delimiters + newline + '\0'
    int pos = 0;
    for (int i = 0; i < 64; ++i)
    {
        buf[pos++] = (char)(((value >> i) & 1ULL) ? '1' : '0');

        // Space after each 4-bit chunk
        if (i % 4 == 3)
            buf[pos++] = ' ';

        // Delimiter after each 9-bit group (i = 8,17,26,35,44,53,62)
        if (i % 9 == 8 && i != 63) // skip after last full group unless you want trailing '|'
            buf[pos++] = '|';
    }
    buf[pos++] = '\n';
    buf[pos] = '\0';
    printf("%s", buf);
}

void print_board_pretty(const char* board81, int id)
{
    printf("Board %d:\n", id);
    printf("+-------+-------+-------+\n");
    for (int r = 0; r < 9; ++r)
    {
        printf("| ");
        for (int c = 0; c < 9; ++c)
        {
            char ch = board81[r * 9 + c];
            if (ch == '0') ch = '.';
            printf("%c ", ch);
            if (c % 3 == 2) printf("| ");
        }
        printf("\n");
        if (r % 3 == 2)
            printf("+-------+-------+-------+\n");
    }
}

/// todo dotad

int main(int argc, char* argv[])
{

    argc = 5;

    argv[1] = "gpu";
    argv[2] = "3";
    argv[3] = "C:\\Users\\przem\\Pulpit\\cuda\\P1\\additional\\sudoku_data\\sudoku_data.csv";

   if (argc != 5) {
        fprintf(stderr, "Invalid arguments. Expected exactly 4 parameters.\n");
        usage();
        return 1;
    }
    

    const char* method = argv[1];
    const char* countStr = argv[2];
    const char* outputPath = argv[4];

    if (strcmp(method, "cpu") != 0 && strcmp(method, "gpu") != 0) {
        fprintf(stderr, "Invalid method: '%s'. Allowed: 'cpu' or 'gpu'.\n", method);
        usage();
		exit(EXIT_FAILURE);
    }

    int count = stoi(countStr);
    if (count < 0) {
        fprintf(stderr, "Invalid count: '%d'. Must be a positive integer.\n", count);
        usage();
        exit(EXIT_FAILURE);
    }


    auto t0 = chrono::high_resolution_clock::now();
    char* buff = read_to_buff(count, argv[3]);
	SudokuInstance si = populate_boards(buff, count);

    auto t1 = chrono::high_resolution_clock::now();
    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    printf("time %.3f ms\n", us / 1000.0);
    
    /*
    for (int j = 0; j < count; ++j)
    {
        const char* line = buff + j * LINE_LEN;

        bool ok = true;
        for (int k = 0; k < 81; ++k)
        {
            char ch = line[k];
            if (ch < '0' || ch > '9')
            {
                ok = false;
				fprintf(stderr, "Invalid character '%c' in board %d at position %d\n", ch, j, k);
            }
        }
        if (!ok)
        {
            fprintf(stderr, "Invalid character in board %d\n", j);
            return 1;
        }

        print_board_pretty(line, j);
    }
    */


    t0 = chrono::high_resolution_clock::now();

    cudaError_t cudaStatus = SudokuCuda(si, buff);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "SudokuCuda failed!");
        return 1;
    }


     t1 = chrono::high_resolution_clock::now();
     us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
     printf("kernel time %.3f ms\n", us / 1000.0);



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}