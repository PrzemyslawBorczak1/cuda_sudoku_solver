#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "stdio.h"


#define GUARD 82
#define NO_SOL 83

#define N 9
#define N2 81
#define SQRTN 3


#define THREADSPERBLOCK 1


#include <chrono>
using namespace std;

struct SudokuInstance {
    char* empty;

    uint16_t* rows;   // size: count * 9
    uint16_t* cols;   // size: count * 9
    uint16_t* boxes;  // size: count * 9

    uint32_t* number; // moze byc zmiejszone na 16 jesli rozwazane co najwyzej 8k watkow

    char* boards;

    uint32_t count;
};

struct Solutions {
    int* flags;   
    char* stacks;       
    char* empties;      
};


__device__ void DFS(uint16_t* rows,
    uint16_t* cols,
    uint16_t* boxes,
    char* empty, char* stack);

__global__ void sudokuKernel(SudokuInstance si, Solutions sl)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    if (gid >= (int)si.count) return;

    // Per-thread shared slice: [empty (N2 bytes) | stack (N2 bytes)]

    extern __shared__ char smem[];
    char* threadBase = smem + tid * (2 * N2);
    char* emptySh = threadBase;
   // char* stackSh = threadBase + N2;


    char* emptyGlobal = si.empty + gid * N2;
    for (int i = 0; i < N2; ++i) {
        emptySh[i] = emptyGlobal[i];
        if (emptyGlobal[i] == GUARD) break;
    }

    emptySh = emptyGlobal;

    // Load 16-bit occupancy masks for this board into local arrays
    uint16_t rowsLoc[9];
    uint16_t colsLoc[9];
    uint16_t boxesLoc[9];
    const int base = gid * 9;
    for (int i = 0; i < 9; ++i) {
        rowsLoc[i] = si.rows[base + i];
        colsLoc[i] = si.cols[base + i];
        boxesLoc[i] = si.boxes[base + i];
    }

	char stackSh[N2];

    DFS(
        rowsLoc, colsLoc, boxesLoc,
		emptyGlobal, stackSh
    );

    if (stackSh[0] == NO_SOL)
        return;

     
    if (atomicExch(sl.flags + gid, 1) != 0) {
        atomicExch(sl.flags + gid, 2);
        return;
    }



	//todo memcpy??
	uint32_t base2 = si.number[gid] * N2;
    for(int i = 0; i < N2; ++i) {
        if (emptySh[i] == GUARD)
            break;

		sl.empties[base2 + i] = emptySh[i];
		sl.stacks[base2 + i] = stackSh[i];
	}


    return;
}


__device__ void DFS(uint16_t* rows,
    uint16_t* cols,
    uint16_t* boxes,
    char* empty, char* stack)
{
    char depth = 0;
    // every iteration starts with stack[] + 1 so the placeholder number needs to be -1
    stack[0] = -1;

    bool found = true;
    while (true) {
        if (depth < 0) {
            // no sol
            stack[0] = NO_SOL;
            return;
        }
        // at the end of indcies of empty indexes should be guard
        char ind = empty[depth];
        if (ind == GUARD) {
            // solved!!
            return;
        }

        // setup in wchich row, col, box we are
        char row = ind / 9;
        char col = ind % 9;
        char box = (char)((row / 3) * 3 + (col / 3));

        // deleteing mask of number that is being backtracked
        if (!found) {
            

            uint16_t mask = (uint16_t)1u << (stack[depth]);

            rows[row] &= (uint16_t)~mask;
            cols[col] &= (uint16_t)~mask;
            boxes[box] &= (uint16_t)~mask;
        }

        found = false;
        for (char num = (char)(stack[depth] + 1); num < 9; num++) {

            uint16_t mask = (uint16_t)1u << (num);

            if ((rows[row] & mask) ||
                (cols[col] & mask) ||
                (boxes[box] & mask))
                continue;

            found = true;
            stack[depth] = num;

            rows[row] |= mask;
            cols[col] |= mask;
            boxes[box] |= mask;

            depth++;
            stack[depth] = -1;
            break;

        }
        if (!found)
            depth--;

    }
}





cudaError_t SudokuCuda(SudokuInstance si, char* board, Solutions* ret)
{

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    SudokuInstance d_si = {};
    Solutions d_sl = {};
	Solutions h_sl = {};



    size_t solCount = (size_t)si.count * N2 * sizeof(char);
    size_t flagCount = (size_t)si.count * sizeof(int);

	h_sl.empties = (char*)malloc(solCount);
    h_sl.stacks = (char*)malloc(solCount);
    h_sl.flags = (int*)malloc(flagCount);
    

    size_t boardsCount = si.count;
    size_t maskCount16x9 = boardsCount * 9 * sizeof(uint16_t);
    size_t emptyCountChar = boardsCount * N2 * sizeof(char);
    size_t numberCount32 = boardsCount * sizeof(uint32_t);
    size_t boardsChars = boardsCount * N2 * sizeof(char);


    if ((cudaStatus = cudaMalloc((void**)&d_si.rows, maskCount16x9)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.cols, maskCount16x9)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.boxes, maskCount16x9)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.empty, emptyCountChar)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_si.number, numberCount32)) != cudaSuccess) goto AllocError;

    if ((cudaStatus = cudaMalloc((void**)&d_sl.empties, solCount)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_sl.stacks, solCount)) != cudaSuccess) goto AllocError;
    if ((cudaStatus = cudaMalloc((void**)&d_sl.flags, solCount)) != cudaSuccess) goto AllocError;
	if ((cudaStatus = cudaMemset(d_sl.flags, 0, flagCount)) != cudaSuccess) goto AllocError;

    if ((cudaStatus = cudaMemcpy(d_si.rows, si.rows, maskCount16x9, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.cols, si.cols, maskCount16x9, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.boxes, si.boxes, maskCount16x9, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.empty, si.empty, emptyCountChar, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(d_si.number, si.number, numberCount32, cudaMemcpyHostToDevice)) != cudaSuccess) goto CopyError;

    d_si.count = si.count;

    int block = d_si.count / THREADSPERBLOCK + 1;
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sudokuKernel!\n", cudaStatus);
        goto Error;
    }


    
    auto t0 = chrono::high_resolution_clock::now();

    sudokuKernel <<< block, THREADSPERBLOCK >>> (d_si, d_sl);


    if ((cudaStatus = cudaMemcpy(h_sl.stacks, d_sl.stacks, solCount, cudaMemcpyDeviceToHost)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(h_sl.empties, d_sl.empties, solCount, cudaMemcpyDeviceToHost)) != cudaSuccess) goto CopyError;
    if ((cudaStatus = cudaMemcpy(h_sl.flags, d_sl.flags, flagCount, cudaMemcpyDeviceToHost)) != cudaSuccess) goto CopyError;

	*ret = h_sl;

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

    auto t1 = chrono::high_resolution_clock::now();
    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    printf("kernell call %.3f ms\n", us / 1000.0);


CopyError:
AllocError:
Error:
    // Cleanup
    cudaFree(d_si.rows);
    cudaFree(d_si.cols);
    cudaFree(d_si.boxes);
    cudaFree(d_si.empty);
    cudaFree(d_si.number);

	cudaFree(d_sl.empties);
    cudaFree(d_sl.stacks);

    return cudaStatus;
}


#include <string>
#include <cstdio> 
#include <chrono>

#define LINE_LEN 83

#define ERR(source) \
    (fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), perror(source), exit(EXIT_FAILURE))


#define BOARDS 40000 //40 tys




SudokuInstance create_instance() {

    SudokuInstance instance{};
    instance.empty = (char*)malloc(BOARDS * N2);

    instance.rows = (uint16_t*)malloc(BOARDS * N * sizeof(uint16_t));
    instance.cols = (uint16_t*)malloc(BOARDS * N * sizeof(uint16_t));
    instance.boxes = (uint16_t*)malloc(BOARDS * N * sizeof(uint16_t));

    instance.number = (uint32_t*)malloc(BOARDS * sizeof(uint32_t));

    instance.boards = (char*)malloc(BOARDS * N2);

    return instance;
}

void free_sudoku_instance(SudokuInstance* si) {
    free(si->empty);
    free(si->rows);
    free(si->cols);
    free(si->boxes);
    free(si->number);

}


bool test_set(uint16_t* part, int index, uint16_t mask) {

    if (part[index] & mask) {
        return false;
    }
    part[index] |= mask;
    return true;

}

int wrtie_board_to_instance(SudokuInstance si, int board_nb, char* board) {
    char empty_ptr = 0;

    // zero out 9 masks for rows/cols/boxes for this board
    const int base = board_nb * 9;
    for (int i = 0; i < 9; ++i) {
        si.rows[base + i] = 0;
        si.cols[base + i] = 0;
        si.boxes[base + i] = 0;
    }

    for (int i = 0; i < N2; ++i) {
        si.boards[board_nb * N2 + i] = board[i];
        char ch = board[i];
        if (ch == '0') {
            si.empty[board_nb * N2 + empty_ptr++] = (char)i;
            continue;
        }

        int num = ch - '1';        // 0..8
        uint16_t mask = (uint16_t)1u << num;

        int row = i / 9;
        int col = i % 9;
        int box = (row / 3) * 3 + (col / 3);

        if (!test_set(si.rows + base, row, mask))
            return 1;

        if (!test_set(si.cols + base, col, mask))
            return 1;

        if (!test_set(si.boxes + base, box, mask))
            return 1;
    }
    si.empty[board_nb * N2 + empty_ptr] = GUARD;
    return 0;
}

// todo przetestowac jeszcze raz
void multiply_boards(SudokuInstance* sr, SudokuInstance* ds)
{
    SudokuInstance& src = *sr;
    SudokuInstance& dst = *ds;

    uint32_t out = 0;

    for (uint32_t i = 0; i < src.count; i++)
    {
        const size_t srcEmptyBase = (size_t)i * N2;
        char firstEmpty = src.empty[srcEmptyBase];

        // If no empties, copy board as-is (already solved or invalid)
        if (firstEmpty == GUARD || out >= BOARDS - N)
        {
            if (out >= BOARDS) {
                fprintf(stderr, "multiply_boards: capacity exceeded (BOARDS)\n");
                break;
            }

            // Copy board chars
            const size_t srcBoardBase = (size_t)i * N2;
            const size_t dstBoardBase = (size_t)out * N2;
            for (int k = 0; k < N2; ++k)
                dst.boards[dstBoardBase + k] = src.boards[srcBoardBase + k];

            // Copy masks (9 entries each)
            const int srcBase = (int)i * 9;
            const int dstBase = (int)out * 9;
            for (int k = 0; k < 9; ++k) {
                dst.rows[dstBase + k] = src.rows[srcBase + k];
                dst.cols[dstBase + k] = src.cols[srcBase + k];
                dst.boxes[dstBase + k] = src.boxes[srcBase + k];
            }

            // Empty list is just GUARD
            dst.empty[out * N2] = GUARD;

            // Preserve identifier
            dst.number[out] = src.number[i];
            ++out;
            continue;
        }

        // Determine row/col/box indices
        int row = firstEmpty / N;
        int col = firstEmpty % N;
        int box = (row / SQRTN) * SQRTN + (col / SQRTN);

        const int srcBase = (int)i * 9;

        // Fetch per-group occupancy bitsets
        uint16_t rowBits = src.rows[srcBase + row];
        uint16_t colBits = src.cols[srcBase + col];
        uint16_t boxBits = src.boxes[srcBase + box];

        for (int num = 0; num < 9; ++num)
        {
            uint16_t mask = (uint16_t)1u << num;

            // Skip digits already present in row/col/box
            if ((rowBits & mask) || (colBits & mask) || (boxBits & mask))
                continue;

            if (out >= BOARDS) {
                fprintf(stderr, "multiply_boards: capacity exceeded while expanding\n");
                break;
            }

            const int dstBase = (int)out * 9;

            // 1) Copy base masks
            for (int k = 0; k < 9; ++k) {
                dst.rows[dstBase + k] = src.rows[srcBase + k];
                dst.cols[dstBase + k] = src.cols[srcBase + k];
                dst.boxes[dstBase + k] = src.boxes[srcBase + k];
            }

            // 2) Set the new digit bits
            dst.rows[dstBase + row] |= mask;
            dst.cols[dstBase + col] |= mask;
            dst.boxes[dstBase + box] |= mask;

            // 3) Copy the source board and write the chosen digit at firstEmpty
            const size_t srcBoardBase = (size_t)i * N2;
            const size_t dstBoardBase = (size_t)out * N2;
            for (int k = 0; k < N2; ++k)
                dst.boards[dstBoardBase + k] = src.boards[srcBoardBase + k];

            // Boards are ASCII '0'..'9'. Place digit (num 0..8 -> '1'+num)
            dst.boards[dstBoardBase + firstEmpty] = (char)('1' + num);

            // 4) Copy empty list excluding the first element (we just filled it)
            //    Rebuild empties by taking src.empty from index 1 onward.
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

            // 5) Preserve board identification
            dst.number[out] = src.number[i];

            ++out;
        }
    }

    dst.count = out;
}



SudokuInstance populate_boards(char* boards, int count) {
    SudokuInstance generation_a = create_instance();

    SudokuInstance generation_b = create_instance();

    for (int i = 0; i < count; i++) {
        if (wrtie_board_to_instance(generation_a, i, boards + i * LINE_LEN) != 0) {
            fprintf(stderr, "Invalid board at line %d\n", i + 1);
            exit(EXIT_FAILURE);
        }
        generation_a.number[i] = i;
    }

    generation_a.count = count;


    const int MAX_ITER = 4; // adjust as needed
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

    free_sudoku_instance(dst);

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
            fprintf(stderr, "Too short file got %d lines but needs %d lines\n", (int)(got / LINE_LEN), (int)(expected / LINE_LEN));
        usage();
        exit(EXIT_FAILURE);
    }

    fclose(source);

    return buff;
}

// todo debuugiing
void print_uint16_bits(uint16_t value)
{
    char buf[16 + 4 + 2]; // bits + spaces + newline + '\0'
    int pos = 0;
    for (int i = 0; i < 16; ++i)
    {
        buf[pos++] = (char)(((value >> i) & 1u) ? '1' : '0');

        // Space after each 4-bit chunk
        if (i % 4 == 3)
            buf[pos++] = ' ';
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

void print_generated(SudokuInstance si)
{
    if (!si.boards) {
        printf("No boards buffer (si.boards == nullptr)\n");
        return;
    }
    if (si.count == 0) {
        printf("No boards generated (si.count == 0)\n");
        return;
    }

    for (uint32_t idx = 0; idx < si.count; ++idx) {
        const char* board81 = si.boards + (size_t)idx * N2;

        // Pretty header
        printf("Generated board %u:\n", idx);
        printf("+-------+-------+-------+\n");
        for (int r = 0; r < 9; ++r) {
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
}

void print_bulk_buffer(const char* bulk, int count)
{
    if (!bulk) {
        printf("Buffer is null\n");
        return;
    }
    if (count <= 0) {
        printf("Nothing to print (count <= 0)\n");
        return;
    }

    for (int i = 0; i < count; ++i) {
        const char* line = bulk + (size_t)i * LINE_LEN;
        printf("Line %d: ", i);

        print_board_pretty(line, i);
        printf("\n");
    }
}



void set_solutions(SudokuInstance si, Solutions sl, char* boards) {


}

int main(int argc, char* argv[])
{
    argc = 5;

    argv[1] = (char*)"gpu";
    argv[2] = (char*)"10";
    argv[3] = (char*)"C:\\Users\\przem\\Pulpit\\cuda\\P1\\additional\\sudoku_data\\sudoku_data.csv";

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

	Solutions sl = {};


    t0 = chrono::high_resolution_clock::now();

    cudaError_t cudaStatus = SudokuCuda(si, buff,&sl);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "SudokuCuda failed!");
        return 1;
    }


    t1 = chrono::high_resolution_clock::now();
    us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    printf("whole kernel time %.3f ms\n", us / 1000.0);

    set_solutions(si, sl, buff);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}