#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "stdio.h"


#define GUARD 82
#define NO_SOL 83

#define N 9
#define N2 81
#define SQRTN 3


#define THREADSPERBLOCK 1

#define LINE_LEN 83


#include <chrono>
using namespace std;

struct SudokuInstance {
    char* empty;

    uint16_t* rows;   
    uint16_t* cols;   
    uint16_t* boxes;  

    uint32_t* number; 
    char* boards;

    // CPU only
    uint32_t count;
};

struct Solutions {
    int* flags;
    char* lines;
};


__device__ void DFS(uint16_t* rows,
    uint16_t* cols,
    uint16_t* boxes,
    char* empty, char* stack);

__global__ void sudokuKernel(SudokuInstance si, Solutions sl)
{
    uint32_t id = blockIdx.x + threadIdx.x;
    if (id >= si.count) 
        return;

    char* empty = si.empty + id * N2;
	char stack[N2];

    DFS(
        si.rows + id * 9,
        si.cols + id * 9,
        si.boxes + id * 9,
		empty, stack
    );

    if (stack[0] == NO_SOL)
        return;


    uint32_t nb = si.number[id];

    int prev = atomicExch(sl.flags + nb, 1);
    if (prev != 0) {
        atomicExch(sl.flags + nb, 2);
        return;
    }

    char i = 0;
	char* brd = si.boards + id * N2;
    while(empty[i] != GUARD) {
        brd[empty[i]] = '1' + stack[i];
        i++;
	}


    char* line = sl.lines + nb * LINE_LEN;
    for (int i = 0; i < 81; ++i)
        line[i] = brd[i];


    line[81] = '\r';
    line[82] = '\n';
}

__device__ void DFS(uint16_t* rows,
    uint16_t* cols,
    uint16_t* boxes,
    char* empty, char* stack)
{
    char depth = 0;
    stack[0] = -1;
    bool found = true;
    while (true) {
        if (depth < 0) {
            stack[0] = NO_SOL;
            return;
        }
        char ind = empty[depth];
        if (ind == GUARD) {
            return;
        }
        char row = ind / 9;
        char col = ind % 9;
        char box = (char)((row / 3) * 3 + (col / 3));

        if (!found) {
            uint16_t mask = (uint16_t)1 << (stack[depth]);
            rows[row] &= ~mask;
            cols[col] &= ~mask;
            boxes[box] &= ~mask;
        }

        found = false;
        for (char num = (char)(stack[depth] + 1); num < 9; num++) {
            uint16_t mask = (uint16_t)1 << (num);
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





cudaError_t prepareDeviceSudokuInstance(SudokuInstance *d_si, SudokuInstance h_si) {
    cudaError_t cuda_status = cudaSuccess;
	uint32_t count = h_si.count;

    // alocate
    size_t mask_count = count * N * sizeof(uint16_t);
    if ((cuda_status = cudaMalloc((void**)&d_si->rows, mask_count)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMalloc((void**)&d_si->cols, mask_count)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMalloc((void**)&d_si->boxes, mask_count)) != cudaSuccess) return cuda_status;

	size_t boards_count = count * N2 * sizeof(char);
    if ((cuda_status = cudaMalloc((void**)&d_si->empty, boards_count)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMalloc((void**)&d_si->boards, boards_count)) != cudaSuccess) return cuda_status;

	size_t number_count = count * sizeof(uint32_t);
    if ((cuda_status = cudaMalloc((void**)&d_si->number, number_count)) != cudaSuccess) return cuda_status;


    // copy

    if ((cuda_status = cudaMemcpy(d_si->rows, h_si.rows, mask_count, cudaMemcpyHostToDevice)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMemcpy(d_si->cols, h_si.cols, mask_count, cudaMemcpyHostToDevice)) != cudaSuccess)  return cuda_status;
    if ((cuda_status = cudaMemcpy(d_si->boxes, h_si.boxes, mask_count, cudaMemcpyHostToDevice)) != cudaSuccess)  return cuda_status;

    if ((cuda_status = cudaMemcpy(d_si->empty, h_si.empty, boards_count, cudaMemcpyHostToDevice)) != cudaSuccess)  return cuda_status;
    if ((cuda_status = cudaMemcpy(d_si->boards, h_si.boards, boards_count, cudaMemcpyHostToDevice)) != cudaSuccess)  return cuda_status;

    if ((cuda_status = cudaMemcpy(d_si->number, h_si.number, number_count, cudaMemcpyHostToDevice)) != cudaSuccess)  return cuda_status;


	d_si->count = h_si.count;

	return cuda_status;
}

SudokuInstance createEmptySudokuInstance() {
	SudokuInstance instance;

	instance.empty = nullptr;
	instance.rows = nullptr;
	instance.cols = nullptr;
	instance.boxes = nullptr;
	instance.number = nullptr;
	instance.boards = nullptr;

    return instance;
}

cudaError_t prepareDeviceSolutions(Solutions* d_ret, uint32_t sol_count) {

    cudaError_t cuda_status = cudaSuccess;

	size_t flag_count = sol_count * sizeof(int);
	size_t line_count = sol_count * LINE_LEN * sizeof(char);

    if ((cuda_status = cudaMalloc((void**)&d_ret->flags, flag_count)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMemset(d_ret->flags, 0, flag_count)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMalloc((void**)&d_ret->lines, line_count)) != cudaSuccess) return cuda_status;

    return cuda_status;
}

Solutions createEmptySolutions() {
    Solutions sl{};
    sl.flags = nullptr;
    sl.lines = nullptr;
    return sl;
}

cudaError_t copySolutions(Solutions d_ret, Solutions* h_ret, uint32_t sol_count) {

    cudaError_t cuda_status = cudaSuccess;

    size_t flag_count = sol_count * sizeof(int);
    size_t line_count = sol_count * LINE_LEN * sizeof(char);


    if ((cuda_status = cudaMemcpy(h_ret->flags, d_ret.flags, flag_count, cudaMemcpyDeviceToHost)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMemcpy(h_ret->lines, d_ret.lines, line_count, cudaMemcpyDeviceToHost)) != cudaSuccess) return cuda_status;

    return cuda_status;
}

cudaError_t prepareAndRunKernel(SudokuInstance h_si, SudokuInstance d_si, Solutions* h_ret, Solutions *d_ret, uint32_t sol_count) {
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return cuda_status;
    }

    if ((cuda_status = prepareDeviceSudokuInstance(&d_si, h_si)) != cudaSuccess) return cuda_status;
    if ((cuda_status = prepareDeviceSolutions(d_ret, sol_count)) != cudaSuccess) return cuda_status;

    // launch
    const int block = (int)(d_si.count / THREADSPERBLOCK + 1);
    auto t0 = chrono::high_resolution_clock::now();
    sudokuKernel << <block, THREADSPERBLOCK >> > (d_si, *d_ret);


    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sudokuKernel!\n", cuda_status);
        return cuda_status;
    }

    auto t1 = chrono::high_resolution_clock::now();
    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    printf("kernell call %.3f ms\n", us / 1000.0);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "sudokuKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        return cuda_status;
    }



	if((cuda_status = copySolutions(*d_ret, h_ret, sol_count)) != cudaSuccess)
		return cuda_status;

	return cudaSuccess;
}


void freeSudokuInstance(SudokuInstance* si);
cudaError_t SudokuCuda(SudokuInstance si, char* board, Solutions* ret, uint32_t sol_count)
{
    cudaError_t cuda_status = cudaSuccess;

    const size_t flag_count  = (size_t)sol_count * sizeof(int);
    const size_t line_count  = (size_t)sol_count * LINE_LEN * sizeof(char);

    ret->flags   = (int*)malloc(flag_count);
    ret->lines   = (char*)malloc(line_count);

	SudokuInstance d_si = createEmptySudokuInstance();
    Solutions d_ret = createEmptySolutions();

    cuda_status = prepareAndRunKernel(si, d_si, ret, &d_ret, sol_count);

	freeSudokuInstance(&si);

    cudaFree(d_si.rows);
    cudaFree(d_si.cols);
    cudaFree(d_si.boxes);
    cudaFree(d_si.empty);
    cudaFree(d_si.number);
    cudaFree(d_si.boards);

    cudaFree(d_ret.lines);
    cudaFree(d_ret.flags);

    return cuda_status;
}







#include <string>
#include <cstdio> 


#define BOARDS 400000 //400 tys




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

void freeSudokuInstance(SudokuInstance* si) {
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

int wrtieBoardToInstance(SudokuInstance si, int board_nb, char* board) {
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

    // Added: static secondary instance to store copies of every newly created board.
    static SudokuInstance mirror;
    static bool mirrorInit = false;
    if (!mirrorInit) {
        mirror = create_instance();
        mirrorInit = true;
    }
    // Reset count for fresh population this invocation.
    mirror.count = 0;

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

            // Mirror copy (new functionality)
            if (mirror.count < BOARDS) {
                uint32_t m = mirror.count;

                // Copy board
                size_t mBoardBase = (size_t)m * N2;
                for (int k = 0; k < N2; ++k)
                    mirror.boards[mBoardBase + k] = dst.boards[dstBoardBase + k];

                // Copy masks
                int mBase = (int)m * 9;
                for (int k = 0; k < 9; ++k) {
                    mirror.rows[mBase + k] = dst.rows[dstBase + k];
                    mirror.cols[mBase + k] = dst.cols[dstBase + k];
                    mirror.boxes[mBase + k] = dst.boxes[dstBase + k];
                }

                // Copy empties (only GUARD)
                mirror.empty[m * N2] = GUARD;

                // Copy number
                mirror.number[m] = dst.number[out];

                mirror.count++;
            }

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

            dst.boards[dstBoardBase + firstEmpty] = (char)('1' + num);

            // 4) Copy empty list excluding the first element (we just filled it)
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

            // Mirror copy (new functionality)
            if (mirror.count < BOARDS) {
                uint32_t m = mirror.count;

                // Copy board
                size_t mBoardBase = (size_t)m * N2;
                for (int k = 0; k < N2; ++k)
                    mirror.boards[mBoardBase + k] = dst.boards[dstBoardBase + k];

                // Copy masks
                int mBase = (int)m * 9;
                for (int k = 0; k < 9; ++k) {
                    mirror.rows[mBase + k] = dst.rows[dstBase + k];
                    mirror.cols[mBase + k] = dst.cols[dstBase + k];
                    mirror.boxes[mBase + k] = dst.boxes[dstBase + k];
                }

                // Copy empties
                size_t mEmptyBase = (size_t)m * N2;
                for (int k = 0; ; ++k) {
                    char v = dst.empty[dstEmptyBase + k];
                    mirror.empty[mEmptyBase + k] = v;
                    if (v == GUARD) break;
                }

                // Copy number
                mirror.number[m] = dst.number[out];

                mirror.count++;
            }

            ++out;
        }
    }

    dst.count = out;

    // (Optional) mirror instance now holds copies in 'mirror'; no external exposure per request.
}


SudokuInstance populate_boards(char* boards, int count) {
    SudokuInstance generation_a = create_instance();
    SudokuInstance generation_b = create_instance();

    for (int i = 0; i < count; i++) {
        if (wrtieBoardToInstance(generation_a, i, boards + i * LINE_LEN) != 0) {
            fprintf(stderr, "Invalid board at line %d\n", i + 1);
            exit(EXIT_FAILURE);
        }
        generation_a.number[i] = i;
    }

    generation_a.count = count;


    const int MAX_ITER = 10; 
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

        if(factor < 1.5)
			break;
        if (src->count >= BOARDS / 0.9)
            break;
    }

    freeSudokuInstance(dst);

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



void print_solutions(SudokuInstance si, Solutions sl, char* boardsm, uint32_t sol_count) 
{
    // Print solved boards as single 81-char lines (matching requested format).
    for (uint32_t i = 0; i < sol_count; ++i) {
        int flag = sl.flags[i];
        if (flag == 0)
            continue;

        const char* line = sl.lines + (size_t)i * LINE_LEN;

        printf("%u: ", i);
        for (int k = 0; k < 81; ++k)
            printf("%c", line[k]);
        putchar('\n');
    }
}

void write_solutions_to_file(SudokuInstance si, Solutions sl, const char* outputPath)
{
    if (!outputPath) {
        fprintf(stderr, "Output path is null\n");
        return;
    }

    FILE* out = fopen(outputPath, "wb");
    if (!out) {
        fprintf(stderr, "Could not open output file: '%s'\n", outputPath);
        return;
    }

    // Write solved boards as single 81-char lines with CRLF.
    for (uint32_t i = 0; i < si.count; ++i) {
        if (sl.flags[i] == 0)
            continue;

        const char* line = sl.lines + (size_t)i * LINE_LEN;
        // write 81 chars
        if (fwrite(line, 1, 81, out) != 81) {
            fprintf(stderr, "Failed to write board %u\n", i);
            break;
        }
        // write CRLF
        static const char crlf[2] = { '\r', '\n' };
        if (fwrite(crlf, 1, 2, out) != 2) {
            fprintf(stderr, "Failed to write CRLF for board %u\n", i);
            break;
        }
    }

    fclose(out);
}

int main(int argc, char* argv[])
{
    argc = 5;

    argv[1] = (char*)"gpu";
    argv[2] = (char*)"1000";
    argv[3] = (char*)"C:\\Users\\przem\\Pulpit\\cuda\\P1\\additional\\sudoku_data\\sudoku_data.csv";
	argv[4] = (char*)"C:\\Users\\przem\\Pulpit\\cuda\\P1\\additional\\sudoku_data\\out2";

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

    cudaError_t cudaStatus = SudokuCuda(si, buff,&sl, count);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "SudokuCuda failed!");
        return 1;
    }


    t1 = chrono::high_resolution_clock::now();
    us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    printf("whole kernel time %.3f ms\n", us / 1000.0);

    print_solutions(si, sl, buff, count);
    write_solutions_to_file(si, sl, outputPath);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}