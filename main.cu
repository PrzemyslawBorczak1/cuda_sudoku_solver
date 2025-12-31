#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "stdio.h"


#include <string>
#include <cstdio> 
#include <chrono>


#define GUARD 82
#define NO_SOL 83

#define N 9
#define N2 81
#define SQRTN 3

#define LINE_LEN 83

#define THREADSPERBLOCK 1

// max amount of boards after multiplication
#define BOARDS 400000 //400 tys
// max aount of multiplication iterations
#define MAX_ITER 7

// fkags for solution status
#define FLAG_SOLVED 1
#define FLAG_MULTIPLE 2
#define FLAG_UNSOLVED 0

using namespace std;
// struct with copied arrays used by kernel and also by host to pupulate data
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

// struct with solution created by kernel
struct Solutions {
    int* flags;
    char* lines;
};








// kernel methods
__device__ void DFS(uint16_t* rows,
    uint16_t* cols,
    uint16_t* boxes,
    char* empty, char* stack);

__global__ void sudokuKernel(SudokuInstance si, Solutions sl)
{
    uint32_t id = blockIdx.x * THREADSPERBLOCK + threadIdx.x;
    if (id >= si.count) 
        return;

	// moving pointers to proper positions
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

	// geting inital number of solved board
    uint32_t nb = si.number[id];

	// setting flag for solved board
    int prev = atomicExch(sl.flags + nb, FLAG_SOLVED);
    if (prev != FLAG_UNSOLVED) {
        atomicExch(sl.flags + nb, FLAG_MULTIPLE);
        return;
    }

	// writing solution to output lines
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

// DFS implementation
__device__ void DFS(uint16_t* rows,
    uint16_t* cols,
    uint16_t* boxes,
    char* empty, char* stack)
{
	// stack holds last tried number for each depth
    char depth = 0;
    stack[0] = -1;
	// cheks if solution was found initaly true
    bool found = true;
    while (true) {
		// no solution found
        if (depth < 0) {
            stack[0] = NO_SOL;
            return;
        }
		// ind of next empty cell
        char ind = empty[depth];
        if (ind == GUARD) {
            return;
        }
		// calculating in wchich row, col, box is this empty cell
        char row = ind / 9;
        char col = ind % 9;
        char box = (char)((row / 3) * 3 + (col / 3));
		// backtracking - removing last tried number from masks (but leaving number on stack for next checking)
        if (!found) {
            uint16_t mask = (uint16_t)1 << (stack[depth]);
            rows[row] &= ~mask;
            cols[col] &= ~mask;
            boxes[box] &= ~mask;
        }

        found = false;
		// checking next numbers
        for (char num = (char)(stack[depth] + 1); num < 9; num++) {
            uint16_t mask = (uint16_t)1 << (num);
			// number cannot be placed here
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

    // alocate memory
    size_t mask_count = count * N * sizeof(uint16_t);
    if ((cuda_status = cudaMalloc((void**)&d_si->rows, mask_count)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMalloc((void**)&d_si->cols, mask_count)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMalloc((void**)&d_si->boxes, mask_count)) != cudaSuccess) return cuda_status;

	size_t boards_count = count * N2 * sizeof(char);
    if ((cuda_status = cudaMalloc((void**)&d_si->empty, boards_count)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMalloc((void**)&d_si->boards, boards_count)) != cudaSuccess) return cuda_status;

	size_t number_count = count * sizeof(uint32_t);
    if ((cuda_status = cudaMalloc((void**)&d_si->number, number_count)) != cudaSuccess) return cuda_status;


	// copy data from host to device
    if ((cuda_status = cudaMemcpy(d_si->rows, h_si.rows, mask_count, cudaMemcpyHostToDevice)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMemcpy(d_si->cols, h_si.cols, mask_count, cudaMemcpyHostToDevice)) != cudaSuccess)  return cuda_status;
    if ((cuda_status = cudaMemcpy(d_si->boxes, h_si.boxes, mask_count, cudaMemcpyHostToDevice)) != cudaSuccess)  return cuda_status;

    if ((cuda_status = cudaMemcpy(d_si->empty, h_si.empty, boards_count, cudaMemcpyHostToDevice)) != cudaSuccess)  return cuda_status;
    if ((cuda_status = cudaMemcpy(d_si->boards, h_si.boards, boards_count, cudaMemcpyHostToDevice)) != cudaSuccess)  return cuda_status;

    if ((cuda_status = cudaMemcpy(d_si->number, h_si.number, number_count, cudaMemcpyHostToDevice)) != cudaSuccess)  return cuda_status;


	d_si->count = h_si.count;

	return cuda_status;
}

// helper mainly used to not perform free on uninitialized pointers
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

// allocate memory for solutions on device
cudaError_t prepareDeviceSolutions(Solutions* d_ret, uint32_t sol_count) {

    cudaError_t cuda_status = cudaSuccess;
   
	size_t flag_count = sol_count * sizeof(int);
	size_t line_count = sol_count * LINE_LEN * sizeof(char);

    if ((cuda_status = cudaMalloc((void**)&d_ret->flags, flag_count)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMemset(d_ret->flags, 0, flag_count)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMalloc((void**)&d_ret->lines, line_count)) != cudaSuccess) return cuda_status;

    return cuda_status;
}

// same as with SudokuInstance
Solutions createEmptySolutions() {
    Solutions sl{};
    sl.flags = nullptr;
    sl.lines = nullptr;
    return sl;
}

// copyies solutions from device to host
cudaError_t copySolutions(Solutions d_ret, Solutions* h_ret, uint32_t sol_count) {

    cudaError_t cuda_status = cudaSuccess;

    size_t flag_count = sol_count * sizeof(int);
    size_t line_count = sol_count * LINE_LEN * sizeof(char);


    if ((cuda_status = cudaMemcpy(h_ret->flags, d_ret.flags, flag_count, cudaMemcpyDeviceToHost)) != cudaSuccess) return cuda_status;
    if ((cuda_status = cudaMemcpy(h_ret->lines, d_ret.lines, line_count, cudaMemcpyDeviceToHost)) != cudaSuccess) return cuda_status;

    return cuda_status;
}

cudaError_t prepareAndRunKernel(SudokuInstance h_si, SudokuInstance d_si, Solutions* h_ret, Solutions *d_ret, uint32_t sol_count) {

    // prepare
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return cuda_status;
    }

    printf("====Preparing data for kernel====\n");
    auto t0 = chrono::high_resolution_clock::now();

    if ((cuda_status = prepareDeviceSudokuInstance(&d_si, h_si)) != cudaSuccess) return cuda_status;
    if ((cuda_status = prepareDeviceSolutions(d_ret, sol_count)) != cudaSuccess) return cuda_status;


    auto t1 = chrono::high_resolution_clock::now();
    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    printf("time %.3f ms\n", us / 1000.0);



    const int block = (int)(d_si.count / THREADSPERBLOCK + 1);

    // run
	printf("====Launching kernel with %d blocks of %d threads====\n", block, THREADSPERBLOCK);
     t0 = chrono::high_resolution_clock::now();
    sudokuKernel << <block, THREADSPERBLOCK >> > (d_si, *d_ret);


    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching sudokuKernel!\n", cuda_status);
        return cuda_status;
    }

     t1 = chrono::high_resolution_clock::now();
     us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    printf("time %.3f ms\n", us / 1000.0);




    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        printf("sudokuKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        return cuda_status;
    }


	// copy solutions back to host
    printf("====Copying solutions from device====\n");
    t0 = chrono::high_resolution_clock::now();
	if((cuda_status = copySolutions(*d_ret, h_ret, sol_count)) != cudaSuccess)
		return cuda_status;


    t1 = chrono::high_resolution_clock::now();
    us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    printf("time %.3f ms\n", us / 1000.0);

	return cudaSuccess;
}

void freeSudokuInstance(SudokuInstance* si);
// main function to be called from host
// prepers data, runs kernel, retrieves results and frees memory
cudaError_t SudokuCuda(SudokuInstance si, char* board, Solutions* ret, uint32_t sol_count)
{
    cudaError_t cuda_status = cudaSuccess;

    const size_t flag_count = (size_t)sol_count * sizeof(int);
    const size_t line_count = (size_t)sol_count * LINE_LEN * sizeof(char);

    ret->flags = (int*)malloc(flag_count);
    ret->lines = (char*)malloc(line_count);

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

















// struct used for sorting empty cells
typedef struct Node {
    Node* next;
    char val;
} Node;

void addNode(Node** head, char value) {
    Node* new_node = (Node*)malloc(sizeof(Node));
    new_node->val = value;

    if(*head == 0) {
        new_node->next = nullptr;
        *head = new_node;
        return;
	}

    new_node->next = *head;
    *head = new_node;
}
char max(char a, char b) {
    return (a > b) ? a : b;
}
char max(char a, char b, char c) {
    return max(max(a, b), c);
}

// moving all data right and last element to first position in array
void cycle(char arr[81], int end) {
    char tmp = arr[end - 1];
    for (int i = end - 1; i > 0; i--) {
        arr[i] = arr[i - 1];
    }
    arr[0] = tmp;
}
//cycles amount times
void cycle(char arr[81], int end, int amount) {
    for (int i = 0; i < amount; i++) {
        cycle(arr, end);
    }
}

// writing linked list to array
void writeToTab(Node* head, char* tab, char* top) {
    if (tab == nullptr || top == nullptr) {
        return;
    }

    while (head != nullptr) {
        tab[*top] = head->val;
        (*top)++;
		Node* temp = head;
        head = head->next;
		free(temp);
    }
}

// creating host SudokuInstance with allocated memory
// this instance will be later populated with data
SudokuInstance createAllocatedInstance() {

    SudokuInstance instance{};
    instance.empty = (char*)malloc(BOARDS * N2);

    instance.rows = (uint16_t*)malloc(BOARDS * N * sizeof(uint16_t));
    instance.cols = (uint16_t*)malloc(BOARDS * N * sizeof(uint16_t));
    instance.boxes = (uint16_t*)malloc(BOARDS * N * sizeof(uint16_t));

    instance.number = (uint32_t*)malloc(BOARDS * sizeof(uint32_t));

    instance.boards = (char*)malloc(BOARDS * N2);

    return instance;
}

// freeing host SudokuInstance
void freeSudokuInstance(SudokuInstance* si) {
    free(si->empty);
    free(si->rows);
    free(si->cols);
    free(si->boxes);
    free(si->number);
}

// try setting bit in mask, returns false if bit was already set
bool testSet(uint16_t* part, int index, uint16_t mask) {

    if (part[index] & mask) {
        return false;
    }
    part[index] |= mask;
    return true;

}


// first write to sudoku insatnce before multiplication
// creates masks and empty cell list in order of most constrained first
int wrtieBoardToInstance(SudokuInstance si, int board_nb, char* board, uint32_t sol_nb) {
	// counter for amount of numbers in rows, cols, boxes
	char counters[27] = { 0 };
	// buff to hold indices of empty cells later copied to sudoku instance
	char buff[81];
	// aray of linked lists for sorting empty cells
    Node* nodes[9] = { 0 };


    char empty_ptr = 0;

	// clearing masks
    const int base = board_nb * 9;
    for (int i = 0; i < 9; i++) {
        si.rows[base + i] = 0;
        si.cols[base + i] = 0;
        si.boxes[base + i] = 0;
    }

    for (int i = 0; i < N2; ++i) {
        si.boards[board_nb * N2 + i] = board[i];
        char ch = board[i];
		// finding empty cells
        if (ch == '0') {
            buff[empty_ptr++] = i;
            continue;
        }

        int num = ch - '1';   
        uint16_t mask = 1 << num;

        int row = i / 9;
        int col = i % 9;
        int box = (row / 3) * 3 + (col / 3);
		// seting masks if board is invalid returns 1
        if (!testSet(si.rows + base, row, mask))
            return 1;
		counters[row]++;


        if (!testSet(si.cols + base, col, mask))
            return 1;
        counters[9 + col]++;

        if (!testSet(si.boxes + base, box, mask))
            return 1;
		counters[18 + box]++;
    }
	buff[empty_ptr] = GUARD;

    // bucket sort
    for (int i = 0; i < empty_ptr; i++) {

        int row = buff[i] / 9;
        int col = buff[i] % 9;
        int box = (row / 3) * 3 + (col / 3);

		char m = max(counters[row], counters[9 + col], counters[18 + box]);

		addNode(&nodes[m], buff[i]);
    }

	empty_ptr = 0;
    for(int i = 8; i >= 0 ; i--) {
        writeToTab(nodes[i], si.empty + board_nb * N2, &empty_ptr);
	}

    si.empty[board_nb * N2 + empty_ptr ] = GUARD;
    if (sol_nb <= 20) {
        cycle(si.empty + board_nb * N2, empty_ptr, 4);
    }


    return 0;
}

// rewrites board from src instance to dst instance without changing it
void copyBoard(SudokuInstance* src, SudokuInstance* dst, uint32_t src_id, uint32_t dst_id) {
    const size_t srcBoardBase = src_id * N2;
    const size_t dstBoardBase = dst_id * N2;

    for (uint32_t i = 0; i < N2; i++)
        dst->boards[dstBoardBase + i] = src->boards[srcBoardBase + i];

    const uint32_t srcBase = src_id * N;
    const uint32_t dstBase = dst_id * N;

    for (uint32_t i = 0; i < 9; i++) {
        dst->rows[dstBase + i] = src->rows[srcBase + i];
        dst->cols[dstBase + i] = src->cols[srcBase + i];
        dst->boxes[dstBase + i] = src->boxes[srcBase + i];
    }

    for (uint32_t i = 0; i < N2; i++) {
		char v = src->empty[srcBoardBase + i];
        dst->empty[dstBoardBase + i] = v;
        if (v == GUARD)
            break;
    }

    dst->number[dst_id] = src->number[src_id];

	dst->count++;
}

// tries to add number to board if board would be illegal does nothing. 
// in oder cases creates new board in dst instance
bool tryAddNumberToBoard(SudokuInstance& src, SudokuInstance& dst, uint32_t src_id, uint32_t dst_id, int num) {
	// setts bit mask for number to add and pointers to proper positions in arrays
    
    uint16_t mask = 1 << num;

    uint32_t dstBase = dst_id * 9;
	uint32_t srcBase = src_id * 9;

	char* empty = src.empty + src_id * N2;
	char firstEmpty = empty[0];

    int row = firstEmpty / N;
    int col = firstEmpty % N;
    int box = (row / SQRTN) * SQRTN + (col / SQRTN);


    uint16_t* src_rows = src.rows + srcBase;
    uint16_t* src_cols = src.cols + srcBase;
    uint16_t* src_boxes = src.boxes + srcBase;

	// checks if number can be placed
    if ((src_rows[row] & mask) || (src_cols[col] & mask) || (src_boxes[box] & mask))
        return false;


	// placing number and copying masks
    uint16_t* dst_rows = dst.rows + dstBase;
    uint16_t* dst_cols = dst.cols + dstBase;
    uint16_t* dst_boxes = dst.boxes + dstBase;

    for (int i = 0; i < 9; ++i) {
        dst_rows[i] = src_rows[i];
        dst_cols[i] = src_cols[i];
        dst_boxes[i] = src_boxes[i];
    }

    dst_rows[row] |= mask;
    dst_cols[col] |= mask;
    dst_boxes[box] |= mask;


	char* dst_board = dst.boards + dst_id * N2;
	char* src_board = src.boards + src_id * N2;
    for (int i = 0; i < N2; ++i)
        dst_board[i] = src_board[i];


    char* dstEmpty = dst.empty + dst_id * N2;
    for (int i = 0; i < N2; i++) {
        dstEmpty[i] = empty[i + 1];
        if (empty[i + 1] == GUARD)
            break;
    }

    dst_board[firstEmpty] = '1' + num;
    dst.number[dst_id] = src.number[src_id];

	return true;
}

// multiplies boards in src instance and creates new boards in dst instance
void multiplyBoards(SudokuInstance* sr, SudokuInstance* ds)
{
    SudokuInstance& src = *sr;
    SudokuInstance& dst = *ds;


    uint32_t out = 0;

    for (uint32_t i = 0; i < src.count; i++)
    {
		char* empty = src.empty + i * N2;
		// if board is already solved or amount of max boards was reached copy  data without changing
        if (empty[0] == GUARD || out >= BOARDS - src.count - N)
        {
			copyBoard(&src, &dst, i, out);
            
            ++out;
            continue;
        }

		// for each board try adding each number to first empty cell
        for (int num = 0; num < 9; ++num)
        {
			if(tryAddNumberToBoard(src, dst, i, out, num) == false)
				continue;

            ++out;
        }
    }

    dst.count = out;
}

// main entry for multiplying boards
SudokuInstance populateBoards(char* boards, int count) {
    SudokuInstance generation_a = createAllocatedInstance();
    SudokuInstance generation_b = createAllocatedInstance();

    for (int i = 0; i < count; i++) {
        if (wrtieBoardToInstance(generation_a, i, boards + i * LINE_LEN, count) != 0) {
            printf("Invalid board at line %d\n", i + 1);
            exit(EXIT_FAILURE);
        }
        generation_a.number[i] = i;
    }

    generation_a.count = count;


    SudokuInstance* src = &generation_a;
    SudokuInstance* dst = &generation_b;

    for (int i = 0; i < MAX_ITER; i++) {
        multiplyBoards(src, dst);
        double factor = (double)dst->count / (double)src->count;
        printf("iter %d: %u -> %u (x%.3f)\n", i, src->count, dst->count, factor);


        SudokuInstance* tmp = src;
        src = dst;
        dst = tmp;

       /* if (factor < 1.5)
            break;*/
        if (dst->count >= BOARDS / 0.9)
            break;
    }


    freeSudokuInstance(dst);


    printf("====Created %u Boards=====\n", src->count);
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

// reading input file to buffer
char* readToBuff(int count, char* path) {


    FILE* source = fopen(path, "rb");
    if (source == NULL) {
        printf("Could not open input file: '%s'\n", path);
        usage();
        exit(EXIT_FAILURE);
    }

    size_t expected = (size_t)count * LINE_LEN * sizeof(char);

    char* buff = (char*)malloc(expected);
    if (!buff) {
        printf("malloc bulk buffer");

        exit(EXIT_FAILURE);
    }

    size_t got = fread(buff, 1, expected, source);
    if (got != expected) {
        if (feof(source))
            printf("Too short file got %d lines but needs %d lines\n", (int)(got / LINE_LEN), (int)(expected / LINE_LEN));
        usage();
        exit(EXIT_FAILURE);
    }

    fclose(source);

    return buff;
}

// writing solutions to output file
void write_solutions_to_file(Solutions sl, const char* outputPath, uint32_t sol_count)
{
    if (!outputPath) {
        printf("Output path is null\n");
        return;
    }

    FILE* out = fopen(outputPath, "wb");
    if (!out) {
        printf("Could not open output file: '%s'\n", outputPath);
        return;
    }

    // Write solved boards as single 81-char lines with CRLF.
    for (uint32_t i = 0; i < sol_count; ++i) {
        if (sl.flags[i] == 0)
            continue;

        const char* line = sl.lines + (size_t)i * LINE_LEN;
        // write 81 chars
        if (fwrite(line, 1, 81, out) != 81) {
            printf("Failed to write board %u\n", i);
            break;
        }
        // write CRLF
        static const char crlf[2] = { '\r', '\n' };
        if (fwrite(crlf, 1, 2, out) != 2) {
            printf("Failed to write CRLF for board %u\n", i);
            break;
        }
    }

    fclose(out);
}

// checking if all boards were solved
void evalSolution(Solutions sl, uint32_t sol_count) 
{
	uint32_t solved = 0;
    for (uint32_t i = 0; i < sol_count; ++i) {
        int flag = sl.flags[i];
        switch (flag) {
            case FLAG_SOLVED:
                solved++;
				break;
            case FLAG_MULTIPLE:
				printf("Board %u has multiple solutions\n", i);
                break;
			case FLAG_UNSOLVED:
                printf("Board %u could not be solved\n", i);
				break;
        }
    }

	printf("Solved boards: %u / %u\n", solved, sol_count);
}

int main(int argc, char* argv[])
{
   
    // checking arguments
    if (argc != 5) {
        printf("Invalid arguments. Expected exactly 4 parameters.\n");
        usage();
        return 1;
    }
  
	printf("Input:\n    method : %s, count : %d, \n    input_file : '%s', \n    output_file : '%s'", argv[1], stoi(argv[2]), argv[3], argv[4]);

	printf("\n\n====Sudoku Solver CUDA====\n\n");

    const char* method = argv[1];
    const char* countStr = argv[2];
    const char* outputPath = argv[4];

    if (strcmp(method, "cpu") != 0 && strcmp(method, "gpu") != 0) {
        printf("Invalid method: '%s'. Allowed: 'cpu' or 'gpu'.\n", method);
        usage();
        exit(EXIT_FAILURE);
    }

    if (strcmp(method, "cpu") == 0){
        return 0;
    }

    int count = stoi(countStr);
    if (count < 0) {
        printf("Invalid count: '%d'. Must be a positive integer.\n", count);
        usage();
        exit(EXIT_FAILURE);
    }

	// main app functionality
    auto t_all_1 = chrono::high_resolution_clock::now();

	printf("====Cpu reading and multiplying====\n");
    auto t0 = chrono::high_resolution_clock::now();
    char* buff = readToBuff(count, argv[3]);
    SudokuInstance si = populateBoards(buff, count);

    auto t1 = chrono::high_resolution_clock::now();
    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    printf("time %.3f ms\n", us / 1000.0);


	Solutions sl = {};
    cudaError_t cudaStatus = SudokuCuda(si, buff,&sl, count);
    if (cudaStatus != cudaSuccess) {
        printf("SudokuCuda failed!");
        return 1;
    }

	// evaluating solutions
    printf("====Cpu evaluating solution====\n");
    evalSolution(sl, count);


    printf("====Saving to file====\n");
    write_solutions_to_file(sl, outputPath, count);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceReset failed!");
        return 1;
    }


    auto t_all_2 = chrono::high_resolution_clock::now();
    us = chrono::duration_cast<chrono::microseconds>(t_all_2 - t_all_1).count();
	printf("====Total time====\n");
    printf("%.3f ms\n", us / 1000.0);

    return 0;
}