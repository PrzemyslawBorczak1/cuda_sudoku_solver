#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "sudoku_types.hpp"

#define LINE_LEN 83
#define BOARDS 80000

static bool test_set(uint64_t* part1, uint64_t* part2, int index, uint64_t mask) {
    if (index < 7) {
        if (*part1 & (mask << (index * 9))) return false;
        *part1 |= (mask << (index * 9));
    } else {
        index -= 7;
        if (*part2 & (mask << (index * 9))) return false;
        *part2 |= (mask << (index * 9));
    }
    return true;
}

static int write_board_to_instance(SudokuInstance& si, int offset, const char* board) {
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
        int num = ch - '1';
        uint64_t mask = (uint64_t)1 << num;
        int row = i / 9;
        int col = i % 9;
        int box = (row / 3) * 3 + (col / 3);
        if (!test_set(si.rows1 + offset, si.rows2 + offset, row, mask)) return 1;
        if (!test_set(si.cols1 + offset, si.cols2 + offset, col, mask)) return 1;
        if (!test_set(si.boxes1 + offset, si.boxes2 + offset, box, mask)) return 1;
    }
    si.empty[offset * N2 + empty_ptr] = GUARD;
    return 0;
}

SudokuInstance create_instance() {
    SudokuInstance si{};
    si.empty  = (char*)malloc(BOARDS * N2);
    si.rows1  = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));
    si.rows2  = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));
    si.cols1  = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));
    si.cols2  = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));
    si.boxes1 = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));
    si.boxes2 = (uint64_t*)malloc(BOARDS * sizeof(uint64_t));
    si.number = (uint32_t*)malloc(BOARDS * sizeof(uint32_t));
    si.count = 0;
    return si;
}

void clear_sudoku_instance(SudokuInstance& si) {
    free(si.empty);
    free(si.rows1);
    free(si.rows2);
    free(si.cols1);
    free(si.cols2);
    free(si.boxes1);
    free(si.boxes2);
    free(si.number);
    si.count = 0;
}

SudokuInstance populate_boards(const char* bulk, int count) {
    SudokuInstance si = create_instance();
    for (int i = 0; i < count; ++i) {
        const char* line = bulk + i * LINE_LEN;
        if (write_board_to_instance(si, i, line) != 0) {
            std::fprintf(stderr, "Invalid board at line %d\n", i + 1);
            exit(EXIT_FAILURE);
        }
        si.number[i] = i;
    }
    si.count = count;
    return si;
}

char* read_to_buffer(int count, const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "Cannot open input file: %s\n", path);
        exit(EXIT_FAILURE);
    }
    size_t total = (size_t)count * LINE_LEN;
    char* buf = (char*)std::malloc(total);
    if (!buf) {
        std::fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
    size_t got = std::fread(buf, 1, total, f);
    std::fclose(f);
    if (got != total) {
        std::fprintf(stderr, "File too short: expected %zu got %zu\n", total, got);
        exit(EXIT_FAILURE);
    }
    return buf;
}