cmake_minimum_required(VERSION 4.0)
project(CudaOneFile CUDA)

set(CMAKE_CUDA_STANDARD 20)

# GPU architecture (change if needed, 75 = RTX 20 series)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Change output target name to "main"
add_executable(main main.cu)

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Release compile flags
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# CUDA compile flags
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -lineinfo       # Keep line info for profiling
        -Xptxas=-v      # Show register usage
    >
)

# Host compiler warnings (Linux/macOS only)
if(NOT MSVC)
    target_compile_options(main PRIVATE
        -Xcompiler -Wall
        -Xcompiler -Wextra
    )
endif()
