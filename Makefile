# Makefile

CUDA = nvcc
TARGET = main
SRC = main.cu

# Release flags
CFLAGS = -O3 -arch=sm_61

# Detect platform
ifeq ($(OS),Windows_NT)
    EXE_EXT = .exe
    RM = del /Q
else
    EXE_EXT =
    RM = rm -f
endif

TARGET_FULL = $(TARGET)$(EXE_EXT)

# Default target: just compile
all: $(TARGET_FULL)

# Compile target
$(TARGET_FULL): $(SRC)
	$(CUDA) $(CFLAGS) -o $(TARGET_FULL) $(SRC)

# Clean target
clean:
	$(RM) $(TARGET_FULL)
