# Compiler
HIPCC = hipcc

# Compiler flags
HIPCCFLAGS = -Wall -Wextra -std=c++11 -Wno-unused-result -Wno-unused-parameter

# Directories
SRC_DIR = .
TESTS_DIR = tests

HELPER_DIR = helper
ARITHMETIC_DIR = $(HELPER_DIR)/arithmetic
MEMORYOPS_DIR  = $(HELPER_DIR)/memoryOps 
STRESS_DIR     = $(HELPER_DIR)/stress
OTHERS_DIR     = $(HELPER_DIR)/others


# Source files
SRC_FILES = $(SRC_DIR)/main.cpp \
	    $(TESTS_DIR)/bandwidth.cpp \
	    $(TESTS_DIR)/stress.cpp \
	    $(TESTS_DIR)/coherency.cpp \
	    $(ARITHMETIC_DIR)/add.cpp \
	    $(ARITHMETIC_DIR)/subtract.cpp \
	    $(STRESS_DIR)/memoryStress.cpp \
	    $(OTHERS_DIR)/getPhysicalAddress.cpp \
	    $(OTHERS_DIR)/boolToString.cpp

# Include directories
INCLUDES = -I$(HOME)/helper/arithmetic \
	   -I$(HOME)/helper/memoryOps \
	   -I$(HOME)/helper/stress \
	   -I$(HOME)/helper/others \
	   -I$(HOME)/test

# Object files (replace .cpp with .o)
OBJ_FILES = $(SRC_FILES:.cpp=.o)

# Executable name
TARGET = stressgfx

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJ_FILES)
	$(HIPCC) $(HIPCCFLAGS) $(INCLUDES) -o $(TARGET) $(OBJ_FILES)

# Compile source files into object files
%.o: %.cpp
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	rm -f $(OBJ_FILES) $(TARGET)

# Phony targets to prevent conflicts with file names
.PHONY: all clean
