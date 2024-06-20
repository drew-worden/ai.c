# Compiler and flags
CC = gcc
CFLAGS = -Wall -Werror -Iincludes
LDFLAGS = -lm 
FORMAT = clang-format

# Directories
SRC_DIR = lib
INCLUDE_DIR = includes
TEST_DIR = tests
DIST_DIR = dist

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c,$(DIST_DIR)/%.o,$(SRC_FILES))

# Test files
TEST_FILES = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJ_FILES = $(patsubst $(TEST_DIR)/%.c,$(DIST_DIR)/%.o,$(TEST_FILES))
TEST_EXEC = $(DIST_DIR)/test_runner

# Targets
.PHONY: all test build clean format

all: build test

build: $(DIST_DIR) $(OBJ_FILES)

$(DIST_DIR):
	mkdir -p $(DIST_DIR)

$(DIST_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

test: build $(TEST_OBJ_FILES)
	$(CC) $(CFLAGS) $(OBJ_FILES) $(TEST_OBJ_FILES) -o $(TEST_EXEC) $(LDFLAGS)
	./$(TEST_EXEC)

$(DIST_DIR)/%.o: $(TEST_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(DIST_DIR)

format:
	$(FORMAT) -i --style=file $(SRC_FILES) $(wildcard $(INCLUDE_DIR)/*.h) $(TEST_FILES)
