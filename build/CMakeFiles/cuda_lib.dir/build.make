# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/random_walk

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/random_walk/build

# Include any dependencies generated for this target.
include CMakeFiles/cuda_lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cuda_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda_lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda_lib.dir/flags.make

CMakeFiles/cuda_lib.dir/helper.cu.o: CMakeFiles/cuda_lib.dir/flags.make
CMakeFiles/cuda_lib.dir/helper.cu.o: ../helper.cu
CMakeFiles/cuda_lib.dir/helper.cu.o: CMakeFiles/cuda_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/random_walk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/cuda_lib.dir/helper.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/random_walk/helper.cu -o CMakeFiles/cuda_lib.dir/helper.cu.o
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/ubuntu/random_walk/helper.cu -MT CMakeFiles/cuda_lib.dir/helper.cu.o -o CMakeFiles/cuda_lib.dir/helper.cu.o.d

CMakeFiles/cuda_lib.dir/helper.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuda_lib.dir/helper.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuda_lib.dir/helper.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuda_lib.dir/helper.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.o: CMakeFiles/cuda_lib.dir/flags.make
CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.o: ../rwalk_optimized.cu
CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.o: CMakeFiles/cuda_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/random_walk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/ubuntu/random_walk/rwalk_optimized.cu -o CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.o
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -M /home/ubuntu/random_walk/rwalk_optimized.cu -MT CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.o -o CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.o.d

CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cuda_lib
cuda_lib_OBJECTS = \
"CMakeFiles/cuda_lib.dir/helper.cu.o" \
"CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.o"

# External object files for target cuda_lib
cuda_lib_EXTERNAL_OBJECTS =

libcuda_lib.a: CMakeFiles/cuda_lib.dir/helper.cu.o
libcuda_lib.a: CMakeFiles/cuda_lib.dir/rwalk_optimized.cu.o
libcuda_lib.a: CMakeFiles/cuda_lib.dir/build.make
libcuda_lib.a: CMakeFiles/cuda_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/random_walk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA static library libcuda_lib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_lib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_lib.dir/build: libcuda_lib.a
.PHONY : CMakeFiles/cuda_lib.dir/build

CMakeFiles/cuda_lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda_lib.dir/clean

CMakeFiles/cuda_lib.dir/depend:
	cd /home/ubuntu/random_walk/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/random_walk /home/ubuntu/random_walk /home/ubuntu/random_walk/build /home/ubuntu/random_walk/build /home/ubuntu/random_walk/build/CMakeFiles/cuda_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuda_lib.dir/depend

