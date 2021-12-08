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
include CMakeFiles/rwalk_op.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/rwalk_op.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/rwalk_op.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rwalk_op.dir/flags.make

CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o: CMakeFiles/rwalk_op.dir/flags.make
CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o: ../rwalk_kernel.cc
CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o: CMakeFiles/rwalk_op.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/random_walk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o -MF CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o.d -o CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o -c /home/ubuntu/random_walk/rwalk_kernel.cc

CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.i"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/random_walk/rwalk_kernel.cc > CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.i

CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.s"
	/usr/bin/x86_64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/random_walk/rwalk_kernel.cc -o CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.s

# Object files for target rwalk_op
rwalk_op_OBJECTS = \
"CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o"

# External object files for target rwalk_op
rwalk_op_EXTERNAL_OBJECTS =

CMakeFiles/rwalk_op.dir/cmake_device_link.o: CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o
CMakeFiles/rwalk_op.dir/cmake_device_link.o: CMakeFiles/rwalk_op.dir/build.make
CMakeFiles/rwalk_op.dir/cmake_device_link.o: libcuda_lib.a
CMakeFiles/rwalk_op.dir/cmake_device_link.o: CMakeFiles/rwalk_op.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/random_walk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/rwalk_op.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rwalk_op.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rwalk_op.dir/build: CMakeFiles/rwalk_op.dir/cmake_device_link.o
.PHONY : CMakeFiles/rwalk_op.dir/build

# Object files for target rwalk_op
rwalk_op_OBJECTS = \
"CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o"

# External object files for target rwalk_op
rwalk_op_EXTERNAL_OBJECTS =

rwalk_op: CMakeFiles/rwalk_op.dir/rwalk_kernel.cc.o
rwalk_op: CMakeFiles/rwalk_op.dir/build.make
rwalk_op: libcuda_lib.a
rwalk_op: CMakeFiles/rwalk_op.dir/cmake_device_link.o
rwalk_op: CMakeFiles/rwalk_op.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/random_walk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable rwalk_op"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rwalk_op.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rwalk_op.dir/build: rwalk_op
.PHONY : CMakeFiles/rwalk_op.dir/build

CMakeFiles/rwalk_op.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rwalk_op.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rwalk_op.dir/clean

CMakeFiles/rwalk_op.dir/depend:
	cd /home/ubuntu/random_walk/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/random_walk /home/ubuntu/random_walk /home/ubuntu/random_walk/build /home/ubuntu/random_walk/build /home/ubuntu/random_walk/build/CMakeFiles/rwalk_op.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rwalk_op.dir/depend

