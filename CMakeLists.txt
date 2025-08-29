cmake_minimum_required(VERSION 3.15)
project(dfnn_mlir)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找LLVM和MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVM config: ${LLVM_DIR}")

# 设置包含目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# 添加一个简单的可执行文件
add_executable(dfnn-mlir-main src/main.cpp)

# 在target_link_libraries部分更新为：
target_link_libraries(dfnn-mlir-main PRIVATE
    MLIRIR
    MLIRSupport
)
