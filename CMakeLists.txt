cmake_minimum_required(VERSION 3.15)
project(dfnn-mlir)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找LLVM和MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVM config: ${LLVM_DIR}")

# LLVM库
llvm_map_components_to_libnames(LLVM_LIBS
    Support
    Core
    IRReader
)

# MLIR库
set(MLIR_LIBS
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRDialect
    MLIRRewrite
    MLIRTransforms
)

# 添加一个简单的可执行文件
add_executable(simple_mlir src/simple_mlir.cpp)

# 设置包含目录
target_include_directories(simple_mlir PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(simple_mlir PRIVATE ${LLVM_LIBS} ${MLIR_LIBS})
