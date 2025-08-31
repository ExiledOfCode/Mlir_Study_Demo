```sh

# 使用的版本是 llvm 11


# llvm 编译指令
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/llvm_install \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD=Native \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  ../llvm


# 使用我们 install 的路径进行 cmake 构建
cmake   -DLLVM_DIR=~/llvm_install/lib/cmake/llvm   -DMLIR_DIR=~/llvm_install/lib/cmake/mlir   ..




```
