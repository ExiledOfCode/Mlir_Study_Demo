

cd /home/wangjiyuan/dfnn_my_mlir/build/
make clean
cmake   -DLLVM_DIR=~/llvm_install/lib/cmake/llvm   -DMLIR_DIR=~/llvm_install/lib/cmake/mlir   ..
make -j8


/home/wangjiyuan/dfnn_my_mlir/setup_code_fix.sh

/home/wangjiyuan/dfnn_my_mlir/build/simple_mlir
