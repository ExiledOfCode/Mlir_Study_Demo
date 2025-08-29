#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h" // 替换原来的Module.h
#include "mlir/IR/MLIRContext.h"

int main() {
    // 1. 创建MLIR上下文
    mlir::MLIRContext context;

    // 2. 创建一个空的MLIR模块
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

    // 3. 创建一个IRBuilder来构建操作
    mlir::OpBuilder builder(&context);

    // 4. 设置模块的插入点
    builder.setInsertionPointToStart(module->getBody());

    // 5. 打印模块
    llvm::outs() << "我们的第一个MLIR模块:\n";
    module->dump();

    return 0;
}
