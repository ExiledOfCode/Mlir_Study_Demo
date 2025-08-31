#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

int main() {
    mlir::MLIRContext context;
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    // 验证模块
    if (mlir::failed(mlir::verify(module))) {
        module->dump();
        return 1;
    }

    // 打印模块
    module->dump();
    return 0;
}
