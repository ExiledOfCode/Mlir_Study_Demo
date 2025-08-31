#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

int main() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();

    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module =
        builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

    // 创建函数类型: (i32, i32) -> i32
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({i32Type, i32Type}, i32Type);

    // 插入函数
    builder.setInsertionPointToEnd(module.getBody());
    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                   "add_example", funcType);

    // 创建函数体块
    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // 获取参数
    mlir::Value a = entryBlock->getArgument(0);
    mlir::Value b = entryBlock->getArgument(1);

    // 创建加法
    mlir::Value sum =
        builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), a, b);

    // 返回
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), sum);

    // 验证并打印
    if (failed(mlir::verify(module))) {
        llvm::errs() << "模块验证失败\n";
        return 1;
    }
    module.print(llvm::outs());
    return 0;
}
