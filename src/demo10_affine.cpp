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

    // 函数类型: () -> i32
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({}, i32Type);

    // 创建函数
    builder.setInsertionPointToEnd(module.getBody());
    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                   "const_add", funcType);

    // 创建函数体
    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // 常量 2 和 3
    mlir::Value c2 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(2));
    mlir::Value c3 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(3));

    // 加法
    mlir::Value sum =
        builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), c2, c3);

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
