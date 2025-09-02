#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

int main() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();

    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module =
        builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

    // 函数类型: (i32) -> i32
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({i32Type}, i32Type);

    // 创建函数
    builder.setInsertionPointToEnd(module.getBody());
    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                   "sum_loop", funcType);

    // 创建函数体
    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // 获取参数
    mlir::Value n = entryBlock->getArgument(0);

    // 常量 0 和 1
    mlir::Value c0 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(0));
    mlir::Value c1 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), i32Type, builder.getI32IntegerAttr(1));

    // 创建 for 循环
    auto forOp = builder.create<mlir::scf::ForOp>(builder.getUnknownLoc(), c0,
                                                  n, c1, c0);
    {
        // 循环体
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(forOp.getBody());
        mlir::Value sum = forOp.getInductionVar();
        mlir::Value newSum = builder.create<mlir::arith::AddIOp>(
            builder.getUnknownLoc(), sum, c1);
        builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), newSum);
    }

    // 返回结果
    builder.setInsertionPointAfter(forOp);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                         forOp.getResult(0));

    // 验证并打印
    if (failed(mlir::verify(module))) {
        llvm::errs() << "模块验证失败\n";
        return 1;
    }
    module.print(llvm::outs());
    return 0;
}
