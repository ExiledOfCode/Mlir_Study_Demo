#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

int main() {
    // 创建MLIR上下文并加载方言
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();

    // 创建OpBuilder和模块
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    // 创建一个函数：无参数，返回i32
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({}, {i32Type}); // 无输入，输出i32
    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);

    // 创建函数体（block）
    mlir::Block *block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);

    // 创建常量42
    mlir::Value constant = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(42));

    // 返回常量
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), constant);

    // 将函数插入模块
    module.push_back(func);

    // 验证模块
    if (mlir::failed(mlir::verify(module))) {
        module->dump();
        return 1;
    }

    // 打印模块
    module->dump();
    return 0;
}
