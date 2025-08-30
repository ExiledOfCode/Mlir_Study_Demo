#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

int main() {
    // 创建MLIR上下文并加载方言
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();

    // 创建OpBuilder和模块
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    // 创建一个函数：无参数，返回i32
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({}, {i32Type});
    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);

    // 创建函数体（block）
    mlir::Block *block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);

    // 创建常量：0（初值和下界），1（步长），10（上界）
    mlir::Value c0 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(0));
    mlir::Value c1 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(1));
    mlir::Value c10 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(10));

    // 创建for循环
    mlir::Value sum = builder
                          .create<mlir::scf::ForOp>(builder.getUnknownLoc(),
                                                    c0,  // 下界
                                                    c10, // 上界
                                                    c1,  // 步长
                                                    c0   // 初始累加器值
                                                    )
                          .getResult(0);

    // 创建循环体
    mlir::Block *loopBody = sum.getDefiningOp<mlir::scf::ForOp>().getBody();
    builder.setInsertionPointToStart(loopBody);

    // 获取循环体的累加器（iter_args）
    mlir::Value acc = loopBody->getArgument(1); // 第0个是迭代变量i，第1个是acc

    // 在循环体内执行加法
    mlir::Value next =
        builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), acc, c1);

    // yield下一次迭代的值
    builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), next);

    // 返回循环结果
    builder.setInsertionPointToEnd(block);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), sum);

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
