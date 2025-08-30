#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();

    // 创建OpBuilder和模块
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    // 创建函数：无参数，返回memref<2x2xi32>
    auto i32Type = builder.getI32Type();
    auto indexType = builder.getIndexType();
    auto memrefType = mlir::MemRefType::get({2, 2}, i32Type);
    auto funcType = builder.getFunctionType({}, {memrefType});
    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "matrix_multiply", funcType);

    // 创建函数体
    mlir::Block *block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);

    // 创建常量
    mlir::Value c0_i32 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(0));
    mlir::Value c1_i32 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(1));
    mlir::Value c2_i32 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(2));

    // 转换为index类型
    mlir::Value c0 = builder.create<mlir::arith::IndexCastOp>(
        builder.getUnknownLoc(), indexType, c0_i32);
    mlir::Value c1 = builder.create<mlir::arith::IndexCastOp>(
        builder.getUnknownLoc(), indexType, c1_i32);
    mlir::Value c2 = builder.create<mlir::arith::IndexCastOp>(
        builder.getUnknownLoc(), indexType, c2_i32);

    // 分配矩阵A, B, C
    mlir::Value A = builder.create<mlir::memref::AllocOp>(
        builder.getUnknownLoc(), memrefType);
    mlir::Value B = builder.create<mlir::memref::AllocOp>(
        builder.getUnknownLoc(), memrefType);
    mlir::Value C = builder.create<mlir::memref::AllocOp>(
        builder.getUnknownLoc(), memrefType);

    // 初始化矩阵A和B（示例值）
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c1_i32, A,
                                          mlir::ValueRange{c0, c0});
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c2_i32, A,
                                          mlir::ValueRange{c0, c1});
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c1_i32, A,
                                          mlir::ValueRange{c1, c0});
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c2_i32, A,
                                          mlir::ValueRange{c1, c1});
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c1_i32, B,
                                          mlir::ValueRange{c0, c0});
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c2_i32, B,
                                          mlir::ValueRange{c0, c1});
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c1_i32, B,
                                          mlir::ValueRange{c1, c0});
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c2_i32, B,
                                          mlir::ValueRange{c1, c1});

    // 嵌套循环计算C = A × B
    // 外层循环（i）
    mlir::scf::ForOp outerLoop =
        builder.create<mlir::scf::ForOp>(builder.getUnknownLoc(), c0, c2, c1);
    mlir::Block *outerBody = outerLoop.getBody();
    builder.setInsertionPointToStart(outerBody);
    mlir::Value i = outerBody->getArgument(0);

    // 中间层循环（j）
    mlir::scf::ForOp innerLoop =
        builder.create<mlir::scf::ForOp>(builder.getUnknownLoc(), c0, c2, c1);
    mlir::Block *innerBody = innerLoop.getBody();

    llvm::outs() << '\n';
    builder.setInsertionPointToStart(innerBody);
    mlir::Value j = innerBody->getArgument(0);

    llvm::outs() << '\n';
    // 创建常量0用于初始化累加器
    mlir::Value zero = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(0));

    llvm::outs() << '\n';
    // 内层循环（k）计算C[i][j]的点积
    mlir::scf::ForOp kLoop = builder.create<mlir::scf::ForOp>(
        builder.getUnknownLoc(), c0, c2, c1, zero);
    mlir::Value sum = kLoop->getResult(0);
    mlir::Block *kLoopBody = sum.getDefiningOp<mlir::scf::ForOp>().getBody();
    builder.setInsertionPointToStart(kLoopBody);

    llvm::outs() << '\n';
    mlir::Value k = kLoopBody->getArgument(0);
    mlir::Value acc = kLoopBody->getArgument(1);

    // 计算A[i][k] * B[k][j]
    mlir::Value a = builder.create<mlir::memref::LoadOp>(
        builder.getUnknownLoc(), A, mlir::ValueRange{i, k});
    mlir::Value b = builder.create<mlir::memref::LoadOp>(
        builder.getUnknownLoc(), B, mlir::ValueRange{k, j});
    mlir::Value prod =
        builder.create<mlir::arith::MulIOp>(builder.getUnknownLoc(), a, b);
    mlir::Value next =
        builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), acc, prod);
    builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), next);

    llvm::outs() << '\n';
    // 将点积结果存储到C[i][j]
    builder.setInsertionPointAfter(kLoop);
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), sum, C,
                                          mlir::ValueRange{i, j});

    llvm::outs() << '\n';
    // 回到主块
    builder.setInsertionPointToEnd(block);

    // 释放矩阵
    builder.create<mlir::memref::DeallocOp>(builder.getUnknownLoc(), A);
    builder.create<mlir::memref::DeallocOp>(builder.getUnknownLoc(), B);

    // 返回C
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), C);

    // 将函数插入模块
    module.push_back(func);

    // 验证模块
    if (mlir::failed(mlir::verify(module))) {
        module->dump();
        llvm::errs() << "Module verification failed\n";
        return 1;
    }

    // 打印模块
    module->dump();
    return 0;
}
