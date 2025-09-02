#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

int main() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::affine::AffineDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();

    mlir::OpBuilder builder(&context);
    mlir::ModuleOp module =
        builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

    // memref 类型: 2x2xi32
    auto memrefType = mlir::MemRefType::get({2, 2}, builder.getI32Type());
    auto funcType =
        builder.getFunctionType({memrefType, memrefType, memrefType}, {});

    // 创建函数
    builder.setInsertionPointToEnd(module.getBody());
    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                   "matmul", funcType);

    // 函数体
    mlir::Block *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value A = entryBlock->getArgument(0);
    mlir::Value B = entryBlock->getArgument(1);
    mlir::Value C = entryBlock->getArgument(2);

    // 常量 0 和 2
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(
        builder.getUnknownLoc(), 0);
    auto c2 = builder.create<mlir::arith::ConstantIndexOp>(
        builder.getUnknownLoc(), 2);

    int64_t c0_i32 = c0.value();
    int64_t c2_i32 = c2.value();
    // 外层循环 i
    auto outerLoop = builder.create<mlir::affine::AffineForOp>(
        builder.getUnknownLoc(), c0_i32, c2_i32, 1);
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(outerLoop.getBody());
        mlir::Value i = outerLoop.getInductionVar();

        // 中层循环 j
        auto innerLoop = builder.create<mlir::affine::AffineForOp>(
            builder.getUnknownLoc(), c0_i32, c2_i32, 1);
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(innerLoop.getBody());
            mlir::Value j = innerLoop.getInductionVar();

            // 常量 0 用于累加
            auto zero = builder.create<mlir::arith::ConstantOp>(
                builder.getUnknownLoc(), builder.getI32Type(),
                builder.getI32IntegerAttr(0));

            // 内层循环 k
            auto kLoop = builder.create<mlir::affine::AffineForOp>(
                builder.getUnknownLoc(), c0_i32, c2_i32, 1, zero->getResult(0));
            {
                mlir::OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToStart(kLoop.getBody());
                mlir::Value k = kLoop.getInductionVar();
                mlir::Value acc = kLoop.getRegionIterArgs()[0];

                // 加载 A[i,k] 和 B[k,j]
                auto a = builder.create<mlir::affine::AffineLoadOp>(
                    builder.getUnknownLoc(), A, mlir::ValueRange{i, k});
                auto b = builder.create<mlir::affine::AffineLoadOp>(
                    builder.getUnknownLoc(), B, mlir::ValueRange{k, j});

                // 乘法和加法
                auto prod = builder.create<mlir::arith::MulIOp>(
                    builder.getUnknownLoc(), a, b);
                auto newAcc = builder.create<mlir::arith::AddIOp>(
                    builder.getUnknownLoc(), acc, prod);

                // 返回累加器
                builder.create<mlir::affine::AffineYieldOp>(
                    builder.getUnknownLoc(), newAcc->getResult(0));
            }

            // 存储结果到 C[i,j]
            builder.create<mlir::affine::AffineStoreOp>(builder.getUnknownLoc(),
                                                        kLoop.getResult(0), C,
                                                        mlir::ValueRange{i, j});
        }
    }

    // 返回
    builder.setInsertionPointAfter(outerLoop);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    // 验证并打印
    if (failed(mlir::verify(module))) {
        llvm::errs() << "模块验证失败\n";
        return 1;
    }
    module.print(llvm::outs());
    return 0;
}
