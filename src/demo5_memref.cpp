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

    // 创建一个函数：无参数，返回i32
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({}, {i32Type});
    mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "main", funcType);

    // 创建函数体（block）
    mlir::Block *block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);

    // 创建常量：0, 1, 2, 3, 4, 5 (i32类型)
    mlir::Value c0_i32 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(0));
    mlir::Value c1_i32 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(1));
    mlir::Value c2_i32 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(2));
    mlir::Value c3_i32 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(3));
    mlir::Value c4_i32 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(4));
    mlir::Value c5_i32 = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getI32IntegerAttr(5));

    // 将i32常量转换为index类型用于索引
    mlir::Value c0 = builder.create<mlir::arith::IndexCastOp>(
        builder.getUnknownLoc(), builder.getIndexType(), c0_i32);
    mlir::Value c1 = builder.create<mlir::arith::IndexCastOp>(
        builder.getUnknownLoc(), builder.getIndexType(), c1_i32);
    mlir::Value c2 = builder.create<mlir::arith::IndexCastOp>(
        builder.getUnknownLoc(), builder.getIndexType(), c2_i32);
    mlir::Value c3 = builder.create<mlir::arith::IndexCastOp>(
        builder.getUnknownLoc(), builder.getIndexType(), c3_i32);
    mlir::Value c4 = builder.create<mlir::arith::IndexCastOp>(
        builder.getUnknownLoc(), builder.getIndexType(), c4_i32);
    mlir::Value c5 = builder.create<mlir::arith::IndexCastOp>(
        builder.getUnknownLoc(), builder.getIndexType(), c5_i32);

    // 分配数组
    auto memrefType = mlir::MemRefType::get({5}, i32Type);
    mlir::Value array = builder.create<mlir::memref::AllocOp>(
        builder.getUnknownLoc(), memrefType);

    // 存储值到数组（使用index类型的索引）
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c1_i32,
                                          array, c0);
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c2_i32,
                                          array, c1);
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c3_i32,
                                          array, c2);
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c4_i32,
                                          array, c3);
    builder.create<mlir::memref::StoreOp>(builder.getUnknownLoc(), c5_i32,
                                          array, c4);

    // 创建for循环累加数组元素
    mlir::Value sum = builder
                          .create<mlir::scf::ForOp>(builder.getUnknownLoc(), c0,
                                                    c5, c1, c0_i32)
                          .getResult(0);
    mlir::Block *loopBody = sum.getDefiningOp<mlir::scf::ForOp>().getBody();
    builder.setInsertionPointToStart(loopBody);

    // 获取循环体的迭代变量和累加器
    mlir::Value i = loopBody->getArgument(0);   // 这是index类型
    mlir::Value acc = loopBody->getArgument(1); // 这是i32类型

    // 加载数组元素并累加
    mlir::Value elem =
        builder.create<mlir::memref::LoadOp>(builder.getUnknownLoc(), array, i);
    mlir::Value next =
        builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), acc, elem);
    builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), next);

    // 释放数组
    builder.setInsertionPointToEnd(block);
    builder.create<mlir::memref::DeallocOp>(builder.getUnknownLoc(), array);

    // 返回结果
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
