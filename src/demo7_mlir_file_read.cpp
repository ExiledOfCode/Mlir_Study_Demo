#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

int main() {
    // 创建 MLIR 上下文
    mlir::MLIRContext context;

    // 加载 func Dialect
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    // 加载 .mlir 文件
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFile("simple_func.mlir");
    if (std::error_code EC = fileOrErr.getError()) {
        llvm::errs() << "无法打开文件: " << EC.message() << "\n";
        return 1;
    }

    // 解析文件
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

    if (!module) {
        llvm::errs() << "解析 MLIR 文件失败\n";
        return 1;
    }

    // 验证模块
    if (failed(mlir::verify(*module))) {
        llvm::errs() << "模块验证失败\n";
        return 1;
    }

    // 打印模块
    module->print(llvm::outs());
    return 0;
}
