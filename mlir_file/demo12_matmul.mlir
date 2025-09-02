module {
  func.func @matmul(%A: memref<2x2xi32>, %B: memref<2x2xi32>, %C: memref<2x2xi32>) {
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 2 {
        %c0 = arith.constant 0 : i32
        %sum = affine.for %k = 0 to 2 iter_args(%acc = %c0) -> i32 {
          %a = affine.load %A[%i, %k] : memref<2x2xi32>
          %b = affine.load %B[%k, %j] : memref<2x2xi32>
          %prod = arith.muli %a, %b : i32
          %new_acc = arith.addi %acc, %prod : i32
          affine.yield %new_acc : i32
        }
        affine.store %sum, %C[%i, %j] : memref<2x2xi32>
      }
    }
    return
  }
}
